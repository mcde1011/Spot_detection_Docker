import time
from datetime import datetime
from pathlib import Path
import threading

import cv2
import numpy as np
import requests
from requests.auth import HTTPDigestAuth
from requests.adapters import HTTPAdapter

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image, CompressedImage

from rclpy.executors import MultiThreadedExecutor
import concurrent.futures

from collections import deque

from cv_bridge import CvBridge


class RicohPublisher(Node):
    def __init__(self):
        super().__init__("ricoh_theta_publisher")

        # -----------------------------
        # Parameter / Konfiguration
        # -----------------------------
        # self.declare_parameter("camera_ip", "10.42.0.17") # Laptop Hotspot
        self.declare_parameter("camera_ip", "192.168.80.101")
        self.declare_parameter("username", "THETAYR30101068")
        self.declare_parameter("password", "30101068")

        # Ausgabe / Bandbreite
        self.declare_parameter("processing_mode", "stitch_local")  # "stitch_local" | "camera_stitch"
        self.declare_parameter("timer_period", 1.0)                # s pro Bild
        self.declare_parameter("output_width", 1280)
        self.declare_parameter("output_height", 640)
        self.declare_parameter("jpeg_quality", 65)
        self.declare_parameter("publish_compressed", True)         # CompressedImage statt Image

        # Orientierung / Stitching-Qualität (nur für stitch_local)
        self.declare_parameter("blend_width_deg", 12.0)
        self.declare_parameter("flip_vertical", True)
        self.declare_parameter("yaw_offset_deg", 0.0)
        self.declare_parameter("swap_halves", False)

        # Speicherung (optional)
        self.declare_parameter("save_enabled", False)
        self.declare_parameter("save_dir", "./ricoh_theta_views")

        # Parameter lesen
        self.CAMERA_IP = self.get_parameter("camera_ip").get_parameter_value().string_value
        self.USERNAME = self.get_parameter("username").get_parameter_value().string_value
        self.PASSWORD = self.get_parameter("password").get_parameter_value().string_value

        self.processing_mode = self.get_parameter("processing_mode").get_parameter_value().string_value
        self.timer_period = float(self.get_parameter("timer_period").value)
        self.output_size = (
            int(self.get_parameter("output_width").value),
            int(self.get_parameter("output_height").value),
        )
        self.jpeg_quality = int(self.get_parameter("jpeg_quality").value)
        self.publish_compressed = bool(self.get_parameter("publish_compressed").value)

        self.blend_width_deg = float(self.get_parameter("blend_width_deg").value)
        self.flip_vertical = bool(self.get_parameter("flip_vertical").value)
        self.yaw_offset_deg = float(self.get_parameter("yaw_offset_deg").value)
        self.swap_halves = bool(self.get_parameter("swap_halves").value)

        self.save_enabled = bool(self.get_parameter("save_enabled").value)
        self.save_base_dir = Path(self.get_parameter("save_dir").get_parameter_value().string_value)
        self.save_base_dir.mkdir(parents=True, exist_ok=True)

        # Auth / HTTP Session mit Timeouts
        self.auth = HTTPDigestAuth(self.USERNAME, self.PASSWORD)
        self._session = requests.Session()
        # etwas großzügigeres Timeout
        self._timeout = (5.0, 20.0)  # (connect, read)
        # kleiner Retry-Puffer auf der Session
        self._session.mount('http://', HTTPAdapter(pool_connections=4, pool_maxsize=8, max_retries=2))

        # QoS
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )

        # Bewegungsstatus (optional via Odom-Subscription nutzbar)
        self.robot_moving = False
        self.robot_was_moved = False

        # Publisher (nur ein Ausgabebild)
        if self.publish_compressed:
            self.pub_view = self.create_publisher(CompressedImage, "/ricoh_theta/image/compressed", qos)
        else:
            self.pub_view = self.create_publisher(Image, "/ricoh_theta/image", qos)

        self.bridge = CvBridge()
        self.counter = 0
        self._last_capture_time = 0.0  # request_to_download (für Log)

        # Pipeline: getrennte Pools für Steuerung und Download/Verarbeitung
        self.control_pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)   # takePicture + status + delete
        self.download_pool = concurrent.futures.ThreadPoolExecutor(max_workers=2)  # download + processing
        self._lock = threading.Lock()
        self._capture_in_flight = False  # genau eine Aufnahme gleichzeitig

        # Serialisierungs-Lock für alle /osc/commands/* Posts
        self._exec_lock = threading.Lock()

        # Kamera je nach Modus konfigurieren
        self.configure_mode()

        # Annahme-Startgröße für Dual-Fisheye (nur relevant für stitch_local)
        self.input_wh = (5376, 2688)

        # Remap-Maps vorbereiten (für stitch_local)
        self._rebuild_maps()

        # Timer
        self.timer = self.create_timer(self.timer_period, self.capture_and_publish)

        # Timestamps
        self.timestamp = deque()

    # -----------------------------
    # HTTP Hilfsmethoden
    # -----------------------------

    def _post_with_retry(self, url: str, json_payload: dict, retries: int = 2, backoff_s: float = 0.5):
        """
        Serialisiert via _exec_lock, loggt 4xx-Body und macht Backoff bei 409/503.
        """
        last_exc = None
        for attempt in range(retries + 1):
            try:
                with self._exec_lock:
                    resp = self._session.post(url, json=json_payload, auth=self.auth, timeout=self._timeout)
                if resp.status_code == 200:
                    return resp
                # Busy/Conflict -> Backoff
                if resp.status_code in (409, 503):
                    time.sleep(backoff_s * (2 ** attempt))
                    continue
                # 4xx/5xx: Body loggen für Diagnose, dann Exception werfen
                self.get_logger().error(f"HTTP {resp.status_code} @ {url} | payload={json_payload} | body={resp.text}")
                resp.raise_for_status()
            except requests.RequestException as e:
                last_exc = e
                time.sleep(backoff_s * (2 ** attempt))
        # wenn alle Versuche fehlschlagen
        if last_exc:
            raise last_exc
        raise RuntimeError("Unbekannter Fehler in _post_with_retry")

    def _post_execute(self, payload: dict):
        url = f"http://{self.CAMERA_IP}/osc/commands/execute"
        return self._post_with_retry(url, payload)

    def _post_status(self, cid: str):
        url = f"http://{self.CAMERA_IP}/osc/commands/status"
        return self._post_with_retry(url, {"id": cid})

    def _get(self, url: str):
        # GET (Download) darf parallel laufen; kein _exec_lock nötig
        resp = self._session.get(url, auth=self.auth, timeout=self._timeout)
        resp.raise_for_status()
        return resp

    def _delete_file(self, file_url: str):
        """Löscht die angegebene Datei auf der Kamera (serialisiert, mit kleinem Delay)."""
        try:
            # sehr kurzer Delay – einige Firmwares brauchen nach Download einen Moment
            time.sleep(0.2)
            payload = {"name": "camera.delete", "parameters": {"fileUrls": [file_url]}}
            self._post_execute(payload)
            self.get_logger().debug(f"camera.delete OK: {file_url}")
        except requests.HTTPError as e:
            code = getattr(e.response, "status_code", None)
            self.get_logger().warning(f"camera.delete fehlgeschlagen (HTTP {code}): {e}")
        except Exception as e:
            self.get_logger().warning(f"camera.delete Fehler: {e}")

    # -----------------------------
    # Kamera-Setup je Modus
    # -----------------------------
    def configure_mode(self):
        """
        - stitch_local:     _imageStitching = "none"   (Kamera liefert Dual-Fisheye)
        - camera_stitch:    _imageStitching = "dynamicSemiAuto" (Kamera liefert bereits equirect, sofern unterstützt)
        """
        try:
            if self.processing_mode == "camera_stitch":
                opt_val = "dynamicSemiAuto"  # je nach Modell ggf. "on"
            else:
                opt_val = "none"

            options_data = {"name": "camera.setOptions", "parameters": {"options": {"_imageStitching": opt_val}}}
            self._post_execute(options_data)
            self.get_logger().info(f"_imageStitching={opt_val} (processing_mode={self.processing_mode})")
        except requests.RequestException as e:
            self.get_logger().error(f"HTTP-Fehler bei setOptions: {e}")
        except Exception as e:
            self.get_logger().error(f"Fehler beim Konfigurieren: {e}")

    # -----------------------------
    # Hauptpipeline (Timer)
    # -----------------------------
    def capture_and_publish(self):
        # Nicht auslösen, wenn Roboter in Bewegung
        if self.robot_moving:
            return
        with self._lock:
            # Nur eine Aufnahme gleichzeitig auslösen/status-pollen
            if self._capture_in_flight:
                return
            self._capture_in_flight = True
        # Auslösen/Status-Polling seriell im Steuerpool
        self.control_pool.submit(self._trigger_and_enqueue_download)

    # -----------------------------
    # Auslösen + Status polling; Download-Job einreihen
    # -----------------------------
    def _trigger_and_enqueue_download(self):
        t_start = time.time()
        file_url = None
        try:
            timestamp = self.get_clock().now().to_msg()
            self.timestamp.append(timestamp)
            take = self._post_execute({"name": "camera.takePicture"})
            data = take.json()
            cid = data.get("id")
            if not cid:
                self.get_logger().error(f"Keine Command-ID erhalten: {data}")
                return

            # Status poll bis done (korrekt über /osc/commands/status)
            deadline = time.time() + 20.0
            while time.time() < deadline:
                st = self._post_status(cid)
                sdata = st.json()
                if sdata.get("state") == "done":
                    file_url = sdata.get("results", {}).get("fileUrl")
                    break
                time.sleep(0.3)

            if not file_url:
                self.get_logger().error("Aufnahme nicht abgeschlossen (Timeout).")
                return

            # Download/Verarbeitung asynchron starten
            self.download_pool.submit(self._download_process_publish, file_url, t_start)

        except requests.RequestException as e:
            self.get_logger().error(f"HTTP-Fehler bei Aufnahme: {e}")
        except Exception as e:
            self.get_logger().error(f"Fehler bei Aufnahme: {e}")
        finally:
            # Freigeben, damit der Timer das nächste Auslösen anstoßen kann
            with self._lock:
                self._capture_in_flight = False

    # -----------------------------
    # Download + Verarbeitung + Publish + Delete
    # -----------------------------
    def _download_process_publish(self, file_url: str, t_start: float):
        # 1) Download
        try:
            r = self._get(file_url)
            img = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                self.get_logger().error("imdecode lieferte None.")
                return
        except Exception as e:
            self.get_logger().error(f"Download/Decode-Fehler: {e}")
            return

        # 2) Nur für lokalen Stitch relevant: Map-Größen ggf. neu berechnen
        try:
            if self.processing_mode == "stitch_local":
                h, w = img.shape[:2]
                if (w, h) != self.input_wh:
                    self.get_logger().warning(f"Unerwartete Rohgröße: {(w, h)} – berechne Maps neu.")
                    self.input_wh = (w, h)
                    self._rebuild_maps()
        except Exception as e:
            self.get_logger().warning(f"Map/Resize-Hinweis: {e}")

        # 3) Verarbeitung
        t_proc0 = time.time()
        mode = self.processing_mode
        try:
            if mode == "stitch_local":
                out_img = self.stitch_dual_fisheye_fast(img)
                if out_img is None:
                    self.get_logger().error("Stitching fehlgeschlagen.")
                    return
            elif mode == "camera_stitch":
                out_img = cv2.resize(img, self.output_size, interpolation=cv2.INTER_AREA)
            else:
                self.get_logger().warning(f"Unbekannter processing_mode='{mode}', fallback stitch_local.")
                out_img = self.stitch_dual_fisheye_fast(img)
                if out_img is None:
                    return
        except Exception as e:
            self.get_logger().error(f"Verarbeitungsfehler: {e}")
            return
        t_proc1 = time.time()

        # 4) Publish
        if not self.robot_was_moved:
            stamp = self.timestamp.popleft()
            self._publish_img("view", out_img, stamp)

        # 5) Optional speichern
        if self.save_enabled:
            try:
                now = datetime.now()
                day_dir = self.save_base_dir / now.strftime("%Y-%m-%d")
                day_dir.mkdir(parents=True, exist_ok=True)
                ts_str = now.strftime("%Y-%m-%d_%H-%M-%S.%f")
                if self.publish_compressed:
                    cv2.imwrite(
                        str(day_dir / f"{ts_str}_{mode}.jpg"),
                        out_img,
                        [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
                    )
                else:
                    cv2.imwrite(str(day_dir / f"{ts_str}_{mode}.png"), out_img)
            except Exception as e:
                self.get_logger().warning(f"Speichern fehlgeschlagen: {e}")

        # 6) Datei auf Kamera löschen (seriell im control_pool)
        try:
            self.control_pool.submit(self._delete_file, file_url)
        except Exception as e:
            self.get_logger().warning(f"Enqueue delete fehlgeschlagen: {e}")

        # 7) Zeiten loggen
        request_to_download = max(0.0, t_proc0 - t_start)  # grob: Auslösen -> Download fertig (bis Start Verarbeitung)
        t_stitch = max(0.0, t_proc1 - t_proc0)
        t_total = (time.time() - t_start)
        self._last_capture_time = request_to_download  # für Abwärtskompatibilität

        self.get_logger().info(
            f"[{mode}] Bild {self.counter} | request_to_download≈{request_to_download:.3f}s | "
            f"request_to_download_plus_processing≈{request_to_download + t_stitch:.3f}s "
            f"(processing={t_stitch:.3f}s) | total_node={t_total:.3f}s (pipeline)"
        )
        self.counter += 1

    # -----------------------------
    # Publish-Helfer
    # -----------------------------
    def _publish_img(self, kind: str, img: np.ndarray, stamp):
        if self.robot_was_moved:
            return
        if self.publish_compressed:
            ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
            if not ok:
                self.get_logger().error(f"JPEG-Encode fehlgeschlagen ({kind}).")
                return
            msg = CompressedImage()
            msg.header.stamp = stamp
            msg.header.frame_id = kind
            msg.format = "jpeg"
            msg.data = np.asarray(buf).tobytes()
            self.pub_view.publish(msg)
        else:
            ros_img = self.bridge.cv2_to_imgmsg(img, encoding="bgr8")
            ros_img.header.stamp = stamp
            ros_img.header.frame_id = kind
            self.pub_view.publish(ros_img)

    # -----------------------------
    # Schnelles Stitching via Remap (lokal)
    # -----------------------------
    @staticmethod
    def _safe_div(a, b, eps=1e-8):
        return a / np.clip(b, eps, None)

    def _rebuild_maps(self):
        self.map_left_x, self.map_left_y, self.map_right_x, self.map_right_y, self.weight_left = \
            self._precompute_equirect_maps(
                in_wh=self.input_wh,
                out_wh=self.output_size,
                blend_width_deg=self.blend_width_deg,
                flip_vertical=self.flip_vertical,
                yaw_offset_deg=self.yaw_offset_deg,
            )

    def _precompute_equirect_maps(self, in_wh, out_wh, blend_width_deg=10.0, flip_vertical=False, yaw_offset_deg=0.0):
        in_w, in_h = in_wh
        out_w, out_h = out_wh

        half_w = in_w // 2
        cx_l, cy_l = half_w * 0.5, in_h * 0.5
        cx_r, cy_r = half_w * 0.5, in_h * 0.5
        rad = min(half_w, in_h) * 0.5

        xs = np.linspace(0.0, 1.0, out_w, dtype=np.float32)
        ys = np.linspace(0.0, 1.0, out_h, dtype=np.float32)
        uu, vv = np.meshgrid(xs, ys)

        yaw = np.deg2rad(yaw_offset_deg)
        theta = (uu * (2.0 * np.pi) + yaw) % (2.0 * np.pi)
        phi = vv * np.pi

        vx = np.sin(phi) * np.cos(theta)
        vy = np.sin(phi) * np.sin(theta)
        vz = np.cos(phi)
        if flip_vertical:
            vz = -vz

        bw_rad = np.deg2rad(blend_width_deg)
        sign = np.tanh(vx / np.sin(bw_rad))
        weight_left = (0.5 * (1.0 + sign)).astype(np.float32)

        f = rad / (np.pi * 0.5)

        alpha_l = np.arccos(np.clip(vx, -1.0, 1.0))
        sin_alpha_l = np.sin(alpha_l)
        nx_l = self._safe_div(vy, sin_alpha_l)
        ny_l = self._safe_div(vz, sin_alpha_l)
        r_l = f * alpha_l
        map_left_x = (nx_l * r_l + cx_l).astype(np.float32)
        map_left_y = (ny_l * r_l + cy_l).astype(np.float32)

        alpha_r = np.arccos(np.clip(-vx, -1.0, 1.0))
        sin_alpha_r = np.sin(alpha_r)
        nx_r = self._safe_div(-vy, sin_alpha_r)
        ny_r = self._safe_div(vz,  sin_alpha_r)
        r_r = f * alpha_r
        map_right_x = (nx_r * r_r + cx_r).astype(np.float32)
        map_right_y = (ny_r * r_r + cy_r).astype(np.float32)

        valid_l = (alpha_l <= (np.pi * 0.5))
        valid_r = (alpha_r <= (np.pi * 0.5))
        map_left_x[~valid_l] = -1
        map_left_y[~valid_l] = -1
        map_right_x[~valid_r] = -1
        map_right_y[~valid_r] = -1

        return map_left_x, map_left_y, map_right_x, map_right_y, weight_left

    def stitch_dual_fisheye_fast(self, dual_fisheye_img: np.ndarray) -> np.ndarray:
        try:
            in_h, in_w = dual_fisheye_img.shape[:2]
            half_w = in_w // 2
            left = dual_fisheye_img[:, :half_w]
            right = dual_fisheye_img[:, half_w:]
            if self.swap_halves:
                left, right = right, left

            eq_left = cv2.remap(
                left, self.map_left_x, self.map_left_y,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            eq_right = cv2.remap(
                right, self.map_right_x, self.map_right_y,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )

            w = self.weight_left
            w3 = w if w.ndim == 3 else np.dstack([w, w, w])
            out = (eq_left.astype(np.float32) * w3 + eq_right.astype(np.float32) * (1.0 - w3))
            out = np.clip(out, 0, 255).astype(np.uint8)

            # Sicherheit: auf Zielgröße bringen (Maps sollten bereits passen)
            if (out.shape[1], out.shape[0]) != self.output_size:
                out = cv2.resize(out, self.output_size, interpolation=cv2.INTER_AREA)
            return out

        except Exception as e:
            self.get_logger().error(f"Fehler beim schnellen Stitching: {e}")
            return None
        
    def destroy_node(self):
        # Timer stoppen
        try:
            self.timer.cancel()
        except Exception:
            pass

        # ThreadPools herunterfahren
        try:
            self.control_pool.shutdown(wait=True, cancel_futures=True)
        except Exception:
            pass
        try:
            self.download_pool.shutdown(wait=True, cancel_futures=True)
        except Exception:
            pass

        # Super-Methode aufrufen
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = RicohPublisher()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    try:
        rclpy.spin(node, executor=executor)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
