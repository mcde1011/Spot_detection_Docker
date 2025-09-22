#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
from typing import Dict, Tuple

import numpy as np
import torch
import cv2

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import Image, CompressedImage
from vision_msgs.msg import (
    Detection2D, Detection2DArray,
    ObjectHypothesisWithPose, BoundingBox2D,
)
from cv_bridge import CvBridge
import py360convert

from detection_msgs.msg import LabeledDetections # Custom-Message (Paket detection_msgs, Msg LabeledDetections)

# Ultralytics YOLO
try:
    from ultralytics import YOLO
except ImportError as e:
    raise RuntimeError("Ultralytics nicht installiert: pip install ultralytics") from e


def _default_model_path() -> str:
    """Share-Verzeichnis bevorzugen, sonst Source-Fallback."""
    try:
        from ament_index_python.packages import get_package_share_directory
        share = get_package_share_directory('ricoh_theta_ros')
        cand = os.path.join(share, 'resource', 'best.pt')
        if os.path.exists(cand):
            return cand
    except Exception:
        pass
    return os.path.expanduser('~/ros2_ws/src/ricoh_theta_ros/resource/best.pt')


class YoloDetector(Node):
    """
    Abonniert EIN Topic (/ricoh_theta/image), erzeugt sechs Views und publiziert pro View:
      /detections/<view>                  (vision_msgs/Detection2DArray, immer)
      /detections/<view>/annotated[... ]  (Image/CompressedImage, nur bei >=1 Box)
      /detections/<view>/labeled          (detection_msgs/LabeledDetections, immer)
    """

    def __init__(self):
        super().__init__("yolo_detector")

        # Parameter
        self.declare_parameter("model_path", _default_model_path())
        self.declare_parameter("input_topic", "/ricoh_theta/image/compressed")
        self.declare_parameter("output_namespace", "/detections")
        self.declare_parameter("publish_annotated", True)  # nur bei >=1 Box
        self.declare_parameter("conf", 0.6)
        self.declare_parameter("iou", 0.45)
        self.declare_parameter("imgsz", 640)
        self.declare_parameter("device", "auto")     # "cpu", "0", "cuda:0", "auto"
        self.declare_parameter("device_index", -1)   # erlaubt int-Override
        self.declare_parameter("half", True)         # FP16 auf CUDA
        self.declare_parameter("publish_annotated_compressed", True)
        self.declare_parameter("annotated_jpeg_quality", 70)

        # Parameter lesen
        model_path = self.get_parameter("model_path").get_parameter_value().string_value
        input_topic: str = self.get_parameter("input_topic").get_parameter_value().string_value
        self.output_ns: str = self.get_parameter("output_namespace").get_parameter_value().string_value
        self.publish_annotated: bool = bool(self.get_parameter("publish_annotated").value)
        self.conf: float = float(self.get_parameter("conf").value)
        self.iou: float = float(self.get_parameter("iou").value)
        self.imgsz: int = int(self.get_parameter("imgsz").value)
        device_param = self.get_parameter("device").get_parameter_value().string_value
        device_index = int(self.get_parameter("device_index").value)
        self.use_half: bool = bool(self.get_parameter("half").value)
        self.publish_annotated_compressed = bool(self.get_parameter("publish_annotated_compressed").value)
        self.annotated_jpeg_quality = int(self.get_parameter("annotated_jpeg_quality").value)

        # Gerät ermitteln
        self.device = self._resolve_device(device_param, device_index)

        self.get_logger().info(f"Lade YOLO Modell: {model_path} auf {self.device}")
        self.model = YOLO(model_path)

        # Modell explizit verschieben (optional, Ultralytics macht es oft selbst)
        try:
            if self.device != "cpu":
                self.model.to(self.device)  # "0" oder "cuda:0"
        except Exception as e:
            self.get_logger().warn(f"Konnte Modell nicht explizit verschieben ({e}); Ultralytics verwaltet das ggf. intern.")

        # Fusing (falls unterstützt)
        try:
            self.model.fuse()
        except Exception:
            pass

        # QoS für Kameradaten
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.bridge = CvBridge()

        # Publisher pro View vorbereiten
        self.views_spec: Dict[str, Tuple[int, int]] = {
            "front": (0, 0),
            "right": (90, 0),
            "back": (180, 0),
            "left": (-90, 0),
            "up": (0, 90),
            "down": (0, -90),
        }
        self.fov_deg: int = 90
        self.out_size: Tuple[int, int] = (800, 800)

        self.pub_dets: Dict[str, rclpy.publisher.Publisher] = {}
        self.pub_imgs: Dict[str, rclpy.publisher.Publisher] = {}
        # NEU: Publisher für gebündelte Custom-Message
        self.pub_labeled: Dict[str, rclpy.publisher.Publisher] = {}

        for view in self.views_spec.keys():
            det_topic = f"{self.output_ns}/{view}"
            ann_topic = f"{self.output_ns}/{view}/annotated"
            labeled_topic = f"{self.output_ns}/{view}/labeled"

            self.pub_dets[view] = self.create_publisher(Detection2DArray, det_topic, 10)

            # optional: klassisches annotiertes Bild-Topic
            if self.publish_annotated:
                if self.publish_annotated_compressed:
                    # ROS-Konvention: /compressed-Suffix
                    self.pub_imgs[view] = self.create_publisher(CompressedImage, ann_topic + "/compressed", 10)
                else:
                    self.pub_imgs[view] = self.create_publisher(Image, ann_topic, 10)

            # NEU: immer publizierbares gebündeltes Topic
            self.pub_labeled[view] = self.create_publisher(LabeledDetections, labeled_topic, 10)

        # Klassenbezeichnungen
        self.class_names = {}
        try:
            if hasattr(self.model, "names"):
                self.class_names = self.model.names
        except Exception:
            pass

        # EIN Subscriber auf das Equirectangular-Topic
        self.sub_image = self.create_subscription(CompressedImage, input_topic, self.cb_image, qos)
        self.get_logger().info(f"Abonniere: {input_topic}")

    def _resolve_device(self, dev_str: str, dev_idx: int) -> str:
        if dev_idx >= 0:
            return str(dev_idx)
        dv = (dev_str or "").strip()
        if dv == "" or dv.lower() == "auto":
            return "0" if torch.cuda.is_available() else "cpu"
        return dv

    def cb_image(self, msg: CompressedImage):
        """Haupt-Callback: Equirectangular rein, 6 Views raus, dann YOLO pro View."""
        try:
            equirect = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().warn(f"cv_bridge Fehler (input): {e}")
            return
        # self.get_logger().info(f"Image erhalten")
        # 6 Perspektiven generieren
        views = self.generate_views(equirect)
        # Pro View Inferenz + Publish
        for view_name, frame in views.items():
            try:
                self._process_view(view_name, frame, msg.header)
            except Exception as e:
                self.get_logger().error(f"Fehler bei Verarbeitung View '{view_name}': {e}")

    def generate_views(self, img: np.ndarray) -> Dict[str, np.ndarray]:
        """Erzeuge perspektivische Projektionen aus Equirectangular."""
        views = {}
        for name, (yaw, pitch) in self.views_spec.items():
            persp_img = py360convert.e2p(
                img,
                fov_deg=self.fov_deg,
                u_deg=yaw,
                v_deg=pitch,
                out_hw=self.out_size
            )
            views[name] = persp_img
        return views

    def _process_view(self, view_name: str, frame: np.ndarray, header):
        """YOLO ausführen, Detections und Bilder publizieren (klassisch & gebündelt)."""

        t0 = time.time()

        # Inferenz
        half_flag = self.use_half and (self.device != "cpu")
        try:
            results = self.model.predict(
                source=frame,
                conf=self.conf,
                iou=self.iou,
                imgsz=self.imgsz,
                device=self.device,
                verbose=False,
                half=half_flag
            )
        except Exception as e:
            self.get_logger().error(f"YOLO predict Fehler ({view_name}): {e}")
            return

        # self.get_logger().info(f"YOLO Ergebnis: {results}")

        det_array = Detection2DArray()
        det_array.header = header

        annotated = None
        r = results[0] if len(results) > 0 else None

        num_boxes = 0
        if r is not None and getattr(r, "boxes", None) is not None and len(r.boxes) > 0:
            xyxy = r.boxes.xyxy.detach().cpu().numpy()
            confs = r.boxes.conf.detach().cpu().numpy()
            clss = r.boxes.cls.detach().cpu().numpy().astype(int)
            num_boxes = len(xyxy)

            if self.publish_annotated:
                annotated = frame.copy()

            for (x1, y1, x2, y2), score, cls in zip(xyxy, confs, clss):
                det = Detection2D()
                det.header = header

                cx = float((x1 + x2) / 2.0)
                cy = float((y1 + y2) / 2.0)
                w  = float(max(0.0, x2 - x1))
                h  = float(max(0.0, y2 - y1))

                # BoundingBox2D robust befüllen
                bbox = BoundingBox2D()
                CenterCls = type(bbox.center)
                center = CenterCls()
                if hasattr(center, 'position'):
                    center.position.x = cx
                    center.position.y = cy
                    center.theta = 0.0
                else:
                    for name, val in (('x', cx), ('y', cy), ('theta', 0.0),
                                      ('x_', cx), ('y_', cy), ('theta_', 0.0)):
                        if hasattr(center, name):
                            setattr(center, name, val)
                bbox.center = center
                bbox.size_x = w
                bbox.size_y = h
                det.bbox = bbox

                hyp = ObjectHypothesisWithPose()
                hyp.hypothesis.class_id = str(cls)
                hyp.hypothesis.score = float(score)
                det.results.append(hyp)

                det_array.detections.append(det)

                # Optionales Zeichnen
                if annotated is not None:
                    label = self.class_names.get(cls, str(cls))
                    self.get_logger().info(
                    f"[{view_name}] Detected: {label} ({score:.2f}) "
                    f"@ (x1={int(x1)}, y1={int(y1)}, x2={int(x2)}, y2={int(y2)})"
                )
                    x1i, y1i, x2i, y2i = int(x1), int(y1), int(x2), int(y2)
                    cv2.rectangle(annotated, (x1i, y1i), (x2i, y2i), (0, 255, 0), 2)
                    cv2.putText(annotated, f"{label} {score:.2f}",
                                (x1i, max(0, y1i - 6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Publish detections (immer) – bestehendes Verhalten
        pub_d = self.pub_dets.get(view_name)
        if pub_d is not None:
            pub_d.publish(det_array)

        # Klassisches annotiertes Topic (nur wenn >=1 Box & aktiviert)
        if self.publish_annotated and annotated is not None and view_name in self.pub_imgs and num_boxes > 0:
            if self.publish_annotated_compressed:
                ok, buf = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), self.annotated_jpeg_quality])
                if not ok:
                    self.get_logger().warn(f"JPEG-Encode für annotiertes Bild fehlgeschlagen ({view_name}).")
                else:
                    msg = CompressedImage()
                    msg.header = header
                    msg.format = "jpeg"
                    msg.data = np.asarray(buf).tobytes()
                    self.pub_imgs[view_name].publish(msg)
            else:
                out_msg = self.bridge.cv2_to_imgmsg(annotated, encoding="bgr8")
                out_msg.header = header
                self.pub_imgs[view_name].publish(out_msg)

        # NEU: Gebündelte Custom-Message (immer)
        try:
            out_comp = CompressedImage()
            out_comp.header = header
            out_comp.format = "jpeg"

            # annotiert falls vorhanden & >=1 Box, sonst Original
            if annotated is not None and num_boxes > 0:
                src_for_jpeg = annotated
                ok, buf = cv2.imencode(".jpg", src_for_jpeg, [int(cv2.IMWRITE_JPEG_QUALITY), self.annotated_jpeg_quality])
                if ok:
                    out_comp.data = np.asarray(buf).tobytes()
                else:
                    self.get_logger().warn(f"JPEG-Encode für LabeledDetections fehlgeschlagen ({view_name}).")
                    out_comp = None

                labeled_msg = LabeledDetections()
                labeled_msg.detections = det_array
                if out_comp is not None:
                    labeled_msg.img = out_comp

                pub_lbl = self.pub_labeled.get(view_name)
                if pub_lbl is not None:
                    pub_lbl.publish(labeled_msg)
        except Exception as e:
            self.get_logger().warn(f"Publizieren LabeledDetections fehlgeschlagen ({view_name}): {e}")

        # Debug: Laufzeit
        dt = time.time() - t0
        if dt > 0:
            fps = 1.0 / dt
            self.get_logger().debug(f"[{view_name}] boxes={num_boxes}  {fps:.1f} FPS ({dt*1000:.1f} ms)")

def main(args=None):
    rclpy.init(args=args)
    node = YoloDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("Programm wurde mit Strg+C beendet.")
    finally:
        node.destroy_node()
        if rclpy.ok():
            try:
                rclpy.shutdown()
            except Exception as e:
                print(f"Shutdown bereits durchgeführt oder fehlgeschlagen: {e}")


if __name__ == "__main__":
    main()
