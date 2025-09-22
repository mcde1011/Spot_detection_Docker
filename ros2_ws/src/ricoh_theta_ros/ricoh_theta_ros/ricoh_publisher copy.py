import requests
import numpy as np
import time
import cv2
import py360convert
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from requests.auth import HTTPDigestAuth

class RicohPublisher(Node):
    def __init__(self):
        super().__init__('ricoh_theta_publisher')

        # Kameraeinstellungen
        self.CAMERA_IP = "192.168.137.244"
        self.USERNAME = "THETAYR30101068"
        self.PASSWORD = "30101068"
        self.auth = HTTPDigestAuth(self.USERNAME, self.PASSWORD)

        # QoS passend für Kameradaten
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5
        )

        # Sechs getrennte Topics anlegen
        self.view_pubs = {
            "front": self.create_publisher(Image, "/ricoh_theta/front", qos),
            "right": self.create_publisher(Image, "/ricoh_theta/right", qos),
            "back":  self.create_publisher(Image, "/ricoh_theta/back", qos),
            "left":  self.create_publisher(Image, "/ricoh_theta/left", qos),
            "up":    self.create_publisher(Image, "/ricoh_theta/up", qos),
            "down":  self.create_publisher(Image, "/ricoh_theta/down", qos),
        }

        self.bridge = CvBridge()

        # Alle N Sekunden Bild aufnehmen und veröffentlichen
        self.timer = self.create_timer(5.0, self.capture_and_publish)

        self.counter = 0

    def capture_and_publish(self):
        # self.get_logger().info(f"Start Timestamp Bild {self.counter}: {time.time()}")
        # self.counter += 1
        # start_time = time.time()
        img = self.capture_image()
        if img is None:
            self.get_logger().error("Kein Bild erhalten.")
            return

        # time_elapsed = time.time() - start_time
        # self.get_logger().info(f"Dauer bis Bild erhalten: {time_elapsed}")

        # 1) 360° -> 6 perspektivische Ansichten
        views = self.generate_views(img)
        # time_elapsed = time.time() - start_time
        # self.get_logger().info(f"Dauer bis Bild geteilt: {time_elapsed}")

        # NEU: ein gemeinsamer Zeitstempel pro 6er-Paket
        stamp = self.get_clock().now().to_msg()

        # 2) HIER: jede Ansicht auf ihr eigenes Topic publishen
        for name, view_img in views.items():
            ros_img = self.bridge.cv2_to_imgmsg(view_img, encoding="bgr8")
            ros_img.header.stamp = stamp
            ros_img.header.frame_id = name  # 'front', 'right', ...
            self.view_pubs[name].publish(ros_img)
            # self.get_logger().info(f"Gesendet: /ricoh_theta/{name}")
        # time_elapsed = time.time() - start_time
        # self.get_logger().info(f"Dauer Aufnahme: {time_elapsed}")

    def capture_image(self):
        # start_time = time.time()
        # Foto aufnehmen
        take_resp = requests.post(
            f"http://{self.CAMERA_IP}/osc/commands/execute",
            json={"name": "camera.takePicture"},
            auth=self.auth
        )
        if take_resp.status_code != 200:
            self.get_logger().error("Fehler beim Auslösen der Kamera.")
            return None

        # ID holen
        try:
            command_id = take_resp.json().get("id")
        except Exception:
            self.get_logger().error("Fehler: Keine ID in der Antwort.")
            return None
        if not command_id:
            self.get_logger().error("Keine gültige Command-ID erhalten.")
            return None

        # time_elapsed = time.time() - start_time
        # self.get_logger().info(f"Dauer bis Bild angefordert: {time_elapsed}")

        # Warten bis fertig
        latest_file = None
        for _ in range(20):
            status_resp = requests.post(
                f"http://{self.CAMERA_IP}/osc/commands/status",
                json={"id": command_id},
                auth=self.auth
            )
            if status_resp.status_code == 200:
                data = status_resp.json()
                if data.get("state") == "done":
                    latest_file = data.get("results", {}).get("fileUrl")
                    break
            time.sleep(0.5)

        if not latest_file:
            self.get_logger().error("Aufnahme nicht abgeschlossen.")
            return None
        
        # time_elapsed = time.time() - start_time
        # self.get_logger().info(f"Dauer bis Bild aufgenommen: {time_elapsed}")
        
        # Bild herunterladen
        img_resp = requests.get(latest_file, auth=self.auth)
        if img_resp.status_code != 200:
            self.get_logger().error("Fehler beim Herunterladen.")
            return None
        
        # time_elapsed = time.time() - start_time
        # self.get_logger().info(f"Dauer bis Bild heruntergeladen: {time_elapsed}")

        img_array = np.asarray(bytearray(img_resp.content), dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        # time_elapsed = time.time() - start_time
        # self.get_logger().info(f"Komplett Bildaufnahme: {time_elapsed}")
        return img

    def generate_views(self, img):
        fov = 90
        out_size = (800, 800)
        directions = {
            "front": (0, 0),
            "right": (90, 0),
            "back": (180, 0),
            "left": (-90, 0),
            "up": (0, 90),
            "down": (0, -90),
        }
        views = {}
        for name, (yaw, pitch) in directions.items():
            persp_img = py360convert.e2p(
                img,
                fov_deg=fov,
                u_deg=yaw,
                v_deg=pitch,
                out_hw=out_size
            )
            views[name] = persp_img
        return views

def main(args=None):
    rclpy.init(args=args)
    node = RicohPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
