#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
from typing import Dict, Tuple, List

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

VIEWS_ORDER = ["front", "right", "back", "left", "up", "down"]

def make_placeholder(size: Tuple[int, int], text: str):
    h, w = size
    img = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.putText(img, text, (10, h//2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (128, 128, 128), 2, cv2.LINE_AA)
    return img

def put_label(img, label: str):
    pad = 26
    vis = img.copy()
    cv2.rectangle(vis, (0,0), (vis.shape[1], pad), (0,0,0), -1)
    cv2.putText(vis, label, (8, int(pad*0.72)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)
    return vis

class MosaicViewer(Node):
    """
    Bezieht für jede View sowohl Roh- als auch Annotated-Topic.
    Anzeige-Regel pro View:
      - Wenn ein frisches Annotated-Frame vorhanden ist (jünger als raw_max_age_ms), zeige Annotated.
      - Sonst zeige Rohbild.
    Dadurch sieht man nur dann Overlays, wenn tatsächlich Detections vorliegen.
    """
    def __init__(self):
        super().__init__("mosaic_viewer")

        # Parameter
        self.declare_parameter("raw_topics", [
            "/ricoh_theta/front",
            "/ricoh_theta/right",
            "/ricoh_theta/back",
            "/ricoh_theta/left",
            "/ricoh_theta/up",
            "/ricoh_theta/down",
        ])
        self.declare_parameter("annotated_topics", [
            "/detections/front/annotated",
            "/detections/right/annotated",
            "/detections/back/annotated",
            "/detections/left/annotated",
            "/detections/up/annotated",
            "/detections/down/annotated",
        ])
        self.declare_parameter("use_annotated", True)       # Annotated bevorzugen
        self.declare_parameter("annotated_fresh_ms", 3000)   # wie frisch annotiert sein muss
        self.declare_parameter("tile_width", 420)
        self.declare_parameter("tile_height", 420)
        self.declare_parameter("grid_rows", 2)
        self.declare_parameter("grid_cols", 3)
        self.declare_parameter("window_name", "Ricoh Theta – Mosaic")
        self.declare_parameter("publish_mosaic", True)
        self.declare_parameter("mosaic_topic", "/ricoh_theta/mosaic")

        raw_topics = [t for t in self.get_parameter("raw_topics").value]
        ann_topics = [t for t in self.get_parameter("annotated_topics").value]
        self.use_annotated = bool(self.get_parameter("use_annotated").value)
        self.ann_fresh_ms = int(self.get_parameter("annotated_fresh_ms").value)
        self.tw = int(self.get_parameter("tile_width").value)
        self.th = int(self.get_parameter("tile_height").value)
        self.rows = int(self.get_parameter("grid_rows").value)
        self.cols = int(self.get_parameter("grid_cols").value)
        self.window_name = self.get_parameter("window_name").get_parameter_value().string_value
        self.publish_mosaic = bool(self.get_parameter("publish_mosaic").value)
        self.mosaic_topic = self.get_parameter("mosaic_topic").get_parameter_value().string_value

        if self.rows * self.cols < len(VIEWS_ORDER):
            self.get_logger().warn("Raster zu klein für 6 Ansichten – passe grid_rows/grid_cols an.")

        self.bridge = CvBridge()

        # Cache: pro View letzter Roh- und Annotated-Frame (+Zeit)
        # Dict[str, Tuple[Image, np.ndarray, float_epoch_ms]]
        self.last_raw: Dict[str, Tuple[Image, np.ndarray, float]] = {}
        self.last_ann: Dict[str, Tuple[Image, np.ndarray, float]] = {}

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5
        )

        # Subscriptions auf beide Topic-Sets
        self.get_logger().info("MosaicViewer: Multi-Topic (raw + annotated)")
        for t in raw_topics:
            name = t.split("/")[-1] or t
            self.create_subscription(Image, t, self._make_cb_raw(name), qos)
            self.get_logger().info(f"  raw  {t} -> {name}")
        for t in ann_topics:
            name = t.split("/")[-2] or t  # .../<view>/annotated → nimm vorletzten Teil
            self.create_subscription(Image, t, self._make_cb_ann(name), qos)
            self.get_logger().info(f"  ann  {t} -> {name}")

        # Publisher (optional)
        self.pub_mosaic = self.create_publisher(Image, self.mosaic_topic, 10) if self.publish_mosaic else None

        # Render-Timer (10 Hz)
        self.timer = self.create_timer(0.1, self.render)

        try:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_EXPANDED)
        except Exception:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        self.get_logger().info(f"Zeige Mosaic-Fenster: {self.window_name}")

    # ------------- Callbacks -------------

    def _now_ms(self) -> float:
        return time.time() * 1000.0

    def _make_cb_raw(self, view_name: str):
        view = view_name.lower()
        def _cb(msg: Image):
            try:
                bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
                self.last_raw[view] = (msg, bgr, self._now_ms())
            except Exception as e:
                self.get_logger().warn(f"raw-Callback Fehler ({view}): {e}")
        return _cb

    def _make_cb_ann(self, view_name: str):
        view = view_name.lower()
        def _cb(msg: Image):
            try:
                bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
                self.last_ann[view] = (msg, bgr, self._now_ms())
            except Exception as e:
                self.get_logger().warn(f"ann-Callback Fehler ({view}): {e}")
        return _cb

    # ------------- Rendering -------------

    def _select_view_image(self, view: str) -> Tuple[np.ndarray, str]:
        """
        Wähle für eine View das angezeigte Bild + Label.
        Bevorzugt Annotated, wenn vorhanden und frisch; sonst Raw.
        """
        ann = self.last_ann.get(view)
        raw = self.last_raw.get(view)

        if self.use_annotated and ann is not None:
            _, ann_img, t_ann = ann
            if (self._now_ms() - t_ann) <= self.ann_fresh_ms:
                return ann_img, f"{view} (annot)"
        if raw is not None:
            _, raw_img, _ = raw
            return raw_img, view
        # Fallback: Platzhalter
        return make_placeholder((self.th, self.tw), f"warte: {view}"), view

    def render(self):
        # Kacheln zusammenstellen
        tiles: List[np.ndarray] = []
        for name in VIEWS_ORDER:
            img, label = self._select_view_image(name)
            tile = cv2.resize(img, (self.tw, self.th), interpolation=cv2.INTER_AREA)
            tile = put_label(tile, label)
            tiles.append(tile)

        # In Grid legen
        grid = []
        idx = 0
        for _ in range(self.rows):
            row_imgs = []
            for _ in range(self.cols):
                if idx < len(tiles):
                    row_imgs.append(tiles[idx])
                else:
                    row_imgs.append(make_placeholder((self.th, self.tw), ""))
                idx += 1
            grid.append(np.hstack(row_imgs))
        mosaic = np.vstack(grid)

        # Anzeigen
        try:
            cv2.imshow(self.window_name, mosaic)
            cv2.waitKey(1)
        except Exception as e:
            self.get_logger().throttle(5000, f"GUI nicht verfügbar: {e}")

        # Optional publish
        if self.pub_mosaic is not None:
            out = self.bridge.cv2_to_imgmsg(mosaic, encoding="bgr8")
            # Header: nimm den aktuellsten vorhandenen (annotated bevorzugt)
            for k in VIEWS_ORDER:
                if k in self.last_ann:
                    out.header = self.last_ann[k][0].header
                    break
                if k in self.last_raw:
                    out.header = self.last_raw[k][0].header
                    break
            self.pub_mosaic.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = MosaicViewer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
