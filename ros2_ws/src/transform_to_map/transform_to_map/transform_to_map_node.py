import rclpy
import numpy as np
import os
import yaml
import cv2
from rclpy.node import Node
from rclpy.time import Time
from vision_msgs.msg import Pose2D
from vision_msgs.msg import Detection2DArray
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from visualization_msgs.msg import MarkerArray, Marker
from tf2_ros import Buffer, TransformListener
import tf2_geometry_msgs
from geometry_msgs.msg import PointStamped, Point
from tf2_geometry_msgs import do_transform_point

from interactive_markers.interactive_marker_server import InteractiveMarkerServer
from visualization_msgs.msg import InteractiveMarker, InteractiveMarkerControl, InteractiveMarkerFeedback

from detection_msgs.msg import LabeledDetections

class TransformToMapNode(Node):
    def __init__(self):
        super().__init__('BB_subscriber')
    
        self.tf_buffer = Buffer(cache_time=rclpy.duration.Duration(seconds=25))
        self.listener = TransformListener(self.tf_buffer, self)
        self.marker_array = MarkerArray()
        self.marker = Marker()
        self.semantic_data = {}
        self.bridge = CvBridge()
        
        # get path to semantic map and load it 
        self.declare_parameter("semantic_map_file", "")
        self.smap_filename = self.get_parameter("semantic_map_file").get_parameter_value().string_value
        if not self.smap_filename:
            self.get_logger().error("No semantic_map_file parameter provided")
        self.load_semantic_map()
        self.timer = self.create_timer(3.0, self.publish_semantic_map)

        # get path to stored detection images
        self.declare_parameter("path_to_images", "")
        self.path_to_annotated_imgs = self.get_parameter("path_to_images").get_parameter_value().string_value
        if not self.path_to_annotated_imgs:
            self.get_logger().error("No path_to_images parameter provided")

        # read params.yaml
        self.declare_parameter("camera_frame", "camera_link")
        self.camera_frame = self.get_parameter("camera_frame").get_parameter_value().string_value
        self.declare_parameter("base_frame", "hkaspot/base_link")
        self.base_frame = self.get_parameter("base_frame").get_parameter_value().string_value
        self.declare_parameter("position_tolerance", 0.4)
        self.position_tolerance = self.get_parameter("position_tolerance").get_parameter_value().double_value

        self.marker_pub = self.create_publisher(MarkerArray, 'utils_rviz_visualization', 10)
        self.detection_img_rviz_pub = self.create_publisher(Image, 'detection_image', 10)
        
        # self.im_server = InteractiveMarkerServer(self, "object_markers")  # eindeutiger Namespace
        # self.im_server.clear()
        # self.im_server.applyChanges()

        self.sub_front = self.create_subscription(
            LabeledDetections,
            '/detections/front/labeled',
            self.front_cb,
            10
        )

        self.sub_front = self.create_subscription(
            LabeledDetections,
            '/detections/back/labeled',
            self.back_cb,
            10
        )

        self.sub_front = self.create_subscription(
            LabeledDetections,
            '/detections/left/labeled',
            self.left_cb,
            10
        )

        self.sub_front = self.create_subscription(
            LabeledDetections,
            '/detections/right/labeled',
            self.right_cb,
            10
        )
        self.sub_front = self.create_subscription(
            LabeledDetections,
            '/detections/up/labeled',
            self.up_cb,
            10
        )

        self.sub_front = self.create_subscription(
            LabeledDetections,
            '/detections/down/labeled',
            self.down_cb,
            10
        )

    ####################################################################
    ###                     SEMANTIC MAP                             ###
    ####################################################################

    def load_semantic_map(self):
        """load YAML-file with graceful handling for empty files"""
        try:
            if not os.path.exists(self.smap_filename):
                self.get_logger().warn(f"YAML file does not exist: {self.smap_filename}")
                return
                
            with open(self.smap_filename, "r", encoding='utf-8') as f:
                loaded_data = yaml.safe_load(f)
                
            # check if file is empty 
            if loaded_data is None:
                self.get_logger().warn("YAML file is empty - using default empty configuration")
                self.semantic_data = {}
            elif isinstance(loaded_data, dict):
                self.semantic_data = loaded_data
                self.get_logger().info(f"Successfully loaded semantic map from: {self.smap_filename}")
            else:
                self.get_logger().warn(f"YAML file contains unexpected data type: {type(loaded_data)}")
                self.semantic_data = {}
                
        except yaml.YAMLError as e:
            self.get_logger().error(f"Error parsing YAML file: {e}")
            self.semantic_data = {}
        except Exception as e:
            self.get_logger().error(f"Error loading YAML file: {e}")
            self.semantic_data = {}

    def addObjToYaml(self, point_in_map, obj_class, saved_pts):
        object_id = ""
        new_pos = np.array([
            float(point_in_map.point.x),
            float(point_in_map.point.y),
            float(point_in_map.point.z)
        ])

        if 'objects' not in self.semantic_data:
            self.semantic_data['objects'] = []

        updated = False

        for obj in self.semantic_data['objects']:
            if obj.get('object_type') != obj_class:
                continue

            old_pos = np.array(obj.get('position', [0.0, 0.0, 0.0]))
            dist = np.linalg.norm(new_pos - old_pos)
            if dist > self.position_tolerance:
                continue

            # Prüfe Überschneidung mit bereits gespeicherten Punkten
            def to_arr(pt_stamped):
                return np.array([
                    float(pt_stamped.point.x),
                    float(pt_stamped.point.y),
                    float(pt_stamped.point.z)
                ])

            skip = any(np.allclose(old_pos, to_arr(saved_pt), rtol=1e-5, atol=1e-8)
                    for saved_pt in saved_pts)

            if skip:
                print("SKIPPE PUNKT, WEIL ÜBERSCHNEIDUNG IN EINEM BILD", flush=True)
                updated = False
                # Wenn du hier GAR NICHTS mehr updaten willst, kannst du auch:
                # break
                continue
            else:
                # print("Überschreibe PT", old_pos, new_pos, flush=True)
                obj['position'] = new_pos.tolist()
                updated = True
                object_id = obj['id']
                break  # wichtig: äußere Schleife verlassen

        print("UPDATED:", updated, flush=True)

        if not updated:
            print("ERZEUGE NEUEN PUNKT", flush=True)
            object_id = self.generate_next_id(obj_class)
            new_object = {
                'object_type': obj_class,
                'id': object_id,
                'position': new_pos.tolist()
            }
            self.semantic_data['objects'].append(new_object)
            self.get_logger().info(f"Added new {obj_class} with id {object_id}")

        return object_id

    def generate_next_id(self, obj_class):
        count = len([obj for obj in self.semantic_data.get('objects', []) if obj.get('object_type') == obj_class])
        next_number = count + 1
        
        # Format: object_type_XXX
        return f"{obj_class}_{next_number:03d}"

    def save_semantic_map(self):
        try:
            with open(self.smap_filename, 'w', encoding='utf-8') as f:
                yaml.dump(self.semantic_data, f, default_flow_style=False, allow_unicode=True)
            # self.get_logger().info(f"Saved semantic map to {self.smap_filename}")
            # return True
        except Exception as e:
            self.get_logger().error(f"Could not save semantic map: {e}")
            return False
        
    def publish_semantic_map(self):
        """Visualize all objects from YAML as markers in RViz"""
        
        if 'objects' not in self.semantic_data:
            return
                
        # Bestehende Marker löschen
        # self.im_server.clear()
        
        for obj in self.semantic_data['objects']:
            obj_class = obj['object_type']
            pos = obj['position']
            obj_id = obj['id']
            
            point_stamped = PointStamped()
            point_stamped.header.stamp = self.get_clock().now().to_msg()
            point_stamped.header.frame_id = "map"
            point_stamped.point.x = pos[0]
            point_stamped.point.y = pos[1]
            point_stamped.point.z = pos[2]
            
            marker = createMarker(obj_class, obj_id, "map", point_stamped)
            marker.lifetime = rclpy.duration.Duration(seconds=0).to_msg()
            self.marker_array.markers.append(marker)

            text_marker = createTextMarker(obj_id, "map", point_stamped)
            self.marker_array.markers.append(text_marker)

        if self.marker_array.markers:
            self.marker_pub.publish(self.marker_array)

        #     im = self.make_interactive_marker(obj_class, obj_id, point_stamped)
            
        #     self.im_server.insert(im)
        #     self.im_server.setCallback(obj_id, self.on_im_feedback)
            

        # self.im_server.applyChanges()

    ####################################################################
    ###               Interactive Markers in RViz                    ###
    ####################################################################

    def on_im_feedback(self, feedback: InteractiveMarkerFeedback):
        if feedback.event_type in (
            InteractiveMarkerFeedback.BUTTON_CLICK,
            InteractiveMarkerFeedback.MOUSE_UP,
        ):
            obj_id = feedback.marker_name
            self.on_object_clicked(obj_id)

    def on_object_clicked(self, obj_id: str):
        img = self.load_image_as_rosmsg(obj_id)
        if img is not None:
            self.detection_img_rviz_pub.publish(img)
            self.get_logger().info(f"Clicked on {obj_id}")
        else:
            self.get_logger().warn(f"Could not load image for {obj_id}")

    ####################################################################
    ###               save and load detection image                  ###
    ####################################################################
    def saveDetectionImg(self, obj_id, ros_img):
        """Save ROS Image message (compressed or uncompressed) as OpenCV image file"""
        try:
            # Check if it's a CompressedImage
            if hasattr(ros_img, 'format'):  # CompressedImage has 'format' attribute
                # Convert CompressedImage to OpenCV format
                cv_image = self.bridge.compressed_imgmsg_to_cv2(ros_img, "bgr8")
            else:
                # Convert regular Image message to OpenCV format
                cv_image = self.bridge.imgmsg_to_cv2(ros_img, "bgr8")
            
            # Add filename to path (with image extension)
            filepath = os.path.join(self.path_to_annotated_imgs, f"{obj_id}.jpg")
            success = cv2.imwrite(filepath, cv_image)
            if not success:
                raise RuntimeError(f"Couldn't save {filepath}")
                
            return filepath
            
        except Exception as e:
            self.get_logger().error(f"Error saving detection image: {e}")
            return None
    
    def load_image_as_rosmsg(self, obj_id):
        """Load saved image and convert to ROS Image message"""
        try:
            # Add filename with extension to path
            filepath = os.path.join(self.path_to_annotated_imgs, f"{obj_id}.jpg")

            if not os.path.exists(filepath):
                self.get_logger().error(f"Image {filepath} does not exist!")
                # Return a dummy image or handle gracefully
                return None

            cv_image = cv2.imread(filepath, cv2.IMREAD_COLOR)
            if cv_image is None:
                self.get_logger().error(f"Couldn't load {filepath}!")
                return None

            ros_image = self.bridge.cv2_to_imgmsg(cv_image, "bgr8")
            return ros_image
            
        except Exception as e:
            self.get_logger().error(f"Error loading image {obj_id}: {e}")
            return None

    ####################################################################
    ###                       CALLBACKS                              ###
    ####################################################################

    def front_cb(self, msg):
        label = "front"
        obj_id = self.drawObjInMap(label, msg.detections, msg.img)
        # if obj_id:
        #     self.saveDetectionImg(obj_id, msg.img)

        # if not success:
        #     self.get_logger().error('ERROR WHILE DRAWING OBJECTS FROM FRONT CAMERA')

    def back_cb(self, msg):
        label = "back"
        obj_id = self.drawObjInMap(label, msg.detections, msg.img)
        # if obj_id:
        #     self.saveDetectionImg(obj_id, msg.img)
        # if not success:
            # self.get_logger().error('ERROR WHILE DRAWING OBJECTS FROM BACK CAMERA')

    def left_cb(self, msg):
        label = "left"
        obj_id = self.drawObjInMap(label, msg.detections, msg.img)
        # if obj_id:
        #     self.saveDetectionImg(obj_id, msg.img)
        # if not success:
            # self.get_logger().error('ERROR WHILE DRAWING OBJECTS FROM LEFT CAMERA')
    
    def right_cb(self, msg):
        label = "right"
        obj_id = self.drawObjInMap(label, msg.detections, msg.img)
        # if obj_id:
        #     self.saveDetectionImg(obj_id, msg.img)
        # if not success:
            # self.get_logger().error('ERROR WHILE DRAWING OBJECTS FROM RIGHT CAMERA')

    def up_cb(self, msg):
        label = "up"
        obj_id = self.drawObjInMap(label, msg.detections, msg.img)
        # if obj_id:
        #     self.saveDetectionImg(obj_id, msg.img)
        # if not success:
            # self.get_logger().error('ERROR WHILE DRAWING OBJECTS FROM UP CAMERA')
        
    def down_cb(self, msg):
        label = "down"
        obj_id = self.drawObjInMap(label, msg.detections, msg.img)
        # if obj_id:
        #     self.saveDetectionImg(obj_id, msg.img)
        # if not success:
            # self.get_logger().error('ERROR WHILE DRAWING OBJECTS FROM DOWN CAMERA')

    ####################################################################
    ###                     DRAW OBJECT IN MAP                       ###
    ####################################################################

    def drawObjInMap(self, label, detections_arr, img):
        self.marker_array.markers.clear()
        pt_camera_frame = PointStamped()
        pt_base_link = PointStamped()
        pt_camera_frame.header = detections_arr.header
        pt_camera_frame.header.frame_id = self.camera_frame
        obj_id = ""
        i = 0
        saved_pts = []
        print("ANZ OBJ: ", len(detections_arr.detections), flush=True)
        for detection in detections_arr.detections:
            if not detection.results:
                self.get_logger().warn("Detection without results – skipping")
                continue

            bbox = detection.bbox
            obj_class = idToString(detection.results[0].hypothesis.class_id)
            dist_approximation = 0.0
            if label != "up" and label != "down":
                dist_approximation = calcDistanceVerticalCam(obj_class, bbox)
                # calculate angle to object by using camera pixels (FoV = 90°, 800x800 Pixel)
                angle_in_image = -(0.1125 * (bbox.center.position.x - 400) * np.pi) / 180
            else:
                dist_approximation, angle_in_image = calcDistanceHorizontalCam(bbox)

            pt_camera_frame.point = self.getPose(label, dist_approximation, angle_in_image)

            pt_base_link = self.transformToBaseFrame(pt_camera_frame)
            pt_map = self.transformToMap(pt_base_link)

            if pt_map is None:
                self.get_logger().warn(
                    f"Skip {obj_class}: no transform {pt_base_link.header.frame_id}->map "
                    f"at t={pt_base_link.header.stamp.sec}.{pt_base_link.header.stamp.nanosec}"
                )
                return False
            else:
                print("Loop NR.", i, flush=True)
                obj_id = self.addObjToYaml(pt_map, obj_class, saved_pts)
                saved_pts.append(pt_map)
                # if saved_to_yaml:
                #     print("SPEICHERE BILD", flush=True)
                self.saveDetectionImg(obj_id, img)
                i +=1

                # self.marker = createMarker(obj_class, obj_id, self.base_frame, pt_base_link)      # comment in to visualize the marker directly in RViz
                # self.marker.lifetime = rclpy.duration.Duration(seconds=1).to_msg()
                # self.marker_array.markers.append(self.marker)
        self.save_semantic_map()

    def transformToBaseFrame(self, pt_camera_frame):
        """
        Transform a PointStamped from camera_frame to base_link,
        for the time of image capture (header.stamp).
        """
        source_frame = pt_camera_frame.header.frame_id
        target_frame = self.base_frame
        stamp = pt_camera_frame.header.stamp  # builtin_interfaces/Time

        can = self.tf_buffer.can_transform(
            target_frame,
            source_frame,
            stamp,
            timeout=rclpy.duration.Duration(seconds=1.0)
        )
        if not can:
            self.get_logger().warn(
                f"No TF {source_frame}->{target_frame} for t={stamp.sec}.{stamp.nanosec}; "
                "can't determine position"
            )
            return None

        tf = self.tf_buffer.lookup_transform(target_frame, source_frame, stamp)
        out = do_transform_point(pt_camera_frame, tf)
        # keep the timestamp
        out.header.stamp = stamp
        return out

    def transformToMap(self, pt_base_link):
        """
        Transform a PointStamped from base_link to map
        for exact the time of image capture (header.stamp).
        """
        target_frame = "map"
        source_frame = pt_base_link.header.frame_id
        stamp = pt_base_link.header.stamp

        can = self.tf_buffer.can_transform(
            target_frame,
            source_frame,
            stamp,
            timeout=rclpy.duration.Duration(seconds=1.0)
        )
        if not can:
            self.get_logger().warn(
                f"No TF {source_frame}->{target_frame} for t={stamp.sec}.{stamp.nanosec} – skip."
            )
            return None

        tf = self.tf_buffer.lookup_transform(target_frame, source_frame, stamp)
        out = do_transform_point(pt_base_link, tf)   # use the same timestamp
        out.header.stamp = stamp
        return out

    def getPose(self, label, dist_approximation, angle_in_image):
        if label == "right":
            angle_in_image = angle_in_image - np.pi/2
        elif label == "back":
            angle_in_image = angle_in_image + np.pi
        elif label == "left":
            angle_in_image = angle_in_image + np.pi/2

        pt = Point()
        pt.x = np.cos(angle_in_image) * dist_approximation
        pt.y = np.sin(angle_in_image) * dist_approximation
        pt.z = 0.0

        if label == "up" or label == "down":
            buffer_x = pt.x
            pt.x = pt.y
            pt.y = buffer_x

        return pt


    # def make_interactive_marker(self, obj_class: str, obj_id: str, point_stamped):
    #     im = InteractiveMarker()
    #     print("FRAME ID: ", point_stamped.header.frame_id, flush=True)
    #     im.header.frame_id = point_stamped.header.frame_id
    #     # im.header.stamp = self.get_clock().now().to_msg()
    #     im.name = str(obj_id)
    #     im.description = str(obj_id)
    #     im.pose.position = point_stamped.point
    #     im.pose.orientation.w = 1.0
    #     im.scale = 0.5
        
    #     shape = Marker()
    #     shape.header = im.header
    #     shape.action = Marker.ADD
    #     shape.pose.orientation.w = 1.0
        
    #     if obj_class == "fire_extinguisher":
    #         shape.id = 1
    #         shape.type = Marker.LINE_STRIP
    #         shape.action = Marker.ADD
    #         shape.pose.orientation.w = 1.0
    #         shape.scale.x = 1.0
    #         shape.scale.y = 1.0
    #         shape.scale.z = 1.0
    #         shape.color.r, shape.color.g, shape.color.b, shape.color.a = 1.0, 0.0, 0.0, 1.0
    #         resolution, radius = 19, 0.25
    #         for i in range(resolution + 1):
    #             angle = 2 * np.pi * i / resolution
    #             p = Point()
    #             p.x = radius * np.cos(angle)  # Relative Position zum Marker
    #             p.y = radius * np.sin(angle)
    #             p.z = 0.5
    #             shape.points.append(p)

    #     elif obj_class == "emergency_exit_sign":
    #         shape.id = 2
    #         shape.type = Marker.CUBE
    #         shape.scale.x, shape.scale.y, shape.scale.z = 0.2, 0.2, 0.01
    #         shape.color.r, shape.color.g, shape.color.b, shape.color.a = 0.0, 1.0, 0.0, 1.0

    #     label = Marker()
    #     label.header.frame_id = im.header.frame_id
    #     label.ns = "labels"
    #     label.id = hash("text_" + str(obj_id)) % (2**31 - 1)
    #     label.type = Marker.TEXT_VIEW_FACING
    #     label.action = Marker.ADD
    #     label.pose.position.x = 0.0  # Relative Position zum InteractiveMarker
    #     label.pose.position.y = 0.0
    #     label.pose.position.z = 0.05
    #     label.pose.orientation.w = 1.0
    #     label.scale.z = 0.15
    #     label.color.r = label.color.g = label.color.b = label.color.a = 1.0
    #     label.text = str(obj_id)

    #     # Haupt-Control für Visualisierung
    #     visual_ctrl = InteractiveMarkerControl()
    #     visual_ctrl.always_visible = True
    #     visual_ctrl.markers.append(shape)
    #     visual_ctrl.markers.append(label)

    #     # Separates Click-Control für bessere Klick-Erkennung
    #     click_ctrl = InteractiveMarkerControl()
    #     click_ctrl.name = "click_control"
    #     click_ctrl.interaction_mode = InteractiveMarkerControl.BUTTON
    #     click_ctrl.always_visible = True
        
    #     # Unsichtbarer Marker für bessere Klick-Erkennung
    #     click_marker = Marker()
    #     click_marker.header.frame_id = im.header.frame_id
    #     click_marker.type = Marker.CYLINDER
    #     click_marker.action = Marker.ADD
    #     click_marker.scale.x = 1.0  # Etwas größer als der visuelle Marker
    #     click_marker.scale.y = 1.0  # für bessere Klickbarkeit
    #     click_marker.scale.z = 1.0
    #     click_marker.color.r = click_marker.color.g = click_marker.color.b = 0.0
    #     click_marker.color.a = 0.0  # Vollständig transparent
    #     click_marker.pose.orientation.w = 1.0
    #     click_ctrl.markers.append(click_marker)

    #     # Beide Controls zum InteractiveMarker hinzufügen
    #     im.controls.append(visual_ctrl)
    #     im.controls.append(click_ctrl)
    #     return im

def idToString(obj_id):
    if obj_id == "0":
        return "emergency_exit_sign"
    elif obj_id == "1":
        return "fire_extinguisher"
    else:
        return "unknown class"

def calcDistanceVerticalCam(obj_class, bbox):
    """Distance approximation with symmetrical sigmoidal"""
    if obj_class == "emergency_exit_sign":
        # symmetrical sigmoidal
        dist_approximation = 877712.8 + (-877710.34/(1+(bbox.center.position.y/986432.3)**1.494542))
    elif obj_class == "fire_extinguisher":
        # symmertrical sogmoidal
        dist_approximation = 0.1385566 + (2088723.861/(1+((bbox.size_y/0.0008264511) ** 1.174847)))
        # dist_approximation = 0.3303583 + (3.82655/(1+((bbox.size_x/56.29718) ** 2.192104)))       # for BB length in x
    return dist_approximation

def calcDistanceHorizontalCam(bbox):
    """Distance approximation with linaer regression"""
    dx_img = bbox.center.position.x - 400
    dy_img = bbox.center.position.y - 400
    
    # transform to base_link orientation
    dx = dy_img
    dy = -dx_img

    angle_in_image = np.arctan2(dx, dy)

    dist_pixel = np.sqrt(dx**2 + dy**2)
    # linaer regression
    dist_approximation = 188645.5 +(-188645.55/(1+(dist_pixel/25053770)**1.015689))
    return dist_approximation, angle_in_image

def createMarker(obj_class, obj_id, frame_id, point_in_map):
    marker = Marker()
    marker.header.frame_id = frame_id
    marker.ns = obj_id
    marker.id = hash(obj_id) % (2**31 - 1) if obj_id else 0

    if obj_class == "fire_extinguisher":
        marker.id = 1
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.025  # Linewidth
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0
        resolution = 20
        radius = 0.25
        for i in range(resolution+ 1):  # +1 to close the circle
            angle = 2 * np.pi * i / resolution
            p = Point()
            p.x = point_in_map.point.x + radius * np.cos(angle)
            p.y = point_in_map.point.y + radius * np.sin(angle)
            p.z = 0.0
            marker.points.append(p)

    elif obj_class == "emergency_exit_sign":
        marker.id = 2
        marker.type = Marker.CUBE
        marker.pose.position = point_in_map.point
        marker.scale.x = 0.2
        marker.scale.y = 0.2
        marker.scale.z = 0.01
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 1.0

    return marker

def createTextMarker(obj_id, frame_id, point_in_map):
    marker = Marker()
    marker.header.frame_id = frame_id
    marker.ns = "labels"
    marker.id = hash("text_" + str(obj_id)) % (2**31 - 1)

    marker.type = Marker.TEXT_VIEW_FACING
    marker.action = Marker.ADD

    marker.pose.position.x = point_in_map.point.x
    marker.pose.position.y = point_in_map.point.y
    marker.pose.position.z = point_in_map.point.z + 0.05

    marker.scale.z = 0.15  # Fontsize in Meters
    marker.color.r = 1.0
    marker.color.g = 1.0
    marker.color.b = 1.0
    marker.color.a = 1.0

    marker.text = str(obj_id)

    return marker



def main(args=None):
    rclpy.init(args=args)
    node = TransformToMapNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
