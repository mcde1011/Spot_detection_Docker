import rclpy
from rclpy.node import Node
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
from geometry_msgs.msg import TransformStamped
import tf_transformations

class StaticTFPublisher(Node):
    def __init__(self):
        super().__init__('static_tf_publisher')
        
        # read params.yaml
        self.declare_parameter("camera_frame", "camera_link")
        self.camera_frame = self.get_parameter("camera_frame").get_parameter_value().string_value
        self.declare_parameter("base_frame", "hkaspot/base_link")
        self.base_frame = self.get_parameter("base_frame").get_parameter_value().string_value
        
        self.broadcaster = StaticTransformBroadcaster(self)
        self.publish_static_transform()

    def publish_static_transform(self):
        t = TransformStamped()
        t.header.stamp = rclpy.time.Time().to_msg()
        t.header.frame_id = self.base_frame
        t.child_frame_id = self.camera_frame
        t.transform.translation.x = -0.117
        t.transform.translation.y = 0.0
        t.transform.translation.z = 0.111

        # Quaternion aus Euler-Winkeln (hier: keine Rotation)
        quat = tf_transformations.quaternion_from_euler(0, 0, 0)
        t.transform.rotation.x = quat[0]
        t.transform.rotation.y = quat[1]
        t.transform.rotation.z = quat[2]
        t.transform.rotation.w = quat[3]

        self.broadcaster.sendTransform(t)

def main():
    rclpy.init()
    node = StaticTFPublisher()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
