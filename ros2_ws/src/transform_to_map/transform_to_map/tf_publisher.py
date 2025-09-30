import rclpy
from rclpy.node import Node
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
from geometry_msgs.msg import TransformStamped
import tf_transformations

class StaticTFPublisher(Node):
    def __init__(self):
        super().__init__('static_tf_publisher')

        # set default parameter
        self.declare_parameter("camera_frame", "camera_link")
        self.declare_parameter("base_frame", "hkaspot/base_link")
        self.declare_parameter("transform.translation", [-0.117, 0.0, 0.111])
        self.declare_parameter("transform.euler", [0.0, 0.0, 0.0])  # roll, pitch, yaw in Radians

        # read config file
        self.camera_frame = self.get_parameter("camera_frame").value
        self.base_frame = self.get_parameter("base_frame").value
        self.transform_translation = self.get_parameter("transform.translation").value
        self.transform_euler = self.get_parameter("transform.euler").value

        q = tf_transformations.quaternion_from_euler(
            float(self.transform_euler[0]),
            float(self.transform_euler[1]),
            float(self.transform_euler[2]),
        )

        self.broadcaster = StaticTransformBroadcaster(self)
        self.publish_static_transform([float(q[0]), float(q[1]), float(q[2]), float(q[3])])

    def publish_static_transform(self, quat):
        t = TransformStamped()
        t.header.stamp = rclpy.time.Time().to_msg()
        t.header.frame_id = self.base_frame
        t.child_frame_id = self.camera_frame

        t.transform.translation.x = float(self.transform_translation[0])
        t.transform.translation.y = float(self.transform_translation[1])
        t.transform.translation.z = float(self.transform_translation[2])

        t.transform.rotation.x = quat[0]
        t.transform.rotation.y = quat[1]
        t.transform.rotation.z = quat[2]
        t.transform.rotation.w = quat[3]

        self.broadcaster.sendTransform(t)
        self.get_logger().info(f"Static transform published: {self.base_frame} -> {self.camera_frame}")

def main():
    rclpy.init()
    node = StaticTFPublisher()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
