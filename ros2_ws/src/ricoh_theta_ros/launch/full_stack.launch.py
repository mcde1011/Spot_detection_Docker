from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory
import os

# Standardpfad: <share>/ricoh_theta_ros/resource/best.pt
default_model = PathJoinSubstitution([
    FindPackageShare('ricoh_theta_ros'),  # Paket-Sharedir
    'resource',                           # oder 'resources' falls so benannt
    'best.pt'
])


def generate_launch_description():
    
    # YOLO-Args
    model_path = LaunchConfiguration('model_path')
    device     = LaunchConfiguration('device')
    imgsz      = LaunchConfiguration('imgsz')
    conf       = LaunchConfiguration('conf')
    iou        = LaunchConfiguration('iou')

    # Publisher-Args
    processing_mode = LaunchConfiguration('processing_mode')  # "stitch_local" | "camera_stitch"
    period_s        = LaunchConfiguration('period_s')
    out_w           = LaunchConfiguration('out_w')
    out_h           = LaunchConfiguration('out_h')
    jpeg_quality    = LaunchConfiguration('jpeg_quality')
    flip_vertical   = LaunchConfiguration('flip_vertical')
    yaw_offset_deg  = LaunchConfiguration('yaw_offset_deg')
    blend_width_deg = LaunchConfiguration('blend_width_deg')
    package_dir = get_package_share_directory('ricoh_theta_ros')
    path_to_images = os.path.join(package_dir, 'camera_images')

    return LaunchDescription([
        # YOLO
        DeclareLaunchArgument('model_path', default_value=default_model),
        DeclareLaunchArgument('device',     default_value='auto'),
        DeclareLaunchArgument('imgsz',      default_value='640'),
        DeclareLaunchArgument('conf',       default_value='0.6'),
        DeclareLaunchArgument('iou',        default_value='0.45'),

        # Publisher
        DeclareLaunchArgument('processing_mode', default_value='camera_stitch'),
        DeclareLaunchArgument('period_s',        default_value='1.0'),
        DeclareLaunchArgument('out_w',           default_value='1280'),
        DeclareLaunchArgument('out_h',           default_value='640'),
        DeclareLaunchArgument('jpeg_quality',    default_value='65'),
        DeclareLaunchArgument('flip_vertical',   default_value='true'),
        DeclareLaunchArgument('yaw_offset_deg',  default_value='0.0'),
        DeclareLaunchArgument('blend_width_deg', default_value='12.0'),

        # Ricoh-Publisher
        Node(
            package='ricoh_theta_ros',
            executable='ricoh_publisher',
            name='ricoh_publisher',
            output='screen',
            parameters=[{
                'processing_mode': processing_mode,         # "stitch_local" oder "camera_stitch"
                'timer_period': period_s,
                'output_width': out_w,
                'output_height': out_h,
                'jpeg_quality': jpeg_quality,
                'publish_compressed': True,                 # wir senden CompressedImage
                'flip_vertical': flip_vertical,
                'yaw_offset_deg': yaw_offset_deg,
                'blend_width_deg': blend_width_deg,
                "path_to_images": path_to_images,

            }],
        ),

        # YOLO-Detector (ein Eingangsthema, komprimiert)
        Node(
            package='ricoh_theta_ros',
            executable='yolo_detector',
            name='yolo_detector',
            output='screen',
            parameters=[{
                'model_path': model_path,
                'device': device,
                'imgsz': imgsz,
                'conf': conf,
                'iou': iou,
                'input_topic': '/ricoh_theta/image/compressed',  # wichtig: komprimiertes Topic
                'output_namespace': '/detections',
                'publish_annotated': True,
                'publish_annotated': True,
                'publish_annotated_compressed': True,
                'annotated_jpeg_quality': 70,
            }],
        ),
    ])
