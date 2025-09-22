# Datei: launch/camera_tf_launch.py
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    package_dir = get_package_share_directory('transform_to_map')
    config = os.path.join(package_dir, 'config', 'params.yaml')
    semantic_map = os.path.join(package_dir, 'config', 'semantic_map.yaml')
    path_to_images = os.path.join(package_dir, 'detection_images')

    return LaunchDescription([
        Node(
            package='transform_to_map',
            executable='transform_to_map_node',
            name='transform_to_map_node',
            output='screen',
            parameters=[
                config, 
                {"semantic_map_file": semantic_map},
                {"path_to_images": path_to_images}
            ]
        ),
        Node(
            package='transform_to_map',
            executable='tf_publisher',
            name='tf_publisher',
            output='screen'
        )
    ])
