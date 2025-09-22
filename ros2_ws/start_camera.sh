#!/bin/bash

# Start camera driver
# source /opt/ros/$ROS_DISTRO/setup.bash
# colcon build --merge-install --symlink-install
source /ros2_ws/install/setup.bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
bash -c "ros2 launch ricoh_theta_ros full_stack.launch.py \
  device:=cuda:0 imgsz:=640 iou:=0.45 period_s:=1.0 \
  log_timing_level:=info processing_mode:=camera_stitch & \
  ros2 launch transform_to_map transform_to_map_launch.py "

# Start transformation to occupancy map

# gnome-terminal -- bash -c "source /opt/ros/humble/setup.bash; source install/setup.bash; ros2 launch transform_to_map transform_to_map_launch.py"
