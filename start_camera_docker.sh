#!/bin/bash
xhost + 

gnome-terminal -- bash -c "docker run --net=host --ipc=host --rm -it --gpus all -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=all -e DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix:rw   --shm-size=1g   --name ros2_dev   my-ros2:humble-gpu ; exec bash"
sleep 4

docker exec -it ros2_dev bash