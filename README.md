# Spot Detection and Mapping
This project identifies fire extinguishers and emergency exit signs within a 360° image. It estimates the spatial position of the detected objects relative to the robot and visualizes them on a 2D occupancy map using RViz.

## Packages
### detection_msgs
Contains a custom message-type named **LabeledDetections.msg** which is used to send results of the deteciton network to transform_to_map_node. The message-type includes:

- **vision_msgs/Detection2DArray:** contains semantic image data like positions, sizes and classifications of Boundingboxes.
- **sensor_msgs/CompressedImage:** result image with drawn in Boundingboxes

### ricoh_publisher
**Input**
- **camera image** Image von camera via HTTP request

**Output:**
- **/ricoh_theta/image/compressed** Compressed Image from Camera
- **/ricoh_theta/image** Not compressed Image

### yolo_detector
**Input**
- **/ricoh_theta/image/compressed** Compressed Image from Camera

**Output**
- **/detections/[image]/labeled** LabeledDetections.msg of the detection results published by ricoh_theta_ros


### transform_to_map
**Input:**
- **/detections/[image]/labeled:** LabeledDetections.msg of the detection results published by ricoh_theta_ros

**Output:**   
- **semantic_map.yaml:** file in which all detections are saved with their approximated position. It's content is constantly published on /semantic_map/visualization.
- **/semantic_map/visualization Topic:** Marker Array which is used by RViz to visualize the detections
- **/detection_images directory:** There is a Image to every detection inside semantic_map.yaml which is saved here. 

## Launch
Before launching with compose up, the project has to be built.

```bash
docker build -t my-ros2:humble-gpu .
docker compose up 
```

