# Spot Detection and Mapping
This Project detects fire extinguishers and Emergency exit signs and approximates their position into an 2D Occupancy Map in Rviz.

## Packages
### detection_msgs
Contains a custom message-type named **LabeledDetections.msg** which is used to send results of the deteciton network to transform_to_map_node. The message-type includes:

- **vision_msgs/Detection2DArray:** contains Detection metadata like positions, sizes and classifications of Boundingboxes in one image.
- **sensor_msgs/CompressedImage:** result image with drawn in Boundingbox

### ricoh_theta_ros

### theta camera


### transform_to_map
**Input:**
- **/detections/[image]/labeled:** LabeledDetections.msg of the detection results published by ricoh_theta_ros

**output:**   
- **semantic_map.yaml:** file in which all detections are saved with their approximated position. It's content is constantly published on /semantic_map/visualization.
- **/semantic_map/visualization Topic:** Marker Array which is used by RViz to visualize the detections
- **/detection_images directory:** There is a Image to every detection inside semantic_map.yaml which is saved here. 

## Launch
Before launching witch compose up, the project has to be built.

```bash
docker build
docker compose up 
```

