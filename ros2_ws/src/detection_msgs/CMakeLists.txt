cmake_minimum_required(VERSION 3.8)
project(detection_msgs)

# Compiler-Standards setzen
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Pakete finden
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(vision_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)

# Interfaces generieren
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/LabeledDetections.msg"
  DEPENDENCIES vision_msgs sensor_msgs
)

# Export-Abhängigkeiten
ament_export_dependencies(rosidl_default_runtime)

ament_package()