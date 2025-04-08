cmake_minimum_required(VERSION 3.5)
project(rover)

# Find packages 
find_package(ament_cmake REQUIRED)
find_package(controller_manager REQUIRED)
find_package(ros2_control REQUIRED)
find_package(ros2_controllers REQUIRED)

# Installation commands
install(
  DIRECTORY config launch meshes urdf
  DESTINATION share/${PROJECT_NAME}
)

# Package setup
ament_package()