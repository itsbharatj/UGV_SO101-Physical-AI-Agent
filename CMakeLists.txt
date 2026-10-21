cmake_minimum_required(VERSION 3.8)
project(so101_rover_description)

find_package(ament_cmake REQUIRED)

install(DIRECTORY urdf meshes launch mujoco
  DESTINATION share/${PROJECT_NAME})

ament_package()
