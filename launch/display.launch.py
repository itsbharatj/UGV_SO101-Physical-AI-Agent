"""Show the robot in RViz with joint sliders:  ros2 launch so101_rover_description display.launch.py"""
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    share = get_package_share_directory("so101_rover_description")
    with open(os.path.join(share, "urdf", "so101_rover.urdf")) as f:
        robot_description = f.read()
    return LaunchDescription([
        Node(package="robot_state_publisher", executable="robot_state_publisher",
             parameters=[{"robot_description": robot_description}]),
        Node(package="joint_state_publisher_gui", executable="joint_state_publisher_gui"),
        Node(package="rviz2", executable="rviz2"),
    ])
