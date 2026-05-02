import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    config = os.path.join(
        get_package_share_directory("egolanes_lite_ros2"), "config", "params.yaml"
    )

    return LaunchDescription(
        [
            Node(
                package="egolanes_lite_ros2",
                executable="egolanes_lite_node",
                name="egolanes_lite_node",
                parameters=[config],
                output="screen",
            ),
            Node(
                package="egolanes_lite_ros2",
                executable="egolanes_ipm_node",
                name="egolanes_ipm_node",
                parameters=[config],
                output="screen",
            ),
            Node(
                package="egolanes_lite_ros2",
                executable="egolanes_vectorizer_node",
                name="egolanes_vectorizer_node",
                parameters=[config],
                output="screen",
            ),
        ]
    )
