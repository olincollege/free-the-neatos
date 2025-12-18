from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    TimerAction,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    """Launch simulator, EKF stack, waypoint server, and cleaning node."""
    pkg_share = FindPackageShare("freato")
    default_map = PathJoinSubstitution([pkg_share, "maps", "big_map.yaml"])
    map_yaml_arg = DeclareLaunchArgument(
        "map_yaml",
        default_value=default_map,
        description="Absolute path to the map YAML file to load",
    )
    map_yaml = LaunchConfiguration("map_yaml")

    big_map_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([pkg_share, "launch", "big_map.launch.py"])
        )
    )

    test_ekf_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([pkg_share, "launch", "test_ekf.launch.py"])
        ),
        launch_arguments={"map_yaml": map_yaml}.items(),
    )

    waypoint_server = Node(
        package="freato",
        executable="waypoint_follow_server",
        name="waypoint_follow_server",
        output="screen",
    )

    cleaning_node = Node(
        package="freato",
        executable="cleaning",
        name="cleaning",
        output="screen",
    )

    return LaunchDescription(
        [
            map_yaml_arg,
            big_map_launch,
            test_ekf_launch,
            TimerAction(period=2.0, actions=[waypoint_server]),
            TimerAction(period=4.0, actions=[cleaning_node]),
        ]
    )
