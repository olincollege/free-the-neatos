from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    """Launch Gazebo, SLAM toolbox, waypoint server, and SLAM exploration."""
    pkg_share = FindPackageShare("freato")

    big_map_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([pkg_share, "launch", "big_map.launch.py"])
        )
    )

    slam_toolbox_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([pkg_share, "launch", "freato.launch.py"])
        )
    )

    waypoint_server = Node(
        package="freato",
        executable="waypoint_follow_server",
        name="waypoint_follow_server",
        output="screen",
    )

    slam_explorer = Node(
        package="freato",
        executable="slam_exploration",
        name="slam_exploration",
        output="screen",
    )

    return LaunchDescription(
        [
            big_map_launch,
            slam_toolbox_launch,
            TimerAction(period=2.0, actions=[waypoint_server]),
            TimerAction(period=4.0, actions=[slam_explorer]),
        ]
    )
