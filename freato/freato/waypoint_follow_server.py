"""
This is a ROS action server to follow waypoints.
"""

import math
import time

import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer
from rclpy.duration import Duration
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, PoseStamped
from nav2_msgs.action import FollowWaypoints
from rclpy.executors import MultiThreadedExecutor

import tf2_ros
from tf2_geometry_msgs import do_transform_pose


class WaypointActionServer(Node):

    def __init__(self):
        """
        Initializes the Letterbox node.
        Starts keyboard listener and letter drawing threads.
        Initializes position tracking and control flags.

        Publishers: `cmd_vel`
        Subscribers: `/odom`
        """

        super().__init__("waypoint_action_server")

        # Set up publishers and subscriptions
        self.vel_pub = self.create_publisher(Twist, "cmd_vel", 10)
        self.create_subscription(Odometry, "/odom", self.process_odom, 10)

        # Setup tf2 to convert from odom to map frame
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Speed parameters
        self.angular_velocity = math.pi / 8
        self.linear_velocity = 0.2

        # Initialize the action server
        self._action_server = ActionServer(
            self, FollowWaypoints, "follow_points", self.execute_callback
        )

        # Position in map frame
        self.position = []

    def execute_callback(self, goal_handle):
        goal_poses = goal_handle.request.poses
        print(f"{len(goal_poses)} Waypoints Received")

        # If not position information from transforms
        if not self.position:
            self.send_drive_command(0.0, 0.0)
            goal_handle.abort()
            result = FollowWaypoints.Result()
            result.missed_waypoints = list(range(0, len(goal_poses)))
            return result

        for i, pose in enumerate(goal_poses):

            # Publish feedback
            feedback = FollowWaypoints.Feedback()
            feedback.current_waypoint = i
            goal_handle.publish_feedback(feedback)
            print(f"Starting waypoint {i}")

            drive_time, rotate_time, rotate_sign = self.go_to_point_math(
                pose.pose.position.x, pose.pose.position.y
            )
            print(f"Turn for: {rotate_time} sec, Drive for: {drive_time} sec")

            # Send rotate command
            self.send_drive_command(0, self.angular_velocity * rotate_sign)
            rotate_start_time = self.get_clock().now()

            # Rotate for that time
            while self.get_clock().now() - rotate_start_time < Duration(
                seconds=rotate_time
            ):
                if goal_handle.is_cancel_requested:
                    self.send_drive_command(0, 0)
                    goal_handle.canceled()
                    print("Canceling action")
                    result = FollowWaypoints.Result()
                    result.missed_waypoints = list(range(i, len(goal_poses)))
                    return result

                # print(self.position)

                time.sleep(0.01)

            # Send drive command
            self.send_drive_command(self.linear_velocity, 0)
            drive_start_time = self.get_clock().now()

            # Drive for that time
            while self.get_clock().now() - drive_start_time < Duration(
                seconds=drive_time
            ):
                if goal_handle.is_cancel_requested:
                    self.send_drive_command(0, 0)
                    goal_handle.canceled()
                    print("Canceling action")
                    result = FollowWaypoints.Result()
                    result.missed_waypoints = list(range(i, len(goal_poses)))
                    return result

                # print(self.position)

                time.sleep(0.01)

            self.send_drive_command(0, 0)

        goal_handle.succeed()
        print("Action finished")
        result = FollowWaypoints.Result()
        return result

    def process_odom(self, msg):
        """
        Callback function for /odom subscription.

        Updates the robot's current position and orientation based on Odometry data.

        Args:
            msg (Odometry): The Odometry message containing pose and orientation.
        """
        map_pose = self.odom_to_map(msg)
        if map_pose:
            q = map_pose.orientation
            yaw = quaternion_to_yaw(q)
            self.position = [
                map_pose.position.x,
                map_pose.position.y,
                yaw,
            ]

    def odom_to_map(self, odom_msg):
        pose_odom = PoseStamped()
        pose_odom.header = odom_msg.header
        pose_odom.pose = odom_msg.pose.pose

        try:
            # Get tranformation from odom to map
            tf_map_from_odom = self.tf_buffer.lookup_transform(
                "map",
                pose_odom.header.frame_id,
                pose_odom.header.stamp,
                timeout=rclpy.duration.Duration(seconds=0.2),
            )
        except:
            print("Transform failed from odom to map failed")
            return None

        pose_map = do_transform_pose(pose_odom.pose, tf_map_from_odom)
        # print("Odom to map success")
        return pose_map

    def send_drive_command(self, linear, angular):
        """Drive with the specified linear and angular velocity.

        Args:
            linear (Float): the linear velocity in m/s
            angular (Float): the angular velocity in radians/s
        """
        msg = Twist()
        msg.linear.x = float(linear)
        msg.angular.z = float(angular)
        self.vel_pub.publish(msg)

    def go_to_point_math(self, target_x, target_y):
        """
        Moves the robot to a desired 2D point using current odometry information.

        The robot first rotates in place to face the target, then drives straight.

        Args:
            desired_x (float): Target X position.
            desired_y (float): Target Y position.
        """
        current_x, current_y, current_angle = self.position

        # print("Current x is: ", current_x)
        # print("Current y is: ", current_y)
        # print("Des x is: ", desired_x)
        # print("Des y is: ", desired_y)

        # First, we calculate the desired orientation to drive in
        # odom gives us quanternion, not yaw (z direction)
        x = target_x - current_x
        y = target_y - current_y

        desired_angle = math.atan2(y, x)
        # print(current_angle, desired_angle)

        current_angle = current_angle % (2 * math.pi)
        desired_angle = desired_angle % (2 * math.pi)

        rotation_needed = (desired_angle - current_angle) % (2 * math.pi)

        # Rotate left or right
        if rotation_needed < math.pi:
            rotate_sign = 1
        else:
            rotation_needed = 2 * math.pi - rotation_needed
            rotate_sign = -1

        # Calculate rotation time
        rotate_time = rotation_needed / self.angular_velocity

        # Calculate drive time
        distance = math.sqrt((x) ** 2 + (y) ** 2)
        drive_time = distance / self.linear_velocity

        return drive_time, rotate_time, rotate_sign


def quaternion_to_yaw(q):
    """
    Converts a quaternion into a yaw (z rotation).

    Args:
        q (geometry_msgs.msg.Quaternion): Orientation as a quaternion.

    Returns:
        float: yaw in radians.
    """
    return math.atan2(2.0 * (q.w * q.z), 1.0 - 2.0 * (q.z * q.z))


def main(args=None):
    """
    Initialize rclpy and Node, then run with a MultiThreadedExecutor
    so action callbacks and /odom callbacks can run concurrently.
    """
    rclpy.init(args=args)

    node = WaypointActionServer()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)

    try:
        executor.spin()
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
