"""
This is a ROS action server to follow waypoints.
"""

import math
import concurrent.futures

import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, PoseStamped, Pose
from nav2_msgs.action import FollowWaypoints
from rclpy.executors import MultiThreadedExecutor
from rclpy.time import Time
from visualization_msgs.msg import Marker

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
        self.marker_pub = self.create_publisher(Marker, "/target", 10)

        # Setup tf2 to convert from odom to map frame
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Run loop
        self.create_timer(0.05, self.run_loop)

        # Speed parameters
        self.angular_velocity = math.pi / 4
        self.linear_velocity = 0.15

        # Initialize the action server
        self._action_server = ActionServer(
            self, FollowWaypoints, "follow_points", self.execute_callback
        )

        # Position in map frame
        self.position = []

        # Reached reached position threshold
        self.position_threshold = 0.15

        # Callback and route progress
        self.goal_handle = None
        self.pose_number = 0
        self.done_future = None

        # Controller constant
        self.proportional_constant = 0.5

        self.run_loop_num = 0

    def run_loop(self):
        self.run_loop_num += 1
        if self.goal_handle:

            if self.goal_handle.is_cancel_requested:
                self.finish("cancel")
                return

            target_pose = self.target_pose

            m = Marker()
            m.header.frame_id = "map"
            m.header.stamp = self.get_clock().now().to_msg()
            m.ns, m.id = "debug", 0
            m.type, m.action = Marker.SPHERE, Marker.ADD

            m.pose.position.x = target_pose.position.x
            m.pose.position.y = target_pose.position.y
            m.pose.position.z = 0.0
            m.pose.orientation.w = 1.0

            m.scale.x = m.scale.y = m.scale.z = 0.2
            m.color.r, m.color.a = 1.0, 1.0  # alpha must be > 0

            self.marker_pub.publish(m)

            distance_to_target_pose = self.distance_to_pose(target_pose)

            if distance_to_target_pose <= self.position_threshold:
                self.pose_number += 1
                if self.pose_number == self.num_poses:
                    self.finish("success")
                    return
                else:
                    print(f"Starting waypoint {self.pose_number}")

            if self.pose_number < self.num_poses:
                target_angle = self.angle_to_pose(target_pose)
                current_angle = self.position[2]

                angle_diff = target_angle - current_angle
                angle_wrap = (angle_diff + math.pi) % (2 * math.pi) - math.pi

                angular_vel_cmd = self.proportional_constant * angle_wrap

                # Clip to range
                angular_vel_cmd = min(angular_vel_cmd, self.angular_velocity)
                angular_vel_cmd = max(angular_vel_cmd, -self.angular_velocity)

                if self.run_loop_num % 10 == 0:
                    print(f"Target angle: {math.degrees(target_angle)}")
                    print(f"Current_angle: {math.degrees(current_angle)}")
                    print(f"Angular Velocity command {angular_vel_cmd}")
                    print(f"Distance to target pose: {distance_to_target_pose}")
                    print()

                linear_vel_cmd = self.linear_velocity
                if abs(angle_diff) > math.radians(20):
                    linear_vel_cmd = 0

                self.send_drive_command(linear_vel_cmd, angular_vel_cmd)

    def execute_callback(self, goal_handle):
        goal_poses = goal_handle.request.poses
        print(f"{len(goal_poses)} Waypoints Received")

        # If existing goal goal handle or no position information from transforms
        if self.goal_handle is not None or not self.position:
            self.send_drive_command(0.0, 0.0)
            goal_handle.abort()
            result = FollowWaypoints.Result()
            result.missed_waypoints = list(range(0, len(goal_poses)))
            return result

        self.goal_handle = goal_handle

        self.done_future = concurrent.futures.Future()
        result = self.done_future.result()
        self.goal_handle = None
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
        # pose_odom.header.stamp = Time()

        try:
            # Get tranformation from odom to map
            tf_map_from_odom = self.tf_buffer.lookup_transform(
                "map",
                pose_odom.header.frame_id,
                Time(),
                timeout=rclpy.duration.Duration(seconds=0.2),
            )
        except:
            print("Transform failed from odom to map failed")
            return None

        pose_map = do_transform_pose(pose_odom.pose, tf_map_from_odom)
        # print("Odom to map succeess")
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

    def angle_to_pose(self, pose: Pose):
        x_distance = pose.position.x - self.position[0]
        y_distance = pose.position.y - self.position[1]
        return math.atan2(y_distance, x_distance)

    def finish(self, resolution):
        self.send_drive_command(0.0, 0.0)
        if resolution == "success":
            self.goal_handle.succeed()
        elif resolution == "abort":
            self.goal_handle.abort()
        elif resolution == "cancel":
            self.goal_handle.canceled()

        self.goal_handle = None
        if self.done_future and not self.done_future.done():
            result = FollowWaypoints.Result()
            if self.pose_number < self.num_poses:
                result.missed_waypoints = list(range(self.pose_number, self.num_poses))
            else:
                result.missed_waypoints = []
            self.pose_number = 0
            self.done_future.set_result(result)

    def distance_to_pose(self, pose: Pose):
        x_distance = pose.position.x - self.position[0]
        y_distance = pose.position.y - self.position[1]
        return math.sqrt((x_distance**2) + (y_distance**2))

    @property
    def target_pose(self) -> Pose:
        if self.goal_handle:
            return self.goal_handle.request.poses[self.pose_number].pose

        return None

    @property
    def num_poses(self) -> int:
        if self.goal_handle:
            return len(self.goal_handle.request.poses)

        return -1


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
