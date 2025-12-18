"""ROS action server that follows a list of waypoints, with bump-triggered retrace."""

import math
import concurrent.futures

import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer
from rclpy.executors import MultiThreadedExecutor
from rclpy.time import Time
from nav_msgs.msg import Odometry
from neato2_interfaces.msg import Bump
from geometry_msgs.msg import Twist, PoseStamped, Pose
from nav2_msgs.action import FollowWaypoints
from visualization_msgs.msg import Marker

import tf2_ros
from tf2_geometry_msgs import do_transform_pose

from freato.angle_helpers import quaternion_to_yaw


class WaypointActionServer(Node):

    def __init__(self):
        """
        Initialize the waypoint server node.

        Publishers:
            cmd_vel: geometry_msgs/Twist drive commands
            /target: visualization_msgs/Marker for RViz

        Subscribers:
            /odom: nav_msgs/Odometry for pose tracking
            /bump: neato2_interfaces/Bump to trigger retrace
        """

        super().__init__("waypoint_action_server")

        # Set up publishers/subscriptions: drive commands, odom tracking, RViz markers,
        # and bump sensor feedback for retracing
        self.vel_pub = self.create_publisher(Twist, "cmd_vel", 10)
        self.create_subscription(Odometry, "/odom", self.process_odom, 10)
        self.marker_pub = self.create_publisher(Marker, "/target", 10)
        self.create_subscription(Bump, "/bump", self.handle_bump, 10)

        # Setup tf2 to convert from odom to map frame
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Run loop drives the proportional controller toward the active waypoint
        self.create_timer(0.05, self.run_loop)

        # Speed parameters for the controller
        self.angular_velocity = math.pi / 4
        self.linear_velocity = 0.15

        # Initialize the action server that accepts waypoint batches
        self._action_server = ActionServer(
            self, FollowWaypoints, "follow_points", self.execute_callback
        )

        # Position in map frame (x, y, yaw); populated once odom transforms work
        self.position = []

        # Distance threshold for waypoint completion
        self.position_threshold = 0.15

        # Callback and route progress tracking
        self.goal_handle = None
        self.pose_number = 0
        self.done_future = None

        # Controller constant for angular velocity
        self.proportional_constant = 0.5

        self.run_loop_num = 0  # used to throttle debug output

        # Estop/retrace bookkeeping for bump handling
        self.estop = False
        self.num_retrace_points = 4
        self.curr_retrace_point = self.num_retrace_points

    def run_loop(self):
        """
        Main control loop that drives toward the current waypoint.
        """
        self.run_loop_num += 1
        if self.goal_handle:

            if self.goal_handle.is_cancel_requested:
                self.finish("cancel")
                return

            target_pose = self.target_pose
            # Publish a marker to visualize the current waypoint target
            self.publish_target_marker(target_pose)

            distance_to_target_pose = self.distance_to_pose(target_pose)

            # Reached waypoint if neato is within threshold
            if distance_to_target_pose <= self.position_threshold:
                if self.estop:
                    # Retrace mode counts down waypoints until the abort logic runs
                    self.pose_number -= 1
                    self.curr_retrace_point -= 1
                    if self.pose_number == -1 or self.curr_retrace_point == 0:
                        self.estop = False
                        self.curr_retrace_point = self.num_retrace_points
                        # Report remaining waypoints starting from the next forward
                        # target (pose_number was decremented above).
                        self.finish("abort", missed_start=self.pose_number + 1)
                    else:
                        print(f"Starting retrace waypoint {self.curr_retrace_point}")
                else:
                    # Forward mode: advance to the next waypoint or end goal
                    self.pose_number += 1
                    if self.pose_number == self.num_poses:
                        self.finish("success")
                        return
                    else:
                        print(f"Starting waypoint {self.pose_number}")

            if self.pose_number < self.num_poses:
                target_angle = self.angle_to_pose(target_pose)
                if self.estop:
                    # Reverse direction by flipping heading 180 degrees
                    target_angle += math.pi
                current_angle = self.position[2]

                angle_diff = target_angle - current_angle
                # Normalize difference to [-pi, pi] for proportional control
                angle_wrap = (angle_diff + math.pi) % (2 * math.pi) - math.pi

                angular_vel_cmd = self.proportional_constant * angle_wrap

                # Clip to be reasonable command
                angular_vel_cmd = min(angular_vel_cmd, self.angular_velocity)
                angular_vel_cmd = max(angular_vel_cmd, -self.angular_velocity)

                if self.run_loop_num % 10 == 0:
                    print(f"Target angle: {math.degrees(target_angle)}")
                    print(f"Current_angle: {math.degrees(current_angle)}")
                    print(f"Angular Velocity command {angular_vel_cmd}")
                    print(f"Distance to target pose: {distance_to_target_pose}")
                    print(f"Angle diff: {angle_diff}")
                    print()

                linear_vel_cmd = self.linear_velocity
                if abs(angle_wrap) > math.radians(20):
                    # If large error, rotate in place
                    linear_vel_cmd = 0

                if self.estop:
                    # If retracing, drive backwards
                    linear_vel_cmd *= -1

                self.send_drive_command(linear_vel_cmd, angular_vel_cmd)

    def execute_callback(self, goal_handle):
        """
        Action callback that receives waypoint lists and starts execution.

        Args:
            goal_handle: message goal handle
        """
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
        self.estop = False  # start moving forward again for new route
        self.curr_retrace_point = self.num_retrace_points

        self.done_future = concurrent.futures.Future()
        result = self.done_future.result()
        self.goal_handle = None
        return result

    def handle_bump(self, msg: Bump):
        """
        React to bumper triggers by beginning a retrace.

        Args:
            msg (Bump): bumper state message
        """
        if self.goal_handle is None:
            return

        if (
            msg.left_front == 1
            or msg.left_side == 1
            or msg.right_front == 1
            or msg.right_side == 1
        ):
            # Begin a short retrace before aborting so the caller can later skip ahead.
            self.send_drive_command(0.0, 0.0)
            self.pose_number -= 1
            if self.pose_number < 0:
                self.finish("abort", missed_start=0)
                return
            self.curr_retrace_point = self.num_retrace_points
            self.estop = True  # run_loop will reverse through previous waypoints

    def process_odom(self, msg):
        """
        Update the server's map-frame pose from odom messages.

        Args:
            msg (Odometry): odom message containing pose/orientation
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
        """
        Transform an odom-frame pose into the map frame.

        Args:
            odom_msg (Odometry): message with pose in odom frame

        Returns:
            Pose: pose transformed into map frame or None if transform fails
        """
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
        """
        Send a velocity command to the robot.

        Args:
            linear (float): linear velocity in m/s
            angular (float): angular velocity in rad/s
        """
        msg = Twist()
        msg.linear.x = float(linear)
        msg.angular.z = float(angular)
        self.vel_pub.publish(msg)

    def angle_to_pose(self, pose: Pose):
        """
        Compute the heading from the robot to the pose.

        Args:
            pose (Pose): target pose to aim at

        Returns:
            float: angle toward the pose in radians
        """
        x_distance = pose.position.x - self.position[0]
        y_distance = pose.position.y - self.position[1]
        return math.atan2(y_distance, x_distance)

    def finish(self, resolution, missed_start=None):
        """
        Finish the current action goal.

        Args:
            resolution (str): either success, abort, or cancel
            missed_start (int): first waypoint index to mark missed
        """
        if self.goal_handle is None:
            return

        self.send_drive_command(0.0, 0.0)
        total_poses = self.num_poses
        if resolution == "success":
            self.goal_handle.succeed()
        elif resolution == "abort":
            self.goal_handle.abort()
        elif resolution == "cancel":
            self.goal_handle.canceled()

        if self.done_future and not self.done_future.done():
            result = FollowWaypoints.Result()
            start = missed_start if missed_start is not None else self.pose_number
            if resolution == "success":
                result.missed_waypoints = []
            else:
                start_idx = max(min(start, total_poses - 1), 0)
                end_idx = max(total_poses, 0)
                result.missed_waypoints = list(range(start_idx, end_idx))
            self.pose_number = 0
            self.done_future.set_result(result)

        self.goal_handle = None

    def distance_to_pose(self, pose: Pose):
        """
        Compute Euclidean distance to a pose.

        Args:
            pose (Pose): pose to measure to

        Returns:
            float: distance in meters
        """
        x_distance = pose.position.x - self.position[0]
        y_distance = pose.position.y - self.position[1]
        return math.sqrt((x_distance**2) + (y_distance**2))

    def publish_target_marker(self, pose: Pose):
        """
        Publish a visualization marker at the target waypoint.

        Args:
            pose (Pose): waypoint pose to visualize
        """
        m = Marker()
        m.header.frame_id = "map"
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns, m.id = "debug", 0
        m.type, m.action = Marker.SPHERE, Marker.ADD
        m.pose = pose
        m.scale.x = m.scale.y = m.scale.z = 0.2
        m.color.r, m.color.a = 1.0, 1.0
        self.marker_pub.publish(m)

    @property
    def target_pose(self) -> Pose:
        """
        Current target pose from the active goal.

        Returns:
            Pose: pose at self.pose_number or None if no goal
        """
        if self.goal_handle:
            return self.goal_handle.request.poses[self.pose_number].pose

        return None

    @property
    def num_poses(self) -> int:
        """
        Count of poses in the active goal.

        Returns:
            int: number of poses, or -1 if no goal
        """
        if self.goal_handle:
            return len(self.goal_handle.request.poses)

        return -1


def main(args=None):
    """
    Initialize rclpy and Node, then run with a with multi threaded
    executor so action callbacks and odom callbacks can run correctly.
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
