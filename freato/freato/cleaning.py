"""This file contains the ROS node to command cleaning through coverage path planning"""

import numpy as np
import rclpy
import tf2_ros
from geometry_msgs.msg import PoseStamped, Twist
from nav2_msgs.action import FollowWaypoints
from nav_msgs.msg import Path, Odometry
from nav_msgs.srv import GetMap
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.time import Time
from tf2_geometry_msgs import do_transform_pose
from visualization_msgs.msg import Marker

from freato.a_star import a_star, inflate_obstacles
from freato.angle_helpers import quaternion_to_yaw
from freato.b_decomp import b_decomp


class Cleaning(Node):
    def __init__(self):
        """
        Initialize the cleaning node.

        Publishers:
            cmd_vel (geometry_msgs/Twist): direct velocity commands
            /target (visualization_msgs/Marker): debug target marker
            /trajectory (nav_msgs/Path): visualization of planned paths

        Subscribers:
            /odom (nav_msgs/Odometry): robot pose updates
        """
        super().__init__("cleaning")

        # Parameters
        self.robot_diameter = 0.4  # width of robot used for A* path planning
        self.cleaning_clearance = 0.3  # extra clearance when generating cleaning lanes
        self.overlap = 0.2  # lane overlap to avoid missed strips
        self.bump_waypoint_skip = 8  # how many waypoints to skip forward after a bump
        self.position = []

        # State: get_to_cell, cleaning
        self.state = "get_to_cell"  # which phase the robot is in
        self.cell_number = 0  # current cleaning cell index
        self.waypoint_num = -1  # waypoint index to resume from after a bump
        self.shutdown = False  # flag to end node
        self.calling_waypoint_server = False  # prevents overlapping action calls

        # Set up publishers and subscriptions
        self.vel_pub = self.create_publisher(Twist, "cmd_vel", 10)
        self.create_subscription(Odometry, "/odom", self.process_odom, 10)
        self.marker_pub = self.create_publisher(Marker, "/target", 10)
        self.path_pub = self.create_publisher(Path, "/trajectory", 10)

        # Initialize the waypoint action server
        self.waypoints_client = ActionClient(self, FollowWaypoints, "follow_points")
        self.waypoint_result_future = None

        # Initialize map client and get map
        self.map_cli = self.create_client(GetMap, "map_server/map")
        self.map, self.map_grid = self.get_map()
        self.map_resolution = self.map.info.resolution
        self.map_x_origin = self.map.info.origin.position.x
        self.map_y_origin = self.map.info.origin.position.y

        self.a_star_inflated_map = inflate_obstacles(
            self.map_grid, self.robot_diameter + 0.1, self.map_resolution
        )  # used when traveling between cells
        self.cleaning_inflated_map = inflate_obstacles(
            self.map_grid, self.cleaning_clearance, self.map_resolution
        )  # used for coverage decomposition

        # Setup tf2 to convert from odom to map frame
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Get decomposition points
        self.cleaning_path = self.get_cleaning_path()
        print("Got cleaning path")

        # callback to run main loop
        self.create_timer(0.01, self.run_loop)

    def run_loop(self):
        """Main state machine loop for navigating to cells and cleaning them"""
        # If we're already waiting on waypoint follower, don't enqueue another goal
        if self.calling_waypoint_server:
            return

        if self.state == "get_to_cell" and self.position:

            if self.cell_number == len(self.cleaning_path):
                # All cells processed; nothing more to send
                print("Finished cleaning")

            start_point = self.coords_to_indices(self.position[0], self.position[1])
            cell = self.cleaning_path[self.cell_number]

            if self.waypoint_num == -1:
                # print(self.cleaning_path)
                if cell:
                    # Normal entry to a cell: head to its first waypoint
                    first_cell_point = cell[0]
                else:
                    # Empty cell, advance to next
                    self.cell_number += 1
                    return
            # Need to get back on track
            else:
                # Skip to the first missed waypoint inside this cell
                first_cell_point = cell[self.waypoint_num]
                print(f"Running A* to get back on track")

            end_point = self.coords_to_indices(first_cell_point[0], first_cell_point[1])
            map_coords_to_start = a_star(
                self.a_star_inflated_map, start_point, end_point
            )
            if map_coords_to_start is None:
                print(
                    f"A* to start cell {self.cell_number} at ({first_cell_point[0]:2f}, {first_cell_point[1]:2f}) failed"
                )

            # Convert grid waypoints back into map-frame meters for navigation
            map_positions_to_start = self.index_coords_to_points(map_coords_to_start)
            map_poses_to_start = self.points_to_pose_stamped(map_positions_to_start)
            self.publish_path(map_poses_to_start)
            self.calling_waypoint_server = True  # block new goals until action returns
            # Hand off to nav stack to reach the cell entry point
            self.call_waypoint_follow_server(map_poses_to_start)
            print(f"Following A* to cell {self.cell_number}")

        elif self.state == "cleaning":

            cleaning_path = self.cleaning_path[self.cell_number]

            if self.waypoint_num > 0:
                # Resume from the bumped waypoint instead of starting at the top
                cleaning_path = cleaning_path[self.waypoint_num :]
                print(f"Resuming cleaning from waypoint {self.waypoint_num}")
            else:
                # Fresh pass through the cell, start at its first waypoint
                self.waypoint_num = 0

            cleaning_poses = self.points_to_pose_stamped(cleaning_path)
            self.publish_path(cleaning_poses)
            self.calling_waypoint_server = True  # avoid queueing overlapping cell goals
            # Call waypoint follower to cover the current cell
            self.call_waypoint_follow_server(cleaning_poses)
            print(f"Cleaning {self.cell_number}")

    def get_cleaning_path(self):
        """Build the list of cleaning cell paths using Boustrophedon decomposition"""
        path = b_decomp(
            self.cleaning_inflated_map,
            self.map_resolution,
            self.robot_diameter,
            self.overlap,
        )
        non_empty_paths = []
        for cell in path:
            cell_coords = []
            for coord in cell:
                if not coord:
                    continue
                x, y = coord
                # Shift decomposition coords (relative to map origin) into map frame
                cell_coords.append((x + self.map_x_origin, y + self.map_y_origin))
            if cell_coords:
                non_empty_paths.append(cell_coords)

        return non_empty_paths

    def process_odom(self, msg):
        """Callback function for /odom subscription"""
        map_pose = self.odom_to_map(msg)
        if map_pose:
            q = map_pose.orientation
            yaw = quaternion_to_yaw(q)
            self.position = [
                map_pose.position.x,
                map_pose.position.y,
                yaw,
            ]

    def get_map(self):
        """Retrieve the static map from the map server and convert to numpy grid"""
        while not self.map_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("service not available, waiting again...")
        map_future = self.map_cli.call_async(GetMap.Request())
        rclpy.spin_until_future_complete(
            self, map_future
        )  # block until map is received
        map = map_future.result().map
        map_grid = np.array(map.data).reshape((map.info.height, map.info.width))
        # Treat occupied cells as 100 and everything else as free
        map_grid = np.where(map_grid == 100, 100, 0).astype(map_grid.dtype)
        return map, map_grid

    def odom_to_map(self, odom_msg):
        """Transform a pose from odom frame into map frame"""
        pose_odom = PoseStamped()
        pose_odom.header = odom_msg.header
        pose_odom.pose = odom_msg.pose.pose

        try:
            # Get tranformation from odom to map
            tf_map_from_odom = self.tf_buffer.lookup_transform(
                "map",
                pose_odom.header.frame_id,
                Time(),
                timeout=rclpy.duration.Duration(seconds=0.2),
            )
        except Exception:
            print("Transform failed from odom to map failed")
            return None

        pose_map = do_transform_pose(pose_odom.pose, tf_map_from_odom)
        # print("Odom to map succeess")
        return pose_map

    def call_waypoint_follow_server(self, poses):
        """Calls the waypoint following server and sets relevant variables"""
        if not self.waypoints_client.wait_for_server(timeout_sec=2.0):
            self.get_logger().error("follow_points action server not available")
            return

        goal = FollowWaypoints.Goal()
        goal.poses = poses

        send_future = self.waypoints_client.send_goal_async(goal)
        # Attach callbacks so we can react once the action server responds/finishes
        send_future.add_done_callback(self._on_goal_response)

    def _on_goal_response(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error("Goal rejected")
            return

        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._on_waypoints_done)

    def _on_waypoints_done(self, future):
        result = future.result().result
        missed_waypoints = list(result.missed_waypoints)
        self.get_logger().info(f"Done. Missed waypoints: {missed_waypoints}")

        print(f"Missed waypoints: {missed_waypoints}")

        if missed_waypoints:

            self.waypoint_num += missed_waypoints[0] + self.bump_waypoint_skip
            if self.waypoint_num < len(self.cleaning_path[self.cell_number]):
                self.state = "get_to_cell"
                # After a bump, jump forward in the same cell to avoid the obstacle
                print(f"Bump sensor triggered, going to waypoint: {self.waypoint_num}")
            else:
                self.waypoint_num = -1
                self.next_state()
        else:
            self.next_state()

        self.calling_waypoint_server = False

    def coords_to_indices(self, x, y):
        """Turns a position in the map frame into map grid row and col
        Args:
            x: the x position in actual distance in the map frame
            y: the y position in actual distance in the map frame
        Returns:
            (row, col): a tuple of the integer grid coordinates on the map grid
        """
        col = int((x - self.map_x_origin) / self.map_resolution)
        row = int((y - self.map_y_origin) / self.map_resolution)

        return (row, col)

    def make_pose(self, x, y, frame_id="odom"):
        """Turns an x-y position into a ROS pose
        Args:
            x: the x position in actual distance in the frame_id frame
            y: the y position in actual distance in the frame_id frame
            frame_id: the ROS reference frame the coordinates are in
        Returns:
            A PoseStamped position that can be understood by ROS
        """
        ps = PoseStamped()
        ps.header.frame_id = frame_id
        ps.header.stamp = self.get_clock().now().to_msg()
        ps.pose.position.x = float(x)
        ps.pose.position.y = float(y)
        ps.pose.position.z = 0.0
        ps.pose.orientation.w = 1.0
        return ps

    def points_to_pose_stamped(self, points):
        """Turns a list of map-frame (x, y) points into a list of ROS poses
        Args:
            points: a list of (x, y) points in the map frame
        Returns:
            the list of points as a list of ROS poses
        """

        poses = []
        for [x, y] in points:
            pose = self.make_pose(x, y, frame_id="map")
            poses.append(pose)

        return poses

    def next_state(self):
        """Advance the cleaning state machine to the next phase"""
        if self.state == "get_to_cell":
            self.state = "cleaning"
        elif self.state == "cleaning":
            self.state = "get_to_cell"
            self.cell_number += 1
            self.waypoint_num = -1
        else:
            print(f"State is abnormal: {self.state}")

    def index_coords_to_points(self, coords):
        """Convert map grid indices back to map-frame (x, y) points"""
        points = []
        for [row, col] in coords:
            x = self.map_x_origin + self.map_resolution * col
            y = self.map_y_origin + self.map_resolution * row
            points.append((x, y))

        return points

    def publish_path(self, poses):
        """Publish a ROS Path message made from a list of poses"""
        path = Path()
        path.header.frame_id = "map"
        path.header.stamp = self.get_clock().now().to_msg()
        path.poses = poses
        self.path_pub.publish(path)


def main(args=None):
    """Boilerplate main method to run the ros2 node"""
    rclpy.init(args=args)
    cleaning = Cleaning()
    while rclpy.ok() and not cleaning.shutdown:
        rclpy.spin_once(cleaning, timeout_sec=0.5)
    print("Shutting down node")
    cleaning.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
