import math
import time

import numpy as np
import rclpy
import tf2_ros
from geometry_msgs.msg import Point, Pose, PoseStamped, Twist
from nav2_msgs.action import FollowWaypoints
from nav_msgs.msg import OccupancyGrid, Path, Odometry
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
        super().__init__("cleaning")

        # Parameters
        self.robot_diameter = 0.4
        self.cleaning_clearance = 0.3
        self.overlap = 0.2
        self.bump_waypoint_skip = 8
        self.position = []

        # State: get_to_cell, cleaning
        self.state = "get_to_cell"
        self.cell_number = 0
        self.waypoint_num = -1
        self.shutdown = False
        self.calling_waypoint_server = False

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
        )
        self.cleaning_inflated_map = inflate_obstacles(
            self.map_grid, self.cleaning_clearance, self.map_resolution
        )

        # Setup tf2 to convert from odom to map frame
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Get decomposition points
        self.cleaning_path = self.get_cleaning_path()
        print("Got cleaning path")

        # callback to run main loop
        self.create_timer(0.01, self.run_loop)

    def run_loop(self):
        # print(self.state)
        if self.calling_waypoint_server:
            return

        if self.state == "get_to_cell" and self.position:

            if self.cell_number == len(self.cleaning_path):
                print("Finished cleaning")

            start_point = self.coords_to_indices(self.position[0], self.position[1])
            cell = self.cleaning_path[self.cell_number]

            if self.waypoint_num == -1:
                # print(self.cleaning_path)
                if cell:
                    first_cell_point = cell[0]
                else:
                    self.cell_number += 1
                    return
            # Need to get back on track
            else:
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

            map_positions_to_start = self.index_coords_to_points(map_coords_to_start)
            map_poses_to_start = self.points_to_pose_stamped(map_positions_to_start)
            self.publish_path(map_poses_to_start)
            self.calling_waypoint_server = True
            self.call_waypoint_follow_server(map_poses_to_start)
            print(f"Following A* to cell {self.cell_number}")

        elif self.state == "cleaning":

            cleaning_path = self.cleaning_path[self.cell_number]

            if self.waypoint_num > 0:
                cleaning_path = cleaning_path[self.waypoint_num :]
                print(f"Resuming cleaning from waypoint {self.waypoint_num}")

            cleaning_poses = self.points_to_pose_stamped(cleaning_path)
            self.publish_path(cleaning_poses)
            self.calling_waypoint_server = True
            self.call_waypoint_follow_server(cleaning_poses)
            print(f"Cleaning {self.cell_number}")

    def get_cleaning_path(self):
        path = b_decomp(
            self.cleaning_inflated_map,
            self.map_resolution,
            self.robot_diameter,
            self.overlap,
        )
        non_empty_paths = []
        # map_width_m = self.map_grid.shape[1] * self.map_resolution
        # map_height_m = self.map_grid.shape[0] * self.map_resolution
        for cell in path:
            cell_coords = []
            for coord in cell:
                if not coord:
                    continue
                x, y = coord
                cell_coords.append((x + self.map_x_origin, y + self.map_y_origin))
                # cell_coords.append(
                #     (
                #         (map_width_m - x) + self.map_x_origin,
                #         y + self.map_y_origin,
                #     )
                # )
            if cell_coords:
                non_empty_paths.append(cell_coords)

        return non_empty_paths

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

    def get_map(self):
        while not self.map_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("service not available, waiting again...")
        map_future = self.map_cli.call_async(GetMap.Request())
        rclpy.spin_until_future_complete(self, map_future)
        map = map_future.result().map
        map_grid = np.array(map.data).reshape((map.info.height, map.info.width))
        map_grid = np.where(map_grid == 100, 100, 0).astype(map_grid.dtype)
        return map, map_grid

    def odom_to_map(self, odom_msg):
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
        if not self.waypoints_client.wait_for_server(timeout_sec=2.0):
            self.get_logger().error("follow_points action server not available")
            return

        goal = FollowWaypoints.Goal()
        goal.poses = poses

        send_future = self.waypoints_client.send_goal_async(goal)
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
                print(f"Bump sensor triggered, going to waypoint: {self.waypoint_num}")
            else:
                self.waypoint_num = -1
                self.next_state()
        else:
            self.next_state()

        self.calling_waypoint_server = False

    def coords_to_indices(self, x, y):
        col = int((x - self.map_x_origin) / self.map_resolution)
        row = int((y - self.map_y_origin) / self.map_resolution)

        return (row, col)

    def make_pose(self, x, y, frame_id="odom"):
        ps = PoseStamped()
        ps.header.frame_id = frame_id
        ps.header.stamp = self.get_clock().now().to_msg()
        ps.pose.position.x = float(x)
        ps.pose.position.y = float(y)
        ps.pose.position.z = 0.0
        ps.pose.orientation.w = 1.0
        return ps

    def points_to_pose_stamped(self, points):

        poses = []
        for [x, y] in points:
            pose = self.make_pose(x, y, frame_id="map")
            poses.append(pose)

        return poses

    def next_state(self):
        if self.state == "get_to_cell":
            self.state = "cleaning"
        elif self.state == "cleaning":
            self.state = "get_to_cell"
            self.cell_number += 1
            self.waypoint_num = -1
        else:
            print(f"State is abnormal: {self.state}")

    def index_coords_to_points(self, coords):
        points = []
        for [row, col] in coords:
            x = self.map_x_origin + self.map_resolution * col
            y = self.map_y_origin + self.map_resolution * row
            points.append((x, y))

        return points

    def publish_path(self, poses):
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
