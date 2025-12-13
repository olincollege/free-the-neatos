from numpy.ma.core import MaskedArrayFutureWarning
import rclpy  # ros2 python
from rclpy.node import Node  # ros node
from nav_msgs.msg import OccupancyGrid  # SLAM map ros message type
from nav2_msgs.msg import ParticleCloud, Particle  # For publishing frontiers to rviz
from geometry_msgs.msg import Pose, Point
from rclpy.qos import qos_profile_sensor_data
import numpy as np
from nav_msgs.srv import GetMap
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped

from freato.a_star import a_star, inflate_obstacles
import time
import math

from rclpy.action import ActionClient
from nav2_msgs.action import FollowWaypoints


class TestAStar(Node):
    def __init__(self):
        super().__init__("test_a_star")

        # # Parameters to change
        # self.map_topic_name = "/map"
        # self.frontiers_topic_name = "NAME"
        # self.map_frame = "map"  # Name of map coordinate frame
        # self.particle_cloud_name = "particle_cloud"
        # self.dtype = [("row", int), ("col", int), ("empty_percent", float)]

        self.robot_diameter = 0.4

        self.cli = self.create_client(GetMap, "map_server/map")
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("service not available, waiting again...")
        self.future = self.cli.call_async(GetMap.Request())
        rclpy.spin_until_future_complete(self, self.future)
        self.map = self.future.result().map
        self.map_grid = np.array(self.map.data).reshape(
            (self.map.info.height, self.map.info.width)
        )

        self.map_resolution = self.map.info.resolution
        self.map_x_origin = self.map.info.origin.position.x
        self.map_y_origin = self.map.info.origin.position.y

        with np.printoptions(threshold=np.inf):
            # print(self.map_grid)
            print(self.map_grid.shape)
            print(np.unique(self.map_grid))

        # print(a_star(self.map_grid, (0, 0), (1600, 560)))

        self.path_pub = self.create_publisher(Path, "/trajectory", 10)
        self.inflated_map_pub = self.create_publisher(
            OccupancyGrid, "/inflated_map", 10
        )

        inflated_map = inflate_obstacles(
            self.map_grid, self.robot_diameter, self.map_resolution
        )

        start = self.coords_to_indices(-0.7, -2.4)
        end = self.coords_to_indices(1.6, -2.63)

        print(start)

        path_points = a_star(inflated_map, start, end)

        print(path_points[0])

        path_msg = self.index_points_to_path(path_points)

        # for _ in range(100):
        print(f"Publishing path with {len(path_points)} points")
        self.path_pub.publish(path_msg)
        self.publish_inflated_map()
        # time.sleep(10)

        self.client = ActionClient(self, FollowWaypoints, "follow_points")

        self.call_waypoint_follow_server(path_points)

        # # Subscriber to map topic: calls function to update frontiers
        # self.map_subscriber = self.create_subscription(
        #     OccupancyGrid, self.map_topic_name, self.update_frontiers, 10
        # )

        # self.map_grid = None

    def make_pose(self, x, y, frame_id="odom"):
        ps = PoseStamped()
        ps.header.frame_id = frame_id
        ps.header.stamp = self.get_clock().now().to_msg()
        ps.pose.position.x = float(x)
        ps.pose.position.y = float(y)
        ps.pose.position.z = 0.0
        ps.pose.orientation.w = 1.0
        return ps

    def index_points_to_pose_stamped(self, points, frame_id="map"):
        poses = []
        for [row, col] in points:
            pose_x = self.map_x_origin + self.map_resolution * (col + 0.5)
            pose_y = self.map_y_origin + self.map_resolution * (row + 0.5)

            pose = self.make_pose(pose_x, pose_y, frame_id=frame_id)

            poses.append(pose)

        return poses

    def index_points_to_path(self, points, frame_id="map"):
        path = Path()
        path.header.frame_id = frame_id
        path.header.stamp = self.get_clock().now().to_msg()
        path.poses = self.index_points_to_pose_stamped(points, frame_id=frame_id)
        return path

    def publish_inflated_map(self):
        inflated_grid = inflate_obstacles(
            self.map_grid, self.robot_diameter, self.map.info.resolution
        )

        msg = OccupancyGrid()
        msg.header.frame_id = self.map.header.frame_id or "map"
        msg.header.stamp = self.get_clock().now().to_msg()

        # Copy map metadata (resolution, width/height, origin)
        msg.info = self.map.info

        # OccupancyGrid.data must be a flat sequence length = width*height
        # Ensure int8-ish values (-1..100) and row-major flatten
        msg.data = inflated_grid.astype(np.int8).ravel(order="C").tolist()

        self.inflated_map_pub.publish(msg)

    def coords_to_indices(self, x, y):
        col = int((x - self.map_x_origin) / self.map_resolution)
        row = int((y - self.map_y_origin) / self.map_resolution)

        return (row, col)

    def update_frontiers(self, msg):
        """
        Callback function that creates a ranked list of unexplored 'frontiers'
        These frontiers are points the robot can occupy that are roughly on
        the boundary of explored and unexplored space. They are ranked based
        on how close they are to a set distance away from the robot. This
        list of frontier coordinates is saved to self.ranked_frontier_coords.
        """
        self.map_grid = np.array(msg.data).reshape((msg.info.height, msg.info.width))

        print(self.map_grid)

    def call_waypoint_follow_server(self, position_indices):
        print("called waypoint follow server")
        if not self.client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("follow_points action server not available")
            return

        goal = FollowWaypoints.Goal()
        goal.poses = self.index_points_to_pose_stamped(position_indices, frame_id="map")

        send_future = self.client.send_goal_async(
            goal, feedback_callback=self.feedback_cb
        )
        rclpy.spin_until_future_complete(self, send_future)

        goal_handle = send_future.result()
        if not goal_handle.accepted:
            self.get_logger().error("Goal rejected")
            return

        self.get_logger().info("Goal accepted")

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)

        result = result_future.result().result
        self.get_logger().info(
            f"Done. Missed waypoints: {list(result.missed_waypoints)}"
        )

    def feedback_cb(self, fb_msg):
        self.get_logger().info(
            f"Currently on waypoint: {fb_msg.feedback.current_waypoint}"
        )


def main(args=None):
    """Boilerplate main method to run the ros2 node"""
    rclpy.init(args=args)
    test_a_star = TestAStar()
    rclpy.spin(test_a_star)

    test_a_star.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
