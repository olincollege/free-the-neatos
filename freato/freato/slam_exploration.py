import rclpy  # ros2 python
from rclpy.action import ActionClient  # used to call waypoint follower
from rclpy.node import Node  # ros node
from nav_msgs.msg import Odometry  # keep track of where robot is to set start of A*
from nav_msgs.msg import OccupancyGrid  # SLAM map ros message type
from nav_msgs.msg import Path  # ros datatype to publish waypoints to rviz
from nav2_msgs.action import FollowWaypoints  # ros action type of waypoint follower
from nav2_msgs.msg import ParticleCloud, Particle  # for publishing frontiers to rviz
from geometry_msgs.msg import Pose, Point  # published frontier representation
from geometry_msgs.msg import PoseStamped  # for publishing waypoints to rviz
from rclpy.qos import qos_profile_sensor_data  # sensor profile for frontier publisher
import numpy as np  # quickly manipulate map array to find frontiers
import math  # various calculations
from freato.a_star import a_star, inflate_obstacles  # a_star implementation functions
from slam_toolbox.srv import SaveMap # server for saving map
from std_msgs.msg import String # for saving map
from datetime import datetime # for saving map with time stamp


class SLAM_Exploration(Node):
    """ A ROS2 node to direct exploration during SLAM mapping
    This node uses the ROS SLAM toolbox and a waypoint action server to generate
    a SLAM map by selecting the most useful 'frontier' to travel to to build the
    map further, then using A* to generate waypoints to send to the action
    server. When there are no 'frontiers' left to pathfind to, it saves the map
    to a file based on the timestamp.
    Publishers:
        self.path_pub: publishes the path the A* algorithm creates
        self.inflated_map_pub: publishes the map A* uses to account for the
            space the robot takes up when finding the best path around obstacles
        self.frontiers_publisher: publishes frontiers as a particle cloud for
            visualization in rviz
    Subscribers: 
        self.map_subscriber: subscribes to the latest maps from the SLAM toolbox
        self.odom_subscriber: subscribes to robot odometry from the SLAM toolbox
    Clients:
        self.waypoints_client: calls waypoint follwing action server
        self.save_map_client: calls action server that saves the latest SLAM map
    
    
    """
    def __init__(self):
        super().__init__("slam_exploration")

        # Parameters to change
        self.map_topic_name = "/map" # Name of map published by SLAM toolbox
        self.odom_topic_name = "/odom" # Name of odom published by SLAM toolbox
        self.map_frame = "map"  # Name of map coordinate frame
        self.frontier_particle_cloud_name = "frontier_particles" # Name of
                                                                 # published
                                                                 # frontiers
                                                                 # for rviz2
        self.robot_grid_width = 8 # width of robot used for finding frontier
                                  # areas the robot can fit in
        self.robot_diameter = 0.4 # width of robot used for A* path planning

        lower_empty_cell_percent = 30 # minimum % of empty cells vs. unknown to
                                      # consider an unoccupied area a frontier
        upper_empty_cell_percent = 95 # maximum % of ^^^

        # Node variables and state tracking

        self.get_new_frontier = True # True when the code to find a new frontier
                                     # on the map should run
        self.unprocessed_map = False # True when a map has been received from
                                     # the SLAM toolbox but no frontier list has
                                     # been made from it yet

        self.latest_map = None 
        self.latest_map_grid = None

        self.end_when_action_done = False # True if there are no frontiers left
                                          # but waypoint follower isn't done
        self.action_is_complete = True # True after latest frontier is reached

        self.waypoints_list = None 
        self.num_waypoints = 0
        self.waypoint_result_future = None
      
        self.latest_odom = None

        # array and datatype for the weighted list of frontier coords
        self.dtype = [("row", int), ("col", int), ("weight", float)]
        self.ranked_frontier_coords = np.array([], dtype=self.dtype)

        # Circular mask of the space needed for robot to fit in any orientation
        self.CIRCLE_MASK = np.ones(
            (self.robot_grid_width, self.robot_grid_width), dtype=bool
        )
        origin = float(self.robot_grid_width) / 2 - 0.5
        with np.nditer(
            self.CIRCLE_MASK, flags=["multi_index"], op_flags=["writeonly"]
        ) as itr:
            for cell in itr:
                cell[...] = (origin - itr.multi_index[0]) ** 2 + (
                    origin - itr.multi_index[1]
                ) ** 2 < (float(self.robot_grid_width) / 2) ** 2
                
        
        self.CIRCLE_CELL_COUNT = np.sum(self.CIRCLE_MASK)
        self.LOWER_EMPTY_CELL_COUNT = int(
            self.CIRCLE_CELL_COUNT * lower_empty_cell_percent / 100
        )
        self.UPPER_EMPTY_CELL_COUNT = int(
            self.CIRCLE_CELL_COUNT * upper_empty_cell_percent / 100
        )
        # print(
            # "Lower and upper empty cell count boundaries for frontier determination:",
            # self.LOWER_EMPTY_CELL_COUNT,
            # self.UPPER_EMPTY_CELL_COUNT,
        # )
        # print("Out of total number of cells:", self.CIRCLE_CELL_COUNT)

        # Subscriber to map topic: calls function to update frontiers
        self.map_subscriber = self.create_subscription(
            OccupancyGrid, self.map_topic_name, self.update_map, 10
        )

        # Subscriber to odom data, for giving starting position of A*
        self.odom_subscriber = self.create_subscription(
            Odometry, self.odom_topic_name, self.odom_cb, 10
        )

        # Publisher for rviz2 visualization of frontier coordinates
        self.frontiers_publisher = self.create_publisher(
            ParticleCloud, self.frontier_particle_cloud_name, qos_profile_sensor_data
        )

        # Publisher for the current waypoint trajectory to frontiers
        self.path_pub = self.create_publisher(Path, "/trajectory", 10)

        # Publisher for the map used by A* that accounts for robot size
        self.inflated_map_pub = self.create_publisher(
            OccupancyGrid, "/inflated_map", 10
        )

        # Create client for the waypoint following server
        self.waypoints_client = ActionClient(self, FollowWaypoints, "follow_points")

        # callback to run main loop
        self.create_timer(0.001, self.run_loop)

        # Map saving service
        self.save_map_client = self.create_client(SaveMap, "/slam_toolbox/save_map")
        while not self.save_map_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("service not available, waiting again...")

        # When ready to shutdown
        self.shutdown = False

    def run_loop(self):
        """Holds the main loop to command driving based on the frontiers list"""
        # wait until map exists
        if self.latest_map is None:
            return
        
        if self.get_new_frontier:
            if self.unprocessed_map:
                print("update map")
                self.update_frontiers()
                self.unprocessed_map = False
                self.waypoints_list = self.select_valid_frontier()
                if (
                    self.waypoints_list is None
                    and self.ranked_frontier_coords.shape[0] == 0
                ):
                    self.end_when_action_done = True
                    if self.action_is_complete:
                        self.save_map_and_end_node()
                    return
            
            if self.action_is_complete:
                if self.end_when_action_done:
                    self.save_map_and_end_node()
                # Tell robot to travel towards a new frontier
                if self.waypoints_list is not None:
                    self.call_waypoint_follow_server(self.waypoints_list)
                    # Display info about the waypoints and map
                    waypoints_path_msg = self.index_points_to_path(self.waypoints_list)
                    self.num_waypoints = len(self.waypoints_list)
                    print(f"Publishing path with {len(self.waypoints_list)} points")
                    self.path_pub.publish(waypoints_path_msg)
                else:
                    print("Frontier finished, A* path find failed")
                self.publish_inflated_map()

    def update_map(self, msg):
        """Callback to save latest map from SLAM toolbox
        self.latest_map holds the ros message and metadata
        self.latest_map_grid holds the map as a numpy array
        """
        self.latest_map = msg
        self.latest_map_grid = np.array(msg.data).reshape(
            (msg.info.height, msg.info.width)
        )
        # print("Map has shape:", self.latest_map_grid.shape)
        self.unprocessed_map = True 
        self.get_logger().info("Received new map")

    def update_frontiers(self):
        """
        Function that creates a ranked list of unexplored 'frontiers'
        These frontiers are points the robot can occupy that are roughly on
        the boundary of explored and unexplored space. They are ranked based
        on how close they are to a set distance away from the robot. The ordered
        list of frontier coordinates is saved to self.ranked_frontier_coords.
        """
        map = self.latest_map
        start_time = self.get_clock().now().nanoseconds
        self.get_logger().info("Updating frontiers list")

        self.ranked_frontier_coords = np.array([], dtype=self.dtype)

        map_grid = self.latest_map_grid
        var = 0

        # loop through every robot-sized spot in the map
        itr = np.nditer(
            map_grid[
                : map_grid.shape[0] - self.robot_grid_width,
                : map_grid.shape[1] - self.robot_grid_width,
            ],
            flags=["multi_index"],
        )
        for i in itr:
            empty_cell_count = 0
            initial_zero = itr.multi_index[0]
            initial_one = itr.multi_index[1]
            additional_0 = itr.multi_index[0] + self.robot_grid_width
            additional_1 = itr.multi_index[1] + self.robot_grid_width
            partial_grid = map_grid[initial_zero:additional_0, initial_one:additional_1]

            # loop through every element in a circular area around each point
            mask_itr = np.nditer(partial_grid, flags=["multi_index"])
            for cell in mask_itr:
                if self.CIRCLE_MASK[mask_itr.multi_index[0]][mask_itr.multi_index[1]]:
                    if cell == 100: # obstacle at this point, not a frontier
                        empty_cell_count = self.UPPER_EMPTY_CELL_COUNT + 1
                        break
                    else:
                        if cell == 0:
                            empty_cell_count += 1
            if ( # determine if this point is a frontier and how useful it is
                self.LOWER_EMPTY_CELL_COUNT
                < empty_cell_count
                < self.UPPER_EMPTY_CELL_COUNT
            ):
                this_weight = self.calculate_frontier_weight(
                    itr.multi_index[0], itr.multi_index[1], empty_cell_count
                )
                self.ranked_frontier_coords = np.append(
                    self.ranked_frontier_coords,
                    np.array(
                        [(itr.multi_index[0], itr.multi_index[1], this_weight)],
                        dtype=self.dtype,
                    ),
                )

        self.ranked_frontier_coords.sort(order="weight")
        # with np.printoptions(threshold=np.inf):
        #     print(self.ranked_frontier_coords)
        self.publish_frontiers(self.ranked_frontier_coords, map.header.stamp, map.info)
        duration = str(
            float(self.get_clock().now().nanoseconds - start_time) / 1_000_000_000
        )
        _ = self.get_logger().info("List updated in %s seconds" % duration)

    def publish_frontiers(self, frontiers, timestamp, map_info):
        """Publish frontiers as a particle cloud for visualization"""
        msg = ParticleCloud()
        msg.header.frame_id = self.map_frame
        msg.header.stamp = timestamp
        for frontier in frontiers[:1]:
            frontier_pose = Pose(
                position=Point(
                    x=self.frontier_list_grid_col_to_x_pos(frontier["col"]),
                    y=self.frontier_list_grid_row_to_y_pos(frontier["row"]),
                    z=0.0,
                )
            )
            msg.particles.append(
                Particle(pose=frontier_pose, weight=1 / (0.00001 + frontier["weight"]))
            )
            self.frontiers_publisher.publish(msg)

    def frontier_list_grid_col_to_x_pos(self, col_index):
        """
        Converts a column in the map grid into its x position in the map frame.
        Handles the offset used by the top-left array indexing when determining
        valid frontiers.
        Args:
            col_index: the column in the map grid
        Returns:
            the x position in the map frame of the center of this column
        """
        return (
            self.latest_map.info.origin.position.x
            + (0.5 + self.robot_grid_width / 2 + float(col_index))
            * self.latest_map.info.resolution
        )

    def frontier_list_grid_row_to_y_pos(self, row_index):
        """
        Converts a column in the map grid into its y position in the map frame
        Also handles the offset used by the array when finding frontiers
        """
        return (
            self.latest_map.info.origin.position.y
            + (0.5 + self.robot_grid_width / 2 + float(row_index))
            * self.latest_map.info.resolution
        )

    def calculate_frontier_weight(self, row, col, empty_cell_count):
        """Determine how useful a frontier is to pathfind to and assign
        a weight to it, with lower weights being better.
        Frontiers closest to the line between explored and unexplored
        and closest to a specific distance from the robot are prioritized.
        Weights are positive somewhat normalized between 0-1,
        but can be larger than 1 if the frontier is far from the robot.
        """
        empty_cell_proportion = float(empty_cell_count) / float(self.CIRCLE_CELL_COUNT)
        # Prioritizes frontiers that are closest to a specific proportion
        # of explored to unexplored space, like 50/50, since these points
        # are most centered on the boundary of explored and unexplored space.
        # The proportion should be about between zero and 1,
        # so multiplying it by 2 makes it better fit this range
        uncertainty_measurement = 2 * math.fabs(empty_cell_proportion - 0.5)

        odom_pose = self.latest_odom.pose.pose.position
        odom_indices = self.coords_to_indices(odom_pose.x, odom_pose.y)

        distance = math.sqrt(
            (row - odom_indices[0]) ** 2 + (col - odom_indices[1]) ** 2
        )
        distance_weight = (distance / self.latest_map_grid.shape[0]) * 2

        # print(f"Uncertainty measurment: {uncertainty_measurement}")
        # print(f"Distance weight: {distance_weight}")

        return uncertainty_measurement + distance_weight

    def select_valid_frontier(self):
        """
        Returns the result of the A* algorithm for the first reachable frontier
        Returns None if no valid frontier is found
        """
        inflated_map = inflate_obstacles(
            self.latest_map_grid, self.robot_diameter, self.latest_map.info.resolution
        )
        odom_pose = self.latest_odom.pose.pose.position
        start_odom_indices = self.coords_to_indices(odom_pose.x, odom_pose.y)

        for frontier in self.lightly_shuffle_frontiers():
            x_coord = self.frontier_list_grid_col_to_x_pos(frontier["col"])
            y_coord = self.frontier_list_grid_row_to_y_pos(frontier["row"])
            end_frontier_indices = self.coords_to_indices(x_coord, y_coord)
            a_star_result = a_star(
                inflated_map, start_odom_indices, end_frontier_indices
            )
            if a_star_result is not None:
                return a_star_result

        return None

    def coords_to_indices(self, x, y):
        map_x_origin = self.latest_map.info.origin.position.x
        map_y_origin = self.latest_map.info.origin.position.y
        map_resolution = self.latest_map.info.resolution
        col = int((x - map_x_origin) / map_resolution)
        row = int((y - map_y_origin) / map_resolution)

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

    def index_points_to_pose_stamped(self, points, frame_id="map"):
        poses = []
        map_x_origin = self.latest_map.info.origin.position.x
        map_y_origin = self.latest_map.info.origin.position.y
        map_resolution = self.latest_map.info.resolution
        for [row, col] in points:
            pose_x = map_x_origin + map_resolution * (col + 0.5)
            pose_y = map_y_origin + map_resolution * (row + 0.5)

            pose = self.make_pose(pose_x, pose_y, frame_id=frame_id)

            poses.append(pose)

        return poses

    def index_points_to_path(self, points, frame_id="map"):
        path = Path()
        path.header.frame_id = frame_id
        path.header.stamp = self.get_clock().now().to_msg()
        path.poses = self.index_points_to_pose_stamped(points, frame_id=frame_id)
        return path

    def lightly_shuffle_frontiers(self, swap_prob=0.25):
        rng = np.random.default_rng()
        coords_copy = self.ranked_frontier_coords.copy()
        n = coords_copy.shape[0]
        for i in range(n):
            if rng.random() < swap_prob:
                j = rng.integers(0, n)
                coords_copy[i], coords_copy[j] = (
                    coords_copy[j].copy(),
                    coords_copy[i].copy(),
                )

        return coords_copy

    def call_waypoint_follow_server(self, position_indices):
        """Calls the waypoint following server and sets relevant variables"""
        print("called waypoint follow server")
        if not self.waypoints_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("follow_points action server not available")
            return

        goal = FollowWaypoints.Goal()
        goal.poses = self.index_points_to_pose_stamped(position_indices, frame_id="map")

        send_future = self.waypoints_client.send_goal_async(
            goal, feedback_callback=self.waypoint_following_feedback_cb
        )
        print("sending future")
        # rclpy.spin_until_future_complete(self, send_future)
        print("sent future")

        print("setting waypoint server initial response callback")
        # rclpy.spin_until_future_complete(self, result_future) # this is blocking
        send_future.add_done_callback(
            self.waypoint_response_callback
        )  # this is non-blocking
        print("response callback set")
        self.get_new_frontier = False
        self.action_is_complete = False

    def waypoint_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error("Goal rejected")
            return

        self.get_logger().info("Goal accepted")

        self.waypoint_result_future = goal_handle.get_result_async()
        self.waypoint_result_future.add_done_callback(
            self.waypoint_following_finished_cb
        )

    def waypoint_following_finished_cb(self, future):
        result = future.result().result
        self.get_logger().info(
            f"Done. Missed waypoints: {list(result.missed_waypoints)}"
        )
        if self.end_when_action_done:
            self.save_map_and_end_node()
        self.action_is_complete = True
        self.get_new_frontier = True

    def waypoint_following_feedback_cb(self, fb_msg):
        self.get_logger().info(
            f"Currently on waypoint: {fb_msg.feedback.current_waypoint}"
        )
        if self.num_waypoints - fb_msg.feedback.current_waypoint <= 2:
            self.get_new_frontier = True

    def publish_inflated_map(self):
        """Publish the map used by A* that accounts for the robot's size
        when checking valid grid locations the robot can be in
        """
        inflated_grid = inflate_obstacles(
            self.latest_map_grid, self.robot_diameter, self.latest_map.info.resolution
        )

        msg = OccupancyGrid()
        msg.header.frame_id = self.latest_map.header.frame_id or "map"
        msg.header.stamp = self.get_clock().now().to_msg()

        # Copy map metadata (resolution, width/height, origin)
        msg.info = self.latest_map.info

        # OccupancyGrid.data must be a flat sequence length = width*height
        # Ensure int8-ish values (-1..100) and row-major flatten
        msg.data = inflated_grid.astype(np.int8).ravel(order="C").tolist()

        self.inflated_map_pub.publish(msg)

    def save_map_and_end_node(self):
        """Saves map with the current timestamp and any map name passed in on startup,
        then shuts down the node"""
        print("Saving map and shutting down")
        request = SaveMap.Request()
        stamp = datetime.now().strftime("%m%d_%H%M")
        map_name = f"slam_explore_map_{stamp}"
        request.name = String(data=map_name)

        future = self.save_map_client.call_async(request)
        future.add_done_callback(self._on_save_map_done)

    def _on_save_map_done(self, future):
        try:
            resp = future.result()
            print(f"Map save response: {resp}")
        except Exception as e:
            print(f"SaveMap failed: {e}")

        self.shutdown = True

    def odom_cb(self, msg):
        """Saves latest odometry message received from subscriber"""
        self.latest_odom = msg


def main(args=None):
    """Boilerplate main method to run the ros2 node"""
    rclpy.init(args=args)
    slam_exploration = SLAM_Exploration()
    while rclpy.ok() and not slam_exploration.shutdown:
        rclpy.spin_once(slam_exploration, timeout_sec=0.5)
    print("Shutting down node")
    slam_exploration.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
