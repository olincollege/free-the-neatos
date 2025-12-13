import rclpy # ros2 python
from rclpy.node import Node # ros node
from nav_msgs.msg import OccupancyGrid # SLAM map ros message type
from nav2_msgs.msg import ParticleCloud, Particle # for publishing frontiers to rviz
from geometry_msgs.msg import Pose, Point # published frontier representation
from rclpy.qos import qos_profile_sensor_data # sensor profile for frontier publisher
import numpy as np # quickly manipulate map array to find frontiers
import math # various calculations

class SLAM_Exploration(Node):
    def __init__(self):
        super().__init__('slam_exploration')
        
        # Parameters to change
        self.map_topic_name = '/map'
        self.frontiers_topic_name = 'NAME'
        self.map_frame = "map" # Name of map coordinate frame
        self.particle_cloud_name = 'particle_cloud'
        self.robot_width = 8
        lower_empty_cell_percent = 30
        upper_empty_cell_percent = 95

        # Node variables

        self.get_new_frontier = True
        self.unprocessed_map = False

        # an array of coordinates followed
        self.dtype = [('row',int),('col',int),('weight',float)]
        self.ranked_frontier_coords = np.array([], dtype=self.dtype)
        
        # Circular mask of the space needed for robot to fit in any orientation
        self.CIRCLE_MASK = np.ones((self.robot_width,self.robot_width), dtype=bool)        
        origin = float(self.robot_width)/2-0.5
        with np.nditer(self.CIRCLE_MASK, flags=['multi_index'], op_flags=['writeonly']) as itr:
            for cell in itr:
                cell[...] = (origin-itr.multi_index[0]) ** 2 + (origin-itr.multi_index[1]) ** 2 < (float(self.robot_width)/2) ** 2
                # print(cell)
        print(self.CIRCLE_MASK)
        self.CIRCLE_CELL_COUNT = np.sum(self.CIRCLE_MASK)
        self.LOWER_EMPTY_CELL_COUNT = int(self.CIRCLE_CELL_COUNT*lower_empty_cell_percent/100)
        self.UPPER_EMPTY_CELL_COUNT = int(self.CIRCLE_CELL_COUNT*upper_empty_cell_percent/100)
        print("Lower and upper empty cell count boundaries for frontier determination:", self.LOWER_EMPTY_CELL_COUNT, self.UPPER_EMPTY_CELL_COUNT)
        print("out of total number of cells:", self.CIRCLE_CELL_COUNT)
        self.latest_map = None

        # Subscriber to map topic: calls function to update frontiers
        self.map_subscriber = self.create_subscription(
            OccupancyGrid,
            self.map_topic_name,
            self.update_map,
            10 )
        # Publisher for rviz2 visualization of frontier coordinates
        self.frontiers_publisher = self.create_publisher(ParticleCloud,
        self.particle_cloud_name, qos_profile_sensor_data)

        # callback to run main loop
        self.create_timer(0.001, self.run_loop)

    def run_loop(self):
        """Holds the main loop to command driving based on the frontiers list"""
        # wait until map exists
        if self.latest_map is None: return

        if self.get_new_frontier:
            if self.unprocessed_map:
                print("update map")
                self.update_frontiers()
                self.unprocessed_map = False
                # a_star_result = select_valid_frontier()
                # if a_star_result is None:
                #     self.end_when_action_done = True 
                #     return
                # self.waypoints_list = a_star_to_waypoints(a_star_result)
            # if self.action_complete:
            #     run new action
                
    def update_map(self, msg):
        self.latest_map = msg
        self.unprocessed_map = True
        self.get_logger().info("Received new map")
        
    def update_frontiers(self):
        """
        Function that creates a ranked list of unexplored 'frontiers'
        These frontiers are points the robot can occupy that are roughly on
        the boundary of explored and unexplored space. They are ranked based
        on how close they are to a set distance away from the robot. This
        list of frontier coordinates is saved to self.ranked_frontier_coords.
        """
        map = self.latest_map
        start_time = self.get_clock().now().nanoseconds
        self.get_logger().info("Updating frontiers list")

        self.ranked_frontier_coords = np.array([],dtype=self.dtype)

        map_grid = np.array(map.data).reshape((map.info.height,map.info.width))
        print(map_grid.shape)

        var = 0


        itr = np.nditer(map_grid[:map_grid.shape[0]-self.robot_width,:map_grid.shape[1]-self.robot_width],flags=['multi_index'])
        for i in itr:
            empty_cell_count = 0
            #print("start location",itr.multi_index)
            initial_zero = itr.multi_index[0]
            initial_one = itr.multi_index[1]
            additional_0 = itr.multi_index[0]+self.robot_width
            additional_1 = itr.multi_index[1]+self.robot_width
            partial_grid = map_grid[initial_zero:additional_0,initial_one:additional_1]
            
            # print("size of partial grid",partial_grid.shape)
            mask_itr = np.nditer(partial_grid, flags=['multi_index'])
            # mask_itr = np.nditer(map_grid[itr.multi_index[0]:itr.multi_index[0]+self.robot_width][itr.multi_index[1]:itr.multi_index[1]+self.robot_width], flags=['multi_index'])
            for cell in mask_itr:
                #print(mask_itr.multi_index)
                #print(cell, type(cell))
                if (self.CIRCLE_MASK[mask_itr.multi_index[0]][mask_itr.multi_index[1]]):
                    if (cell == 100):
                        empty_cell_count = self.UPPER_EMPTY_CELL_COUNT + 1
                        break
                    else:
                        if (cell == 0):
                            empty_cell_count += 1
            if (self.LOWER_EMPTY_CELL_COUNT < empty_cell_count < self.UPPER_EMPTY_CELL_COUNT):
                this_weight = self.calculate_frontier_weight(itr.multi_index[0], itr.multi_index[1], empty_cell_count)
                self.ranked_frontier_coords = np.append(self.ranked_frontier_coords,np.array([(itr.multi_index[0],itr.multi_index[1],this_weight)],dtype=self.dtype))

        

        self.ranked_frontier_coords.sort(order='weight')
        # with np.printoptions(threshold=np.inf):
        #     print(self.ranked_frontier_coords)
            # Remember to convert them!!!!!!!!!!!!
        self.publish_frontiers(self.ranked_frontier_coords, map.header.stamp, map.info)
        duration = str(float(self.get_clock().now().nanoseconds - start_time)/1_000_000_000)
        _ = self.get_logger().info("List updated in %s seconds" % duration)

    def publish_frontiers(self, frontiers, timestamp, map_info):
        msg = ParticleCloud()
        msg.header.frame_id = self.map_frame
        msg.header.stamp = timestamp
        for frontier in frontiers[:1]:
            # TODO: Turn into method to offset by the correct amount
            frontier_pose = Pose(position=Point(x=self.frontier_list_grid_col_to_x_pos(frontier['col']),y=self.frontier_list_grid_row_to_y_pos(frontier['row']),z=0.0))
            msg.particles.append(Particle(pose=frontier_pose, weight=1/(0.00001+frontier['weight'])))
            self.frontiers_publisher.publish(msg)

    def frontier_list_grid_col_to_x_pos(self, col_index):
        """
        Converts a column in the map grid into its x position in the map frame
        Also handles the offset used by the array when finding frontiers
        """
        return self.latest_map.info.origin.position.x + (0.5 + self.robot_width/2 + float(col_index))*self.latest_map.info.resolution
            
    def frontier_list_grid_row_to_y_pos(self, row_index):
        """
        Converts a column in the map grid into its y position in the map frame
        Also handles the offset used by the array when finding frontiers
        """
        return self.latest_map.info.origin.position.y + (0.5 + self.robot_width/2 + float(row_index))*self.latest_map.info.resolution

    def calculate_frontier_weight(self, row, col, empty_cell_count):
        """Determine how useful a frontier is to pathfind to and assign
        a weight to it, with lower weights being better. 
        Frontiers closest to the line between explored and unexplored
        and closest to a specific distance from the robot are prioritized.
        Weights are positive somewhat normalized between 0-1,
        but can be larger than 1 if the frontier is far from the robot. 
        """
        empty_cell_proportion = float(empty_cell_count)/float(self.CIRCLE_CELL_COUNT)
        # Prioritizes frontiers that are closest to a specific proportion
        # of explored to unexplored space, like 50/50, since these points
        # are most centered on the boundary of explored and unexplored space.
        # The proportion should be about between zero and 1, 
        # so multiplying it by 2 makes it better fit this range
        uncertainty_measurement = 2*math.fabs(empty_cell_proportion-0.5)
        return uncertainty_measurement
        # TODO: Incorporate distance measurement into determination of best frontiers
        

def main(args=None):
    """Boilerplate main method to run the ros2 node
    """
    rclpy.init(args=args)
    slam_exploration = SLAM_Exploration()
    rclpy.spin(slam_exploration)

    slam_exploration.destroy_node()
    rclpy.shutdown()
    
if __name__ == '__main__':
    main()
