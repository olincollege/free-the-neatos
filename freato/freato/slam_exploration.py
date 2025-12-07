import rclpy # ros2 python
from rclpy.node import Node # ros node
from nav_msgs.msg import OccupancyGrid # SLAM map ros message type
import numpy as np

class SLAM_Exploration(Node):
    def __init__(self):
        super().__init__('slam_exploration')
        
        # Parameters to change
        self.map_topic_name = '/map'
        self.frontiers_topic_name = 'NAME'
        self.robot_width = 64

        # Node variables
        self.ranked_frontier_coords:list[tuple[int,int]]
        
        # Circular mask of the space needed for robot to fit in any orientation
        self.CIRCLE_MASK = np.ones((self.robot_width,self.robot_width), dtype=bool)        
        origin = float(self.robot_width)/2-0.5
        with np.nditer(self.CIRCLE_MASK, flags=['multi_index'], op_flags=['writeonly']) as itr:
            for cell in itr:
                cell[...] = (origin-itr.multi_index[0]) ** 2 + (origin-itr.multi_index[1]) ** 2 < (float(self.robot_width)/2) ** 2
                print(cell)
        print(self.CIRCLE_MASK)
            

        # Subscriber to map topic: calls function to update frontiers
        self.map_subscriber = self.create_subscription(
            OccupancyGrid,
            self.map_topic_name,
            self.update_frontiers,
            10 )
        # Publisher for ranked list of frontier coordinates
        # self.frontiers_publisher = self.create_publisher(TYPE,
        # 'self.frontiers_topic_name'
        # , 10)
        
        
    def update_frontiers(self,msg):
        """
        Callback function that creates a ranked list of unexplored 'frontiers'
        These frontiers are points the robot can occupy that are roughly on
        the boundary of explored and unexplored space. They are ranked based
        on how close they are to a set distance away from the robot. This
        list of frontier coordinates is saved to self.ranked_frontier_coords.
        """
        start_time = self.get_clock().now().nanoseconds
        self.get_logger().info("Updating frontiers list")

        map_grid = np.array(msg.data).reshape((msg.info.height,msg.info.width))
        print(map_grid.shape)

        var = 0
        itr = np.nditer(map_grid[:map_grid.shape[0]-self.robot_width][:map_grid.shape[1]-self.robot_width],flags=['multi_index'])
        for i in itr:
            print("start location",itr.multi_index)
            initial_zero = itr.multi_index[0]
            initial_one = itr.multi_index[1]
            additional_0 = itr.multi_index[0]+self.robot_width
            additional_1 = itr.multi_index[1]+self.robot_width
            print(itr.multi_index[0])
            print(itr.multi_index[1])
            print(initial_zero)
            print(initial_one)
            print(additional_0)
            print(additional_1)

            partial_grid = map_grid[initial_zero:additional_0][initial_one:additional_1]
            print("map grid",map_grid.shape)
            partial_grid_2 = map_grid[0:64:1][1:64:1]
            
            print("size of partial grid",partial_grid.shape, partial_grid_2.shape)
            mask_itr = np.nditer(partial_grid, flags=['multi_index'])
            
            # mask_itr = np.nditer(map_grid[itr.multi_index[0]:itr.multi_index[0]+self.robot_width][itr.multi_index[1]:itr.multi_index[1]+self.robot_width], flags=['multi_index'])
            for cell in mask_itr:
                print(mask_itr.multi_index)
                print(cell, type(cell))
                if self.CIRCLE_MASK[mask_itr.multi_index[0]][mask_itr.multi_index[1]] and (cell == 100):
                    break;
                else:
                    var += 1
        print(var)

        duration = str(float(self.get_clock().now().nanoseconds - start_time)/1_000_000_000)
        _ = self.get_logger().info("List updated in %s seconds" % duration)

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
