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

        # Node variables
        self.ranked_frontier_coords:list[tuple[int,int]]

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

        for  in flattened_map:
            for i in flattened_map[0:64*64]:
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
