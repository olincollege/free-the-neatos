#!/usr/bin/env python3

import rclpy
from threading import Thread
from rclpy.time import Time
from rclpy.node import Node
from std_msgs.msg import Header
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import (
    PoseWithCovarianceStamped,
    Pose,
    Point,
    Quaternion,
    Twist,
    TransformStamped,
)
from nav_msgs.msg import Odometry
from tf2_ros import StaticTransformBroadcaster
from rclpy.duration import Duration
import math
import time
import numpy as np
from freato.occupancy_field import OccupancyField
import numpy as np
import matplotlib.pyplot as plt
from freato.helper_functions import TFHelper
from freato.angle_helpers import quaternion_from_euler, euler_from_quaternion
from freato.icp import tf_from_icp as tf_icp
import copy


class ExtendedKalmanFilter(Node):

    def __init__(self):
        super().__init__("ekf")

        # defining reference frames for transformations
        self.base_frame = "base_footprint"  # the frame of the robot base
        self.map_frame = "map"  # the name of the map coordinate frame
        self.odom_frame = "odom"  # the name of the odometry coordinate frame

        self.tf_helper = TFHelper(self)  # TF helper for transforms

        # laser_subscriber listens for data from the lidar
        self.create_subscription(LaserScan, "scan", self.receive_scan, 10)
        self.scan_to_process = None  # latest scan to process
        self.last_scan_timestep = None  # time of last processed scan

        # initialize control input
        self.u = np.array([0.0, 0.0])

        # publisher the estimated pose to the 'ekf_pose' topic
        self.pose_publisher = self.create_publisher(
            PoseWithCovarianceStamped, "ekf_pose", 10
        )

        # subscribe to odometry data
        self.last_odom_pose = None  # last odom pose received from odom topic
        self.create_subscription(Odometry, "odom", self.receive_odom, 10)

        # define occupancy field object
        self.occupancy_field = OccupancyField(self)

        # extract dense point cloud from the occupancy grid
        self.map_points = self.extract_map_points()
        self.get_logger().info(
            f"Extracted {self.map_points.shape[0]} occupied points from the map."
        )

        # declare parameters with defaults
        self.declare_parameter("x_init", 0.0)
        self.declare_parameter("y_init", 0.0)
        self.declare_parameter("theta_init", 0.0)
        self.declare_parameter("p_init_diag", [0.05, 0.05, 0.05])

        # read parameters
        x0 = self.get_parameter("x_init").value
        y0 = self.get_parameter("y_init").value
        theta0 = self.get_parameter("theta_init").value
        p_diag = self.get_parameter("p_init_diag").value

        # initialize EKF state and covariance
        self.X = np.array([x0, y0, theta0], dtype=float)
        self.P = np.diag(p_diag)

        self.get_logger().info(
            "EKF Initialized at: x={0: .3f}, y={1: .3f}, theta={2: .3f}".format(
                x0, y0, theta0
            )
        )

        # create timer that runs main loop at the set interval dt
        self.timer = self.create_timer(0.2, self.main_loop)

        # create timer to publish latest transform
        self.transform_timer = self.create_timer(0.1, self.pub_latest_transform)

        # FOR DEBUGGING
        # self.visualize_occupancy_field() # visualize occupancy field via matplotlib

    def pub_latest_transform(self):
        """Publish the latest map to odom transform."""
        if self.last_scan_timestep is None:
            return
        tf_time = self.last_scan_timestep + Duration(seconds=0.1)
        self.tf_helper.send_last_map_to_odom_transform(
            self.map_frame, self.odom_frame, tf_time
        )

    def main_loop(self):
        """Main loop of the EKF."""

        if self.scan_to_process is None:
            return  # skip if no scan to process

        # if valid scan received, process it
        scan_msg = copy.deepcopy(self.scan_to_process)
        processed_scan = self.process_scan(scan_msg)

        # prediction step
        X_pred_next, P_pred_next = self.prediction(self.X, self.u, self.P)

        # get measurement (scan to map ICP)
        X_measured = self.scan_to_map_ICP(processed_scan, X_pred_next)

        # correction step
        self.X, self.P = self.correction(X_measured, X_pred_next, P_pred_next)

        # publish updated pose
        self.pose_publisher.publish(self.format_pose_msg(self.X, self.P))

        # **ADD THIS: Update the map->odom transform**
        robot_pose = Pose()
        robot_pose.position = Point(x=self.X[0], y=self.X[1], z=0.0)
        quat = quaternion_from_euler(0, 0, self.X[2])
        robot_pose.orientation = Quaternion(x=quat[0], y=quat[1], z=quat[2], w=quat[3])

        # Get current odom pose
        odom_result = self.tf_helper.get_matching_odom_pose(
            self.odom_frame, self.base_frame, scan_msg.header.stamp
        )

        if odom_result[0] is not None:
            self.tf_helper.fix_map_to_odom_transform(robot_pose, odom_result[0])

        self.get_logger().info(
            f"Estimated: [{self.X[0]:.3f}, {self.X[1]:.3f}, {np.rad2deg(self.X[2]):.3f}]"
        )

    def prediction(self, X, u, P):
        """
        Motion update step of the EKF.
        The motion model is based off of velocity commands.

        Args:
            state: current state vector [x, y, theta]
            vel_cmd: control input vector [v, omega]

        Returns:
            x_next: predicted next state vector [x, y, theta]
            P_next: predicted next covariance matrix [3x3]
        """

        # predict next state using motion model
        x_next = X[0] + u[0] * math.cos(X[2] + u[1] / 2)
        y_next = X[1] + u[0] * math.sin(X[2] + u[1] / 2)
        theta_next = X[2] + u[1]

        # wrap theta
        theta_next = self.tf_helper.angle_normalize(theta_next)

        # formulate into array
        X_pred_next = np.array([x_next, y_next, theta_next])

        # Updated Jacobian for arc motion
        F = np.array(
            [
                [1, 0, -u[0] * math.sin(X[2] + u[1] / 2)],
                [0, 1, u[0] * math.cos(X[2] + u[1] / 2)],
                [0, 0, 1],
            ]
        )

        # Jacobian wrt control
        L = np.array(
            [
                [math.cos(X[2] + u[1] / 2), -0.5 * u[0] * math.sin(X[2] + u[1] / 2)],
                [math.sin(X[2] + u[1] / 2), 0.5 * u[0] * math.cos(X[2] + u[1] / 2)],
                [0, 1],
            ]
        )

        # Velocity-dependent noise
        sigma_v = 0.05
        sigma_w = np.deg2rad(5)
        M = np.diag([sigma_v**2, sigma_w**2])

        Q = L @ M @ L.T
        P_pred_next = F @ P @ F.T + Q

        return X_pred_next, P_pred_next

    def correction(self, X_measured, X_pred_next, P_pred_next):
        """
        Measurement update step of the EKF.

        Args:
            x_pred: predicted state vector [x, y, theta]
            P_pred: predicted covariance matrix [3x3]
            z: measured state vector [x, y, theta]

        Returns:
            X_updated_next: updated state vector after measurement [x, y, theta]
            P_updated_next: updated covariance matrix after measurement [3x3]
        """

        H = np.eye(3)  # measurement matrix
        y = X_measured - (H @ X_pred_next)  # measurement residual
        y[2] = self.tf_helper.angle_normalize(y[2])  # normalize angle difference

        # Increase uncertainty with velocity
        pos_std = 0.2
        ang_std = np.deg2rad(15)

        R = np.diag(
            [pos_std**2, pos_std**2, ang_std**2]
        )  # measurement noise covariance, tune as needed
        S = H @ P_pred_next @ H.T + R

        K = P_pred_next @ H.T @ np.linalg.inv(S)  # Kalman gain

        X_updated_next = X_pred_next + K @ y  # update state estimate

        P_updated_next = (np.eye(3) - K @ H) @ P_pred_next  # update covariance

        return X_updated_next, P_updated_next

    def scan_to_map_ICP(self, scan, x):
        """
        Performs scan to map ICP for an absolute pose measurement.

        Args:
            scan: Nx2 array of scan points in robot base frame
            x: current predicted state vector [x, y, theta]

        Returns:
            x_icp: measured state vector [x, y, theta]
        """

        # format current pose estimate into transformation matrix
        T_estimate = np.array(
            [
                [np.cos(x[2]), -np.sin(x[2]), x[0]],
                [np.sin(x[2]), np.cos(x[2]), x[1]],
                [0, 0, 1],
            ]
        )

        # perform ICP between scan and occupancy field occupied points to get
        # corrected pose
        T_x = tf_icp(scan, self.map_points, T_estimate, 50, 1e-6, 0.4)

        # self.get_logger().info(f"T_estimate:\n{T_estimate}")
        # self.get_logger().info(f"T_returned:\n{T_x}")
        # self.get_logger().info(f"Difference:\n{T_x - T_estimate}")

        # format into state vector
        x_icp = np.array([T_x[0, 2], T_x[1, 2], math.atan2(T_x[1, 0], T_x[0, 0])])

        self.get_logger().info(
            f"Pred: [{x[0]:.3f}, {x[1]:.3f}, {np.rad2deg(x[2]):.3f}]"
        )
        self.get_logger().info(
            f"ICP:  [{x_icp[0]:.3f}, {x_icp[1]:.3f}, {np.rad2deg(x_icp[2]):.3f}]"
        )

        # Visualize first scan alignment (only once)
        # if not hasattr(self, '_visualized_first_scan'):
        #     self._visualize_scan_alignment(scan, x, x_icp)
        #     self._visualized_first_scan = True

        return x_icp

    def format_pose_msg(self, x, P):
        """
        Formats the state and covariance into a PoseWithCovarianceStamped message.

        Args:
            x: state vector [x, y, theta]
            P: covariance matrix [3x3]

        Returns:
            pose_msg: PoseWithCovarianceStamped message
        """

        pose_msg = PoseWithCovarianceStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = "map"

        pose_msg.pose.pose.position = Point(x=x[0], y=x[1], z=0.0)
        quat = quaternion_from_euler(0, 0, x[2])
        pose_msg.pose.pose.orientation = Quaternion(
            x=quat[0], y=quat[1], z=quat[2], w=quat[3]
        )

        # fill in covariance (6x6 matrix flattened to 36 elements)
        cov = np.zeros((6, 6))
        cov[0, 0] = P[0, 0]  # x variance
        cov[1, 1] = P[1, 1]  # y variance
        cov[5, 5] = P[2, 2]  # theta variance
        pose_msg.pose.covariance = cov.flatten().tolist()

        return pose_msg

    def receive_scan(self, msg):
        """
        Process incoming laser scan data.

        Args:
            msg: LaserScan message
        """
        if msg == None:
            return
        else:
            self.scan_to_process = msg
            self.last_scan_timestep = Time.from_msg(msg.header.stamp)

    def receive_odom(self, msg):
        """
        Process incoming odometry data.

        Args:
            msg: Odometry message
        """

        # first time receiving odom
        if self.last_odom_pose is None:
            self.last_odom_pose = msg
            return

        # compute change in position and orientation
        dx = msg.pose.pose.position.x - self.last_odom_pose.pose.pose.position.x
        dy = msg.pose.pose.position.y - self.last_odom_pose.pose.pose.position.y
        ds = math.sqrt(dx**2 + dy**2)

        # extract yaw angles
        q_now = msg.pose.pose.orientation
        q_prev = self.last_odom_pose.pose.pose.orientation

        yaw_now = euler_from_quaternion(q_now.x, q_now.y, q_now.z, q_now.w)[2]

        yaw_prev = euler_from_quaternion(q_prev.x, q_prev.y, q_prev.z, q_prev.w)[2]

        # compute change in orientation
        dtheta = self.tf_helper.angle_normalize(yaw_now - yaw_prev)

        # assign to control input
        self.u[0] = ds
        self.u[1] = dtheta

        # update last odom pose
        self.last_odom_pose = msg

    def process_scan(self, msg):
        """
        Process the laser scan message to extract valid range and angle data
        and converts it to Cartesian coordinates in the base link frame.

        Args:
            msg: LaserScan message
        Returns:
            scan: Nx2 array of scan points in robot base frame
        """

        # convert scan to polar coordinates in robot frame
        ranges, angles = self.tf_helper.convert_scan_to_polar_in_robot_frame(
            msg, self.base_frame
        )

        # subdivide into n beams
        beam_step = max(1, int(len(ranges) / 180))

        lx = []
        ly = []

        # sort through laser beams
        for i in range(0, len(ranges), beam_step):
            r_i = ranges[i]
            theta_i = angles[i]

            # check for valid range
            if not math.isfinite(r_i) or r_i <= 0.0:
                continue

            # Convert to Cartesian coordinates
            x = r_i * math.cos(theta_i) - 0.08  # 8 cm offset from base_link to laser
            y = r_i * math.sin(theta_i)

            # Filter out NaN/inf
            if math.isfinite(x) and math.isfinite(y):
                lx.append(x)
                ly.append(y)

        scan = np.vstack((lx, ly)).T  # Nx2 array

        return scan

    def extract_map_points(self):
        """
        Extract occupied cells from the map as a dense point cloud.

        Returns:
            occupied_cells: Mx2 array of occupied cell coordinates in world frame
        """
        map_data = self.occupancy_field.map
        resolution = map_data.info.resolution
        width = map_data.info.width
        height = map_data.info.height
        origin_x = map_data.info.origin.position.x
        origin_y = map_data.info.origin.position.y

        occupied_cells = []

        # Iterate through map grid
        for i in range(width):
            for j in range(height):
                idx = j * width + i
                if (
                    idx < len(map_data.data) and map_data.data[idx] > 50
                ):  # Occupied threshold
                    # Convert grid coordinates to world coordinates
                    x = i * resolution + origin_x
                    y = j * resolution + origin_y
                    occupied_cells.append([x, y])

        return np.array(occupied_cells)

    # VISUALIZATION FOR DEBUGGING

    def visualize_occupancy_field(self):
        """Visualize occupancy field with robot position."""

        plt.figure(figsize=(8, 8))

        # Plot occupied points
        plt.plot(
            self.occupancy_field.occupied[:, 0],
            self.occupancy_field.occupied[:, 1],
            "k.",
            markersize=0.5,
        )

        # Plot robot position
        plt.plot(self.X[0], self.X[1], "ro", markersize=10)

        plt.axis("equal")
        plt.show()
        self.get_logger().info("Displayed occupancy field visualization.")

    def _visualize_scan_alignment(self, scan, x_pred, x_icp):
        """Visualize scan alignment for debugging."""
        import matplotlib.pyplot as plt

        # Transform scan with predicted pose
        T_pred = np.array(
            [
                [np.cos(x_pred[2]), -np.sin(x_pred[2]), x_pred[0]],
                [np.sin(x_pred[2]), np.cos(x_pred[2]), x_pred[1]],
                [0, 0, 1],
            ]
        )
        scan_pred = (T_pred @ np.hstack((scan, np.ones((scan.shape[0], 1)))).T).T[:, :2]

        # Transform scan with ICP pose
        T_icp = np.array(
            [
                [np.cos(x_icp[2]), -np.sin(x_icp[2]), x_icp[0]],
                [np.sin(x_icp[2]), np.cos(x_icp[2]), x_icp[1]],
                [0, 0, 1],
            ]
        )
        scan_icp = (T_icp @ np.hstack((scan, np.ones((scan.shape[0], 1)))).T).T[:, :2]

        plt.figure(figsize=(12, 5))

        # Plot with predicted pose
        plt.subplot(1, 2, 1)
        plt.plot(
            self.map_points[:, 0],
            self.map_points[:, 1],
            "k.",
            markersize=0.5,
            alpha=0.3,
        )
        plt.plot(
            scan_pred[:, 0],
            scan_pred[:, 1],
            "r.",
            markersize=2,
            label="Scan (predicted)",
        )
        plt.plot(x_pred[0], x_pred[1], "ro", markersize=10)
        plt.axis("equal")
        plt.legend()
        plt.title("Predicted Pose")

        # Plot with ICP pose
        plt.subplot(1, 2, 2)
        plt.plot(
            self.map_points[:, 0],
            self.map_points[:, 1],
            "k.",
            markersize=0.5,
            alpha=0.3,
        )
        plt.plot(scan_icp[:, 0], scan_icp[:, 1], "b.", markersize=2, label="Scan (ICP)")
        plt.plot(x_icp[0], x_icp[1], "bo", markersize=10)
        plt.axis("equal")
        plt.legend()
        plt.title("ICP Pose")

        plt.tight_layout()
        plt.show()


def main(args=None):
    rclpy.init()
    n = ExtendedKalmanFilter()
    rclpy.spin(n)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
