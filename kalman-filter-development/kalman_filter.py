#!/usr/bin/env python3

import rclpy
from threading import Thread
from rclpy.time import Time
from rclpy.node import Node
from std_msgs.msg import Header
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseWithCovarianceStamped, Pose, Point, Quaternion
from rclpy.duration import Duration
import math
import time
import numpy as np
from occupancy_field import OccupancyField
import numpy as np
import matplotlib.pyplot as plt
from helper_functions import TFHelper
from rclpy.qos import qos_profile_sensor_data
from angle_helpers import quaternion_from_euler
from icp import tf_from_icp as tf_icp

class ExtendedKalmanFilter(Node):
    
    def __init__(self):
        super().__init__('extended_kalman_filter')
        self.base_frame = "base_footprint"   # the frame of the robot base
        self.map_frame = "map"          # the name of the map coordinate frame
        self.odom_frame = "odom"        # the name of the odometry coordinate frame
        self.scan_topic = "scan"        # the topic where we will get laser scans from

        # laser_subscriber listens for data from the lidar
        self.create_subscription(LaserScan, self.scan_topic, self.receive_scan, 10)
        self.scan_to_process = None 

        # publisher the estimated pose to the 'ekf_pose' topic
        self.pose_publisher = self.create_publisher(PoseWithCovarianceStamped, 'ekf_pose', 10)

        self.occupancy_field = OccupancyField(self) # define occupancy field object

        self.dt = 0.25 # time step between filter updates

        # create timer that runs main loop at the set interval dt
        self.timer = self.create_timer(self.dt, self.main_loop)

        self.X = X_init
        self.P = P_init

    def main_loop(self):
        """
        Main loop of the EKF.

        """
        
        if self.scan_to_process is None:
            return # skip if no scan to process
        
        # if valid scan received, process it
        processed_scan = self.process_scan(self.scan_to_process)
        
        self.scan_to_process = None # reset scan after processing

        # get control input (odometry)
        u = self.get_odometry_control() # placeholder for odometry control input

        # prediction step
        X_pred_next, P_pred_next = self.prediction(self.X, u, self.P)

        # get measurement (scan to map ICP)
        X_measured = self.scan_to_map_ICP(processed_scan, X_pred_next)

        # correction step
        self.X, self.P = self.correction(X_measured, X_pred_next, P_pred_next)

        self.pose_publisher.publish(self.format_pose_msg(self.X, self.P))

    def prediction(self, X, u, P):
        """
        Motion update step of the EKF.

        Args:
            state: current state [x, y, theta]
            vel_cmd: control input [v, omega]
        
        Returns:
            x_next: predicted next state
            P_next: predicted next covariance
        """

        # predict next state
        x_next = X[0] + u[0] * self.dt * np.cos(X[2])
        y_next = X[1] + u[0] * self.dt * np.sin(X[2])
        theta_next = X[2] + u[1] * self.dt

        # wrap theta to [-pi, pi]
        theta_next = (theta_next + np.pi) % (2 * np.pi) - np.pi

        # setup next state matrix
        X_pred_next = np.array([x_next, y_next, theta_next])

        # Jacobian of the motion model wrt state
        F = np.array([[1, 0, -u[0] * self.dt * np.sin(X[2])], 
                      [0, 1,  u[0] * self.dt * np.cos(X[2])],
                      [0, 0, 1]])
        
        # Jacobian of the motion model wrt control input
        L = np.array([[self.dt * np.cos(X[2]), 0],
                      [self.dt * np.sin(X[2]), 0],
                      [0, self.dt]])

        sigma_v = 0.05  # m/s
        sigma_w = np.deg2rad(5)  # rad/s
        W = np.diag([sigma_v**2, sigma_w**2]) # control noise covariance, tune as needed

        Q = L @ W @ L.T  # process noise covariance

        P_pred_next = F @ P @ F.T + Q  # update covariance

        return X_pred_next, P_pred_next

    def correction(self, X_measured, X_pred_next, P_pred_next):
        """
        Measurement update step of the EKF.

        Args:
            x_pred (np.array): Predicted state [x, y, theta]
            P_pred (np.array): Predicted covariance
            z (np.array): Measurement [x_meas, y_meas]
        
        Returns:
            np.array: Updated state after measurement
        """

        H = np.eye(3)  # measurement matrix
        y = X_measured - (H @ X_pred_next)  # measurement residual
        y[2] = (y[2] + np.pi) % (2 * np.pi) - np.pi  # normalize angle difference

        R = np.diag([0.05**2, 0.05**2, (np.deg2rad(4))**2]) # measurement noise covariance, tune as needed
        S = H @ P_pred_next @ H.T + R

        K = P_pred_next @ H.T @ np.linalg.inv(S)  # Kalman gain

        X_updated_next = X_pred_next + K @ y # update state estimate

        P_updated_next = (np.eye(3) - K @ H) @ P_pred_next  # update covariance

        return X_updated_next, P_updated_next

    def scan_to_map_ICP(self, scan, x):
        """
        Performs scan to map ICP for an absolute pose measurement.

        """

        # transform scan to map frame using current pose estimate
        scan_tf = self.transform_scan_to_map(scan, x)

        # perform ICP between scan and occupancy field occupied points
        T_correction = tf_icp(scan_tf, self.occupancy_field.occupied, np.eye(3), max_iterations=50, tolerance=1e-6)

        # format current pose estimate into transformation matrix
        T_estimate = np.array([[np.cos(x[2]), -np.sin(x[2]), x[0]],
                               [np.sin(x[2]),  np.cos(x[2]), x[1]],
                               [0, 0, 1]])

        # combine transformations to get corrected pose
        T_x = T_correction @ T_estimate

        # format into state vector
        x_icp = np.array([T_x[0,2], T_x[1,2], math.atan2(T_x[1,0], T_x[0,0])])

        return x_icp

    def receive_scan(self, msg):
        """
        Process incoming laser scan data.

        """
        if msg == None:
            return
        else:
            self.scan_to_process = msg
    
    def process_scan(self, msg):
        """
        Process the laser scan message to extract valid range and angle data
        and converts it to Cartesian coordinates in the base link frame.
        """

        ranges = np.array(msg.ranges)
        angles = np.linspace(msg.angle_min, msg.angle_max, len(ranges))

        beam_step = max(1, int(len(ranges) / 120))  # subdivide into n beams

        lx = []
        ly = []

        # sort through laser beams
        for i in range(0, len(ranges), beam_step):
            r_i = ranges[i]
            theta_i = angles[i]
            
            # check for valid range
            if not math.isfinite(r_i) or r_i <= 0.0:
                continue

            # convert to Cartesian coordinates in laser frame
            lx.append(r_i * math.cos(theta_i) - 0.08)  # account for laser offset
            ly.append(r_i * math.sin(theta_i))

        scan = np.vstack((lx, ly)).T  # Nx2 array

        return scan
    
    def transform_scan_to_map(self, scan, x):
        """
        Transforms the scan points from the base frame to the map frame
        using the current estimated pose of the robot.
        """

        # construct transformation matrix from robot pose
        T = np.array([[np.cos(x[2]), -np.sin(x[2]), x[0]],
                      [np.sin(x[2]),  np.cos(x[2]), x[1]],
                      [0, 0, 1]])

        # apply transformation to scan points
        scan_homogeneous = np.hstack((scan, np.ones((scan.shape[0], 1))))
        scan_transformed = (T @ scan_homogeneous.T).T[:, 0:2]

        return scan_transformed

def main(args=None):
    rclpy.init()
    n = ExtendedKalmanFilter()
    rclpy.spin(n)
    rclpy.shutdown()

if __name__ == '__main__':
    main()