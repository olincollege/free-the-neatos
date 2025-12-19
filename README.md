# Free the Neatos

Implementing robot vacuum software in ROS 2 for Neato vacuum robots.

## Overview

This project implements a complete autonomous cleaning system for Neato vacuum robots using ROS 2. The system enables the robots to autonomously map their environment and perform intelligent coverage path planning for cleaning operations.

See our [website](https://olincollege.github.io/free-the-neatos/index.html) for the full project writeup

<div align="center">
  <img src="docs/images/mapping.gif" alt="Mapping mode demo" width="350" />
  <img src="docs/images/cleaning.gif" alt="Cleaning mode demo" width="350" />
</div>

## Features

- **SLAM (Simultaneous Localization and Mapping)**: Real-time mapping of unknown environments using the ROS slam_toolbox
- **Exploration Path Planning**: Autonomous exploration to build complete maps using frontier-based exploration
- **Coverage Path Planning**: Intelligent cleaning path planning using Boustrophedon Decomposition
- **A\* Path Planning**: Obstacle-avoiding pathfinding between waypoints
- **Extended Kalman Filter (EKF)**: Improved odometry estimation for better localization
- **Waypoint Following**: Low-level control system for following planned paths

## Requirements

- ROS 2 (Humble or later)
- Python 3.8+
- Neato robot hardware (or compatible simulator)

## Dependencies

- `rclpy` - ROS 2 Python client library
- `std_msgs` - Standard ROS message types
- `geometry_msgs` - Geometry message types
- `sensor_msgs` - Sensor message types
- `nav_msgs` - Navigation message types
- `nav2_msgs` - Navigation2 message types
- `slam_toolbox` - SLAM implementation
- `launch` / `launch_ros` - ROS 2 launch system
- `neato_packages` - Custom Neato ROS interface. Install [here](https://github.com/comprobo25/neato_packages).

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd free-the-neatos
```

2. Source your ROS 2 workspace:
```bash
source /opt/ros/humble/setup.bash
```

3. Build the package:
```bash
cd freato
colcon build
source install/setup.bash
```

## Usage

### Mapping Mode

To run the robot in mapping/exploration mode:

```bash
ros2 launch freato freato_mapping.launch.py
```

### Cleaning Mode

To run the robot in cleaning mode (requires a pre-existing map):

```bash
ros2 launch freato freato_cleaning.launch.py
```

## Project Structure

```
free-the-neatos/
├── freato/                   # Main ROS 2 package
│   ├── freato/               # Python package
│   │   ├── a_star.py         # A* path planning implementation
│   │   ├── angle_helpers.py  # Angle utility functions
│   │   ├── b_decomp.py       # Boustrophedon decomposition
│   │   ├── cleaning.py       # Cleaning mode controller
│   │   ├── ekf.py            # Extended Kalman Filter
│   │   ├── helper_functions.py # Utility functions
│   │   ├── icp.py            # Iterative Closest Point algorithm
│   │   ├── occupancy_field.py # Occupancy grid utilities
│   │   ├── slam_exploration.py # SLAM exploration controller
│   │   └── waypoint_follow_server.py # Waypoint following action server
│   ├── launch/               # Launch files
│   ├── maps/                 # Pre-built maps
│   ├── worlds/               # Simulation world files
│   └── test/                 # Unit tests
├── docs/                     # Website
```

## Architecture

  <img src="docs/images/image4.jpg" alt="Mapping mode demo" width="650" />

### Mapping Components
- **SLAM**: Uses ROS slam_toolbox for simultaneous localization and mapping
- **Exploration Path Planning**: Frontier-based exploration algorithm

### Cleaning Components
- **Coverage Path Planning**: Boustrophedon Decomposition for efficient coverage
- **EKF**: Custom extended kalman filter using iterative closest point 

### Components for Both Modes
- **Waypoint Following**: Action server for executing planned paths
- **A\* Path Planning**: Obstacle-avoiding pathfinding

## Documentation

See our [website](https://olincollege.github.io/free-the-neatos/index.html) for the full project write-up with:
- Project overview and goals
- Milestone documentation
- Software architecture details
