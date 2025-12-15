"""
Library of code to run A* on a occupancy grid map, with supporting functions
"""

import heapq
import math
from collections import defaultdict

import numpy as np


def inflate_obstacles(map_grid, robot_diameter, resolution):
    """
    Simple obstacle inflation using a square robot footprint.

    Args:
        map: (2D numpy array), OccupancyGrid values
        robot_diameter: meters
        resolution: meters/cell

    Returns:
        New 2D numpy array with inflated obstacles set to 100.
    """
    inflated = map_grid.copy()

    # robot radius in cells
    r = int(math.ceil((robot_diameter / 2.0) / resolution))
    if r <= 0:
        return inflated

    height, width = map_grid.shape

    obstacle_cells = np.argwhere(map_grid == 100)

    for row, col in obstacle_cells:
        r0 = max(0, row - r)
        r1 = min(height, row + r + 1)
        c0 = max(0, col - r)
        c1 = min(width, col + r + 1)

        inflated[r0:r1, c0:c1] = 100

    return inflated


def a_star(map_grid, start, target):
    """
    Run A* using octile movement on 2d map

    Args:
        map_grid (2d numpy array): Map
        start (tuple of int): Starting coordinates
        target (tuple of int): Targate coordinates

    Returns:
        []: Array of path location map indices. Empty if no possible path
    """
    visited = set()

    priority_queue = [(0, start[0], start[1])]

    came_from = {}
    g_score = defaultdict(lambda: float("inf"))

    g_score[start] = 0

    f_score = {}
    f_score[start] = octile_distance(start, target)

    neighbor_diff = [
        (0, 1, 1.0),
        (1, 0, 1.0),
        (-1, 0, 1.0),
        (0, -1, 1.0),
        (1, 1, math.sqrt(2)),
        (1, -1, math.sqrt(2)),
        (-1, 1, math.sqrt(2)),
        (-1, -1, math.sqrt(2)),
    ]

    heuristic_weight = 1.01

    while priority_queue:
        current = heapq.heappop(priority_queue)

        current = (current[1], current[2])

        if current in visited:
            continue

        if current == target:
            return path_reconstruction(came_from, current)

        visited.add((current))

        for [row_diff, col_diff, cost] in neighbor_diff:

            neighbor = (current[0] + row_diff, current[1] + col_diff)

            # If out of bounds
            if (
                neighbor[0] < 0
                or neighbor[0] >= map_grid.shape[0]
                or neighbor[1] < 0
                or neighbor[1] >= map_grid.shape[1]
            ):
                continue

            # If obstacle
            if map_grid[neighbor[0], neighbor[1]] == 100:
                continue

            # If visited
            if neighbor in visited:
                continue

            neighbor_g_score = g_score[current] + cost

            if neighbor_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = neighbor_g_score
                f_score[neighbor] = (
                    neighbor_g_score
                    + octile_distance(neighbor, target) * heuristic_weight
                )

                heapq.heappush(
                    priority_queue, (f_score[neighbor], neighbor[0], neighbor[1])
                )

    return None


def octile_distance(current, target):
    """
    Calculate the octile distance between two points.

    Octile distance is optimal for 8-direction movement, meaning diagonals
    are included.

    Args:
        current (tuple of ints): Current point coordinates
        target (tuple of ints): Target point coordinates

    Returns:
        int: Octile distance between coords
    """
    row_diff = abs(target[0] - current[0])
    col_diff = abs(target[1] - current[1])

    return (max(row_diff, col_diff) - min(row_diff, col_diff)) + (
        min(row_diff, col_diff) * math.sqrt(2)
    )


def path_reconstruction(came_from, current):
    """
    Reconstruct the A* path

    Args:
        came_from (dict): A mapping from location to previous node
        current (tuple of ints): The current node

    Returns:
        list of tuples of ints: Path in order of visited node
    """
    complete_path = [current]

    while current in came_from:
        current = came_from[current]
        complete_path.append(current)

    return complete_path[::-1]
