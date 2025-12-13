"""
Library of code to run A* on a map, with supporting functions
"""

import random
import heapq
import math
import numpy as np
from collections import defaultdict

# random.seed(1)


def generate_test_map(width, height, blocked_chance=0.5):

    graph = [[0] * width for _ in range(height)]

    for r in range(height):
        for c in range(width):
            if random.random() < blocked_chance:
                graph[r][c] = 1

    return graph


def inflate_obstacles(map, robot_diameter, resolution):
    """
    Simple obstacle inflation using a square robot footprint.

    Args:
        map: (2D numpy array), OccupancyGrid values
        robot_diameter: meters
        resolution: meters/cell

    Returns:
        New 2D numpy array with inflated obstacles set to 100.
    """
    inflated = map.copy()

    # robot radius in cells
    r = int(math.ceil((robot_diameter / 2.0) / resolution))
    if r <= 0:
        return inflated

    H, W = map.shape

    obstacle_cells = np.argwhere(map == 100)

    for row, col in obstacle_cells:
        r0 = max(0, row - r)
        r1 = min(H, row + r + 1)
        c0 = max(0, col - r)
        c1 = min(W, col + r + 1)

        inflated[r0:r1, c0:c1] = 100

    return inflated


def print_map(map):
    for row in map:
        print_row = ""
        for col in row:
            if len(str(col)) == 1:
                print_row += f" {str(col)} "
            elif len(str(col)) == 2:
                print_row += f"{str(col)} "
            else:
                print_row += str(col)
        print(print_row)


def a_star(map, start, target):
    """
    _summary_

    Args:
        map (2d numpy array): Map
        start (tuple of int): Starting coordinates
        target (tuple of int): Targate coordinates

    Returns:
        _type_: _description_
    """
    visited = set()

    priority_queue = [(0, start[0], start[1])]

    came_from = {}
    g_score = defaultdict(lambda: float("inf"))

    g_score[start] = 0

    f_score = {}
    f_score[start] = manhattan_distance(start, target)

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
                or neighbor[0] >= map.shape[0]
                or neighbor[1] < 0
                or neighbor[1] >= map.shape[1]
            ):
                continue

            # If obstacle
            if map[neighbor[0], neighbor[1]] == 100:
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

    return []


def manhattan_distance(current, target):
    return abs(target[0] - current[0]) + abs(target[1] - current[1])


def octile_distance(current, target):
    # a,b are (row,col)
    row_diff = abs(target[0] - current[0])
    col_diff = abs(target[1] - current[1])
    # (max-min)*1 + min*sqrt(2)
    return (max(row_diff, col_diff) - min(row_diff, col_diff)) + (
        min(row_diff, col_diff) * math.sqrt(2)
    )


def path_reconstruction(came_from, current):
    complete_path = [current]

    while current in came_from:
        current = came_from[current]
        complete_path.append(current)

    return complete_path[::-1]


# start = (0, 0)
# end = (29, 29)

# test_map = generate_test_map(30, 30, blocked_chance=0.2)

# test_map[start[0]][start[1]] = 0
# test_map[end[0]][end[1]] = 0

# visited, route = a_star(test_map, start, end)

# for r in range(len(test_map)):
#     for c in range(len(test_map[0])):
#         if test_map[r][c] == 1:
#             char = "X"
#         else:
#             char = " "
#         test_map[r][c] = char

# test_map[start[0]][start[1]] = "S"
# test_map[end[0]][end[1]] = "E"

# for [r, c] in visited:
#     test_map[r][c] = "v"

# if route:
#     for i, [r, c] in enumerate(route[1:-1]):
#         test_map[r][c] = i

# print_map(test_map)

# print()

# for row in test_map:
#     print(row)
