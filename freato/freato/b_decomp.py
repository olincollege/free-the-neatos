import numpy as np
import matplotlib.pyplot as plt


def b_decomp(map, map_res, neato_width, overlap):
    """Run Boustrophedon Decomposition on Occupancy Map from Lidar
        will generate path
    Inputs:
        map: occupancy grid (0 is free, 100 is occupied)
    Output:
        paths: list of paths (list of points) to traverse the map, hitting all the area
    """
    cells = build_bd_cells(map)
    print(f"Num Cells: {len(cells)}")
    paths = cells_to_paths(cells, map_res, neato_width, overlap)
    if overlap > neato_width:
        return []
    # print(paths)
    plot_paths(map, paths, map_res)

    return paths


def cells_to_paths(cells, map_res, n_width, overlap):
    """Convert all cells to a list of list of points (x,y)
    Inputs:
        cells: list of lists of tuples
        map_res: resolution of the occupancy grid (size of the square)
        n_width: width of neato (m)
        overlap: how much we want the path to overlap (m)
    Outputs:
        paths = a list of lists of tuples with x,y values for points for the neato to go to
    """
    paths = [cell_to_path(cell, map_res, n_width, overlap) for cell in cells]
    return paths


def cell_to_path(cell, map_res, n_width, overlap):
    """Convert a cell to a list of points (x,y)
    Inputs:
        cell: list of 3 tuples
        tuple format (column, start, end)
        example of one cell
        [ (0, 1, 4),   # column 0, rows 1 to 3 are free
          (1, 2, 5),   # column 1, rows 2 to 4 are free
          (2, 2, 4)    # column 2, rows 2 to 3 are free
        ]
        map_res: resolution of the occupancy grid (size of the square)
        n_width: width of neato (m)
        overlap: how much we want the path to overlap (m)
    Outputs:
        path: list of tuples with x,y values for points for the neato to go to
    """
    n_rad = n_width / 2
    step = n_width - overlap
    path = []

    # Compute X bounds of the cell
    start_col = cell[0][0]
    end_col = cell[-1][0]
    start_x = start_col * map_res + n_rad
    end_x = end_col * map_res + map_res - n_rad

    switch = True  # True = up, False = down
    x = start_x
    prev_y = None  # track last Y to connect safely

    while x <= end_x + 1e-6:
        col = int(np.floor(x / map_res))

        # find free interval in this column
        interval = None
        for c, start, end in cell:
            if c == col:
                interval = (start, end)
                break
        if interval is None:
            x += step
            continue

        y_min = interval[0] * map_res + n_rad
        y_max = interval[1] * map_res - n_rad

        if y_min >= y_max:
            x += step
            continue

        # Determine vertical endpoints for this column
        if switch:
            top_y = y_max
            bottom_y = y_min
        else:
            top_y = y_min
            bottom_y = y_max

        # Connect safely from previous point
        if prev_y is not None:
            # Horizontal move at a safe y (its not gonna crash while moving between christmas)
            safe_y = min(max(prev_y, y_min), y_max)
            xs = np.linspace(prev_x, x, max(2, int(abs(x - prev_x) / (map_res / 2))))
            for xi in xs:
                path.append((round(xi, 3), round(safe_y, 3)))

        # Vertical move
        path.append((round(x, 3), round(top_y, 3)))
        path.append((round(x, 3), round(bottom_y, 3)))

        prev_x = x
        prev_y = bottom_y  # last Y after vertical sweep
        switch = not switch
        x += step

    return path


def build_bd_cells(map):
    """Build the BD cells by matching free areas in columns to adjacent columns
    Inputs:
        map: occupancy grid
    Outputs:
        cells: list of lists of 3 tuples
        tuple format (column, start, end)
        example of one cell
        [ (0, 1, 4),   # column 0, rows 1 to 3 are free
          (1, 2, 5),   # column 1, rows 2 to 4 are free
          (2, 2, 4)    # column 2, rows 2 to 3 are free
        ]
    """
    cells = []
    current_cells = []  # each column in map
    for col in range(map.shape[1]):
        intervals = get_col_intervals(map, col)
        new_cells = []  # each interval of free space in the column
        for interval in intervals:
            matched = False
            # check compatibility with current cells (do they overlap?)
            for cell in current_cells:
                # unpack previous column cells
                prev_col = cell[-1][0]
                prev_start = cell[-1][1]
                prev_end = cell[-1][2]
                # Do the current interval and previous overlap?
                if not (interval[1] <= prev_start or interval[0] >= prev_end):
                    # add cell to new cells
                    cell.append((col, interval[0], interval[1]))
                    new_cells.append(cell)
                    matched = True
                    break
            # No matches???
            if not matched:
                # Add new cell
                new_cell = [(col, interval[0], interval[1])]
                cells.append(new_cell)
                new_cells.append(new_cell)
        # Update cells from this column
        current_cells = new_cells
    return cells


def get_col_intervals(map, col):
    """Get the intervals of free space for a column
    Inputs:
        col: column index to search for intervals
    Outputs:
        intervals: list of tuples with start and end indexes for intervals of free space
        For obstacle 5 to 9 [(3, 5), (10, 20)]
        [(start, end), (start, end)]
    """
    intervals = []
    start = None
    end = None
    column = map[:, col]
    switch = False  # false --> looking for start #true --> looking for end
    for i in range(map.shape[0] - 1):
        if switch:  # looking for end
            if (column[i] == 0) & (column[i + 1] == 100):  # 00
                end = i + 1
                intervals.append((start, end))
                switch = False
        else:  # looking for start
            if column[i] == 0:
                start = i
                switch = True
    return intervals


def sample_map():
    """Generate a sample occupancy grid (map)
    Output:
        map: occupancy grid (0 is free, 100 is occupied)
    """
    h = 20  # height
    w = 30  # width
    map = np.zeros((h, w), dtype=np.int32)
    # Add obstacles
    map[0:h, 0] = 100  # left wall
    map[0:h, w - 1] = 100  # right wall
    map[0, 1 : w - 1] = 100  # right wall
    map[h - 1, 1 : w - 1] = 100  # right wall
    map[5:11, 10:11] = 100  # obstacle 1
    map[11:12, 5:15] = 100  # obstacle 2
    map[18:19, 1:10] = 100  # obstacle 3
    map[5:11, 21:29] = 100  # obstacle 4
    # print(map) #vis map
    return map


def plot_paths(map, paths, map_res):
    """Plot occupancy grid and overlay paths"""
    plt.figure(figsize=(10, 6))
    plt.imshow(
        map,
        cmap="gray_r",
        origin="upper",
        extent=[0, map.shape[1] * map_res, 0, map.shape[0] * map_res],
    )

    # Plot each cell's path
    for path in paths:
        if len(path) == 0:
            continue
        xs = [p[0] for p in path]
        ys = [map.shape[0] * map_res - p[1] for p in path]
        plt.plot(xs, ys)  # marker optional

    plt.title("Boustrophedon Coverage Paths")
    plt.xlabel("X (meters)")
    plt.ylabel("Y (meters)")
    plt.grid(True)
    plt.show()


# # Testing
# sample = sample_map()
# neato_width = 0.06  # m
# overlap = 0.01  # m
# map_res = 0.05  # m

# if __name__ == "__main__":
#     b_decomp(sample, map_res, neato_width, overlap)
