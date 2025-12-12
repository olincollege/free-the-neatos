import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------------
# 1. Generate a simple occupancy grid
# -------------------------------------------------------

grid = np.zeros((20, 30), dtype=np.int32)
# Add obstacles
grid[5:10, 10:11] = 1
grid[11:12, 5:15] = 1
grid[18:10, 1:10] = 1

# -------------------------------------------------------
# 2. Find connected free-space intervals in each column
#    This is the "sweep-line" step of BD
# -------------------------------------------------------

def get_intervals(col):
    """Return continuous free-space intervals in a column."""
    intervals = []
    in_free = False
    start = None

    for r in range(grid.shape[0]):
        if grid[r, col] == 0 and not in_free:
            in_free = True
            start = r
        if (grid[r, col] == 1 or r == grid.shape[0] - 1) and in_free:
            end = r if grid[r, col] == 1 else r + 1
            intervals.append((start, end))
            in_free = False
    return intervals

# -------------------------------------------------------
# 3. Build BD cells by tracking when intervals split/merge
# -------------------------------------------------------

cells = []          # list of cells (each is a list of (col, start, end) )
active_cells = []   # active cells from previous column

for c in range(grid.shape[1]):
    intervals = get_intervals(c)

    new_active = []

    for interval in intervals:
        matched = False
        # Try to match interval with an active cell (continuation)
        for cell in active_cells:
            last_col, last_start, last_end = cell[-1]
            # Overlap rule
            if not (interval[1] <= last_start or interval[0] >= last_end):
                cell.append((c, interval[0], interval[1]))
                new_active.append(cell)
                matched = True
                break

        # No match = a new cell appears
        if not matched:
            new_cell = [(c, interval[0], interval[1])]
            cells.append(new_cell)
            new_active.append(new_cell)

    active_cells = new_active

# -------------------------------------------------------
# 4. Generate a boustrophedon path for each cell
# -------------------------------------------------------

def cell_to_path(cell):
    """Convert a cell (column intervals) into a back-and-forth sweep path."""
    path = []
    direction = 1  # 1 = top→bottom, -1 = bottom→top

    for col, start, end in cell:
        if direction == 1:
            for r in range(start, end):
                path.append((r, col))
        else:
            for r in range(end - 1, start - 1, -1):
                path.append((r, col))
        direction *= -1  # flip direction each column
    return path

paths = [cell_to_path(cell) for cell in cells]

# -------------------------------------------------------
# 5. Visualize
# -------------------------------------------------------

plt.imshow(grid, cmap='gray_r')

for path in paths:
    ys = [p[0] for p in path]
    xs = [p[1] for p in path]
    plt.plot(xs, ys)

plt.title("Boustrophedon Coverage Path")
plt.gca().invert_yaxis()
plt.show()
