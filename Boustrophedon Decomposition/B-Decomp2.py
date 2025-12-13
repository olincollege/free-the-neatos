import numpy as np
import matplotlib.pyplot as plt


def b_decomp(map):
    """
    Run Boustrophedon Decomposition on Occupancy Map from Lidar
    will generate path
        
    Inputs:
        map: occupancy grid (0 is free, 100 is occupied)

    Output:
        path: path (list of points) to traverse the map, hitting all the area
    """
    print(get_col_intervals(10, map))
def build_bd_cells(map):
    """
    Build the BD cells by matching free areas in columns to adjacent columns

    Inputs:
        map: occupancy grid
    Outputs:
        cells: list of lists of 3 tuples
        tuple format (column, start, end)
        example of one cell
            [
            (0, 1, 4),   # column 0, rows 1 to 3 are free
            (1, 2, 5),   # column 1, rows 2 to 4 are free
            (2, 2, 4)    # column 2, rows 2 to 3 are free
            ]
        
    """
    cells = []
    current_cells = []

    #each column in map
    for col in range(map.shape[1]):
        intervals = get_col_intervals(col)
        new_cells = []
        #each interval of free space in the column
        for interval in intervals:
            matched = False
            #check compatibility with current cells (do they overlap?)
            for cell in current_cells:
                #unpack previous column cells
                prev_col = cell[-1][0]
                prev_start = cell[-1][1]
                prev_end = cell[-1][2]

                #Do the current interval and previous overlap?
                if not (interval[1] <= prev_start or interval[0] >= prev_end):
                    #add cell to new cells
                    cell.append((col, interval[0], interval[1]))
                    new_cells.append(cell)
                    matched = True
                    break

            #No matches??? ;)
            if not matched:
                #Add new cell
                new_cell = [(col, interval[0], interval[1])]
                cells.append(new_cell)
                new_cells.append(new_cell)

        #Update cells from this column
        current_cells = new_cells










def get_col_intervals(col, map):
    """
    Get the intervals of free space for a column

    Inputs:
        col: column index to search for intervals

    Outputs:
        intervals: list of tuples with start and end indexes for intervals
                   of free space
                    For obstacle 5 to 9
                   [(3, 5), (10, 20)]
                   [(start, end), (start, end)]
    """
    intervals = []
    start = None
    end = None
    column = map[:, col]
    switch = False #false --> looking for start
                   #true --> looking for end

    for i in range(map.shape[0]):
        if switch: #looking for end
            if (column[i] == 0)&(column[i+1] == 1):
                end = i+1
                intervals.append((start, end))
                switch = False

        else: #looking for start
            if column[i] == 0:
                start = i
                switch = True

    return intervals

def sample_map():
    """
    Generate a sample occupancy grid (map)

    Output:
        map: occupancy grid (0 is free, 100 is occupied)
    """

    h = 20 #height
    w = 30 #width
    map = np.zeros((h, w), dtype=np.int32)
    # Add obstacles
    map[0:h, 0] = 1#00 #left wall
    map[0:h, w-1] = 1#00 #right wall
    map[0, 1:w-1] = 1#00 #right wall
    map[h-1, 1:w-1] = 1#00 #right wall


    map[5:10, 10:11] = 1#00 #obstacle 1
    map[11:12, 5:15] = 1#00 #obstacle 2
    map[18:19, 1:10] = 1#00 #obstacle 3
    map[5:8, 21:24] = 1#00 #obstacle 4
    print(map) #vis map
    return map


sample = sample_map()
if __name__ == "__main__":
    b_decomp(sample)