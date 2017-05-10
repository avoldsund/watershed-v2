import numpy as np
from lib import river_analysis, plot, util
from matplotlib import pyplot as plt
import scipy.io
import math


def create_grid_set_heights(n, h, outlet):
    indices_rc = []

    for i in range(n):
        for j in range(n):
            indices_rc.append((i, j))

    coords = [(i[0] * h, i[1] * h) for i in indices_rc]

    # z = [abs(c[0] - outlet[0]) for c in coords]
    z = [math.sqrt((c[0] - outlet[0])**2 + (c[1] - outlet[1])**2) for c in coords]

    return indices_rc, z

# Grid metadata
N = 100 + 2
size = 1000  # in meters
step_size = size/(N-2)
outlet_rc = np.zeros(2, dtype=int)
outlet_rc[0] = N - 2
outlet_rc[1] = (N - 2)/2


# Heights
coord, z = create_grid_set_heights(N, step_size, outlet_rc * step_size)
heights = np.reshape(z, [N, N])
plot.plot_heights(heights)
print heights
ws_of_node, traps, trap_heights, trap_indices_in_ws, steepest_spill_pairs, flow_directions, heights = river_analysis.\
    calculate_watershed_of_node_no_landscape_input(heights, N, N, step_size, outlet_rc, d4=False)

#plot.plot_flow_directions(flow_directions[1:-1, 1:-1])
# plot.plot_watershed_of_node(ws_of_node, N, N)

"Pre-process output data"
nr_of_cells_in_each_trap = [len(traps[i]) for i in trap_indices_in_ws]
total_trap_cells = np.sum(nr_of_cells_in_each_trap)
traps = [util.map_1d_to_2d(traps[i], N) for i in trap_indices_in_ws]
nr_of_traps = len(traps)
trap_heights = [trap_heights[t] for t in trap_indices_in_ws]

ws = util.map_1d_to_2d(ws_of_node, N)

# Add flow directions for traps
flow_directions = river_analysis.add_trap_flow_directions(flow_directions, steepest_spill_pairs)

zero_wrapping = np.zeros((N, N), dtype=int)
zero_wrapping[1:-1, 1:-1] = flow_directions[1:-1, 1:-1]
flow_directions = zero_wrapping
steepest_spill_pairs = [util.map_1d_to_2d(steepest_spill_pairs[i][0], N) for i in range(len(steepest_spill_pairs)) if i in trap_indices_in_ws]

# Create cell array structure for traps
new_traps = np.zeros((len(traps), 2), dtype=object)
for i in range(len(traps)):
    new_traps[i][0] = traps[i][0]
    new_traps[i][1] = traps[i][1]

# Change outlet to smaller grid

scipy.io.savemat('watershed.mat', dict(watershed=ws, outlet=outlet_rc, stepSize=step_size))
scipy.io.savemat('heights.mat', dict(heights=heights))
scipy.io.savemat('traps.mat', dict(traps=new_traps, totalTrapCells=total_trap_cells,
                                   nrOfTraps=nr_of_traps, nrOfCellsInEachTrap=nr_of_cells_in_each_trap,
                                   trapHeights=trap_heights))
scipy.io.savemat('steepest.mat', dict(spillPairs=steepest_spill_pairs))
scipy.io.savemat('flowDirections.mat', dict(flowDirections=flow_directions))