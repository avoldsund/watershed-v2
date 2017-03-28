import numpy as np
from matplotlib import pyplot as plt
from lib import load_data, util, river_analysis, plot
import scipy.io

# Different examples to show how the velocity field will be if the vectors point in the direction of steepest descent
heights_one = np.array([[20, 20, 20],
                        [20, 15, 20],
                        [20, 5, 20]])
heights_two = np.array([[20, 15, 10],
                        [15, 10, 5],
                        [10, 5, 0]])
heights_three = np.array([[20, 20, 20],
                          [5, 5, 5],
                          [0, 0, 0]])
heights_four = np.array([[0, 5, 20],
                         [0, 5, 20],
                         [0, 5, 20]])
heights_five = np.array([[0, 0, 0],
                         [20, 20, 20],
                         [0, 0, 0]])
heights_six = np.array([[10, 15],
                        [0, 10]])
heights_seven = np.array([[20, 20, 20, 20, 20],
                         [20, 15, 10, 15, 20],
                         [20, 10, 5, 10, 20]])

heights_trap = np.array([[20, 20, 20, 20, 20],
                         [20, 15, 10, 15, 20],
                         [20, 15, 10, 15, 20],
                         [20, 10, 0, 10, 20],
                         [20, 5, 0, 5, 20]])

heights_three_traps_div = np.array([[10, 10, 10, 10, 10, 10],
                                    [10, 9, 9, 9, 7, 10],
                                    [10, 9, 10, 9, 7, 10],
                                    [8, 10, 10, 10, 7, 10],
                                    [10, 4, 4, 4, 4.5, 10],
                                    [10, 4, 10, 10, 10, 10]])

print heights_three_traps_div

outlet = (4, 1)
step_size = 10
ny, nx = np.shape(heights_three_traps_div)

ws_of_node, traps, trap_heights, trap_indices_in_ws, steepest_spill_pairs, flow_directions, heights = river_analysis.\
    calculate_watershed_of_node_no_landscape_input(heights_three_traps_div, nx, ny, step_size, outlet, d4=False)
# print ws_of_node
# print traps
# print trap_heights
# print trap_indices_in_ws
# print steepest_spill_pairs
# print flow_directions

# Pre-processing
# boundary = util.get_domain_boundary_indices(nx, ny)

# for i in trap_indices_in_ws:
#     s = util.map_1d_to_2d(steepest_spill_pairs[i][1], ny)
#     if heights[s] == trap_heights[i] and steepest_spill_pairs[i][1] not in boundary:
#         traps[i] = np.append(traps[i], steepest_spill_pairs[i][1])

# Pre-processing of traps
# If a trap is spilling over into a trap of equal height,
# print traps.append((np.array([4,5,6,2,3]), np.array([0,1,2,3,4])))
# Watershed output

"Pre-process output data"
nr_of_cells_in_each_trap = [len(traps[i]) for i in trap_indices_in_ws]
total_trap_cells = np.sum(nr_of_cells_in_each_trap)
traps = [util.map_1d_to_2d(traps[i], nx) for i in trap_indices_in_ws]
nr_of_traps = len(traps)
trap_heights = [trap_heights[t] for t in trap_indices_in_ws]

ws = util.map_1d_to_2d(ws_of_node, nx)

# Add flow directions for traps
flow_directions = river_analysis.add_trap_flow_directions(flow_directions, steepest_spill_pairs)

zero_wrapping = np.zeros((ny, nx), dtype=int)
zero_wrapping[1:-1, 1:-1] = flow_directions[1:-1, 1:-1]
flow_directions = zero_wrapping
steepest_spill_pairs = [util.map_1d_to_2d(steepest_spill_pairs[i][0], nx) for i in range(len(steepest_spill_pairs)) if i in trap_indices_in_ws]

# Create cell array structure for traps
new_traps = np.zeros((len(traps), 2), dtype=object)
for i in range(len(traps)):
    new_traps[i][0] = traps[i][0]
    new_traps[i][1] = traps[i][1]

scipy.io.savemat('watershed.mat', dict(watershed=ws, outlet=outlet))
scipy.io.savemat('heights.mat', dict(heights=heights))
scipy.io.savemat('traps.mat', dict(traps=new_traps, totalTrapCells=total_trap_cells,
                                   nrOfTraps=nr_of_traps, nrOfCellsInEachTrap=nr_of_cells_in_each_trap,
                                   trapHeights=trap_heights))
scipy.io.savemat('steepest.mat', dict(spillPairs=steepest_spill_pairs))
scipy.io.savemat('flowDirections.mat', dict(flowDirections=flow_directions))
