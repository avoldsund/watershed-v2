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

outlet_coords = (4, 1)
step_size = 10
ny, nx = np.shape(heights_three_traps_div)

ws_of_node, traps, trap_heights, trap_indices_in_ws, steepest_spill_pairs, flow_directions, heights = river_analysis.\
    calculate_watershed_of_node_no_landscape_input(heights_three_traps_div, nx, ny, step_size, outlet_coords)

# Pre-processing
boundary = util.get_domain_boundary_indices(nx, ny)

# for i in trap_indices_in_ws:
#     s = util.map_1d_to_2d(steepest_spill_pairs[i][1], ny)
#     if heights[s] == trap_heights[i] and steepest_spill_pairs[i][1] not in boundary:
#         traps[i] = np.append(traps[i], steepest_spill_pairs[i][1])


nr_of_cells_in_each_trap = [len(traps[i]) for i in trap_indices_in_ws]
total_trap_cells = np.sum(nr_of_cells_in_each_trap)
traps = [util.map_1d_to_2d(traps[i], nx) for i in trap_indices_in_ws]
nr_of_traps = len(traps)
trap_heights = [trap_heights[t] for t in trap_indices_in_ws]
# print 'traps: ', traps
# print 'total_trap_cells: ', total_trap_cells
# print 'nr_of_cells_in_each_trap: ', nr_of_cells_in_each_trap
# print 'nrOfTraps: ', nrOfTraps
# print 'trap_heights: ', trap_heights

# Pre-processing of traps
# If a trap is spilling over into a trap of equal height,
# print traps.append((np.array([4,5,6,2,3]), np.array([0,1,2,3,4])))
# Watershed output

# Create cell array structure for traps
new_traps = np.zeros((len(traps), 2), dtype=object)
for i in range(len(traps)):
    new_traps[i][0] = traps[i][0]
    new_traps[i][1] = traps[i][1]

ws = util.map_1d_to_2d(ws_of_node, nx)

zero_wrapping = np.zeros((ny, nx), dtype=int)
zero_wrapping[1:-1, 1:-1] = flow_directions[1:-1, 1:-1]
flow_directions = zero_wrapping

scipy.io.savemat('watershed.mat', dict(watershed=ws))
scipy.io.savemat('heights.mat', dict(heights=heights))
scipy.io.savemat('traps.mat', dict(traps=new_traps, totalTrapCells=total_trap_cells,
                                   nrOfTraps=nr_of_traps, nrOfCellsInEachTrap=nr_of_cells_in_each_trap,
                                   trapHeights=trap_heights))
scipy.io.savemat('flowDirections.mat', dict(flowDirections=flow_directions))

print 'ws', ws
print 'heights', heights
print 'new_traps', new_traps
print 'total_trap_cells', total_trap_cells
print 'nr_of_traps', nr_of_traps
print 'nr_of_cells_in_each_trap', nr_of_cells_in_each_trap
print 'trap_heights', trap_heights
print 'flow_directions', flow_directions
print 'spill_pairs', steepest_spill_pairs