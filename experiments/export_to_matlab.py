import numpy as np
from matplotlib import pyplot as plt
from lib import load_data, util, river_analysis, plot
import scipy.io
from scipy.ndimage.filters import maximum_filter

# Different examples to show how the velocity field will be if the vectors point in the direction of steepest descent
heights_one = np.array([[20, 20, 20],
                        [20, 15, 20],
                        [20, 5, 20]])
# heights_two = np.array([[20, 15, 10],
#                         [15, 10, 5],
#                         [10, 5, 0]])
# heights_three = np.array([[20, 20, 20],
#                           [5, 5, 5],
#                           [0, 0, 0]])
# heights_four = np.array([[0, 5, 20],
#                          [0, 5, 20],
#                          [0, 5, 20]])
# heights_five = np.array([[0, 0, 0],
#                          [20, 20, 20],
#                          [0, 0, 0]])
# heights_six = np.array([[10, 15],
#                         [0, 10]])
file_name = '/home/anderovo/Dropbox/watershedLargeFiles/anders_hoh.tiff'
landscape = load_data.get_smallest_test_landscape_tyrifjorden(file_name)
# print heights

r, c = np.shape(landscape.heights)
# plot.plot_landscape_2d(landscape, 1)
# acc_flow = river_analysis.calculate_accumulated_flow(landscape)
# plot.plot_accumulated_flow(acc_flow)

outlet = (208, 152)

ws_of_node, traps, trap_heights, trap_indices_in_ws, steepest_spill_pairs, flow_directions, heights = river_analysis.\
    calculate_watershed_of_node(landscape, outlet)

plot.plot_watersheds_2d([ws_of_node], outlet, landscape, 1)
# Pre-processing
boundary = util.get_domain_boundary_indices(landscape.nx, landscape.ny)
all_trap_nodes = np.concatenate(traps)


print [util.map_1d_to_2d(np.array([steepest_spill_pairs[el][0], steepest_spill_pairs[el][1]]), landscape.ny) for el in trap_indices_in_ws]
print [traps[i] for i in trap_indices_in_ws]
print 'heights[201, 156]: ', heights[201, 156]
print 'heights[202, 156]: ', heights[202, 156]
print 'util.map_1d_to_2d(np.array([50406]), landscape.ny): ', util.map_1d_to_2d(np.array([50406]), landscape.ny)
for i in trap_indices_in_ws:
    s = util.map_1d_to_2d(steepest_spill_pairs[i][1], landscape.ny)
    if heights[s] == trap_heights[i] and steepest_spill_pairs[i][1] not in boundary and steepest_spill_pairs[i][1] \
            not in all_trap_nodes:
        traps[i] = np.append(traps[i], steepest_spill_pairs[i][1])

nr_of_cells_in_each_trap = [len(traps[i]) for i in trap_indices_in_ws]
total_trap_cells = np.sum(nr_of_cells_in_each_trap)

traps = [util.map_1d_to_2d(traps[i], landscape.nx) for i in trap_indices_in_ws]
nrOfTraps = len(traps)
trap_heights = [trap_heights[t] for t in trap_indices_in_ws]

new_traps = np.zeros((len(traps), 2), dtype=object)
for i in range(len(traps)):
    new_traps[i][0] = traps[i][0]
    new_traps[i][1] = traps[i][1]

print flow_directions[200:205, 154:158]
ws = util.map_1d_to_2d(ws_of_node, landscape.nx)

zero_wrapping = np.zeros((landscape.ny, landscape.nx), dtype=int)
zero_wrapping[1:-1, 1:-1] = flow_directions[1:-1, 1:-1]
flow_directions = zero_wrapping

scipy.io.savemat('watershed.mat', dict(watershed=ws))
scipy.io.savemat('heights.mat', dict(heights=landscape.heights))
scipy.io.savemat('traps.mat', dict(traps=new_traps, totalTrapCells=total_trap_cells,
                                   nrOfTraps=nrOfTraps, nrOfCellsInEachTrap=nr_of_cells_in_each_trap,
                                   trapHeights=trap_heights))
scipy.io.savemat('flowDirections.mat', dict(flowDirections=flow_directions))


print flow_directions[200:205, 154:158]
print heights[200:205, 154:158]

