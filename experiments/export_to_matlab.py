import numpy as np
from lib import load_data, util, river_analysis, plot
import scipy.io


file_name = '/home/anderovo/Dropbox/watershedLargeFiles/anders_hoh.tiff'
# landscape = load_data.get_landscape_tyrifjorden(file_name)
# file_name = 'E:\Dropbox\watershedLargeFiles/anders_hoh.tiff'
landscape = load_data.get_smallest_test_landscape_tyrifjorden(file_name)


r, c = np.shape(landscape.heights)
# plot.plot_landscape_2d(landscape, 1)
# acc_flow = river_analysis.calculate_accumulated_flow(landscape, d4=False)
# plot.plot_accumulated_flow(acc_flow)

step_size = 10
outlet = (172, 494)  # LandscapeLarge
# outlet = (235, 272) # Works, same tof regardless of max_time
# outlet = (454, 114) # works!
# outlet = (335, 182)  # Traps haven't been merged!!!
# outlet = (377, 310) # Works
# outlet = (204, 612)
# outlet = (618, 1992) 2kx2k
# outlet = (3269, 1041)

ws_of_node, traps, trap_heights, trap_indices_in_ws, steepest_spill_pairs, flow_directions, heights = river_analysis.\
    calculate_watershed_of_node(landscape, outlet, d4=False)
plot.plot_watersheds_2d([ws_of_node], outlet, landscape, 1)


"Pre-process output data"
nr_of_cells_in_each_trap = [len(traps[i]) for i in trap_indices_in_ws]
total_trap_cells = np.sum(nr_of_cells_in_each_trap)

traps = [util.map_1d_to_2d(traps[i], landscape.nx) for i in trap_indices_in_ws]
nr_of_traps = len(traps)
trap_heights = [trap_heights[t] for t in trap_indices_in_ws]

ws = util.map_1d_to_2d(ws_of_node, landscape.nx)

flow_directions = river_analysis.add_trap_flow_directions(flow_directions, steepest_spill_pairs)

zero_wrapping = np.zeros((landscape.ny, landscape.nx), dtype=int)
zero_wrapping[1:-1, 1:-1] = flow_directions[1:-1, 1:-1]
flow_directions = zero_wrapping
steepest_spill_pairs = [util.map_1d_to_2d(steepest_spill_pairs[i][0], landscape.nx) for i in range(len(steepest_spill_pairs)) if i in trap_indices_in_ws]

# Create cell array structure for traps
new_traps = np.zeros((len(traps), 2), dtype=object)
for i in range(len(traps)):
    new_traps[i][0] = traps[i][0]
    new_traps[i][1] = traps[i][1]

# new_spill_pairs = np.zeros((len(steepest_spill_pairs), 2), dtype=object)
# for i in range(len(steepest_spill_pairs)):
#     new_spill_pairs[i][0] = steepest_spill_pairs[i][0]
#     new_spill_pairs[i][1] = steepest_spill_pairs[i][1]

# The matrix is transposed if the type is masked array, so we use np.asarray!!!
heights = np.asarray(heights)


# Save all landscape information
scipy.io.savemat('landscape.mat', dict(watershed=ws, outlet=outlet, stepSize=step_size, heights=heights,
                                       traps=new_traps, totalTrapCells=total_trap_cells, nrOfTraps=nr_of_traps,
                                       nrOfCellsInEachTrap=nr_of_cells_in_each_trap, trapHeights=trap_heights,
                                       spillPairs=steepest_spill_pairs, flowDirections=flow_directions))
