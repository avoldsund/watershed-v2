from lib import load_data, util, river_analysis, plot
import numpy as np
import time


"""
Measure the time for each step in the algorithm
"""

saved_files = '/home/anderovo/Dropbox/watershedLargeFiles/'
file_name = saved_files + 'anders_hoh.tiff'

# landscape = load_data.get_landscape_tyrifjorden(file_name)  # Size 4000x4000
landscape = load_data.get_smallest_test_landscape(file_name)  # Set to size 500x500

start = time.time()
landscape.heights = util.fill_single_cell_depressions(landscape.heights, landscape.ny, landscape.nx)
end = time.time()
print 'Time for filling depressions: ', end-start

start = time.time()
flow_directions = util.get_flow_direction_indices(landscape.heights, landscape.step_size, landscape.ny, landscape.nx)
node_endpoints = util.get_node_endpoints(flow_directions)
end = time.time()
print 'Time for computing flow paths: ', end-start

start = time.time()
local_watersheds = util.get_local_watersheds(node_endpoints)
local_minima = np.asarray(local_watersheds.keys())
combined_minima = util.combine_minima(local_minima, landscape.ny, landscape.nx)
watersheds = util.combine_watersheds(local_watersheds, combined_minima)
end = time.time()
print 'Time for computing watersheds for each collection of minima: ', end-start
#
start = time.time()
watersheds, steepest_spill_pairs = util.combine_watersheds_spilling_into_each_other(watersheds, landscape.heights)
end = time.time()
print 'Time for computing steepest spill, combining watersheds and removing cycles: ', end-start

start = time.time()
conn_mat = util.create_watershed_conn_matrix(watersheds, steepest_spill_pairs, landscape.ny, landscape.nx)
end = time.time()
print 'Time for creating connectivity matrix for watersheds: ', end-start

old_watersheds = list(watersheds)  # Make copy of watersheds before thresholding
start = time.time()
threshold = 20
spill_heights = util.get_spill_heights(watersheds, landscape.heights, steepest_spill_pairs)
all_traps, size_of_traps = util.get_all_traps(watersheds, landscape.heights, spill_heights)
conn_mat, new_watersheds = util.remove_watersheds_below_threshold(
    watersheds, conn_mat, size_of_traps, threshold)
end = time.time()

print 'Time for thresholding: ', end-start

start = time.time()
all_rivers = river_analysis.get_rivers(old_watersheds, new_watersheds, steepest_spill_pairs, all_traps,
                                       flow_directions, landscape.heights)
end = time.time()
print 'Time for calculating rivers: ', end-start
thresholded_traps = [all_traps[i] for i in range(len(all_traps)) if size_of_traps[i] > threshold]

if len(all_rivers) > 0:
    all_rivers = np.concatenate(all_rivers)
plot.plot_traps_and_rivers(new_watersheds, thresholded_traps, all_rivers, landscape, 1)
