from lib import load_data, plot, util
import numpy as np
import cPickle as pickle

"""

"""

saved_files = '/home/anderovo/Dropbox/watershedLargeFiles/'
file_name = saved_files + 'anders_hoh.tiff'

# Get all watersheds before preprocessing them
landscape = load_data.get_landscape_tyrifjorden(file_name)
watersheds = pickle.load(open(saved_files + 'watersheds.pkl', 'rb'))
# steepest_spill_pairs = pickle.load(open(saved_files + 'steepestSpillPairs.pkl', 'rb'))
# combined_minima = pickle.load(open(saved_files + 'combinedMinima.pkl', 'rb'))
print len(watersheds)
# # Fill single cell depressions
landscape.heights = util.fill_single_cell_depressions(landscape.heights, landscape.ny, landscape.nx)

# Get flow directions and initial watersheds
flow_directions = util.get_flow_direction_indices(landscape.heights, landscape.step_size, landscape.ny, landscape.nx)
print len(np.where(flow_directions == -1)[0])
node_endpoints = util.get_node_endpoints(flow_directions)
local_watersheds = util.get_local_watersheds(node_endpoints)
local_minima = np.asarray(local_watersheds.keys())
print len(local_minima)
combined_minima = util.combine_minima(local_minima, landscape.ny, landscape.nx)
print len(combined_minima)
#print combined_minima
# pickle.dump(combined_minima, open('combinedMinima.pkl', 'wb'))
# watersheds = util.combine_watersheds(local_watersheds, combined_minima)
#
# # Preprocess watersheds:
# watersheds, steepest_spill_pairs = util.combine_watersheds_spilling_into_each_other(watersheds, landscape.heights)

# conn_mat = util.create_watershed_conn_matrix(watersheds, steepest_spill_pairs, landscape.ny, landscape.nx)
# spill_heights = util.get_spill_heights(watersheds, landscape.heights, steepest_spill_pairs)
# size_of_traps = util.get_size_of_traps(watersheds, landscape.heights, spill_heights)
# threshold_size = 1000
# total_nodes = landscape.nx * landscape.ny
# conn_mat, watersheds = util.remove_watersheds_below_threshold(watersheds, conn_mat, size_of_traps, threshold_size, total_nodes)
#
# pickle.dump(conn_mat, open('connMatThreshold1000.pkl', 'wb'))
#pickle.dump(watersheds, open('watershedsThreshold1000.pkl', 'wb'))

minima_size = 500
largest_minima = [c for c in combined_minima if len(c) > minima_size]

plot.plot_watersheds_and_minima_2d(watersheds, largest_minima, landscape, 1)

# Plot the watersheds
# plot.plot_watersheds_2d(watersheds, landscape, 1)
