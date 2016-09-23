from lib import load_data, plot, util, analysis
import time
import numpy as np

"""
Plot the upslope/downslope watersheds, or both
"""

# Fetch elevation data
saved_files = '/home/anderovo/Dropbox/watershedLargeFiles/'
file_name = saved_files + 'anders_hoh.tiff'

# Get initial watersheds, remove single cell depressions
landscape = load_data.get_smallest_test_landscape(file_name)
landscape.heights = util.fill_single_cell_depressions(landscape.heights, landscape.ny, landscape.nx)
flow_directions = util.get_flow_direction_indices(landscape.heights, landscape.step_size, landscape.ny, landscape.nx)
node_endpoints = util.get_node_endpoints(flow_directions)
local_watersheds = util.get_local_watersheds(node_endpoints)
local_minima = np.asarray(local_watersheds.keys())
combined_minima = util.combine_minima(local_minima, landscape.ny, landscape.nx)
watersheds = util.combine_watersheds(local_watersheds, combined_minima)

# Get all spill pairs and create connectivity matrix
watersheds, steepest_spill_pairs = util.combine_watersheds_spilling_into_each_other(watersheds, landscape.heights)
conn_matrix = util.create_watershed_conn_matrix(watersheds, steepest_spill_pairs, landscape.ny, landscape.nx)

# Find watershed nr of interest
w_nr = util.get_watershed_nr_by_rc(watersheds, landscape, 300, 300)
upslope_indices = analysis.get_upslope_watersheds(conn_matrix, w_nr)
downslope_indices = analysis.get_downslope_watersheds(conn_matrix, w_nr)

# Plot the result. Watershed colored gold, upslope watersheds colored dodgerblue
plot.plot_upslope_watersheds(watersheds, upslope_indices, w_nr, landscape)

# For plotting both upslope and downslope watersheds, upslope dodgerblue, downslope darkorchid
# plot.plot_up_and_downslope_watersheds(watersheds, upslope_indices, downslope_indices, w_nr, landscape)
