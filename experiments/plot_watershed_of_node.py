from lib import util, river_analysis, load_data, plot

"""
Plot the upslope area (watershed) of a single node.
"""

saved_files = '/home/anderovo/Dropbox/watershedLargeFiles/'
file_name = saved_files + 'anders_hoh.tiff'


# Get all necessary information
landscape = load_data.get_smallest_test_landscape_tyrifjorden(file_name)
landscape.heights = util.fill_single_cell_depressions(landscape.heights, landscape.ny, landscape.nx)
watersheds, steepest, flow_dir = util.calculate_watersheds(landscape.heights, landscape.nx, landscape.ny, landscape.step_size)
spill_heights = util.get_spill_heights(watersheds, landscape.heights, steepest)
traps, size_of_traps = util.get_all_traps(watersheds, landscape.heights, spill_heights)

# Increase heights of traps and recalculate flow. Remove flow from some indices in traps.
util.make_landscape_depressionless(watersheds, steepest, landscape)
flow = util.get_flow_direction_indices(landscape.heights, landscape.step_size, landscape.ny, landscape.nx)
for i in range(len(traps)):
    trap_in_2d = util.map_1d_to_2d(traps[i], landscape.nx)
    flow[trap_in_2d] = -1

# Create connections and expand matrix to accommodate trap nodes
node_conn_mat = util.make_sparse_node_conn_matrix(flow, landscape.ny, landscape.nx)
node_conn_mat = river_analysis.expand_conn_mat(node_conn_mat, len(traps))
expanded_conn_mat = river_analysis.reroute_trap_connections(node_conn_mat, landscape.ny, landscape.nx, traps, steepest)

# Get the watershed of the node and plot it
node_coords_r_c = (220, 16)
ws_of_node = river_analysis.get_watershed_of_node(node_coords_r_c, expanded_conn_mat, traps, landscape.ny, landscape.nx)
plot.plot_watersheds_2d([ws_of_node], landscape, 1)
