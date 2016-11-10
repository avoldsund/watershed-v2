
from lib import load_data, plot, util, analysis
import cPickle as pickle
import numpy as np

"""
Plot the landscape in two dimensions
"""

saved_files = '/home/anderovo/Dropbox/watershedLargeFiles/'
file_name = saved_files + 'anders_hoh.tiff'


# Get all watersheds before preprocessing them
landscape = load_data.get_landscape_tyrifjorden(file_name)
watersheds = pickle.load(open(saved_files + 'watershedsThreshold2500.pkl', 'rb'))
conn_mat = pickle.load(open(saved_files + 'connMatThreshold2500.pkl', 'rb'))
threshold = 2500

plot.plot_watersheds_2d(watersheds, landscape, 4)

#c = conn_mat.todense()
#print c.nonzero()
# watersheds = pickle.load(open(saved_files + 'watersheds.pkl', 'rb'))
# steepest_spill_pairs = pickle.load(open(saved_files + 'steepestSpillPairs.pkl', 'rb'))
#
# # Use threshold to remove smaller watersheds (in terms of spill volume)
# conn_mat = util.create_watershed_conn_matrix(watersheds, steepest_spill_pairs, landscape.ny, landscape.nx)
# spill_heights = util.get_spill_heights(watersheds, landscape.heights, steepest_spill_pairs)
# size_of_traps = util.get_size_of_traps(watersheds, landscape.heights, spill_heights)
# threshold_size = 25
# total_nodes = landscape.nx * landscape.ny
# conn_mat, watersheds = util.remove_watersheds_below_threshold(watersheds, conn_mat,
#                                                               size_of_traps, threshold_size, total_nodes)
#
# Choose threshold size in the name of the pickled files
# pickle.dump(conn_mat, open('connMatThreshold500.pkl', 'wb'))
# pickle.dump(watersheds, open('watershedsThreshold500.pkl', 'wb'))

# Find watershed nr of interest
# w_nr = util.get_watershed_nr_by_rc(watersheds, landscape, 1500, 1400)
# upslope_indices, node_levels = analysis.get_upslope_watersheds(conn_mat, w_nr)
# downslope_indices = analysis.get_downslope_watersheds(conn_mat, w_nr)

# Plot the result. Watershed colored gold, upslope watersheds colored dodgerblue
#plot.plot_downslope_watersheds(watersheds, downslope_indices, landscape, ds=8)

# plot.plot_up_and_downslope_watersheds(watersheds, upslope_indices, downslope_indices, node_levels, landscape, ds=8)

#plot.plot_watersheds_2d(watersheds, landscape, 1)
