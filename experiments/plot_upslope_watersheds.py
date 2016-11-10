from lib import load_data, plot, util, analysis
import time
import numpy as np
import cPickle as pickle

"""
Plot the upslope/downslope watersheds, or both
"""

# Fetch elevation data
saved_files = '/home/anderovo/Dropbox/watershedLargeFiles/'
file_name = saved_files + 'anders_hoh.tiff'

landscape = load_data.get_landscape_tyrifjorden(file_name)
watersheds = pickle.load(open(saved_files + 'watershedsThreshold1000.pkl', 'rb'))
conn_mat = pickle.load(open(saved_files + 'connMatThreshold1000.pkl', 'rb'))

# Find watershed nr of interest
# Vaelern (1700, 1400)
# Tyrifjorden (2000, 2000)
w_nr = util.get_watershed_nr_by_rc(watersheds, landscape, 1700, 1400)
upslope_indices, node_levels = analysis.get_upslope_watersheds(conn_mat, w_nr)
downslope_indices = analysis.get_downslope_watersheds(conn_mat, w_nr)

# Upslope watersheds
# plot.plot_upslope_watersheds(watersheds, upslope_indices, node_levels, landscape)

# For plotting both upslope and downslope watersheds
plot.plot_up_and_downslope_watersheds(watersheds, upslope_indices, downslope_indices, node_levels, w_nr, landscape)
