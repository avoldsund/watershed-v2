from lib import load_data, plot, util, river_analysis
import cPickle as pickle
import numpy as np
import time

"""
Plot all traps in the landscape above a specific threshold
"""

saved_files = '/home/anderovo/Dropbox/watershedLargeFiles/'

# Load all data
landscape = pickle.load(open(saved_files + 'landscape.pkl', 'rb'))
watersheds = pickle.load(open(saved_files + 'watersheds.pkl', 'rb'))
watershedsThresholded = pickle.load(open(saved_files + 'watershedsThreshold2500.pkl', 'rb'))
conn_mat = pickle.load(open(saved_files + 'connMatThreshold2500.pkl', 'rb'))
steepest_spill_pairs = pickle.load(open(saved_files + 'steepestSpillPairs.pkl', 'rb'))
spill_heights = pickle.load(open(saved_files + 'spillHeights.pkl', 'rb'))
size_of_traps = pickle.load(open(saved_files + 'sizeOfTraps.pkl', 'rb'))

# Recalculate steepest_steepest_spill_pairs for remaining watersheds
steepest_steepest_spill_pairs = util.remap_steepest_spill_pairs(
    watershedsThresholded, steepest_spill_pairs, landscape.ny, landscape.nx)

threshold = 2500
traps = util.get_threshold_traps(watersheds, size_of_traps, threshold, landscape.heights)

plot.plot_traps(watershedsThresholded, traps, threshold, landscape, 4)
