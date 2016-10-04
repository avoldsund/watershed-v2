from lib import load_data, plot, util, analysis
import cPickle as pickle
import numpy as np

"""
Plot all traps in the landscape above a specific threshold
"""

saved_files = '/home/anderovo/Dropbox/watershedLargeFiles/'
file_name = saved_files + 'anders_hoh.tiff'

landscape = load_data.get_landscape_tyrifjorden(file_name)

watersheds = pickle.load(open(saved_files + 'watershedsThreshold1000.pkl', 'rb'))
conn_mat = pickle.load(open(saved_files + 'connMatThreshold1000.pkl', 'rb'))
spill_pairs = pickle.load(open(saved_files + 'steepestSpillPairs.pkl', 'rb'))


# This will remove some not spilling out of the boundary as well
# Remove the watersheds spilling out of the boundary
# a = np.nonzero(conn_mat)
# new_watersheds = [watersheds[i] for i in range(len(watersheds)) if i in a[0]]

# Recalculate steepest_spill_pairs for remaining watersheds
steepest_spill_pairs = util.remap_steepest_spill_pairs(watersheds, spill_pairs, landscape.ny, landscape.nx)
spill_heights = util.get_spill_heights(watersheds, landscape.heights, steepest_spill_pairs)
size_of_traps = util.get_size_of_traps(watersheds, landscape.heights, spill_heights)
threshold = 1000

plot.plot_traps(watersheds, spill_heights, threshold, landscape, 4)
