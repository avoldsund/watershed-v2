from lib import load_data, plot, util, analysis
import cPickle as pickle
import numpy as np

"""
Plot the size of the traps for the watersheds
"""

saved_files = '/home/anderovo/Dropbox/watershedLargeFiles/'
file_name = saved_files + 'anders_hoh.tiff'

landscape = load_data.get_landscape_tyrifjorden(file_name)
watersheds = pickle.load(open(saved_files + 'watersheds.pkl', 'rb'))
steepest_spill_pairs = pickle.load(open(saved_files + 'steepestSpillPairs.pkl'))

spill_heights = util.get_spill_heights(watersheds, landscape.heights, steepest_spill_pairs)
size_of_traps = util.get_size_of_traps(watersheds, landscape.heights, spill_heights)
size_order = np.argsort(size_of_traps)

plot.plot_trap_sizes(size_of_traps, size_order)
