
from lib import load_data, plot, util, analysis
import cPickle as pickle
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

"""
Plot all rivers from the spill points
"""

saved_files = '/home/anderovo/Dropbox/watershedLargeFiles/'

landscape = pickle.load(open(saved_files + 'landscape.pkl', 'rb'))
watersheds = pickle.load(open(saved_files + 'watershedsThreshold0.pkl', 'rb'))
thresholded_watersheds = pickle.load(open(saved_files + 'watershedsThreshold2500.pkl', 'rb'))
steepest_spill_pairs = pickle.load(open(saved_files + 'steepestSpillPairs.pkl', 'rb'))
flow_directions = pickle.load(open(saved_files + 'flowDirections.pkl', 'rb'))
spill_heights = pickle.load(open(saved_files + 'spillHeights.pkl', 'rb'))
all_traps = pickle.load(open(saved_files + 'allTraps.pkl', 'rb'))
size_of_traps = pickle.load(open(saved_files + 'sizeOfTraps.pkl', 'rb'))

threshold = 2500
thresholded_traps = [all_traps[i] for i in range(len(all_traps)) if size_of_traps[i] > threshold]

for t in thresholded_traps:
    row_col = util.map_1d_to_2d(t, landscape.nx)
    plt.scatter(landscape.x_min + row_col[1][0::1] * landscape.step_size,
                landscape.y_max - row_col[0][0::1] * landscape.step_size,
                color='gold', s=10, lw=0, alpha=1)
    plt.show()

# rivers = analysis.get_rivers(watersheds, thresholded_watersheds, steepest_spill_pairs, all_traps,
#                              flow_directions, landscape.heights)
#
# all_rivers = np.concatenate(rivers)
#
# plot.plot_traps_and_rivers(thresholded_watersheds, thresholded_traps, all_rivers, threshold, landscape, 1)
#