
from lib import plot, analysis, util
import cPickle as pickle
import numpy as np

"""
Plot all rivers from the spill points
"""

saved_files = '/home/anderovo/Dropbox/watershedLargeFiles/'

landscape = pickle.load(open(saved_files + 'landscape.pkl', 'rb'))
watersheds = pickle.load(open(saved_files + 'watersheds.pkl', 'rb'))
thresholded_watersheds = pickle.load(open(saved_files + 'watershedsThreshold500.pkl', 'rb'))
steepest_spill_pairs = pickle.load(open(saved_files + 'steepestSpillPairs.pkl', 'rb'))
flow_directions = pickle.load(open(saved_files + 'flowDirections.pkl', 'rb'))
spill_heights = pickle.load(open(saved_files + 'spillHeights.pkl', 'rb'))
all_traps = pickle.load(open(saved_files + 'allTraps.pkl', 'rb'))
size_of_traps = pickle.load(open(saved_files + 'sizeOfTraps.pkl', 'rb'))

threshold = 500
thresholded_traps = [all_traps[i] for i in range(len(all_traps)) if size_of_traps[i] > threshold]

all_rivers = analysis.get_rivers(watersheds, watersheds, steepest_spill_pairs, all_traps,
                                 flow_directions, landscape.heights)

all_rivers = [r for r in all_rivers if len(r) > 250]
if len(all_rivers) > 0:
    all_rivers = np.concatenate(all_rivers)

plot.plot_traps_and_rivers(thresholded_watersheds, thresholded_traps, all_rivers, threshold, landscape, 4)