
from lib import plot, analysis, util
import cPickle as pickle
import numpy as np

"""
Plot the trap and the river in one watershed
"""

saved_files = '/home/anderovo/Dropbox/watershedLargeFiles/'

landscape = pickle.load(open(saved_files + 'landscape.pkl', 'rb'))
watersheds = pickle.load(open(saved_files + 'watersheds.pkl', 'rb'))
thresholded_watersheds = pickle.load(open(saved_files + 'watershedsThreshold2500.pkl', 'rb'))
steepest_spill_pairs = pickle.load(open(saved_files + 'steepestSpillPairs.pkl', 'rb'))
flow_directions = pickle.load(open(saved_files + 'flowDirections.pkl', 'rb'))
spill_heights = pickle.load(open(saved_files + 'spillHeights.pkl', 'rb'))
all_traps = pickle.load(open(saved_files + 'allTraps.pkl', 'rb'))
size_of_traps = pickle.load(open(saved_files + 'sizeOfTraps.pkl', 'rb'))

index_of_ws = 8
threshold = 2500
thresholded_traps = [all_traps[i] for i in range(len(all_traps)) if size_of_traps[i] > threshold]

all_rivers = analysis.get_rivers(watersheds, [thresholded_watersheds[index_of_ws]], steepest_spill_pairs, all_traps,
                                 flow_directions, landscape.heights)

if len(all_rivers) > 0:
    all_rivers = np.concatenate(all_rivers)

# Watersheds in thresholded watershed
mapping = util.map_nodes_to_watersheds(watersheds, landscape.ny, landscape.nx)
new_mapping = util.map_nodes_to_watersheds(thresholded_watersheds, landscape.ny, landscape.nx)
watershed_indices = np.unique([mapping[el] for el in thresholded_watersheds[index_of_ws]])
selected_watersheds = [watersheds[i] for i in watershed_indices]

# Get the watersheds the river flows through
ws_river_passes_through = np.unique([mapping[el] for el in all_rivers])
print ws_river_passes_through
river_watersheds = [watersheds[i] for i in ws_river_passes_through]

# Traps of watersheds the river passes through
traps_river_passes_through = [all_traps[i] for i in range(len(all_traps)) if i in ws_river_passes_through]

plot.plot_traps_and_river_in_one_watershed_2d(traps_river_passes_through, river_watersheds, selected_watersheds, all_rivers, landscape, 1)
