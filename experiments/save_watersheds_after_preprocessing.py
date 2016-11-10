from lib import load_data, plot, util
import numpy as np
import cPickle as pickle
import time


"""
Save the watersheds after preprocessing
"""

saved_files = '/home/anderovo/Dropbox/watershedLargeFiles/'
file_name = saved_files + 'anders_hoh.tiff'

# Save the landscape with depressions filled
landscape = load_data.get_landscape_tyrifjorden(file_name)
# landscape.heights = util.fill_single_cell_depressions(landscape.heights, landscape.ny, landscape.nx)
# pickle.dump(landscape, open('landscape.pkl', 'wb'))
# Load the landscape with the depressions filled
landscape = pickle.load(open(saved_files + 'landscape.pkl', 'rb'))

# Save the flow directions
# flow_directions = util.get_flow_direction_indices(landscape.heights, landscape.step_size, landscape.ny, landscape.nx)
# pickle.dump(flow_directions, open('flowDirections.pkl', 'wb'))
# Load the flow directions
flow_directions = pickle.load(open(saved_files + 'flowDirections.pkl', 'rb'))

node_endpoints = util.get_node_endpoints(flow_directions)
local_watersheds = util.get_local_watersheds(node_endpoints)
local_minima = np.asarray(local_watersheds.keys())
combined_minima = util.combine_minima(local_minima, landscape.ny, landscape.nx)
watersheds = util.combine_watersheds(local_watersheds, combined_minima)

# Preprocess watersheds:
watersheds, steepest_spill_pairs = util.combine_watersheds_spilling_into_each_other(watersheds, landscape.heights)
# Save the preprocessed watersheds and the steepest spill pairs
pickle.dump(watersheds, open('watersheds.pkl', 'wb'))
pickle.dump(steepest_spill_pairs, open('steepestSpillPairs.pkl', 'wb'))
