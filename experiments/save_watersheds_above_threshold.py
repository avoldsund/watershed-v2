
from lib import load_data, plot, util, river_analysis
import cPickle as pickle
import numpy as np
import time

"""
Save the watersheds above the threshold
"""

saved_files = '/home/anderovo/Dropbox/watershedLargeFiles/'

landscape = pickle.load(open(saved_files + 'landscape.pkl', 'rb'))
watersheds = pickle.load(open(saved_files + 'watersheds.pkl', 'rb'))
steepest_spill_pairs = pickle.load(open(saved_files + 'steepestSpillPairs.pkl', 'rb'))

# Save the connectivity matrix between the watersheds
# conn_mat = util.create_watershed_conn_matrix(watersheds, steepest_spill_pairs, landscape.ny, landscape.nx)
# pickle.dump(conn_mat, open('connMat.pkl', 'wb'))

# Load the connectivity matrix
conn_mat = pickle.load(open(saved_files + 'connMat.pkl', 'rb'))

# Save spill_heights, all_traps and size_of_traps
# spill_heights = util.get_spill_heights(watersheds, landscape.heights, steepest_spill_pairs)
# pickle.dump(spill_heights, open('spillHeights.pkl', 'wb'))
# all_traps, size_of_traps = util.get_all_traps(watersheds, landscape.heights, spill_heights)
# pickle.dump(all_traps, open('allTraps.pkl', 'wb'))
# pickle.dump(size_of_traps, open('sizeOfTraps.pkl', 'wb'))

# Load spill_heights, all_traps and size_of_traps
spill_heights = pickle.load(open(saved_files + 'spillHeights.pkl', 'rb'))
all_traps = pickle.load(open(saved_files + 'allTraps.pkl', 'rb'))
size_of_traps = pickle.load(open(saved_files + 'sizeOfTraps.pkl', 'rb'))

# Remove watersheds below the threshold
threshold = 25
start = time.time()
conn_mat_after_thresholding, watersheds_after_thresholding = util.remove_watersheds_below_threshold(
    watersheds, conn_mat, size_of_traps, threshold, landscape.heights, landscape.total_nodes)
end = time.time()
print end-start

# Save the new connMat and watersheds
pickle.dump(conn_mat_after_thresholding, open('connMatThreshold25.pkl', 'wb'))
pickle.dump(watersheds_after_thresholding, open('watershedsThreshold25.pkl', 'wb'))
