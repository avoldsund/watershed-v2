from lib import load_data, plot, util
import numpy as np
import cPickle as pickle

"""
Plot the landscape in two dimensions
"""

saved_files = '/home/anderovo/Dropbox/watershedLargeFiles/'
file_name = saved_files + 'anders_hoh.tiff'

# Get all watersheds before preprocessing them
landscape = load_data.get_landscape_tyrifjorden(file_name)
watersheds = pickle.load(open(saved_files + 'watershedsThreshold0.pkl', 'rb'))
steepest = pickle.load(open(saved_files + 'steepestSpillPairs.pkl', 'rb'))

# Merge the watersheds that spill into each other (connected components)
watersheds = util.merge_watersheds(watersheds, list(steepest), landscape.nx, landscape.ny)

# Plot the watersheds
plot.plot_watersheds_2d(watersheds, landscape, 4)
