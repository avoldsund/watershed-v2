from lib import load_data, plot, util
import numpy as np
import cPickle as pickle

"""
Plot the landscape in two dimensions
"""

saved_files = '/home/anderovo/Dropbox/watershedLargeFiles/'

# Get all watersheds before preprocessing them
landscape = pickle.load(open(saved_files + 'landscape.pkl', 'rb'))
watersheds = pickle.load(open(saved_files + 'watersheds.pkl', 'rb'))
steepest = pickle.load(open(saved_files + 'steepestSpillPairs.pkl', 'rb'))

# Merge the watersheds that spill into each other (connected components)
watersheds = util.merge_watersheds(watersheds, list(steepest), landscape.nx, landscape.ny)

# Plot the watersheds
plot.plot_watersheds_2d(watersheds, landscape, 4)
