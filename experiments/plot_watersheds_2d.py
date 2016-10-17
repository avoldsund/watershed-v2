from lib import plot, util
import cPickle as pickle

"""
Plot the watersheds, either thresholded or not
"""

saved_files = '/home/anderovo/Dropbox/watershedLargeFiles/'

landscape = pickle.load(open(saved_files + 'landscape.pkl', 'rb'))
watersheds = pickle.load(open(saved_files + 'watersheds.pkl', 'rb'))
steepest = pickle.load(open(saved_files + 'steepestSpillPairs.pkl', 'rb'))

watersheds = util.merge_watersheds(watersheds, steepest, landscape.nx, landscape.ny)

plot.plot_watersheds_2d(watersheds, landscape, 8)
