import os
from lib import load_data, plot

"""
Plot the heights of the landscape in two dimensions
"""

file_name = '/home/anderovo/Dropbox/watershedLargeFiles/anders_hoh.tiff'
landscape = load_data.get_landscape_tyrifjorden(file_name)

ds = 8
#plot.plot_landscape_3d(landscape, ds)

plot.plot_hillshade(landscape.heights)