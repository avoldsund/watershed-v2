import os
from lib import load_data, plot

"""
Plot the landscape in two dimensions
"""

file_name = '/home/anderovo/Dropbox/watershedLargeFiles/anders_hoh.tiff'
landscape = load_data.get_landscape_tyrifjorden(file_name)

ds = 16
plot.plot_landscape_2d(landscape, ds)
