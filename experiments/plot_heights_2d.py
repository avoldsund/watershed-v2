import os
from lib import load_data, plot

"""
Plot the landscape in two dimensions
"""
import numpy as np

file_name = '/home/anderovo/Dropbox/watershedLargeFiles/anders_hoh.tiff'
landscape = load_data.get_landscape_tyrifjorden(file_name)

ds = 1
plot.plot_landscape_2d(landscape, ds)
