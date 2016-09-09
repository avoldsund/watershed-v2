from lib import load_data, plot, util
from matplotlib import pyplot as plt
from matplotlib import colors
import numpy as np
import seaborn as sns; sns.set()
from matplotlib.colors import LinearSegmentedColormap

"""
Plot the landscape in two dimensions
"""

saved_files = '/home/anderovo/Dropbox/watershedLargeFiles/'
file_name = saved_files + 'anders_hoh.tiff'
landscape = load_data.get_landscape_tyrifjorden(file_name)

flow_directions = util.get_flow_directions(landscape.heights, landscape.step_size,
                                           landscape.interior_ny, landscape.interior_nx)

plot.plot_flow_directions(flow_directions, landscape)
