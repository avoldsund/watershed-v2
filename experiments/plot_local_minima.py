import os
from lib import load_data, plot, util
import time
from scipy.sparse import csr_matrix, find
import numpy as np
from scipy import sparse, io
import cPickle as pickle

"""
Plot the landscape in two dimensions
"""

saved_files = '/home/anderovo/Dropbox/watershedLargeFiles/'
landscape = pickle.load(open(saved_files + 'landscape.pkl', 'rb'))

flow_dir = util.get_flow_directions(landscape.heights, landscape.step_size, landscape.ny, landscape.nx)
minima = np.where(flow_dir == -1)[0]

plot.plot_local_minima(minima, landscape)
