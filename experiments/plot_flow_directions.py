from lib import plot, util
import seaborn as sns; sns.set()
import cPickle as pickle

"""
Plot the flow directions
"""

saved_files = '/home/anderovo/Dropbox/watershedLargeFiles/'
landscape = pickle.load(open(saved_files + 'landscape.pkl', 'rb'))

flow_directions = util.get_flow_directions(landscape.heights, landscape.step_size,
                                           landscape.ny, landscape.nx)

plot.plot_flow_directions(flow_directions[1:-1, 1:-1])
