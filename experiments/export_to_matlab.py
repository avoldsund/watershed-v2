import numpy as np
from matplotlib import pyplot as plt
from lib import load_data, util, river_analysis, plot
import scipy.io

# Different examples to show how the velocity field will be if the vectors point in the direction of steepest descent
# heights_one = np.array([[20, 20, 20],
#                         [20, 15, 20],
#                         [20, 5, 20]])
# heights_two = np.array([[20, 15, 10],
#                         [15, 10, 5],
#                         [10, 5, 0]])
# heights_three = np.array([[20, 20, 20],
#                           [5, 5, 5],
#                           [0, 0, 0]])
# heights_four = np.array([[0, 5, 20],
#                          [0, 5, 20],
#                          [0, 5, 20]])
# heights_five = np.array([[0, 0, 0],
#                          [20, 20, 20],
#                          [0, 0, 0]])
# heights_six = np.array([[10, 15],
#                         [0, 10]])
file_name = '/home/anderovo/Dropbox/watershedLargeFiles/anders_hoh.tiff'
landscape = load_data.get_smallest_test_landscape_tyrifjorden(file_name)
r, c = np.shape(landscape.heights)

ws_of_node = river_analysis.calculate_watershed_of_node(landscape, (230, 125))
ws_of_node_2d = util.map_1d_to_2d(ws_of_node, landscape.nx)
# river_analysis.calculate_nr_of_upslope_cells()
acc_flow = river_analysis.calculate_accumulated_flow(landscape)
# plot.plot_accumulated_flow(acc_flow)

print ws_of_node
# Change name to obtain different field
y, x = np.gradient(landscape.heights, edge_order=1)
y /= 10
x /= 10
# plot.plot_watersheds_2d_coords([ws_of_node], landscape, 1)

x = -x
# print 'dx: '
# print x
# print 'dy: '
# print y
#
# print ws_of_node_2d
# Plot vector field
plt.figure()

X = np.zeros((r, c))
Y = np.zeros((r, c))
X[ws_of_node_2d] = x[ws_of_node_2d]
Y[ws_of_node_2d] = y[ws_of_node_2d]
Q = plt.quiver(X, Y, pivot='mid', color='r', units='inches')
plt.axis([-1, c, -1, r])
plt.gca().invert_yaxis()

plt.show()

scipy.io.savemat('gradient.mat', dict(dx=X, dy=Y))
scipy.io.savemat('watershed.mat', dict(ws=ws_of_node_2d))
