import numpy as np
from lib import plot

grid500 = np.array([0.31, 2.66, 0.66, 4.22, 0.003, 1.52, 0.85])
grid1000 = np.array([1.33, 13.81, 2.73, 17.83, 0.019, 7.17, 6.43])
grid2000 = np.array([5.83, 59.41, 13.32, 79.79, 0.098, 44.80, 68.47])
grid4000 = np.array([20.97, 278.69, 60.59, 496.99, 0.39, 318.27, 438.71])

grids = [grid500, grid1000, grid2000, grid4000]

plot.plot_running_times(grids)

#10.2
#49.3
#271.7
#1614.6