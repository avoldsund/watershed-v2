import numpy as np
import matplotlib.pyplot as plt
from lib import util


def plot_landscape_2d(landscape, ds):

    x = np.linspace(landscape.x_min, landscape.x_max, landscape.nx)
    y = np.linspace(landscape.y_max, landscape.y_min, landscape.ny)

    cmap = plt.get_cmap('terrain')
    min_height = np.min(landscape.heights)
    max_height = np.max(landscape.heights)
    v = np.linspace(min_height, max_height, 100, endpoint=True)

    plt.contourf(x[0::ds], y[0::ds], landscape.heights[0::ds, 0::ds], v, cmap=cmap)
    plt.colorbar(label='Height', spacing='uniform')

    plt.show()


def plot_watersheds_2d(watersheds, landscape, ds):

    # Construct the (x, y)-coordinate system
    x_grid = np.linspace(landscape.x_min, landscape.x_max, landscape.interior_nx)
    y_grid = np.linspace(landscape.y_max, landscape.y_min, landscape.interior_ny)
    x, y = np.meshgrid(x_grid[0::ds], y_grid[0::ds])
    z = landscape.interior_heights[0::ds, 0::ds]

    # Plotting the terrain in the background
    cmap = plt.get_cmap('terrain')
    v = np.linspace(np.min(landscape.interior_heights), np.max(landscape.interior_heights), 100, endpoint=True)
    plt.contourf(x, y, z, v, cmap=cmap)
    plt.colorbar(label='Height', spacing='uniform')

    # Only plot the n largest watersheds
    ws_above_n_nodes = 1000
    large_watersheds = [ws for ws in watersheds if len(ws) > ws_above_n_nodes]
    large_watersheds.sort(key=len)
    nr_of_large_watersheds = len(large_watersheds)

    color_list = ['red', 'green', 'blue', 'yellow']
    color_list = iter(color_list * (nr_of_large_watersheds/3))

    for i in range(len(large_watersheds)):
        row_col = util.map_1d_to_2d(large_watersheds[i], landscape.interior_nx)
        plt.scatter(landscape.x_min + row_col[1][0::ds] * landscape.step_size,
                    landscape.y_max - row_col[0][0::ds] * landscape.step_size,
                    color=next(color_list), s=30, lw=0, alpha=0.7)

    plt.show()
