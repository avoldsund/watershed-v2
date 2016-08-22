import numpy as np
import matplotlib.pyplot as plt


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

    #cbar = fig.colorbar(cax, ticks=[-1, 0, 1])
    #cbar.ax.set_yticklabels(['< -1', '0', '> 1', 'a', 'b', 'c'])  # vertically oriented colorbar


def plot_local_minima(local_minima):

    plt.matshow(local_minima)
    plt.show()
