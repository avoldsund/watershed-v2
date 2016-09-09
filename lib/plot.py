import numpy as np
import matplotlib.pyplot as plt
from lib import util


def discrete_cmap(N, base_cmap=None):
    """Create an N-bin discrete colormap from the specified input map"""

    # Note that if base_cmap is a string or None, you can simply do
    #    return plt.cm.get_cmap(base_cmap, N)
    # The following works for string, None, or a colormap instance:
    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)

    return base.from_list(cmap_name, color_list, N)


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
    # z = landscape.interior_heights[0::ds, 0::ds]
    #
    # # Plotting the terrain in the background
    # cmap = plt.get_cmap('terrain')
    # v = np.linspace(np.min(landscape.interior_heights), np.max(landscape.interior_heights), 100, endpoint=True)
    # plt.contourf(x, y, z, v, cmap=cmap)
    # plt.colorbar(label='Height', spacing='uniform')

    # Only plot the n largest watersheds
    ws_above_n_nodes = 100
    large_watersheds = [ws for ws in watersheds if len(ws) > ws_above_n_nodes]
    large_watersheds.sort(key=len)
    nr_of_large_watersheds = len(large_watersheds)

    color_list = ['red', 'green', 'blue', 'yellow']
    color_list = iter(color_list * (nr_of_large_watersheds/3))

    nr_of_largest_indigo = 10

    for i in range(0, nr_of_large_watersheds - nr_of_largest_indigo):
        row_col = util.map_1d_to_2d(large_watersheds[i], landscape.interior_nx)
        plt.scatter(landscape.x_min + row_col[1][0::ds] * landscape.step_size,
                    landscape.y_max - row_col[0][0::ds] * landscape.step_size,
                    color=next(color_list), s=30, lw=0, alpha=0.7)

    for i in range(nr_of_large_watersheds - nr_of_largest_indigo, nr_of_large_watersheds):
        print len(large_watersheds[i])
        row_col = util.map_1d_to_2d(large_watersheds[i], landscape.interior_nx)
        plt.scatter(landscape.x_min + row_col[1][0::ds] * landscape.step_size,
                    landscape.y_max - row_col[0][0::ds] * landscape.step_size,
                    color='indigo', s=30, lw=0, alpha=0.7)

    #plt.rcParams.update({'font.size': 14})

    if ws_above_n_nodes == 0:
        plt.title('All watersheds in the landscape')
    else:
        plt.title('All watersheds with over %s nodes in the landscape'%str(ws_above_n_nodes))
    plt.xlabel('x')
    plt.ylabel('y')

    plt.show()


def plot_flow_directions(flow_directions, landscape):

    (r, c) = np.shape(flow_directions)
    N = 9
    plt.matshow(flow_directions, cmap=discrete_cmap(N, "RdBu_r"))
    plt.grid(False)
    cb = plt.colorbar(ticks=[-1, 0, 1, 2, 3, 4, 5, 6, 7], )
    cb.set_ticklabels(['N/A', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW', 'N'])
    plt.clim(-1.5, 7.5)

    x_labels = np.arange(landscape.x_min, landscape.x_max, 500, dtype=int)
    y_labels = np.arange(landscape.y_min, landscape.y_max, 500, dtype=int)
    x_ticks = np.arange(0, c, 500)
    y_ticks = np.arange(3998, 0, -500)
    plt.xticks(x_ticks, x_labels)
    plt.yticks(y_ticks, y_labels)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.tick_params(labelbottom='on', labeltop='off')
    plt.title('Flow directions')
    plt.show()


def plot_local_minima(minima, landscape):

    (r, c) = np.shape(minima)
    N = 2
    plt.matshow(minima, cmap=discrete_cmap(N, "Blues"))
    plt.grid(False)
    #cb = plt.colorbar(ticks=[0, 1], )
    #cb.set_ticklabels(['Not Minima', 'Minima'])
    #plt.clim(-0.5, 1.5)

    x_labels = np.arange(landscape.x_min, landscape.x_max, 500, dtype=int)
    y_labels = np.arange(landscape.y_min, landscape.y_max, 500, dtype=int)
    x_ticks = np.arange(0, c, 500)
    y_ticks = np.arange(3998, 0, -500)
    plt.xticks(x_ticks, x_labels)
    plt.yticks(y_ticks, y_labels)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.tick_params(labelbottom='on', labeltop='off')

    plt.title('Minima in the landscape')

    plt.show()
