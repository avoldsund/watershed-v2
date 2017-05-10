import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from lib import util
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LightSource
from matplotlib.cbook import get_sample_data


def discrete_cmap(N, base_cmap=None):
    """Create an N-bin discrete colormap from the specified input map"""
    # Developed by jakevdp
    # https://gist.github.com/jakevdp/91077b0cae40f8f8244a

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

    plt.title('The elevations of the landscape')
    font = {'family': 'normal',
            'size': 24}
    plt.rc('font', **font)

    plt.show()


def plot_landscape_3d(landscape, ds):
    """
    Plot the height of the landscape in 3 dimensions, given a landscape object.
    :param landscape: Landscape object with all data.
    :param ds: Downsampling factor for only plotting every ds point.
    :return: Plot the landscape in the x-y-z-coordinate system.
    """

    # Construct the (x, y)-coordinate system
    x_grid = np.linspace(landscape.x_min, landscape.x_max, landscape.nx)
    y_grid = np.linspace(landscape.y_max, landscape.y_min, landscape.ny)
    x, y = np.meshgrid(x_grid[0::ds], y_grid[0::ds])
    z = landscape.heights[0::ds, 0::ds]

    # Plot (x, y, z) in 3D
    fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
    ax.plot_surface(x, y, z, cmap=plt.get_cmap('terrain'))

    ax.axis('off')

    plt.show()


def plot_watersheds_2d(watersheds, outlet, landscape, ds):
    """
    Plot all watersheds and their boundary nodes
    :param watersheds: List of arrays where each array is a watershed
    :param landscape: A landscape object holding metadata
    :param ds: Downsampling factor if only every ds node shall be plotted
    :return: Plots the watersheds
    """

    nr_of_watersheds = len(watersheds)
    # boundary_pairs = util.get_boundary_pairs_in_watersheds(watersheds, landscape.nx, landscape.ny)

    color_hex = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33', '#a65628', '#f781bf']
    if len(watersheds) < 3:
        color_small = iter(color_hex)
    else:
        color_small = iter(color_hex * (len(watersheds)/3))
    fig = plt.figure()
    ax = fig.gca()  # fig.add_subplot(111, aspect=1)

    # Plot the watersheds
    for i in range(nr_of_watersheds):
        print i
        row_col = util.map_1d_to_2d(watersheds[i], landscape.nx)
        plt.scatter(landscape.x_min + row_col[1][0::ds],
                    landscape.y_max - row_col[0][0::ds],
                    color=next(color_small), s=5, lw=0, alpha=1)

    # Outlet is of the form (row, col) <-> (y, x)
    plt.scatter(landscape.x_min + outlet[1], landscape.y_max - outlet[0],
                color='b', s=10, lw=0, alpha=1)

    # Plot the boundary nodes
    # for i in range(nr_of_watersheds):
    #     b_p = boundary_pairs[i]
    #     u = np.unique(np.concatenate((b_p[0], b_p[1])))
    #     row_col = util.map_1d_to_2d(u, landscape.nx)
    #     plt.scatter(landscape.x_min + row_col[1][0::ds] * landscape.step_size,
    #                 landscape.y_max - row_col[0][0::ds] * landscape.step_size,
    #                 color='black', s=2, lw=0, alpha=1)

    fig.tight_layout()
    ax.axis('off')

    plt.show()


def plot_watersheds_2d_coords(watersheds, landscape, ds):
    """
    Plot all watersheds and their boundary nodes
    :param watersheds: List of arrays where each array is a watershed
    :param landscape: A landscape object holding metadata
    :param ds: Downsampling factor if only every ds node shall be plotted
    :return: Plots the watersheds
    """

    nr_of_watersheds = len(watersheds)

    color_hex = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33', '#a65628', '#f781bf']
    if len(watersheds) < 3:
        color_small = iter(color_hex)
    else:
        color_small = iter(color_hex * (len(watersheds)/3))
    fig = plt.figure()
    ax = fig.add_subplot(111, aspect=1)

    # Plot the watersheds
    for i in range(nr_of_watersheds):
        row_col = util.map_1d_to_2d(watersheds[i], landscape.nx)
        plt.scatter(row_col[1][0::ds], row_col[0][0::ds],
                    color=next(color_small), s=25, lw=0, alpha=1)

    ax.axis('off')

    plt.show()


def plot_traps_and_river_in_one_watershed_2d(selected_traps, river_watersheds, selected_watersheds, rivers, landscape, ds):
    """
    Plot the traps and river in one of the thresholded watersheds.
    :param selected_traps: The traps in the 'river watersheds', i.e. where a river passes through
    :param river_watersheds: All watersheds a river passes through
    :param selected_watersheds: All watersheds in the thresholded watershed
    :param rivers: The rivers in the thresholded watershed
    :param landscape: The landscape object
    :param ds: Downsampling factor
    :return: Plots the rivers and the traps in the river watersheds for a watershed with trap above the threshold
    """

    # Only get the boundary points of the sub-all_watersheds
    boundary_pairs = util.get_boundary_pairs_in_watersheds(selected_watersheds, landscape.nx, landscape.ny)

    color_hex = ['#d9d9d9', '#bdbdbd', '#969696', '#737373', '#525252', '#252525']
    color_ws = iter(color_hex * (len(selected_watersheds) / 3))

    fig = plt.figure()
    ax = fig.add_subplot(111, aspect=1)

    # Plot the watersheds in the thresholded watershed with shades of grey
    for ws in selected_watersheds:
        row_col = util.map_1d_to_2d(ws, landscape.nx)
        plt.scatter(landscape.x_min + row_col[1][0::ds] * landscape.step_size,
                    landscape.y_max - row_col[0][0::ds] * landscape.step_size,
                    color=next(color_ws), s=10, lw=0, alpha=1)

    # Plot the watersheds the rivers passes through in a different color
    for ws in river_watersheds:
        row_col = util.map_1d_to_2d(ws, landscape.nx)
        plt.scatter(landscape.x_min + row_col[1][0::ds] * landscape.step_size,
                    landscape.y_max - row_col[0][0::ds] * landscape.step_size,
                    color='#b2df8a', s=10, lw=0, alpha=1)

    # Plot the boundary nodes
    for i in range(len(boundary_pairs)):
        b_p = boundary_pairs[i]
        u = np.unique(np.concatenate((b_p[0], b_p[1])))
        row_col = util.map_1d_to_2d(u, landscape.nx)
        plt.scatter(landscape.x_min + row_col[1][0::ds] * landscape.step_size,
                    landscape.y_max - row_col[0][0::ds] * landscape.step_size,
                    color='black', s=10, lw=0, alpha=1)

    # Plot the traps in the river watersheds
    for t in selected_traps:
        row_col = util.map_1d_to_2d(t, landscape.nx)
        plt.scatter(landscape.x_min + row_col[1][0::ds] * landscape.step_size,
                    landscape.y_max - row_col[0][0::ds] * landscape.step_size,
                    color='#1f78b4', s=10, lw=0, alpha=1)

    # Plot the river in this watershed
    river_row_col = util.map_1d_to_2d(rivers, landscape.nx)
    plt.scatter(landscape.x_min + river_row_col[1][0::1] * landscape.step_size,
                landscape.y_max - river_row_col[0][0::1] * landscape.step_size,
                color='#a6cee3', s=10, lw=0, alpha=1)

    ax.axis('off')

    plt.show()


def plot_watersheds_before_thresholding(watersheds, landscape, ds):
    """
    Plot all watersheds before any thresholding. Do not plot watersheds below the a set number of nodes.
    :param watersheds: All watersheds before thresholding
    :param landscape: The landscape object
    :param ds: Downsampling factor
    :return: Plot all watersheds
    """

    ws_above_n_nodes = 100
    large_watersheds = [ws for ws in watersheds if len(ws) > ws_above_n_nodes]
    large_watersheds.sort(key=len)

    color_small = ['red', 'green', 'blue', 'yellow']
    colors = iter(color_small * (len(watersheds)/3))

    fig = plt.figure()
    ax = fig.add_subplot(111, aspect=1)

    for i in range(len(large_watersheds)):
        row_col = util.map_1d_to_2d(large_watersheds[i], landscape.nx)
        plt.scatter(landscape.x_min + row_col[1][0::ds] * landscape.step_size,
                    landscape.y_max - row_col[0][0::ds] * landscape.step_size,
                    color=next(colors), s=2, lw=0, alpha=0.7)

    ax.axis('off')

    plt.show()


def plot_flow_directions(flow_directions):
    """
    Plot the flow directions in the landscape
    :param flow_directions: Flow directions given from -1 to 7, i.e 'N/A' to 'N'
    :return: Plot the flow directions
    """

    flow_directions = flow_directions.astype(int)
    # Convert matrix to -1 to 7 format (in contrast to -1 to 128)
    neg_indices = np.where(flow_directions < 0)
    flow_directions = np.log2(flow_directions)
    flow_directions[neg_indices] = -1

    plt.figure(figsize=(15, 15))
    #(r, c) = np.shape(flow_directions)
    N = 9
    plt.matshow(flow_directions, cmap=discrete_cmap(N, "RdBu_r"), fignum=1)
    plt.grid(False)
    cb = plt.colorbar(fraction=0.046, pad=0.04, ticks=[-1, 0, 1, 2, 3, 4, 5, 6, 7])
    cb.set_ticklabels(['N/A', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW', 'N'])
    cb.ax.tick_params(labelsize=40)
    plt.clim(-1.5, 7.5)
    plt.axis('off')
    plt.savefig('flowDir.png', format='png', dpi=300, bbox_inches='tight')
    #x_ticks = np.arange(0, c, 500)
    #y_ticks = np.arange(3998, 0, -500)
    #plt.xticks(x_ticks, [])
    #plt.yticks(y_ticks, [])

    plt.show()


def plot_local_minima(minima):
    """
    Plot all local minima in the landscape, i.e. nodes without a downslope
    :param minima: A 2D-array with ones where there is a minima, zero if not
    :return: Plot all nodes without a downslope direction
    """

    (r, c) = np.shape(minima)
    N = 2
    plt.matshow(minima, cmap=discrete_cmap(N, "Blues"))
    plt.grid(False)

    x_ticks = np.arange(0, c, 500)
    y_ticks = np.arange(3998, 0, -500)
    plt.xticks(x_ticks, [])
    plt.yticks(y_ticks, [])

    plt.show()


def plot_up_and_downslope_watersheds(watersheds, upslope_indices, downslope_indices, node_levels, w_nr, landscape, ds=4):
    """
    Plot the upslope and downslope watersheds of a selected watershed
    :param watersheds: All watersheds, thresholded or not
    :param upslope_indices: Watershed indices of upslope watersheds
    :param downslope_indices: Watershed indices of downslope watersheds
    :param node_levels: Amount of steps each watershed is from the selected watershed.
    :param w_nr: The index of a selected watershed
    :param landscape: The landscape object
    :param ds: Downsampling factor
    :return: Plot all up- and downslope watersheds for a selected watershed
    """

    nr_of_watersheds = len(watersheds)
    boundary_pairs = util.get_boundary_pairs_in_watersheds(watersheds, landscape.nx, landscape.ny)

    color_hex = ['#d9d9d9', '#bdbdbd', '#969696', '#737373', '#525252', '#252525']
    colors = iter(color_hex * (nr_of_watersheds/3))
    upslope_colors = iter(cm.Blues_r(np.linspace(0, 1, len(node_levels))))
    not_upslope = np.setdiff1d(np.arange(0, len(watersheds), 1), upslope_indices)

    fig = plt.figure()
    ax = fig.add_subplot(111, aspect=1)

    # Plot all non upslope watersheds in different colors
    for el in not_upslope:
        row_col = util.map_1d_to_2d(watersheds[el], landscape.nx)
        plt.scatter(landscape.x_min + row_col[1][0::ds] * landscape.step_size,
                    landscape.y_max - row_col[0][0::ds] * landscape.step_size,
                    color=next(colors), s=2, lw=0, alpha=1)

    # Plot the upslope watersheds in different shades of blue to show the sequence of flow
    for i in range(len(node_levels)):
        if i == 0:
            continue
        else:
            level_color = next(upslope_colors)
            for ws in np.asarray(node_levels[i]):
                row_col = util.map_1d_to_2d(watersheds[ws], landscape.nx)
                plt.scatter(landscape.x_min + row_col[1][0::ds] * landscape.step_size,
                            landscape.y_max - row_col[0][0::ds] * landscape.step_size,
                            color=level_color, s=2, lw=0, alpha=1)

    downslope_colors = iter(cm.RdPu_r(np.linspace(0, 1, len(downslope_indices))))

    # Plot the upslope watersheds in different hues of blue to show the sequence of flow
    for el in downslope_indices:
            if el == w_nr:
                row_col = util.map_1d_to_2d(watersheds[el], landscape.nx)
                plt.scatter(landscape.x_min + row_col[1][0::ds] * landscape.step_size,
                            landscape.y_max - row_col[0][0::ds] * landscape.step_size,
                            color='gold', s=2, lw=0, alpha=1)
            else:
                row_col = util.map_1d_to_2d(watersheds[el], landscape.nx)
                plt.scatter(landscape.x_min + row_col[1][0::ds] * landscape.step_size,
                            landscape.y_max - row_col[0][0::ds] * landscape.step_size,
                            color=next(downslope_colors), s=2, lw=0, alpha=1)

    # Plot the boundary nodes
    for i in range(nr_of_watersheds):
        b_p = boundary_pairs[i]
        u = np.unique(np.concatenate((b_p[0], b_p[1])))
        row_col = util.map_1d_to_2d(u, landscape.nx)
        plt.scatter(landscape.x_min + row_col[1][0::ds] * landscape.step_size,
                    landscape.y_max - row_col[0][0::ds] * landscape.step_size,
                    color='black', s=2, lw=0, alpha=1)

    ax.axis('off')

    plt.show()


def plot_trap_sizes(size_of_traps, size_order):
    """
    Plot the size of the traps as a graph
    :param size_of_traps: The size of the traps for each index
    :param size_order: The sort order from smallest to largest
    :return: Plot the trap sizes for all watersheds
    """

    plt.plot(np.arange(0, len(size_order), 1), size_of_traps[size_order])
    plt.xlabel('Watershed nr')
    plt.ylabel('Area of traps')
    plt.title('Area of trap regions')

    font = {'family': 'normal',
            'size': 36}

    plt.rc('font', **font)

    plt.show()


def plot_trap_sizes_histogram(size_of_traps):
    """
    Plots the trap sizes as a histogram
    :param size_of_traps: The size of the traps for all watersheds
    :return: Plots the trap sizes as a histogram
    """

    plt.hist(size_of_traps, bins=30, color=(1, 0.3984375, 0))
    plt.xlabel('10-logarithm of the trap sizes')
    plt.ylabel('Number of watersheds')
    plt.title('Histogram of trap sizes')

    font = {'family': 'normal',
            'size': 36}

    plt.rc('font', **font)

    plt.show()


def plot_traps_and_rivers(watersheds, traps, rivers, landscape, ds=1):
    """
    Possibility to plot watersheds, traps, rivers and boundary nodes
    :param watersheds: Watersheds in the landscape
    :param traps: The traps in the landscape
    :param rivers: All rivers
    :param landscape: The landscape object
    :param ds: Downsampling factor
    :return: Plot all traps and rivers, and potentially more
    """

    nr_of_watersheds = len(watersheds)

    color_hex = ['#d9d9d9', '#bdbdbd', '#969696', '#737373', '#525252', '#252525']
    color_small = iter(color_hex * (len(watersheds) / 3))

    fig = plt.figure()
    ax = fig.add_subplot(111, aspect=1)

    # # Plot the watersheds
    # for i in range(nr_of_watersheds):
    #     ws = watersheds[i]
    #     row_col = util.map_1d_to_2d(ws, landscape.nx)
    #     plt.scatter(landscape.x_min + row_col[1][0::ds] * landscape.step_size,
    #                 landscape.y_max - row_col[0][0::ds] * landscape.step_size,
    #                 color=next(color_small), s=2, lw=0, alpha=1)

    # Plot all traps
    for i in range(len(traps)):
        trap = traps[i]
        row_col = util.map_1d_to_2d(trap, landscape.nx)
        plt.scatter(landscape.x_min + row_col[1][0::1] * landscape.step_size,
                    landscape.y_max - row_col[0][0::1] * landscape.step_size,
                    color='red', s=2, lw=0, alpha=1)
                    #color='#034e7b', s=2, lw=0, alpha=1)

    # # Plot the boundary nodes
    # boundary_pairs = util.get_boundary_pairs_in_watersheds(watersheds, landscape.nx, landscape.ny)
    # for i in range(nr_of_watersheds):
    #     b_p = boundary_pairs[i]
    #     u = np.unique(np.concatenate((b_p[0], b_p[1])))
    #     row_col = util.map_1d_to_2d(u, landscape.nx)
    #     plt.scatter(landscape.x_min + row_col[1][0::1] * landscape.step_size,
    #                 landscape.y_max - row_col[0][0::1] * landscape.step_size,
    #                 color='black', s=2, lw=0, alpha=1)

    # Plot rivers
    row_col = util.map_1d_to_2d(rivers, landscape.nx)
    plt.scatter(landscape.x_min + row_col[1][0::1] * landscape.step_size,
                landscape.y_max - row_col[0][0::1] * landscape.step_size,
                color='red', s=2, lw=0, alpha=1)
                #color='#3690c0', s=2, lw=0, alpha=1)

    ax.axis('off')

    plt.show()


def plot_watersheds_and_minima_2d(watersheds, minima, landscape, ds):
    # This method is not in use, but can plot the watersheds and the nodes without downslope neighbors

    ws_above_n_nodes = 0
    large_watersheds = [ws for ws in watersheds if len(ws) > ws_above_n_nodes]
    large_watersheds.sort(key=len)
    nr_of_large_watersheds = len(large_watersheds)

    print 'nr of large watersheds: ', nr_of_large_watersheds

    color_small = ['red', 'green', 'blue', 'yellow']
    color_small = iter(color_small * (len(watersheds)/3))

    nr_of_largest = 5

    for i in range(0, nr_of_large_watersheds - nr_of_largest):
        row_col = util.map_1d_to_2d(large_watersheds[i], landscape.nx)
        plt.scatter(landscape.x_min + row_col[1][0::ds] * landscape.step_size,
                    landscape.y_max - row_col[0][0::ds] * landscape.step_size,
                    color=next(color_small), s=20, lw=0, alpha=1)

    color_large = ['gold', 'darkgreen', 'darkorange', 'darkorchid', 'dodgerblue']
    color_large = iter(color_large * (len(watersheds)/3))

    for i in range(nr_of_large_watersheds - nr_of_largest, nr_of_large_watersheds):
        row_col = util.map_1d_to_2d(large_watersheds[i], landscape.nx)
        plt.scatter(landscape.x_min + row_col[1][0::ds] * landscape.step_size,
                    landscape.y_max - row_col[0][0::ds] * landscape.step_size,
                    color=next(color_large), s=20, lw=0, alpha=1)

    for i in range(len(minima)):
        row_col = util.map_1d_to_2d(minima[i], landscape.nx)
        plt.scatter(landscape.x_min + row_col[1][0::ds] * landscape.step_size,
                    landscape.y_max - row_col[0][0::ds] * landscape.step_size,
                    color='black', s=20, lw=0, alpha=1)

    plt.xlabel('x')
    plt.ylabel('y')

    plt.show()


def plot_traps(watersheds, traps, threshold, landscape, ds):
    # This method is not in use, but can plot all traps

    nr_of_watersheds = len(watersheds)

    color_hex = ['#d9d9d9', '#bdbdbd', '#969696', '#737373', '#525252', '#252525']
    color_small = iter(color_hex * (len(watersheds) / 3))

    # Plot the watersheds
    for i in range(nr_of_watersheds):
        ws = watersheds[i]
        row_col = util.map_1d_to_2d(ws, landscape.nx)
        plt.scatter(landscape.x_min + row_col[1][0::ds] * landscape.step_size,
                    landscape.y_max - row_col[0][0::ds] * landscape.step_size,
                    color=color_hex[0], s=2, lw=0, alpha=1)
                    # color=next(color_small), s=2, lw=0, alpha=1)

    # Plot all traps
    for i in range(len(traps)):
        trap = traps[i]
        row_col = util.map_1d_to_2d(trap, landscape.nx)
        plt.scatter(landscape.x_min + row_col[1][0::ds] * landscape.step_size,
                    landscape.y_max - row_col[0][0::ds] * landscape.step_size,
                    color='gold', s=20, lw=0, alpha=1)

    ## Plot the boundary nodes
    #boundary_pairs = util.get_boundary_pairs_in_watersheds(watersheds, landscape.nx, landscape.ny, False)
    #
    #for i in range(nr_of_watersheds):
    #    b_p = boundary_pairs[i]
    #    u = np.unique(np.concatenate((b_p[0], b_p[1])))
    #    row_col = util.map_1d_to_2d(u, landscape.nx)
    #    plt.scatter(landscape.x_min + row_col[1][0::ds] * landscape.step_size,
    #                landscape.y_max - row_col[0][0::ds] * landscape.step_size,
    #                color='black', s=2, lw=0, alpha=1)

    plt.title('The traps of the watersheds with over %s cells in the landscape' % str(threshold))
    font = {'family': 'normal',
            'size': 18}
    plt.rc('font', **font)
    plt.xlabel('x')
    plt.ylabel('y')

    plt.show()


def plot_running_times(grids):

    colors = ['#1b9e77', '#d95f02', '#7570b3', '#e7298a']
    labels = ['500x500', '1000x1000', '2000x2000', '4000x4000']
    steps = np.array([1, 2, 3, 4, 5, 6, 7])

    ax = plt.subplot(211)
    # plt.plot(steps, grids[2], label=labels[2], color=colors[2], linewidth=4)
    ax.bar(steps, grids[2], width=0.5, color=colors[2], align='center',
           label=labels[2])
    plt.xlabel('Step nr.')
    plt.ylabel('Time (seconds)')
    plt.legend(loc=2)

    ax = plt.subplot(212)
    ax.bar(steps - 0.3, np.divide(grids[0], grids[0]), width=0.2, color=colors[0], align='center', label=labels[0])
    ax.bar(steps - 0.10, np.divide(grids[1], grids[0]), width=0.2, color=colors[1], align='center', label=labels[1])
    ax.bar(steps + 0.10, np.divide(grids[2], grids[0]), width=0.2, color=colors[2], align='center', label=labels[2])
    ax.bar(steps + 0.3, np.divide(grids[3], grids[0]), width=0.2, color=colors[3], align='center', label=labels[3])
    plt.yscale('log', basey=2)
    ax.set_ylim(ymin=-0.5)

    plt.xlabel('Step nr.')
    plt.ylabel('Time relative to 500x500')
    plt.legend(loc=2)

    font = {'family': 'normal',
            'size': 20}
    plt.rc('font', **font)

    plt.show()


def plot_hillshade(heights):

    cmap = plt.cm.gist_earth
    ve = 0.5
    ls = LightSource(azdeg=300, altdeg=20)
    # ls = LightSource(azdeg=315, altdeg=45)
    rgb = ls.shade(heights, cmap=cmap, vert_exag=ve, blend_mode='overlay')
    plt.imshow(rgb)

    plt.axis('off')

    plt.show()


def plot_accumulated_flow_above_threshold(acc_flow):

    threshold = 100000
    river_nodes = np.where(acc_flow > threshold)

    plt.scatter(river_nodes[1] * 10, river_nodes[0] * 10, color='black', s=2, lw=0, alpha=1)
    plt.gca().invert_yaxis()
    plt.show()


def plot_accumulated_flow(acc_flow):

    rows, cols = np.shape(acc_flow)
    # threshold = rows * cols * 0.01
    threshold = 50000
    chosen_nodes = np.where(acc_flow > threshold)
    remove_nodes = np.where(acc_flow <= threshold)
    acc_flow[chosen_nodes] = threshold
    acc_flow[remove_nodes] = 0
    fig = plt.figure()
    ax = fig.gca()
    fig.tight_layout()
    ax.axis('off')
    plt.imshow(acc_flow, cmap='Blues')
    #plt.colorbar()

    plt.savefig("/home/anderovo/Dropbox/masters-thesis/thesis/flowAccumulation.png", dpi=300, bbox_inches='tight',
                pad_inches=0)

    plt.show()

    # plt.scatter(river_nodes[1] * 10, river_nodes[0] * 10, color='black', s=2, lw=0, alpha=1)
    # plt.gca().invert_yaxis()
    # plt.show()


def plot_difference_d4_and_d8(watersheds, outlet, landscape, ds):
    """
    Plot all watersheds and their boundary nodes
    :param watersheds: List of arrays where each array is a watershed
    :param outlet: The outlet cell of the watershed
    :param ds: Downsampling factor if only every ds node shall be plotted
    :return: Plots the watersheds
    """

    nr_of_watersheds = len(watersheds)

    color_hex = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33', '#a65628', '#f781bf']
    if len(watersheds) < 3:
        color_small = iter(color_hex)
    else:
        color_small = iter(color_hex * (len(watersheds)/3))
    fig = plt.figure()
    ax = fig.gca()  # fig.add_subplot(111, aspect=1)

    # Plot first watershed
    row_col = util.map_1d_to_2d(watersheds[0], landscape.nx)
    plt.scatter(landscape.x_min + row_col[1][0::ds],
                landscape.y_max - row_col[0][0::ds],
                color=next(color_small), s=10, lw=0, alpha=1)

    # Plot second watershed
    row_col = util.map_1d_to_2d(watersheds[1], landscape.nx)
    plt.scatter(landscape.x_min + row_col[1][0::ds],
                landscape.y_max - row_col[0][0::ds],
                color=next(color_small), s=10, lw=0, alpha=1)

    # Plot intersection of 1. and 2. watershed
    common_elements = np.intersect1d(watersheds[0], watersheds[1])
    row_col = util.map_1d_to_2d(common_elements, landscape.nx)
    plt.scatter(landscape.x_min + row_col[1][0::ds],
                landscape.y_max - row_col[0][0::ds],
                color=next(color_small), s=10, lw=0, alpha=1)

    # Outlet is of the form (row, col) <-> (y, x)
    plt.scatter(landscape.x_min + outlet[1], landscape.y_max - outlet[0],
                color='k', s=20, lw=0, alpha=1)

    fig.tight_layout()
    ax.axis('off')

    plt.savefig("/home/anderovo/Dropbox/masters-thesis/thesis/differenceD8andD4.png", dpi=300, bbox_inches='tight',
                pad_inches=0)

    plt.show()


def plot_watershed_of_node(ws, rows, cols):

    ws_2d = util.map_1d_to_2d(ws, cols)
    ws_mat = np.zeros((rows, cols))
    ws_mat[ws_2d] = 1
    discrete_matshow(ws_mat)
    plt.axis('off')

    plt.show()


def plot_flow_acc_no_landscape(flow_acc):

    discrete_matshow(flow_acc[1:-1, 1:-1])
    plt.axis('off')
    plt.show()


def discrete_matshow(data):
    # get discrete colormap
    cmap = plt.get_cmap('Blues', np.max(data) - np.min(data) + 1)
    # set limits .5 outside true range

    mat = plt.matshow(data, cmap=cmap, vmin=np.min(data) - .5, vmax=np.max(data) + .5)

    # tell the colorbar to tick at integers
    ticks = np.arange(np.min(data), np.max(data) + 1)

    # cax = plt.colorbar(mat, ticks=ticks)
    # cax.ax.tick_params(labelsize=22)

    # cax.set_ticklabels(['Not in watershed', 'In watershed'], update_ticks=True)


def plot_heights(heights):

    plt.figure(figsize=(15, 15))
    plt.matshow(heights[1:-1, 1:-1], cmap='terrain', fignum=1)
    cb = plt.colorbar(fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=40)
    plt.axis('off')

    plt.savefig('heightsRegular.png', format='png', dpi=300, bbox_inches='tight')
    plt.show()