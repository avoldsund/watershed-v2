import numpy as np
import math
import scipy.ndimage.filters as filters
import scipy.ndimage.morphology as morphology


def detect_local_minima(arr):
    # Found at: http://stackoverflow.com/questions/3986345/how-to-find-the-local-minima-of-a-smooth-
    # multidimensional-array-in-numpy-efficie
    # http://stackoverflow.com/questions/3684484/peak-detection-in-a-2d-array/3689710#3689710
    """
    Takes an array and detects the troughs using the local maximum filter.
    Returns a boolean mask of the troughs (i.e. 1 when
    the pixel's value is the neighborhood maximum, 0 otherwise)
    """
    # define a connected neighborhood
    # http://www.scipy.org/doc/api_docs/SciPy.ndimage.morphology.html#generate_binary_structure
    neighborhood = morphology.generate_binary_structure(len(arr.shape), 2)
    # apply the local minimum filter; all locations of minimum value
    # in their neighborhood are set to 1
    # http://www.scipy.org/doc/api_docs/SciPy.ndimage.filters.html#minimum_filter
    local_min = (filters.minimum_filter(arr, footprint=neighborhood) == arr)
    # local_min is a mask that contains the peaks we are
    # looking for, but also the background.
    # In order to isolate the peaks we must remove the background from the mask.
    #
    # we create the mask of the background
    background = (arr == 0)
    #
    # a little technicality: we must erode the background in order to
    # successfully subtract it from local_min, otherwise a line will
    # appear along the background border (artifact of the local minimum filter)
    # http://www.scipy.org/doc/api_docs/SciPy.ndimage.morphology.html#binary_erosion
    eroded_background = morphology.binary_erosion(
        background, structure=neighborhood, border_value=1)
    #
    # we obtain the final mask, containing only peaks,
    # by removing the background from the local_min mask
    detected_minima = local_min - eroded_background

    return np.where(detected_minima)


def fill_single_cell_depressions(heights, rows, cols):
    """
    Preprocessing to reduce local minima in the terrain.
    :param heights: Heights of the landscape.
    :param rows: Nr of interior nodes in x-dir
    :param cols: Nr of interior nodes in y-dir
    :return heights: Updated heights with single cell depressions removed
    """
    nbr_heights = get_neighbor_heights(heights, rows, cols)
    interior_heights = heights[1:-1, 1:-1]
    delta = np.repeat(interior_heights[:, :, np.newaxis], 8, axis=2) - nbr_heights
    local_minima = np.where(np.max(delta, axis=2) < 0)  # The single cell depressions to be raised

    raised_elevations = np.min(nbr_heights[local_minima], axis=1)  # The elevations they are raised to
    interior_heights[local_minima] = raised_elevations

    return heights


def get_neighbor_heights(heights, rows, cols):
    """
    Returns the heights of the neighbors for all nodes
    :param heights: Heights of the landscape
    :param rows: Number of rows in the 2D-grid
    :param cols: Number of columns in the 2D-grid
    :return nbr_heights: nx x ny x 8 grid
    """

    nbr_heights = np.zeros((rows, cols, 8), dtype=int)

    nbr_heights[:, :, 0] = heights[0:-2, 2:]    # 1
    nbr_heights[:, :, 1] = heights[1:-1, 2:]    # 2
    nbr_heights[:, :, 2] = heights[2:, 2:]      # 4
    nbr_heights[:, :, 3] = heights[2:, 1:-1]    # 8
    nbr_heights[:, :, 4] = heights[2:, 0:-2]    # 16
    nbr_heights[:, :, 5] = heights[1:-1, 0:-2]  # 32
    nbr_heights[:, :, 6] = heights[0:-2, 0:-2]  # 64
    nbr_heights[:, :, 7] = heights[0:-2, 1:-1]  # 128

    return nbr_heights


def get_derivatives(heights, nbr_heights, step_size):
    """
    Returns the derivatives between all nodes and their neighbors
    :param heights:
    :param nbr_heights:
    :param step_size: Step size in the grid
    :return derivatives: The slope to the neighbors for all nodes, nx x ny x 8
    """

    interior_heights = heights[1:-1, 1:-1]
    delta = np.repeat(interior_heights[:, :, np.newaxis], 8, axis=2) - nbr_heights

    diag = math.sqrt(step_size ** 2 + step_size ** 2)
    card = step_size
    distance = np.array([diag, card, diag, card, diag, card, diag, card])

    derivatives = np.divide(delta, distance)

    return derivatives


def get_flow_directions(heights, step_size, rows, cols):
    """
    Returns the steepest directions for all nodes
    :param heights: The heights for all nodes in the 2D-grid
    :param step_size: Step size in the grid
    :param rows: Nr of rows
    :param cols: Nr of columns
    :return max_indices/flow_directions: The neighbor index indicating steepest slope
    """

    nbr_heights = get_neighbor_heights(heights, rows, cols)
    derivatives = get_derivatives(heights, nbr_heights, step_size)
    max_indices = np.argmax(derivatives, axis=2)

    #flow_directions = 2 ** max_indices

    return max_indices


def get_positive_flow_directions(heights, step_size, rows, cols):
    """
    Returns the steepest directions for all nodes and setting -1 for nodes with no lower neighbors
    :param heights: The heights for all nodes in the 2D-grid
    :param step_size: Step size in the grid
    :param rows: Nr of rows for interior
    :param cols: Nr of columns for interior
    :return: max_indices/flow_directions: The neighbor index indicating steepest slope
    """

    nbr_heights = get_neighbor_heights(heights, rows, cols)
    derivatives = get_derivatives(heights, nbr_heights, step_size)

    max_indices = np.ones((rows, cols), dtype=int) * -1
    pos_derivatives = np.max(derivatives, axis=2) > 0
    max_indices[pos_derivatives] = np.argmax(derivatives, axis=2)[pos_derivatives]

    return max_indices


def get_downslope_neighbors(heights, step_size, rows, cols):

    nbr_heights = get_neighbor_heights(heights, rows, cols)
    derivatives = get_derivatives(heights, nbr_heights, step_size)
    


def get_node_endpoints(cols, rows, downslope_neighbors):
    """
    Returns the end node if one follows the down slope until reaching a local minimum, for every node
    :param cols: Number of nodes is x-direction
    :param rows: Number of nodes in y-direction
    :param downslope_neighbors: Indices of the downslope neighbor for each node. Equal to -1 if the node is a minimum.
    :return terminal_nodes: The indices of the end nodes
    """

    total_nodes = cols * rows
    terminal_nodes = np.empty(total_nodes, dtype=object)
    indices_in_terminal = np.zeros(total_nodes, dtype=bool)

    # The nodes itself are minimums
    indices_node_is_minimum = np.where(downslope_neighbors == -1)[0]
    terminal_nodes[indices_node_is_minimum] = indices_node_is_minimum
    indices_in_terminal[indices_node_is_minimum] = True

    num_of_end_nodes_inserted = len(indices_node_is_minimum)

    while num_of_end_nodes_inserted > 0:
        num_of_end_nodes_inserted, terminal_nodes = update_terminal_nodes(terminal_nodes, downslope_neighbors,
                                                                          indices_in_terminal)

    # DO WE REALLY NEED BOTH TERMINAL NODES AND INDICES_IN_TERMINAL????????????????????????????????????????????????
    # YES, BECAUSE WE CAN HAVE 0 (FALSE) AS AN ENDPOINT...?
    # POSSIBLE TO WORK AROUND THIS PROBLEM, I THINK
    return terminal_nodes


def update_terminal_nodes(terminal_nodes, downslope_neighbors, indices_in_terminal):
    """
    Returns an updated terminal_nodes and the number of new end points found.
    The method finds all indices which haven't gotten end nodes yet, and takes a step in the down slope direction. If
    the down slope is a local minimum, these indices will now get end points in terminal_nodes. These end points will be
    found from terminal_nodes.
    :param downslope_neighbors: Indices of the downslope neighbor for each node. Equal to -1 if the node is a minimum.
    :return terminal_nodes: The indices of the end nodes
    """

    indices_end_points_not_localized = np.where(indices_in_terminal == False)[0]
    indices_to_check_if_downslopes_are_minimum = downslope_neighbors[indices_end_points_not_localized]
    downslope_is_minimum = np.concatenate((np.where(terminal_nodes[indices_to_check_if_downslopes_are_minimum] == 0)[0],
                                           np.nonzero(terminal_nodes[indices_to_check_if_downslopes_are_minimum])[0]))

    indices = indices_end_points_not_localized[downslope_is_minimum]
    values = terminal_nodes[indices_to_check_if_downslopes_are_minimum[downslope_is_minimum]]

    terminal_nodes[indices] = values
    indices_in_terminal[indices] = True

    return len(values), terminal_nodes