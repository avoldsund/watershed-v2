import numpy as np
import math
import scipy.ndimage.filters as filters
import scipy.ndimage.morphology as morphology
from scipy.sparse import csr_matrix, identity, csgraph
import itertools
import networkx
import time
import matplotlib.pyplot as plt

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

    nbr_heights = np.zeros((rows, cols, 8))

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


def get_flow_directions(heights, step_size, int_rows, int_cols):
    """
    Returns the steepest directions for all nodes and setting -1 for local minima and flat areas
    :param heights: The heights for all nodes in the 2D-grid
    :param step_size: Step size in the grid
    :param int_rows: Nr of int_rows for interior
    :param int_cols: Nr of columns for interior
    :return: flow_directions: The neighbor index indicating steepest slope
    """

    nbr_heights = get_neighbor_heights(heights, int_rows, int_cols)
    derivatives = get_derivatives(heights, nbr_heights, step_size)

    flow_directions = np.ones((int_rows, int_cols), dtype=int) * -1
    pos_derivatives = np.max(derivatives, axis=2) > 0
    flow_directions[pos_derivatives] = np.argmax(derivatives, axis=2)[pos_derivatives]
    flow_directions[pos_derivatives] = 2 ** flow_directions[pos_derivatives]

    return flow_directions


def remove_out_of_boundary_flow(flow_directions):
    """
    Replaces flow out of the boundary by flagging node as minimum/flat area (-1)
    :param flow_directions: The directions of flow for every node
    :return: Void function that alters flow_directions
    """

    # If flow direction is out of the boundary, set flow direction to -1

    change_top = np.concatenate((np.where(flow_directions[0, :] == 1)[0],
                                np.where(flow_directions[0, :] == 64)[0],
                                np.where(flow_directions[0, :] == 128)[0]))
    change_right = np.concatenate((np.where(flow_directions[:, -1] == 1)[0],
                                  np.where(flow_directions[:, -1] == 2)[0],
                                  np.where(flow_directions[:, -1] == 4)[0]))
    change_bottom = np.concatenate((np.where(flow_directions[-1, :] == 4)[0],
                                   np.where(flow_directions[-1, :] == 8)[0],
                                   np.where(flow_directions[-1, :] == 16)[0]))
    change_left = np.concatenate((np.where(flow_directions[:, 0] == 16)[0],
                                 np.where(flow_directions[:, 0] == 32)[0],
                                 np.where(flow_directions[:, 0] == 64)[0]))

    flow_directions[0, change_top] = -1
    flow_directions[change_right, -1] = -1
    flow_directions[-1, change_bottom] = -1
    flow_directions[change_left, 0] = -1
    # This function does not return something, just change the input flow_directions


def get_flow_direction_indices(heights, step_size, int_rows, int_cols):

    flow_directions = get_flow_directions(heights, step_size, int_rows, int_cols)
    remove_out_of_boundary_flow(flow_directions)

    values = [1, 2, 4, 8, 16, 32, 64, 128]
    translations = [-int_cols + 1, 1, int_cols + 1, int_cols, int_cols - 1, -1, -int_cols - 1, -int_cols]

    for ix in range(len(values)):
        coords = np.where(flow_directions == values[ix])
        if len(coords[0]) > 0:
            from_ix = coords[0] * int_cols + coords[1]
            to_ix = from_ix + translations[ix]
            flow_directions[coords] = to_ix

    return flow_directions


def map_2d_to_1d(coords, cols):
    """
    Map from a 2D-grid to a 1D-array
    :param coords: Coords in a tuple (coords_rows, coords_cols)
    :param cols: Nr of columns
    :return indices: Indices in the 1D-array
    """
    indices = coords[1] + coords[0] * cols

    return indices


def map_1d_to_2d(indices, cols):
    """
    Map from a 1D-array to a 2D-grid
    :param indices: Indices in the 1D-array
    :param cols: Number of columns in the 2D-grid
    :return rows, cols: Tuple containing the row and col indices
    """

    rows = np.divide(indices, cols)
    cols = indices % cols

    return rows, cols


def get_node_endpoints(downslope_neighbors):

    rows, cols = np.shape(downslope_neighbors)
    terminal_nodes = np.empty((rows, cols), dtype=object)

    # Get all minima as starting points for stepwise algorithm
    minima = np.where(downslope_neighbors == -1)
    terminal_nodes[minima] = map_2d_to_1d(minima, cols)
    num_inserted = len(minima[0])

    while num_inserted > 0:
        num_inserted, terminal_nodes = update_terminal_nodes(terminal_nodes, downslope_neighbors, cols)

    return terminal_nodes


def update_terminal_nodes(terminal_nodes, downslope_neighbors, cols):

    # IMPLEMENT HAS TERMINAL NODES
    no_terminal_nodes = np.where(terminal_nodes < 0)  # Indices of all nodes without terminal nodes yet

    if len(no_terminal_nodes[0]) == 0:
        return 0, terminal_nodes
    next_nodes = downslope_neighbors[no_terminal_nodes]  # Check if these nodes are minima, or if they have endpoints

    # The next point is a minimum
    are_minima = np.where(downslope_neighbors[map_1d_to_2d(next_nodes, cols)] == -1)[0]
    terminal_nodes[(no_terminal_nodes[0][are_minima], no_terminal_nodes[1][are_minima])] = next_nodes[are_minima]

    # The next point might have an end node
    undecided = np.setdiff1d(range(len(next_nodes)), are_minima)  # Might be nodes already assigned end points
    are_end_points = undecided[np.where(terminal_nodes[map_1d_to_2d(next_nodes[undecided], cols)] >= 0)[0]]
    terminal_nodes[(no_terminal_nodes[0][are_end_points], no_terminal_nodes[1][are_end_points])] = \
        terminal_nodes[map_1d_to_2d(next_nodes[are_end_points], cols)]

    num_inserted = len(are_minima) + len(are_end_points)

    return num_inserted, terminal_nodes


def get_local_watersheds(node_endpoints):

    endpoints = node_endpoints.flatten()
    unique, counts = np.unique(endpoints, return_counts=True)
    sorted_indices = np.argsort(endpoints)
    indices_to_endpoints = np.split(sorted_indices, np.cumsum(counts))[0:-1]
    local_watersheds = dict(zip(unique, indices_to_endpoints))

    return local_watersheds


def map_1d_interior_to_2d_exterior(node_index, number_of_cols):

    r = node_index/number_of_cols + 1
    c = node_index % number_of_cols + 1
    row_col = zip(r, c)

    return row_col


def map_2d_exterior_to_1d_interior(coords, cols):

    indices = []
    for c in coords:
        ix = c[1] - 1 + (c[0] - 1) * cols
        indices.append(ix)

    return indices


def combine_minima(local_minima, rows, cols):
    """
    Return the combined minima in the landscape as a list of arrays
    :param local_minima: 1D-array with indices of all minima
    :param rows: Nr of rows
    :param cols: Nr of columns
    :return combined_minima: List of arrays containing the combined minima
    """

    # local_minima = (rows, cols)
    local_minima_2d = map_1d_to_2d(local_minima, cols)
    one = (local_minima_2d[0] - 1, local_minima_2d[1] + 1)
    two = (local_minima_2d[0], local_minima_2d[1] + 1)
    four = local_minima_2d[0] + 1, local_minima_2d[1] + 1
    eight = local_minima_2d[0] + 1, local_minima_2d[1]
    sixteen = local_minima_2d[0] + 1, local_minima_2d[1] - 1
    thirtytwo = local_minima_2d[0], local_minima_2d[1] - 1
    sixtyfour = local_minima_2d[0] - 1, local_minima_2d[1] - 1
    onetwentyeight = local_minima_2d[0] - 1, local_minima_2d[1]

    nbrs_to_minima = np.hstack((one, two, four, eight, sixteen, thirtytwo, sixtyfour, onetwentyeight))
    valid_nbrs_to_minima = np.where(np.logical_and(np.logical_and(nbrs_to_minima[0] >= 0, nbrs_to_minima[0] < rows),
                                                   np.logical_and(nbrs_to_minima[1] >= 0, nbrs_to_minima[1] < cols)))[0]

    from_min = np.concatenate([local_minima for i in range(8)])
    to = map_2d_to_1d(nbrs_to_minima[:, valid_nbrs_to_minima], cols)
    from_min = from_min[valid_nbrs_to_minima]

    # Remove all connections not between minima
    valid_pairs = np.where(np.in1d(to, local_minima))[0]
    to = to[valid_pairs]
    from_min = from_min[valid_pairs]
    data = np.ones(len(to))

    # Make connectivity matrix between pairs of minima
    conn = csr_matrix((data, (from_min, to)), shape=(rows*cols, rows*cols), dtype=int)
    n_components, labels = csgraph.connected_components(conn, directed=False)
    unique, counts = np.unique(labels, return_counts=True)

    sorted_indices = np.argsort(labels)

    nodes_in_comb_min = np.split(sorted_indices, np.cumsum(counts))[0:-1]

    combined_minima = [ws for ws in nodes_in_comb_min if len(ws) > 1]

    already_located_minima = np.concatenate(combined_minima)
    remaining_minima = np.setdiff1d(local_minima, already_located_minima)
    [combined_minima.append(np.array([remaining_minima[i]])) for i in range(len(remaining_minima))]

    return combined_minima


def combine_watersheds(local_watersheds, combined_minima):

    watersheds = []

    for i in range(len(combined_minima)):
        if len(combined_minima[i]) == 1:
            watersheds.append(local_watersheds[list(combined_minima[i])[0]])
        else:
            ws = np.concatenate(list((local_watersheds[i] for i in combined_minima[i])))
            watersheds.append(ws)

    return watersheds


def create_nbr_connectivity_matrix(flow_directions, nx, ny):
    # Note: This is a version without 1 on the diagonal
    """
    Create a connectivity matrix between all nodes using the flow
    :param flow_directions: 2D-grid showing the flow
    :param nx: Number of cols
    :param ny: Number of rows
    :return A: Returns a sparse adjacency matrix
    """

    # Start by removing the flow out of the boundary
    remove_out_of_boundary_flow(flow_directions)

    values = [1, 2, 4, 8, 16, 32, 64, 128]
    translations = [-nx + 1, 1, nx + 1, nx, nx - 1, -1, -nx - 1, -nx]
    from_indices = []
    to_indices = []
    total_nodes = nx * ny

    for ix in range(len(values)):
        coords = np.where(flow_directions == values[ix])
        if len(coords[0]) > 0:
            from_ix = coords[0] * nx + coords[1]
            to_ix = from_ix + translations[ix]
            from_indices.append(from_ix)
            to_indices.append(to_ix)

    rows = np.concatenate(from_indices)
    cols = np.concatenate(to_indices)
    data = np.ones(len(rows))

    adj_mat = csr_matrix((data, (rows, cols)), shape=(total_nodes, total_nodes))

    return adj_mat


def get_minima(adj_mat):
    """
    Returns the indices of the local minima
    :param adj_mat: Matrix showing where flow occurs
    :return minima: Indices of the minima
    """

    minima = np.where(np.diff(adj_mat.indptr) == 0)[0]

    return minima


def get_downslope_rivers(adj_mat):
    """
    Returns a matrix showing all downslope nodes for each node
    :param adj_mat: Matrix showing where flow occurs
    :return downslope_rivers: Sparse matrix with ones at downslope nodes
    """

    rows, cols = np.shape(adj_mat)
    changes = True
    id_matrix = identity(rows, dtype=int, format='csr')
    downslope_rivers = adj_mat + id_matrix  # Set diagonal to 1

    AA = adj_mat

    while changes:
        changes = False
        AA = csr_matrix.dot(AA, adj_mat)

        if AA.nnz > 0:
            changes = True
            downslope_rivers = AA + downslope_rivers

    return downslope_rivers


#def get_local_watersheds(downslope_rivers, local_minima):
#
#    ws = csr_matrix.dot(downslope_rivers, csr_matrix.transpose(downslope_rivers))
#
#    local_watersheds = {}
#
#    for m in local_minima:
#        local_watersheds[m] = csr_matrix.getrow(ws, m).indices
#
#    return local_watersheds


def get_row_and_col_from_indices(node_indices, number_of_cols):
    """
    Return (r, c) for all indices in node_indices.
    :param node_indices: Indices in the 1d-grid.
    :param number_of_cols: Number of columns in the 2d-grid.
    :return row_col: (r, c) for every index
    """

    row_col = np.empty((len(node_indices), 2), dtype=int)
    row_col[:, 0] = np.divide(node_indices, number_of_cols)
    row_col[:, 1] = node_indices % number_of_cols

    return row_col


def get_watersheds_with_combined_minima(combined_minima, local_watersheds):

    watersheds = []
    for c in combined_minima:
        watersheds.append(np.concatenate([local_watersheds[el] for el in c]))

    return watersheds


