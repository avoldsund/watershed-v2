import numpy as np
import math
from scipy.sparse import csr_matrix, identity, csgraph, identity
import itertools
import networkx
import time
import matplotlib.pyplot as plt
from operator import itemgetter


def get_watershed_nr_by_rc(watersheds, landscape, r, c):
    # Simple help method

    mapping = map_nodes_to_watersheds(watersheds, landscape.ny, landscape.nx)
    ix = map_2d_to_1d((r, c), landscape.nx)
    w_nr = mapping[ix]

    return w_nr


def fill_single_cell_depressions(heights, rows, cols):
    """
    Preprocessing to reduce local minima in the terrain.
    :param heights: Heights of the landscape.
    :param rows: Nr of interior nodes in x-dir
    :param cols: Nr of interior nodes in y-dir
    :return heights: Updated heights with single cell depressions removed
    """

    nbr_heights = get_neighbor_heights(heights, rows, cols)
    delta = np.repeat(heights[1:-1, 1:-1, np.newaxis], 8, axis=2) - nbr_heights[1:-1, 1:-1]
    local_minima = np.where(np.max(delta, axis=2) < 0)  # The single cell depressions to be raised

    # Coords of local minima is for the interior, need to map to exterior
    local_minima = (local_minima[0] + 1, local_minima[1] + 1)

    raised_elevations = np.min(nbr_heights[local_minima], axis=1)  # The elevations they are raised to
    heights[local_minima] = raised_elevations

    return heights


def get_neighbor_heights(heights, rows, cols):
    """
    Returns the heights of the neighbors for all nodes
    :param heights: Heights of the landscape
    :param rows: Number of rows in the 2D-grid
    :param cols: Number of columns in the 2D-grid
    :return nbr_heights: nx x ny x 8 grid
    """

    nbr_heights = np.empty((rows, cols, 8), dtype=object)

    nbr_heights[1:-1, 1:-1, 0] = heights[0:-2, 2:]    # 1
    nbr_heights[1:-1, 1:-1, 1] = heights[1:-1, 2:]    # 2
    nbr_heights[1:-1, 1:-1, 2] = heights[2:, 2:]      # 4
    nbr_heights[1:-1, 1:-1, 3] = heights[2:, 1:-1]    # 8
    nbr_heights[1:-1, 1:-1, 4] = heights[2:, 0:-2]    # 16
    nbr_heights[1:-1, 1:-1, 5] = heights[1:-1, 0:-2]  # 32
    nbr_heights[1:-1, 1:-1, 6] = heights[0:-2, 0:-2]  # 64
    nbr_heights[1:-1, 1:-1, 7] = heights[0:-2, 1:-1]  # 128

    return nbr_heights


def get_neighbor_indices(indices, cols):
    """
    Given a list of neighbors, returns their neighbor indices
    :param indices: Array of indices
    :param cols: Nr of columns
    :return nbrs: The neighbors
    """

    nbrs = np.zeros((len(indices), 8), dtype=int)

    nbrs[:, 0] = indices - cols + 1
    nbrs[:, 1] = indices + 1
    nbrs[:, 2] = indices + cols + 1
    nbrs[:, 3] = indices + cols
    nbrs[:, 4] = indices + cols - 1
    nbrs[:, 5] = indices - 1
    nbrs[:, 6] = indices - cols - 1
    nbrs[:, 7] = indices - cols

    return nbrs


def get_domain_boundary_indices(cols, rows):

    top = np.arange(0, cols, 1)
    bottom = np.arange(cols * rows - cols, cols * rows, 1)
    left = np.arange(cols, cols * rows - cols, cols)
    right = np.arange(2 * cols - 1, cols * rows - 1, cols)

    boundary_indices = np.concatenate((top, bottom, left, right))
    boundary_indices.sort()

    return boundary_indices


def get_domain_boundary_coords(cols, rows):
    """
    Returns the coordinates of the domain boundary as a tuple (rows, cols) consisting of
    numpy arrays.
    :param cols: Nr of grid points in x-dir
    :param rows: Nr of grid points in y-dir
    :return boundary_coordinates: Coordinates of domain boundary
    """

    top = (np.zeros(cols, dtype=int), np.arange(0, cols, 1))
    bottom = (np.ones(cols, dtype=int) * (rows - 1), np.arange(0, cols, 1))
    left = (np.arange(1, rows - 1, 1), np.zeros(rows - 2, dtype=int))
    right = (np.arange(1, rows - 1, 1), np.ones(rows - 2, dtype=int) * (cols - 1))

    boundary_coordinates = (np.concatenate([top[0], bottom[0], left[0], right[0]]),
                            np.concatenate([top[1], bottom[1], left[1], right[1]]))

    return boundary_coordinates


def get_derivatives(heights, nbr_heights, step_size):
    """
    Returns the derivatives as a r x c x 8 grid, where all boundary coordinates
    have None as derivatives,
    :param heights:
    :param nbr_heights:
    :param step_size: Step size in the grid
    :return derivatives: The slope to the neighbors for all nodes, nx x ny x 8
    """

    (r, c) = np.shape(heights)

    delta = np.repeat(heights[1:-1, 1:-1, np.newaxis], 8, axis=2) - nbr_heights[1:-1, 1:-1]
    diag = math.sqrt(step_size ** 2 + step_size ** 2)
    card = step_size
    distance = np.array([diag, card, diag, card, diag, card, diag, card])
    calc_derivatives = np.divide(delta, distance)

    derivatives = np.empty((r, c, 8), dtype=object)
    derivatives[1:-1, 1:-1] = calc_derivatives

    return derivatives


def get_flow_directions(heights, step_size, rows, cols):
    """
    Returns the steepest directions for all nodes and setting -1 for local minima and flat areas
    :param heights: The heights for all nodes in the 2D-grid
    :param step_size: Step size in the grid
    :param rows: Nr of rows
    :param cols: Nr of columns
    :return: flow_directions: The neighbor index indicating steepest slope
    """

    nbr_heights = get_neighbor_heights(heights, rows, cols)
    derivatives = get_derivatives(heights, nbr_heights, step_size)

    flow_directions = np.empty((rows, cols), dtype=object)
    flow_directions[1:-1, 1:-1] = -1

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

    # If flow direction is out of the interior, set flow direction to -1

    change_top = np.concatenate((np.where(flow_directions[1, :] == 1)[0],
                                np.where(flow_directions[1, :] == 64)[0],
                                np.where(flow_directions[1, :] == 128)[0]))
    change_right = np.concatenate((np.where(flow_directions[:, -2] == 1)[0],
                                  np.where(flow_directions[:, -2] == 2)[0],
                                  np.where(flow_directions[:, -2] == 4)[0]))
    change_bottom = np.concatenate((np.where(flow_directions[-2, :] == 4)[0],
                                   np.where(flow_directions[-2, :] == 8)[0],
                                   np.where(flow_directions[-2, :] == 16)[0]))
    change_left = np.concatenate((np.where(flow_directions[:, 1] == 16)[0],
                                 np.where(flow_directions[:, 1] == 32)[0],
                                 np.where(flow_directions[:, 1] == 64)[0]))

    flow_directions[1, change_top] = -1
    flow_directions[change_right, -2] = -1
    flow_directions[-2, change_bottom] = -1
    flow_directions[change_left, 1] = -1

    # This function does not return something, just change the input flow_directions


def get_flow_direction_indices(heights, step_size, rows, cols):
    """
    For every coordinate specifies the next index it flows to. If no flow, the index is set as -1.
    All boundary nodes have a flow set to None, as there's not enough information to determine.
    :param heights: Heights of grid
    :param step_size: Length between grid points
    :param rows: Nodes in y-direction
    :param cols: Nodes in x-direction
    :return flow_directions: Next node it flows to
    """

    flow_directions = get_flow_directions(heights, step_size, rows, cols)
    remove_out_of_boundary_flow(flow_directions)

    values = [1, 2, 4, 8, 16, 32, 64, 128]
    translations = [-cols + 1, 1, cols + 1, cols, cols - 1, -1, -cols - 1, -cols]

    for ix in range(len(values)):
        coords = np.where(flow_directions == values[ix])
        if len(coords[0]) > 0:
            from_ix = coords[0] * cols + coords[1]
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
    cols = (indices % cols)

    return rows, cols


def get_node_endpoints(downslope_neighbors):
    """
    Returns a 2d array specifying node endpoint for the coordinate
    :param downslope_neighbors: Downslope index for each coordinate
    :return terminal_nodes: The end point for each node
    """

    rows, cols = np.shape(downslope_neighbors)
    domain_boundary = get_domain_boundary_coords(cols, rows)

    terminal_nodes = np.empty((rows, cols), dtype=object)

    # Get all minima as starting points for stepwise algorithm
    minima = np.where(downslope_neighbors == -1)
    terminal_nodes[minima] = map_2d_to_1d(minima, cols)
    terminal_nodes[domain_boundary] = -2  # Finding terminal nodes is harder with None at the boundary

    num_inserted = len(minima[0])

    while num_inserted > 0:
        num_inserted, terminal_nodes = update_terminal_nodes(terminal_nodes, downslope_neighbors, cols)

    terminal_nodes[domain_boundary] = None

    return terminal_nodes


def update_terminal_nodes(terminal_nodes, downslope_neighbors, cols):
    """
    Help method for get_node_endpoints. Returns updated terminal nodes.
    :param terminal_nodes: Array specifying endpoint for each node
    :param downslope_neighbors: Downslope index for each coordinate
    :param cols: Nr of nodes in x-direction
    :return num_inserted, terminal_nodes: Nr of new coords with detected endpoint. 2d-array indicating endpoints
    """

    no_terminal_nodes = np.where(terminal_nodes < -2)  # Indices of all nodes without terminal nodes yet

    if len(no_terminal_nodes[0]) == 0:
        return 0, terminal_nodes
    next_nodes = downslope_neighbors[no_terminal_nodes]  # Check if these nodes are minima, or if they have endpoints

    # The next point is a minimum
    next_nodes = next_nodes.astype(int)  # Mapping for type object doesn't work
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
    del local_watersheds[None]  # All nodes with None as endpoint aren't of interest

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
    """
    Combine all watersheds with adjacent minima, leave the rest as is
    :param local_watersheds: Watersheds leading to a minima
    :param combined_minima: Collection of adjacent minima
    :return watersheds: The combined watersheds
    """

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


def get_boundary_pairs_in_watersheds(watersheds, nx, ny):
    """
    Return all boundary pairs between all watersheds. If domain pairs should be excluded,
    remove comments at indicated places.
    :param watersheds: All watersheds of the domain.
    :param nx: Nr of nodes in x-direction
    :param ny: Nr of nodes in y-direction
    :return boundary_pairs: List of lists where each list contain a tuple of two arrays
    """

    indices = np.arange(0, nx * ny, 1)
    nbrs = get_neighbor_indices(indices, nx)

    # N.B: If boundary pairs to domain should be removed, include line below
    # domain_bnd_nodes = get_domain_boundary_indices(nx, ny)

    boundary_pairs = []

    for watershed in watersheds:

        watershed = np.sort(watershed)
        nbrs_for_ws = nbrs[watershed]
        nbrs_for_ws_1d = np.concatenate(nbrs_for_ws)

        # Find nodes not in the watershed which aren't at the domain boundary
        not_in_watershed_arr = np.in1d(nbrs_for_ws_1d, watershed, invert=True)

        # N.B: If boundary pairs to domain should be removed, include lines below
        # at_dom_boundary = np.in1d(nbrs_for_ws_1d, domain_bnd_nodes)
        # valid_nodes = np.where((not_in_watershed_arr - at_dom_boundary) == True)[0]

        # Pairs in from-to format
        repeat_from = np.repeat(watershed, 8)
        from_indices = repeat_from[not_in_watershed_arr]
        to_indices = nbrs_for_ws_1d[not_in_watershed_arr]
        boundary_pairs_for_ws = [from_indices, to_indices]
        boundary_pairs.append(boundary_pairs_for_ws)

    return boundary_pairs


def get_boundary_pairs_for_specific_watersheds(specific_watersheds, nx):
    """
    Only find boundary pairs for specified watersheds.
    :param specific_watersheds: Selection of watersheds
    :param nx: Number of nodes in x-direction
    :return boundary_pairs: Boundary pairs for specified watersheds
    """

    boundary_pairs = []

    for watershed in specific_watersheds:

        nbrs = get_neighbor_indices(watershed, nx)
        nbrs_for_ws_1d = np.concatenate(nbrs)
        valid_nodes = np.in1d(nbrs_for_ws_1d, watershed, invert=True)

        # Pairs in from-to format
        repeat_from = np.repeat(watershed, 8)
        from_indices = repeat_from[valid_nodes]
        to_indices = nbrs_for_ws_1d[valid_nodes]
        boundary_pairs_for_ws = [from_indices, to_indices]
        boundary_pairs.append(boundary_pairs_for_ws)

    return boundary_pairs


def get_possible_spill_pairs(heights, boundary_pairs):
    """
    Returns a list of lists where each list contains the possible spill pairs from one watershed to another
    :param heights: Heights of terrain
    :param boundary_pairs: Pairs between watershed boundaries
    :return spill_pairs: Possible spill pairs
    :return spill_height: Height of point where it will pour out
    """

    rows, cols = np.shape(heights)
    heights = np.reshape(heights, rows * cols)
    heights_pairs = [heights[np.vstack((arr[0], arr[1]))] for arr in boundary_pairs]

    # Max elevation of each pair, get min of that
    min_of_max = [np.min(np.max(heights_pairs[i], axis=0)) for i in range(len(heights_pairs))]

    # If x -> y, both x and y have heights <= min_of_max
    indices = [np.where(np.logical_and(heights_pairs[i][0] <= min_of_max[i], heights_pairs[i][1] <= min_of_max[i]))[0]
               for i in range(len(heights_pairs))]

    spill_pairs = [[boundary_pairs[i][0][indices[i]], boundary_pairs[i][1][indices[i]]] for i in range(len(indices))]

    spill_height = min_of_max

    return spill_height, spill_pairs


def get_steepest_spill_pair(heights, spill_pairs):
    """
    Return a list of tuples where each tuple is the spill pair for each watershed
    :param heights: Heights of terrain
    :param spill_pairs: List of lists. Each list contains two arrays in from-to format
    :return steepest_spill_pairs: Set containing the steepest spill pairs
    """

    rows, cols = np.shape(heights)
    heights = np.reshape(heights, rows * cols)
    steepest_spill_pairs = [None] * len(spill_pairs)

    for i in range(len(spill_pairs)):

        diff = abs(spill_pairs[i][0] - spill_pairs[i][1])
        derivatives = np.array([None] * len(spill_pairs[i][0]), dtype=object)

        card_indices = np.where(np.logical_or(diff == 1, diff == cols + 1))[0]
        diag_indices = np.setdiff1d(np.arange(0, len(spill_pairs[i][0]), 1), card_indices)
        card_der = (heights[spill_pairs[i][0][card_indices]] - heights[spill_pairs[i][1][card_indices]])/10
        diag_der = (heights[spill_pairs[i][0][diag_indices]] - heights[spill_pairs[i][1][diag_indices]])/math.sqrt(200)
        derivatives[card_indices] = card_der
        derivatives[diag_indices] = diag_der

        max_index = np.argmax(derivatives)
        steepest_spill_pairs[i] = (spill_pairs[i][0][max_index], spill_pairs[i][1][max_index])

    return set(steepest_spill_pairs)


def map_nodes_to_watersheds(watersheds, rows, cols):
    """
    Map between node indices and watershed number
    :param watersheds: List of arrays containing watersheds
    :param rows: Nr of rows
    :param cols: Nr of columns
    :return mapping_nodes_to_watersheds: Array where index gives watershed nr
    """

    mapping_nodes_to_watersheds = np.ones(rows * cols, dtype=int) * -1

    for i in range(len(watersheds)):
        mapping_nodes_to_watersheds[watersheds[i]] = i

    return mapping_nodes_to_watersheds


def merge_watersheds_flowing_into_each_other(watersheds, steepest_spill_pairs, rows, cols):

    map_node_to_ws = map_nodes_to_watersheds(watersheds, rows, cols)

    # Dictionary used for removing merged spill_pairs
    d = {}
    for s_p in steepest_spill_pairs:
        key = (map_node_to_ws[s_p[0]], map_node_to_ws[s_p[1]])
        value = s_p
        d[key] = value

    # Use set operations to find pairs of watersheds spilling to each other
    temp = set([(map_node_to_ws[el[0]], map_node_to_ws[el[1]]) for el in steepest_spill_pairs])
    temp_rev = set([(map_node_to_ws[el[1]], map_node_to_ws[el[0]]) for el in steepest_spill_pairs])
    pairs_to_each_other = temp.intersection(temp_rev)

    # Remove (y, x) when (x, y) is in the set
    merge_pairs = set(tuple(sorted(x)) for x in pairs_to_each_other)

    # Create the new list of watersheds
    merged_watersheds = [np.concatenate((watersheds[el[0]], watersheds[el[1]])) for el in merge_pairs]

    merged_indices = np.unique(list(merge_pairs))
    if len(pairs_to_each_other) > 0:
        removed_spill_pairs = set(itemgetter(*pairs_to_each_other)(d))
    else:
        removed_spill_pairs = {}

    return merged_watersheds, removed_spill_pairs, merged_indices


def combine_watersheds_spilling_into_each_other(watersheds, heights):
    """
    Iterative process to combine all watersheds spilling into each other
    :param watersheds: Different areas flowing to the same area
    :param heights: Heights of terrain
    :return watersheds: New collection of watersheds where some have been merged
    """

    ny, nx = np.shape(heights)

    remaining_spill_pairs = {}
    merged_watersheds = watersheds
    steepest_spill_pairs = None
    it = 0

    while len(merged_watersheds) > 0:
        print it

        # Find spill pairs for given watersheds
        boundary_pairs = get_boundary_pairs_for_specific_watersheds(merged_watersheds, nx)
        spill_height, spill_pairs = get_possible_spill_pairs(heights, boundary_pairs)
        steepest_spill_pairs = get_steepest_spill_pair(heights, spill_pairs)

        # Add the new spill pairs to the unaltered ones
        steepest_spill_pairs = steepest_spill_pairs.union(remaining_spill_pairs)

        # Merge watersheds spilling into each other
        merged_watersheds, removed_spill_pairs, merged_indices = merge_watersheds_flowing_into_each_other(
            watersheds, steepest_spill_pairs, ny, nx)
        remaining_spill_pairs = steepest_spill_pairs.difference(removed_spill_pairs)

        # Remove the merged watersheds from watersheds. Add the new ones to the end
        watersheds = [watersheds[i] for i in range(len(watersheds)) if i not in merged_indices]
        watersheds.extend(merged_watersheds)

        it += 1
        if len(merged_watersheds) == 0:  # Remove cycles at last iteration
            merged_watersheds, removed_spill_pairs, merged_indices = remove_cycles(
                watersheds, steepest_spill_pairs, ny, nx)
            remaining_spill_pairs = steepest_spill_pairs.difference(removed_spill_pairs)
            watersheds = [watersheds[i] for i in range(len(watersheds)) if i not in merged_indices]
            watersheds.extend(merged_watersheds)

    return watersheds, steepest_spill_pairs


def remove_cycles(watersheds, steepest, ny, nx):
    # Remove cycles by combining the watersheds involved in a cycle

    steepest = list(steepest)
    mapping = map_nodes_to_watersheds(watersheds, ny, nx)

    # Only spill_pairs going from a ws to another ws, no paths between ws and boundary
    spill_pairs = [(mapping[steepest[i][0]], mapping[steepest[i][1]]) for i in range(len(steepest))
                   if (mapping[steepest[i][0]] != -1 and mapping[steepest[i][1]] != -1)]

    DG = networkx.DiGraph()
    DG.add_edges_from(spill_pairs)

    # Remove cycles
    cycles = sorted(networkx.simple_cycles(DG))

    merged_indices = sorted([x for l in cycles for x in l])
    ws_not_being_merged = np.setdiff1d(np.arange(0, len(watersheds), 1), merged_indices)
    merged_watersheds = [np.concatenate([watersheds[el] for el in c]) for c in cycles]

    # Remove the no longer valid spill pairs
    d = {}
    for s_p in steepest:
        key = (mapping[s_p[0]], mapping[s_p[1]])
        value = s_p
        d[key] = value

    removed_spill_pairs = set([d[el] for el in spill_pairs if el[0] in merged_indices])

    return merged_watersheds, removed_spill_pairs, merged_indices


def merge_watersheds(watersheds, steepest, nx, ny):

    mapping = map_nodes_to_watersheds(watersheds, ny, nx)

    # Only spill_pairs going from a ws to another ws, no paths between ws and boundary
    spill_pairs = [(mapping[steepest[i][0]], mapping[steepest[i][1]]) for i in range(len(steepest))
                   if (mapping[steepest[i][0]] != -1 and mapping[steepest[i][1]] != -1)]

    DG = networkx.DiGraph()
    DG.add_edges_from(spill_pairs)

    G = DG.to_undirected()
    watershed_indices = sorted(networkx.connected_components(G))

    ws_being_merged = sorted([x for l in watershed_indices for x in l])
    ws_not_being_merged = np.setdiff1d(np.arange(0, len(watersheds), 1), ws_being_merged)
    merged_watersheds = [np.concatenate([watersheds[el] for el in ws_set]) for ws_set in watershed_indices]

    not_merged_watersheds = [watersheds[el] for el in ws_not_being_merged]
    merged_watersheds.extend(not_merged_watersheds)

    watersheds = merged_watersheds

    return watersheds


def create_watershed_conn_matrix(watersheds, steepest_spill_pairs, rows, cols):
    """
    Returns a connectivity matrix in csr_matrix format. This shows which watersheds are connected.
    :param watersheds: List of arrays where each array contains node indices for the watershed.
    :param steepest_spill_pairs: List of pairs where each pair is a spill pair between two node indices.
    :param rows: Nr of rows
    :param cols: Nr of cols
    :return conn_mat: The connectivity matrix showing connections between watersheds.
    """

    map_ix_to_ws = map_nodes_to_watersheds(watersheds, rows, cols)
    steepest_pairs_ws_nr = [(map_ix_to_ws[p[0]], map_ix_to_ws[p[1]]) for p in steepest_spill_pairs
                            if map_ix_to_ws[p[1]] != -1]
    from_ws = [p[0] for p in steepest_pairs_ws_nr]
    to_ws = [p[1] for p in steepest_pairs_ws_nr]

    nr_of_watersheds = len(watersheds)

    row_indices = from_ws
    col_indices = to_ws
    data = np.ones(len(row_indices))

    conn_mat = csr_matrix((data, (row_indices, col_indices)), shape=(nr_of_watersheds, nr_of_watersheds))

    return conn_mat
