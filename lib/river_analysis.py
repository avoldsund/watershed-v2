from lib import util, plot
import networkx as nx
from scipy.sparse import csr_matrix
import numpy as np
from math import sqrt


def get_upslope_watersheds(conn_mat, ws_nr):
    """
    Returns a list of watersheds that are upslope for watershed nr ws_nr
    :param conn_mat: Connectivity matrix for watersheds
    :param ws_nr: Watershed of interest, the one you want the upslope for
    :return upslope_watersheds: Indices of all upslope watersheds
    """

    initial_upslope = conn_mat[:, ws_nr].nonzero()[0]

    not_visited_ws = initial_upslope.tolist()
    visited_ws = [ws_nr]

    if len(initial_upslope) == 0:  # There are no upslope neighbors
        return visited_ws, None

    visited_ws.extend(initial_upslope)

    # To be able to give a sense of distance away from watershed
    node_levels = [[ws_nr], list(not_visited_ws)]
    level = 1

    while not_visited_ws:  # As long as new upslope ws are found
        ix = not_visited_ws.pop(0)  # Use it as a queue to keep levels
        new_upslope_ws = conn_mat[:, ix].nonzero()[0]

        if ix not in node_levels[level]:
            level += 1
            node_levels.append(new_upslope_ws.tolist())
        else:
            next_level_nr = level + 1  # You fill up next level before switching
            if len(node_levels) > next_level_nr:
                node_levels[next_level_nr].extend(new_upslope_ws.tolist())
            else:
                node_levels.append(new_upslope_ws.tolist())

        not_visited_ws.extend(new_upslope_ws)
        visited_ws.extend(new_upslope_ws)

    upslope_watersheds = visited_ws

    return upslope_watersheds, node_levels


def get_downslope_watersheds(conn_mat, ws_nr):
    """
    Return the indices of all downslope watersheds
    :param conn_mat: Connectivity matrix for watersheds
    :param ws_nr: Watershed you want the downslope watersheds for
    :return downslope_watersheds: Indices of all downslope watersheds
    """

    initial_downslope = conn_mat[ws_nr, :].nonzero()[1]

    not_visited_ws = [initial_downslope]
    visited_ws = [ws_nr]

    if len(initial_downslope) == 0:  # There are no downslope neighbors
        return visited_ws

    visited_ws.extend(initial_downslope)

    while not_visited_ws:  # As long as new downslope ws are found
        ix = not_visited_ws.pop()
        new_downslope_ws = conn_mat[ix, :].nonzero()[1]

        not_visited_ws.extend(new_downslope_ws)
        visited_ws.extend(new_downslope_ws)

    downslope_watersheds = visited_ws

    return downslope_watersheds


def get_all_rivers_before_thresholding(watersheds, heights, steepest_spill_pairs, spill_heights, flow_direction_indices):
    """
    Find all rivers for the watersheds
    :param watersheds: The different watersheds
    :param heights: Heights of landscape
    :param steepest_spill_pairs: Steepest spill pairs for each watershed
    :param spill_heights: Heights of the steepest spill pairs
    :param flow_direction_indices: Indices of the flow directions
    :return rivers: All river nodes
    """

    r, c = np.shape(heights)
    mapping = util.map_nodes_to_watersheds(watersheds, r, c)

    spill_to = np.asarray([el[1] for el in steepest_spill_pairs])
    order = np.argsort([mapping[s] for s in spill_to])
    rivers = []
    # Remove all spill points at the edge
    it = 0
    for start in order:
        it += 1
        print it
        start = spill_to[start]
        # Find which watershed the river is in
        ws_nr = mapping[start]
        if ws_nr > -1:
            ws = watersheds[ws_nr]
            traps_in_ws = ws[np.where(heights[util.map_1d_to_2d(ws, c)] <= spill_heights[ws_nr])[0]]
            river = [start]
            ix = util.map_1d_to_2d(start, c)
            next_node = flow_direction_indices[util.map_1d_to_2d(start, c)]
            while next_node:
                river.append(next_node)
                next_node = flow_direction_indices[util.map_1d_to_2d(next_node, c)]
                if next_node in traps_in_ws or next_node == -1:  # If the next node is in the trap we're at the end of the river
                    next_node = False
            rivers.append(river)

    return np.concatenate(rivers)


def get_rivers(watersheds, new_watersheds, steepest_spill_pairs, traps, downslope_indices, heights):
    """
    Returns all rivers between traps and from traps and to the boundary
    :param watersheds: The nodes in the different watersheds
    :param new_watersheds: The thresholded watersheds
    :param steepest_spill_pairs: The steepest spill pair for every watershed
    :param traps: The trap in every watershed
    :param downslope_indices: The downslope node for all nodes
    :param heights: Elevations in the landscape
    :return all_rivers: The rivers between the traps and from traps to the boundary
    """

    rows, cols = np.shape(heights)
    # new_steepest = util.remap_steepest_spill_pairs(new_watersheds, steepest_spill_pairs, rows, cols)
    # new_mapping = util.map_nodes_to_watersheds(new_watersheds, rows, cols)
    # spill_pairs_between_thresholded = [(new_mapping[el[0]], new_mapping[el[1]]) for el in new_steepest]

    mapping = util.map_nodes_to_watersheds(watersheds, rows, cols)
    merged_watersheds = [np.unique(mapping[ws]) for ws in new_watersheds]
    all_rivers = []

    for i in range(len(merged_watersheds)):  # Iterate over the thresholded watersheds
        # A new river for the watershed
        small_watersheds = merged_watersheds[i]
        spill_pairs_merged_watersheds = [(mapping[s[0]], mapping[s[1]]) for s in steepest_spill_pairs
                                         if mapping[s[0]] in small_watersheds or mapping[s[1]] in small_watersheds]

        # River must go from start to end
        start = [el for el in spill_pairs_merged_watersheds if el[0] not in small_watersheds]
        # Note: There is always a maximum of one end watershed
        end = [el for el in spill_pairs_merged_watersheds if el[1] not in small_watersheds][0]

        for r in range(len(start)):  # A trap might have several rivers flowing to it
            large_river = []
            G = nx.Graph()
            G.add_edges_from(spill_pairs_merged_watersheds)
            river_ws = nx.shortest_path(G, start[r][0], end[0])

            for j in range(len(river_ws) - 1):  # The watersheds that are part of the river
                spill_start = steepest_spill_pairs[river_ws[j]][1]
                spill_end = steepest_spill_pairs[river_ws[j+1]][0]
                trap_in_ws = traps[river_ws[j+1]]
                river = []
                new_river_node = spill_start
                while new_river_node:
                    river.append(new_river_node)
                    if new_river_node in trap_in_ws:
                        if j != len(river_ws) - 2:
                            river_through_trap = get_river_in_trap(trap_in_ws, new_river_node,
                                                                   spill_end, cols)
                            river.extend(river_through_trap)
                        new_river_node = False
                    else:
                        new_river_node = downslope_indices[util.map_1d_to_2d(new_river_node, cols)]
                large_river.extend(river[:-1])  # Remove the last node in the river as it will be in the trap/lake

            all_rivers.append(large_river)

    return all_rivers


def get_river_in_trap(trap, start, end, cols, d4):
    """
    Returns the river through a trap as an array
    :param trap: The indices of the nodes in the trap
    :param start: Starting point of the river
    :param end: End point of the river
    :param cols: Number of cols in the data set
    :param d4: Use the D4-method instead of D8
    :return river: The river through the trap
    """

    # Get the neighbors of the trap nodes
    nbrs_of_trap_indices = util.get_neighbor_indices(trap, cols, d4)
    nbrs = np.hstack(nbrs_of_trap_indices)
    if d4:
        repeat_trap = np.repeat(trap, 4)
    else:
        repeat_trap = np.repeat(trap, 8)
    # Find the pairs of all nodes in the trap that are neighbors
    are_pairs = np.in1d(nbrs_of_trap_indices, trap)
    trap_indices = repeat_trap[are_pairs]
    trap_nbrs = nbrs[are_pairs]

    # Find the weights of each pair
    if d4:
        distance = np.ones(4) * 10
    else:
        distance = np.array([sqrt(200), 10, sqrt(200), 10, sqrt(200), 10, sqrt(200), 10])
    repeat_distance = np.tile(distance, len(trap))
    weights = repeat_distance[are_pairs]

    # Create the graph and add pairs and weights
    pairs = zip(trap_indices, trap_nbrs, weights)
    T = nx.Graph()
    T.add_weighted_edges_from(pairs)

    river = nx.shortest_path(T, start, end, weight='weight')

    return river


def calculate_nr_of_upslope_cells(node_conn_mat, rows, cols, traps, steepest_spill_pairs, d4):
    """
    Calculate the nr of upslope cells for all nodes in the landscape. Returns a 2D-array with a number
    indicating the nr of upslope cells for each coordinate.
    :param node_conn_mat: Based on downslope nodes. How the nodes are connected.
    :param rows: Nr of rows in landscape
    :param cols: Nr of cols in landscape
    :param traps: All traps in landscape
    :param steepest_spill_pairs: Steepest spill pairs for each watershed
    :param d4: Use the D4-method instead of D8
    :return flow_acc: Accumulated flow for each node
    """

    # Retrieve the expanded connectivity matrix with traps as nodes
    node_conn_mat = expand_conn_mat(node_conn_mat, len(traps))
    conn_mat = reroute_trap_connections(node_conn_mat, rows, cols, traps, steepest_spill_pairs, d4)

    # The flow starts in the start_cells. These are the cells without flow leading in to them
    start_nodes = calculate_flow_origins(conn_mat, traps, rows, cols)
    flow_acc, one_or_trap_size = assign_initial_flow_acc(traps, start_nodes, rows, cols)
    _, next_nodes = conn_mat[start_nodes, :].nonzero()
    next_nodes = np.unique(next_nodes)

    current_nodes = next_nodes
    it = 0

    while len(current_nodes) > 0:
        print 'Iteration: ', it
        # Current nodes cannot be assigned flow without previous nodes having flow assigned
        previous_nodes, corr_current_index = conn_mat[:, current_nodes].nonzero()
        _, flow_to_each_current = np.unique(corr_current_index, return_counts=True)
        previous_nodes_with_flow = flow_acc[previous_nodes] > 0
        remove_indices = corr_current_index[previous_nodes_with_flow == False]
        keep_indices = np.setdiff1d(np.arange(0, len(current_nodes), 1), remove_indices)
        sorting_order = np.argsort(corr_current_index)
        previous_nodes = previous_nodes[sorting_order]
        assign_flow_indices = np.setdiff1d(current_nodes, current_nodes[remove_indices])

        # Calculate flow to current nodes having previous nodes with assigned flow
        splits = np.cumsum(flow_to_each_current)
        nodes_to_each_current = np.split(previous_nodes, splits)[:-1]
        flow_to_each_current = np.asarray([np.sum(flow_acc[el]) for el in nodes_to_each_current])
        flow_acc[current_nodes[keep_indices]] = flow_to_each_current[keep_indices]

        # Add one or the trap size
        flow_acc[assign_flow_indices] += one_or_trap_size[assign_flow_indices]

        it += 1
        if len(assign_flow_indices) > 0:
            _, next_nodes = conn_mat[assign_flow_indices, :].nonzero()
            next_nodes = np.unique(next_nodes)
            unassigned_current_nodes = current_nodes[remove_indices]
            current_nodes = np.union1d(next_nodes, unassigned_current_nodes)

        else:
            current_nodes = []

    # Map from trap nodes back to traps
    for i in range(len(traps)):
        trap = traps[i]
        flow_acc[trap] = flow_acc[rows * cols + i]

    flow_acc = flow_acc[:rows * cols]
    flow_acc = flow_acc.reshape(rows, cols)

    return flow_acc


def expand_conn_mat(conn_mat, nr_of_traps):
    """
    Adds zero columns and rows to represent the trap nodes to the connectivity matrix
    :param conn_mat: Connections between the nodes
    :param nr_of_traps: Nr of trap nodes
    :return expanded_mat: Expanded matrix to account for trap nodes
    """

    r, c = conn_mat._shape
    new_indptr = np.hstack((conn_mat.indptr, np.asarray([conn_mat.indptr[-1]] * nr_of_traps)))
    expanded_mat = csr_matrix((conn_mat.data, conn_mat.indices, new_indptr),
                              shape=(r + nr_of_traps, c + nr_of_traps))

    return expanded_mat


def reroute_trap_connections(conn_mat, rows, cols, traps, steepest_spill_pairs, d4):
    """
    Reroute connections going from nodes to traps so that they go to trap nodes instead. Remove the old connection.
    :param conn_mat: Connectivity matrix between all nodes
    :param rows: Nr of rows in landscape
    :param cols: Nr of cols in landscape
    :param traps: All traps in landscape
    :param steepest_spill_pairs: Spill pair from each trap
    :param d4: Use the D4-method instead of D8
    :return conn_mat: Updated connectivity matrix
    """

    # rows and cols is the original size (nx x ny)
    r, c = np.shape(conn_mat)

    # Add the connections: trap_nodes -> downslope_indices
    trap_indices = np.arange(r - len(traps), r, 1)
    downslope_indices = np.asarray([el[1] for el in steepest_spill_pairs])
    add_conn_from_trap_nodes = csr_matrix((np.ones(len(trap_indices), dtype=int), (trap_indices, downslope_indices)), shape=(r, c))
    conn_mat = conn_mat + add_conn_from_trap_nodes

    # Add the connections: nodes_to_trap -> trap_nodes
    trap_boundaries = np.concatenate(get_traps_boundaries(traps, cols, rows, d4))
    conn_node_to_trap = conn_mat[:, trap_boundaries].nonzero()
    nodes_to_trap = conn_node_to_trap[0]
    nodes_in_trap = trap_boundaries[conn_node_to_trap[1]]
    map_nodes_to_trap = util.map_nodes_to_watersheds(traps, rows, cols)
    trap_nodes = map_nodes_to_trap[nodes_in_trap] + rows * cols

    # Get nodes_to_trap's corresponding trap indices
    add_conn_to_trap_nodes = csr_matrix(((np.ones(len(nodes_to_trap), dtype=int)), (nodes_to_trap, trap_nodes)), shape=(r, c))
    conn_mat = conn_mat + add_conn_to_trap_nodes

    # Remove the connections: nodes_to_trap -> nodes_in_trap

    remove_conn_to_traps = csr_matrix((np.ones(len(nodes_to_trap), dtype=int) * -1, (nodes_to_trap, nodes_in_trap)), shape=(r, c))
    conn_mat = conn_mat + remove_conn_to_traps

    # Remove flow out of boundary
    domain_boundary = util.get_domain_boundary_indices(cols, rows)
    conn_nodes_to_boundary = conn_mat[:, domain_boundary].nonzero()
    nodes_to_boundary = conn_nodes_to_boundary[0]
    boundary_nodes = domain_boundary[conn_nodes_to_boundary[1]]
    remove_conn_to_boundary = csr_matrix(((np.ones(len(nodes_to_boundary), dtype=int) * -1), (nodes_to_boundary, boundary_nodes)), shape=(r, c))
    conn_mat = conn_mat + remove_conn_to_boundary

    return conn_mat


def calculate_flow_origins(expanded_conn_mat, traps, rows, cols):
    """
    Returns the starting nodes for flow in a landscape, i.e., the nodes without any upslope
    :param expanded_conn_mat: Connectivity matrix between all nodes, including trap nodes
    :param traps: All traps in the landscape
    :param rows: Nr of nodes in y-direction
    :param cols: Nr of nodes in x-direction
    :return origin_nodes: Starting nodes for flow. The node can be a trap node
    """

    # Remove boundary nodes and nodes with flow to them
    boundary_indices = util.get_domain_boundary_indices(cols, rows)
    nodes_with_upslope = expanded_conn_mat.nonzero()[1]
    original_trap_nodes = np.concatenate(traps)
    not_starting_nodes = np.unique(np.hstack((boundary_indices, nodes_with_upslope, original_trap_nodes)))
    origin_nodes = np.setdiff1d(np.arange(rows * cols + len(traps)), not_starting_nodes).astype(int)

    return origin_nodes


def assign_initial_flow_acc(traps, start_nodes, rows, cols):
    """
    The initial flow starting in each start node. Starting nodes that are trap nodes must be handled differently
    :param traps: All traps in landscape
    :param start_nodes: The flow start nodes
    :param rows: Rows in landscape grid
    :param cols: Cols in landscape grid
    :return acc_flow: Array for flow accumulation in the whole landscape. All nodes besides start nodes are None
    """

    nr_of_nodes = int(rows * cols)
    # Get indices of the trap nodes that are starting indices
    starting_trap_nodes = np.array(start_nodes[start_nodes >= nr_of_nodes])

    # Initialize accumulation flow array, and calculate the size of each trap
    acc_flow = np.zeros(rows * cols + len(traps), dtype=int)
    trap_sizes = np.asarray([len(t) for t in traps])

    one_or_trap_size = np.ones(len(acc_flow), dtype=int)
    one_or_trap_size[nr_of_nodes:] = trap_sizes

    # Assign flow to the starting trap nodes, and the other starting nodes
    acc_flow[starting_trap_nodes] = trap_sizes[starting_trap_nodes - nr_of_nodes]
    acc_flow[np.setdiff1d(start_nodes, starting_trap_nodes)] = 1

    return acc_flow, one_or_trap_size


def get_traps_boundaries(traps, nx, ny, d4):
    """
    Returns all nodes in the trap boundary
    :param traps: All traps in landscape
    :param nx: Nr of columns in grid
    :param ny: Nr of rows in grid
    :param d4: Use the D4-method instead of D8
    :return trap_boundary: The trap boundary nodes in each trap
    """

    indices = np.arange(0, nx * ny, 1)
    nbrs = util.get_neighbor_indices(indices, nx, d4)

    # N.B: If boundary pairs to domain should be removed, include line below
    # domain_bnd_nodes = get_domain_boundary_indices(nx, ny)

    trap_boundary = []

    for trap in traps:
        nbrs_for_each_node_in_trap = nbrs[trap]
        nbr_is_in_trap = np.split(np.in1d(nbrs_for_each_node_in_trap, trap), len(trap))
        node_is_in_trap_boundary = ~np.all(nbr_is_in_trap, axis=1)

        # It is not possible that no elements are in trap boundary
        trap_boundary.append(trap[node_is_in_trap_boundary])

    return trap_boundary


def get_watershed_of_node(node_coords_r_c, expanded_conn_mat, traps, r, c):
    """
    :param node_coords_r_c: The coords for the outlet node
    :param expanded_conn_mat: Connectivity matrix for nodes and trap nodes
    :param traps: All traps in landscape
    :param r: y-dim of landscape
    :param c: x-dim of landscape
    :return watershed_of_node: The watershed for the chosen node
    """

    total_nodes = r * c
    node_1d = util.map_2d_to_1d((node_coords_r_c[0], node_coords_r_c[1]), c)
    map_nodes_to_trap = util.map_nodes_to_watersheds(traps, r, c)

    prev_nodes = expanded_conn_mat[:, node_1d].nonzero()[0]

    if len(prev_nodes) == 0:  # Node is trap node, or node has no upslope nodes
        node_in_trap_ix = map_nodes_to_trap[node_1d]

        if node_in_trap_ix == -1:  # Node not in trap
            return np.array([node_1d]), np.array([])
        else:  # Node in trap, check if trap node has upslope nodes
            trap_node = node_in_trap_ix + total_nodes
            prev_nodes = expanded_conn_mat[:, trap_node].nonzero()[0]
            if len(prev_nodes) == 0:
                return traps[node_in_trap_ix], np.array([node_in_trap_ix])

    if map_nodes_to_trap[node_1d] != -1:
        trap_node = map_nodes_to_trap[node_1d] + total_nodes
    else:
        trap_node = node_1d

    watershed_of_node = [np.array([trap_node]), prev_nodes]
    while prev_nodes.size:  # Upslope nodes are added until there are no more
        prev_nodes = expanded_conn_mat[:, prev_nodes].nonzero()[0]
        if prev_nodes.size:  # No empty array is added
            watershed_of_node.append(prev_nodes)

    watershed_of_node = np.concatenate(watershed_of_node)

    # If there are any trap nodes, map them to traps and add them
    traps_in_upslope = []
    trap_nodes_in_ws = []

    for j in range(len(watershed_of_node)):
        if watershed_of_node[j] >= total_nodes:
            traps_in_upslope.append(traps[watershed_of_node[j] - total_nodes])
            trap_nodes_in_ws.append(watershed_of_node[j] - total_nodes)
    watershed_of_node = watershed_of_node[watershed_of_node < total_nodes]
    if len(traps_in_upslope) > 0:
        traps_in_upslope = np.concatenate(traps_in_upslope)
        watershed_of_node = np.concatenate((watershed_of_node, traps_in_upslope))

    trap_nodes_in_ws = np.sort(np.array(trap_nodes_in_ws))

    return watershed_of_node, trap_nodes_in_ws


def calculate_watershed_of_node(landscape, outlet_coords_r_c, d4):

    landscape.heights = util.fill_single_cell_depressions(landscape.heights, landscape.ny, landscape.nx)
    watersheds, steepest, flow_dir = util.calculate_watersheds(landscape.heights, landscape.nx, landscape.ny, landscape.step_size, d4)
    spill_heights = util.get_spill_heights(watersheds, landscape.heights, steepest)
    traps, size_of_traps = util.get_all_traps(watersheds, landscape.heights, spill_heights)

    # Increase heights of traps and recalculate flow. Remove flow from some indices in traps.
    util.make_landscape_depressionless(watersheds, steepest, landscape)
    flow = util.get_flow_direction_indices(landscape.heights, landscape.step_size, landscape.ny, landscape.nx, d4)
    flow_dir = util.get_flow_directions(landscape.heights, landscape.step_size, landscape.ny, landscape.nx, d4)

    for i in range(len(traps)):
        trap_in_2d = util.map_1d_to_2d(traps[i], landscape.nx)
        flow[trap_in_2d] = -1
        flow_dir[trap_in_2d] = -1

    # Create connections and expand matrix to accommodate trap nodes
    node_conn_mat = util.make_sparse_node_conn_matrix(flow, landscape.ny, landscape.nx)
    node_conn_mat = expand_conn_mat(node_conn_mat, len(traps))
    expanded_conn_mat = reroute_trap_connections(node_conn_mat, landscape.ny, landscape.nx, traps, steepest)

    # Get the watershed of the node
    ws_of_node, trap_nodes_in_ws = get_watershed_of_node(outlet_coords_r_c, expanded_conn_mat, traps, landscape.ny, landscape.nx)

    return ws_of_node, traps, spill_heights, trap_nodes_in_ws, steepest, flow_dir, landscape.heights


def calculate_accumulated_flow(landscape):

    landscape.heights = util.fill_single_cell_depressions(landscape.heights, landscape.ny, landscape.nx)
    watersheds, steepest, flow_dir = util.calculate_watersheds(landscape.heights, landscape.nx, landscape.ny,
                                                               landscape.step_size)
    spill_heights = util.get_spill_heights(watersheds, landscape.heights, steepest)
    traps, size_of_traps = util.get_all_traps(watersheds, landscape.heights, spill_heights)

    # Increase heights of traps and recalculate flow. Remove flow from some indices in traps.
    util.make_landscape_depressionless(watersheds, steepest, landscape)
    flow = util.get_flow_direction_indices(landscape.heights, landscape.step_size, landscape.ny, landscape.nx)
    for i in range(len(traps)):
        trap_in_2d = util.map_1d_to_2d(traps[i], landscape.nx)
        flow[trap_in_2d] = -1

    node_conn_mat = util.make_sparse_node_conn_matrix(flow, landscape.ny, landscape.nx)
    upslope_cells = calculate_nr_of_upslope_cells(node_conn_mat, landscape.ny, landscape.nx, traps, steepest)

    return upslope_cells


def calculate_watershed_of_node_no_landscape_input(heights, nx, ny, step_size, outlet_coords_r_c):
    # Does not require landscape as input

    heights = util.fill_single_cell_depressions(heights, ny, nx)
    watersheds, steepest, flow_dir = util.calculate_watersheds(heights, nx, ny, step_size)
    spill_heights = util.get_spill_heights(watersheds, heights, steepest)
    traps, size_of_traps = util.get_all_traps(watersheds, heights, spill_heights)

    # Increase heights of traps and recalculate flow. Remove flow from some indices in traps.
    util.make_landscape_depressionless_no_landscape_input(watersheds, steepest, heights, ny)
    flow = util.get_flow_direction_indices(heights, step_size, ny, nx)
    flow_dir = util.get_flow_directions(heights, step_size, ny, nx)

    for i in range(len(traps)):
        trap_in_2d = util.map_1d_to_2d(traps[i], nx)
        flow[trap_in_2d] = -1

    # Create connections and expand matrix to accommodate trap nodes
    node_conn_mat = util.make_sparse_node_conn_matrix(flow, ny, nx)
    node_conn_mat = expand_conn_mat(node_conn_mat, len(traps))
    expanded_conn_mat = reroute_trap_connections(node_conn_mat, ny, nx, traps, steepest)

    # Get the watershed of the node
    ws_of_node, trap_nodes_in_ws = get_watershed_of_node(outlet_coords_r_c, expanded_conn_mat, traps, ny, nx)

    return ws_of_node, traps, spill_heights, trap_nodes_in_ws, steepest, flow_dir, heights
