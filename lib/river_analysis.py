from lib import util
import networkx as nx
from scipy.sparse import csr_matrix, lil_matrix, identity, csgraph, identity
import numpy as np
from math import sqrt
import time


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
    print upslope_watersheds

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


def get_river_in_trap(trap, start, end, cols):
    """
    Returns the river through a trap as an array
    :param trap: The indices of the nodes in the trap
    :param start: Starting point of the river
    :param end: End point of the river
    :param cols: Number of cols in the data set
    :return river: The river through the trap
    """

    # Get the neighbors of the trap nodes
    nbrs_of_trap_indices = util.get_neighbor_indices(trap, cols)
    nbrs = np.hstack(nbrs_of_trap_indices)
    repeat_trap = np.repeat(trap, 8)

    # Find the pairs of all nodes in the trap that are neighbors
    are_pairs = np.in1d(nbrs_of_trap_indices, trap)
    trap_indices = repeat_trap[are_pairs]
    trap_nbrs = nbrs[are_pairs]

    # Find the weights of each pair
    distance = np.array([sqrt(200), 10, sqrt(200), 10, sqrt(200), 10, sqrt(200), 10])
    repeat_distance = np.tile(distance, len(trap))
    weights = repeat_distance[are_pairs]

    # Create the graph and add pairs and weights
    pairs = zip(trap_indices, trap_nbrs, weights)
    T = nx.Graph()
    T.add_weighted_edges_from(pairs)

    river = nx.shortest_path(T, start, end, weight='weight')

    return river


def calculate_nr_of_upslope_cells(node_conn_mat, traps, steepest_spill_pairs):

    rows, cols = node_conn_mat._shape
    # Retrieve the expanded connectivity matrix with traps as nodes
    node_conn_mat = expand_conn_mat(node_conn_mat, len(traps))
    expanded_conn_mat = reroute_trap_connections(node_conn_mat, traps, steepest_spill_pairs)

    # The flow starts in the start_cells. These are the cells without flow leading in to them
    start_nodes = calculate_flow_origins(expanded_conn_mat, traps, sqrt(rows), sqrt(cols))
    flow_acc, one_or_trap_size = assign_initial_flow_acc(traps, start_nodes, sqrt(rows), sqrt(cols))
    indices_with_next, next_nodes = expanded_conn_mat[start_nodes, :].nonzero()

    it = 0
    while len(next_nodes) > 0:
        print 'Iteration: ', it
        # For each next node, check if flow is defined for all upslope nodes
        next_nodes_unique = np.unique(next_nodes)
        prev_nodes, prev_flowing_to = expanded_conn_mat[:, next_nodes_unique].nonzero()
        valid = flow_acc[prev_nodes] > 0
        remove_indices_of_nodes = prev_flowing_to[valid == False]

        nodes_getting_flow_acc = np.setdiff1d(next_nodes_unique, next_nodes_unique[remove_indices_of_nodes])

        # If there are no nodes in nodes_getting_flow_acc, we are done
        if len(nodes_getting_flow_acc) > 0:
            for i in range(len(nodes_getting_flow_acc)):
                node_getting_flow = nodes_getting_flow_acc[i]
                flow_to_node = expanded_conn_mat[:, node_getting_flow].nonzero()[0]
                flow_acc[node_getting_flow] = one_or_trap_size[node_getting_flow] + np.sum(flow_acc[flow_to_node])
            indices_with_next_one, next_nodes_from_assigned = expanded_conn_mat[nodes_getting_flow_acc, :].nonzero()
            if len(remove_indices_of_nodes) > 0:
                indices_with_next_two, next_nodes_from_unassigned = expanded_conn_mat[remove_indices_of_nodes, :].nonzero()
                next_nodes = np.hstack((next_nodes_from_assigned, next_nodes_from_unassigned))
            else:
                next_nodes = next_nodes_from_assigned
        else:
            print 'Before breeeeeeeeeeak: ', nodes_getting_flow_acc
            break
        it += 1

    for i in range(len(traps)):
        trap = traps[i]
        flow_acc[trap] = flow_acc[rows + i]
    flow_acc = flow_acc[:rows]
    flow_acc = flow_acc.reshape(sqrt(rows), sqrt(cols))

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


def reroute_trap_connections(trap_node_conn_mat, traps, steepest_spill_pairs):
    """
    Turns all traps into a single node so that the flow is defined in the entire grid.
    NB: Flow to boundary is removed
    :param trap_node_conn_mat: Initial conn. matrix between nodes before adding trap nodes and rerouting connections
    :param traps: All traps in the landscape
    :param steepest_spill_pairs: The steepest spill pairs to all traps
    :return trap_node_conn_mat: Updated connectivity matrix
    """

    r, c = np.shape(trap_node_conn_mat)

    # Set trap's downslope to the spill point
    trap_indices = np.arange(r - len(traps), r, 1)
    downslope_indices = np.asarray([el[1] for el in steepest_spill_pairs])
    trap_node_conn_mat[trap_indices, downslope_indices] = 1

    # Get trap boundary
    trap_boundary = get_traps_boundaries(traps, sqrt(c), sqrt(r))
    print 'trap_boundary[9]: ', trap_boundary[9]
    # Get all indices flowing to traps, and to which they flow
    #print len(traps)
    #print [len(t) for t in traps]
    indices_flowing_to_traps = trap_node_conn_mat[:, trap_boundary[9]].nonzero()
    print indices_flowing_to_traps
    nr_of_nonzero_per_column = np.diff(trap_node_conn_mat.tocsc().indptr)
    print traps[9]
    print np.intersect1d(traps[9], indices_flowing_to_traps)
    nr_of_nodes_to_each_trap = [(nr_of_nonzero_per_column[traps[i]]) for i in range(len(traps))]
    #print nr_of_nodes_to_each_trap
    #print sum(nr_of_nodes_to_each_trap)
    trap_indices = np.concatenate([[r - len(traps) + i] * nr_of_nodes_to_each_trap[i]
                                   for i in range(len(traps))]).astype(int)

    assert len(trap_indices) == len(indices_flowing_to_traps)

    # Create new sparse matrix
    trap_node_conn_mat = trap_node_conn_mat.tolil()
    start = time.time()
    print 'len(indices_flowing_to_traps): ', len(indices_flowing_to_traps)
    trap_node_conn_mat[indices_flowing_to_traps, :] = 0
    end = time.time()

    print 'trap_node_conn_mat[indices_flowing_to_traps, :] = 0: ', end-start

    start = time.time()

    print 'len(trap_indices)', len(trap_indices)
    trap_node_conn_mat[indices_flowing_to_traps, trap_indices] = 1
    end = time.time()
    print 'trap_node_conn_mat[indices_flowing_to_traps, trap_indices] = 1: ', end-start
    print 'c'
    # Remove flow out of boundary
    domain_boundary = util.get_domain_boundary_indices(sqrt(c - len(traps)), sqrt(r - len(traps)))
    trap_node_conn_mat[:, domain_boundary] = 0

    trap_node_conn_mat = trap_node_conn_mat.tocsr()

    return trap_node_conn_mat


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


def get_traps_boundaries(traps, nx, ny):
    """
    Returns all nodes in the trap boundary
    :param traps: All traps in landscape
    :param nx: Nr of columns in grid
    :param ny: Nr of rows in grid
    :return trap_boundary: The trap boundary nodes in each trap
    """

    indices = np.arange(0, nx * ny, 1)
    nbrs = util.get_neighbor_indices(indices, nx)

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
