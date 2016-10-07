from lib import util
import networkx as nx
from scipy.sparse import csr_matrix, identity, csgraph, identity
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
        return visited_ws

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


def get_rivers_between_spill_points(watersheds, heights, steepest_spill_pairs, spill_heights, flow_direction_indices):

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

# def get_upslope_nodes():


def get_rivers(watersheds, new_watersheds, steepest_spill_pairs, traps, downslope_indices, heights):

    # Note: Assume steepest_spill_pairs is sorted
    rows, cols = np.shape(heights)
    mapping = util.map_nodes_to_watersheds(watersheds, rows, cols)
    merged_watersheds = [np.unique(mapping[ws]) for ws in new_watersheds]
    all_rivers = []

    for i in range(len(merged_watersheds)):  # Iterate over the thresholded watersheds
        # A new river for the watershed
        river_in_large_ws = []
        small_watersheds = merged_watersheds[i]
        spill_pairs_merged_watersheds = [(mapping[s[0]], mapping[s[1]]) for s in steepest_spill_pairs
                                         if mapping[s[0]] in small_watersheds or mapping[s[1]] in small_watersheds]

        # River must go from start to end
        start = [el for el in spill_pairs_merged_watersheds if el[0] not in small_watersheds]
        if len(start) == 0:  # There is no river flowing across
            continue
        else:
            start = start[0]
        end = [el for el in spill_pairs_merged_watersheds if el[1] not in small_watersheds][0]

        G = nx.Graph()
        G.add_edges_from(spill_pairs_merged_watersheds)
        river_ws = nx.shortest_path(G, start[0], end[0])

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
            river_in_large_ws.extend(river)

        all_rivers.append(river_in_large_ws)

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
