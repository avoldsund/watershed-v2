from lib import util
import networkx
from scipy.sparse import csr_matrix, identity, csgraph, identity


def get_upslope_watersheds(conn_mat, ws_nr):
    """
    Returns a list of watersheds that are upslope for watershed nr ws_nr
    :param conn_mat: Connectivity matrix for watersheds
    :param ws_nr: Watershed of interest, the one you want the upslope for
    :return upslope_watersheds: Indices of all upslope watersheds
    """

    initial_upslope = conn_mat[:, ws_nr].nonzero()[0]
    not_visited_ws = [initial_upslope]
    visited_ws = [ws_nr]

    if len(initial_upslope) == 0:  # There are no upslope neighbors
        return visited_ws

    visited_ws.extend(initial_upslope)

    while not_visited_ws:  # As long as new upslope ws are found
        ix = not_visited_ws.pop()
        new_upslope_ws = conn_mat[:, ix].nonzero()[0]

        not_visited_ws.extend(new_upslope_ws)
        visited_ws.extend(new_upslope_ws)

    upslope_watersheds = visited_ws

    return upslope_watersheds


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



