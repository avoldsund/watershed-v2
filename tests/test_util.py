from lib import util, compare_methods
import numpy as np
import math
from scipy.sparse import csr_matrix


def test_map_2d_to_1d():

    rows = 3
    cols = 4
    coords_rows = np.array([0, 0, 0, 1, 1, 2, 2])
    coords_cols = np.array([0, 2, 3, 1, 3, 0, 3])
    coords = (coords_rows, coords_cols)

    indices = np.array([0, 2, 3, 5, 7, 8, 11])

    result_indices = util.map_2d_to_1d(coords, cols)

    assert np.array_equal(indices, result_indices)


def test_map_1d_to_2d():

    rows = 3
    cols = 4
    indices = np.array([0, 2, 3, 5, 7, 8, 11])

    coords_rows = np.array([0, 0, 0, 1, 1, 2, 2])
    coords_cols = np.array([0, 2, 3, 1, 3, 0, 3])
    coords = (coords_rows, coords_cols)

    result_coords = util.map_1d_to_2d(indices, cols)

    assert np.array_equal(coords, result_coords)


def test_get_neighbor_heights():

    cols = 2
    rows = 2
    heights = np.array([[1, 2, 3, 4],
                        [5, 6, 7, 8],
                        [9, 10, 11, 12],
                        [13, 14, 15, 16]])
    neighbors = np.zeros((2, 2, 8), dtype=int)
    neighbors[0, 0] = np.array([3, 7, 11, 10, 9, 5, 1, 2])
    neighbors[0, 1] = np.array([4, 8, 12, 11, 10, 6, 2, 3])
    neighbors[1, 0] = np.array([7, 11, 15, 14, 13, 9, 5, 6])
    neighbors[1, 1] = np.array([8, 12, 16, 15, 14, 10, 6, 7])

    result_neighbors = util.get_neighbor_heights(heights, rows, cols)

    assert np.array_equal(neighbors, result_neighbors)


def test_get_neighbor_heights_rect():

    cols = 2
    rows = 1
    heights = np.array([[0, 5, 10, 15],
                        [1, 2, 3, 4],
                        [50, 40, 30, 20]])
    neighbors = np.zeros((rows, cols, 8), dtype=int)
    neighbors[0, 0] = np.array([10, 3, 30, 40, 50, 1, 0, 5])
    neighbors[0, 1] = np.array([15, 4, 20, 30, 40, 2, 5, 10])

    result_neighbors = util.get_neighbor_heights(heights, rows, cols)

    assert np.array_equal(neighbors, result_neighbors)


def test_get_derivatives():

    heights = np.array([[18, 63, 59, 3],
                        [6, 77, 59, 75],
                        [68, 65, 7, 35],
                        [11, 31, 69, 20]])
    nbr_heights = np.array([[[59, 59, 7, 65, 68, 6, 18, 63], [3, 75, 35, 7, 65, 77, 63, 59]],
                            [[59, 7, 69, 31, 11, 68, 6, 77], [75, 35, 20, 69, 31, 65, 77, 59]]])

    h = 10.0
    d = math.sqrt(h ** 2 + h ** 2)
    derivatives = np.array([[[18/d, 18/h, 70/d, 12/h, 9/d, 71/h, 59/d, 14/h],
                             [56/d, -16/h, 24/d, 52/h, -6/d, -18/h, -4/d, 0/h]],
                            [[6/d, 58/h, -4/d, 34/h, 54/d, -3/h, 59/d, -12/h],
                             [-68/d, -28/h, -13/d, -62/h, -24/d, -58/h, -70/d, -52/h]]], dtype=float)

    result_derivatives = util.get_derivatives(heights, nbr_heights, h)

    assert np.array_equal(derivatives, result_derivatives)


def test_get_flow_directions():

    rows = 2
    cols = 2
    step_size = 10
    heights = np.array([[18, 63, 59, 3],
                        [6, 77, 59, 75],
                        [68, 65, 7, 35],
                        [11, 31, 69, 20]])
    pos_flow_directions = np.array([[32, 8],
                                    [2, -1]])

    result_pos_flow_directions = util.get_flow_directions(heights, step_size, rows, cols)

    assert np.array_equal(pos_flow_directions, result_pos_flow_directions)


def test_get_flow_directions_advanced():

    rows = 3
    cols = 4
    step_size = 10
    heights = np.array([[5, 7, 8, 7, 6, 0],
                        [7, 2, 10, 10, 7, 6],
                        [7, 2, 4, 5, 5, 4],
                        [7, 7, 3.9, 4, 0, 0],
                        [6, 5, 4, 4, 0, 0]])
    flow_directions = np.array([[-1, 32, 8, 1],
                                [-1, 32, 4, 8],
                                [128, 64, 2, -1]])

    result_pos_flow_directions = util.get_flow_directions(heights, step_size, rows, cols)

    assert np.array_equal(flow_directions, result_pos_flow_directions)


def test_get_flow_directions_float():

    rows = 2
    cols = 2
    step_size = 10
    heights = np.array([[399.8, 396.9, 392.7, 386.9],
                        [402.5, 399.4, 394.8, 388.8],
                        [405.4, 402.0, 396.5, 389.5],
                        [407.9, 404.8, 398.4, 389.6]])
    flow_directions = np.array([[1, 2],
                                [2, 2]])

    result_pos_flow_directions = util.get_flow_directions(heights, step_size, rows, cols)

    assert np.array_equal(flow_directions, result_pos_flow_directions)


def test_get_flow_direction_indices():

    rows = 3
    cols = 4
    step_size = 10
    heights = np.array([[5, 7, 8, 7, 6, 0],
                        [7, 2, 10, 10, 7, 6],
                        [7, 2, 4, 5, 5, 4],
                        [7, 7, 3.9, 4, 0, 0],
                        [6, 5, 4, 4, 0, 0]])

    flow_direction_indices = np.array([[-1, 0, 6, -1],
                                       [-1, 4, 11, 11],
                                       [4, 4, 11, -1]])

    result_flow_direction_indices = util.get_flow_direction_indices(heights, step_size, rows, cols)

    assert np.array_equal(flow_direction_indices, result_flow_direction_indices)


def test_fill_single_cell_depressions():

    rows = 1
    cols = 1
    heights = np.array([[5, 6, 7], [12, 2, 8], [11, 10, 9]])
    filled = np.array([[5, 6, 7], [12, 5, 8], [11, 10, 9]])

    result_filled = util.fill_single_cell_depressions(heights, rows, cols)

    assert np.array_equal(filled, result_filled)


def test_fill_single_cell_depressions_advanced():

    rows = 3
    cols = 2
    heights = np.array([[1, 3, 5, 6], [3, 4, 2, 7], [5, 5, 5, 7],
                        [7, 1, 1, 7], [7, 7, 7, 7]])
    filled = np.array([[1, 3, 5, 6], [3, 4, 3, 7], [5, 5, 5, 7],
                       [7, 1, 1, 7], [7, 7, 7, 7]])

    result_filled = util.fill_single_cell_depressions(heights, rows, cols)

    assert np.array_equal(filled, result_filled)


def test_fill_single_cell_depressions_three_minima():

    rows = 3
    cols = 3
    heights = np.array([[5, 5, 5, 5, 5], [5, 0, 2, -1, 5], [5, 2, 2, 2, 5],
                        [5, 2, 0, 1, 5], [5, 5, 5, 5, 5]])
    filled = np.array([[5, 5, 5, 5, 5], [5, 2, 2, 2, 5], [5, 2, 2, 2, 5],
                       [5, 2, 1, 1, 5], [5, 5, 5, 5, 5]])

    result_filled = util.fill_single_cell_depressions(heights, rows, cols)

    assert np.array_equal(filled, result_filled)


def test_get_node_endpoints():

    downslope_neighbors = np.array([[-1, 0, 6, -1],
                                    [-1, 4, 11, 11],
                                    [4, 4, 11, -1]])
    node_endpoints = np.array([[0, 0, 11, 3],
                               [4, 4, 11, 11],
                               [4, 4, 11, 11]])

    result_node_endpoints = util.get_node_endpoints(downslope_neighbors)

    assert np.array_equal(node_endpoints, result_node_endpoints)


def test_remove_out_of_boundary_flow():

    flow_directions = np.array([[-1, 32, 8, 1],
                                [-1, 32, 4, 8],
                                [128, 64, 2, -1]])

    new_flow_directions = np.array([[-1, 32, 8, -1],
                                    [-1, 32, 4, 8],
                                    [128, 64, 2, -1]])

    util.remove_out_of_boundary_flow(flow_directions)

    assert np.array_equal(flow_directions, new_flow_directions)


def test_remove_out_of_boundary_flow_advanced():

    flow_directions = np.array([[2, 64, 4, 128, 1],
                                [64, 16, 4, 32, 32],
                                [32, 64, 128, 1, 2],
                                [16, 16, 8, 4, 128],
                                [4, 32, 16, 8, 4]])

    new_flow_directions = np.array([[2, -1, 4, -1, -1],
                                    [-1, 16, 4, 32, 32],
                                    [-1, 64, 128, 1, -1],
                                    [-1, 16, 8, 4, 128],
                                    [-1, 32, -1, -1, -1]])

    util.remove_out_of_boundary_flow(flow_directions)

    assert np.array_equal(flow_directions, new_flow_directions)


def test_get_local_watersheds():

    cols = 4
    node_endpoints = np.array([[0, 0, 11, 3],
                               [4, 4, 11, 11],
                               [4, 4, 11, 11]])
    local_watersheds = {0: np.array([0, 1]),
                        3: np.array([3]),
                        4: np.array([4, 5, 8, 9]),
                        11: np.array([2, 6, 7, 10, 11])}

    result_local_watersheds = util.get_local_watersheds(node_endpoints)

    assert compare_methods.compare_two_dictionaries_where_values_are_arrays(local_watersheds, result_local_watersheds)


def test_combine_watersheds():

    cols = 4
    node_endpoints = np.array([[0, 0, 11, 3],
                               [4, 4, 11, 11],
                               [4, 4, 11, 11]])
    combined_minima = [np.array([0, 4]),
                       np.array([3]),
                       np.array([11])]
    local_watersheds = {0: np.array([0, 1]),
                        3: np.array([3]),
                        4: np.array([4, 5, 8, 9]),
                        11: np.array([2, 6, 7, 10, 11])}
    combined_watersheds = [np.array([0, 1, 4, 5, 8, 9]),
                           np.array([3]),
                           np.array([2, 6, 7, 10, 11])]

    result_combined_watersheds = util.combine_watersheds(local_watersheds, combined_minima)

    assert compare_methods.compare_two_lists_of_arrays(combined_watersheds, result_combined_watersheds)


def test_create_nbr_connectivity_matrix():

    flow_directions = np.array([[8, -1],
                                [2, -1]])

    conn_mat = np.array([[0, 0, 1, 0],
                         [0, 0, 0, 0],
                         [0, 0, 0, 1],
                         [0, 0, 0, 0]])

    result_conn_mat = util.create_nbr_connectivity_matrix(flow_directions, 2, 2)

    assert np.array_equal(conn_mat, result_conn_mat.todense())


def test_create_nbr_connectivity_matrix_square():

    flow_directions = np.array([[2, 8, 8],
                                [8, 2, -1],
                                [2, 2, 128]])

    conn_mat = np.array([[0, 1, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 1, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 1, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 1, 0, 0],
                         [0, 0, 0, 0, 0, 1, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 1, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 1],
                         [0, 0, 0, 0, 0, 1, 0, 0, 0]])

    result_conn_mat = util.create_nbr_connectivity_matrix(flow_directions, 3, 3)

    assert np.array_equal(conn_mat, result_conn_mat.todense())


def test_create_nbr_connectivity_matrix_advanced():

    flow_directions = np.array([[-1, 32, 8, 1],
                                [-1, 32, 4, 8],
                                [128, 64, 2, -1]])

    conn_mat = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                         [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

    result_conn_mat = util.create_nbr_connectivity_matrix(flow_directions, 4, 3)

    assert np.array_equal(conn_mat, result_conn_mat.todense())


def test_get_minima():

    rows = np.array([0, 2])
    cols = np.array([2, 3])
    data = np.array([1, 1])
    adj_mat = csr_matrix((data, (rows, cols)), shape=(4, 4))

    minimums = np.array([1, 3])

    result_minimums = util.get_minima(adj_mat)

    assert np.array_equal(minimums, result_minimums)


def test_get_minima_advanced():

    rows = np.array([0, 1, 2, 3, 4, 6, 7, 8])
    cols = np.array([1, 4, 5, 6, 5, 7, 8, 5])
    data = np.array([1, 1, 1, 1, 1, 1, 1, 1])
    adj_mat = csr_matrix((data, (rows, cols)), shape=(9, 9))

    minimums = np.array([5])

    result_minimums = util. get_minima(adj_mat)

    assert np.array_equal(minimums, result_minimums)


def test_get_downslope_rivers():

    rows = np.array([0, 2])
    cols = np.array([2, 3])
    data = np.array([1, 1])
    adj_mat = csr_matrix((data, (rows, cols)), shape=(4, 4))

    rows_d_r = np.array([0, 0, 0, 1, 2, 2, 3])
    cols_d_r = np.array([0, 2, 3, 1, 2, 3, 3])
    data_d_r = np.array([1, 1, 1, 1, 1, 1, 1])
    downslope_rivers = csr_matrix((data_d_r, (rows_d_r, cols_d_r)), shape=(4, 4))

    result_downslope_rivers = util.get_downslope_rivers(adj_mat)

    assert np.array_equal(downslope_rivers.todense(), result_downslope_rivers.todense())


def test_get_downslope_rivers_advanced():

    rows = np.array([0, 1, 2, 3, 4, 6, 7, 8])
    cols = np.array([1, 4, 5, 6, 5, 7, 8, 5])
    data = np.array([1, 1, 1, 1, 1, 1, 1, 1])
    adj_mat = csr_matrix((data, (rows, cols)), shape=(9, 9))

    rows_d_r = np.array([0, 0, 0, 0, 1, 1, 1, 2, 2, 3, 3, 3, 3, 3, 4, 4, 5, 6, 6, 6, 6, 7, 7, 7, 8, 8])
    cols_d_r = np.array([0, 1, 4, 5, 1, 4, 5, 2, 5, 3, 5, 6, 7, 8, 4, 5, 5, 5, 6, 7, 8, 5, 7, 8, 5, 8])
    data_d_r = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    downslope_rivers = csr_matrix((data_d_r, (rows_d_r, cols_d_r)), shape=(9, 9))

    result_downslope_rivers = util.get_downslope_rivers(adj_mat)

    assert np.array_equal(downslope_rivers.todense(), result_downslope_rivers.todense())


#def test_get_local_watersheds_two_minima():
#
#    minimums = np.array([1, 3])
#
#    rows = np.array([0, 0, 0, 1, 2, 2, 3])
#    cols = np.array([0, 2, 3, 1, 2, 3, 3])
#    data = np.array([1, 1, 1, 1, 1, 1, 1])
#    downslope_rivers = csr_matrix((data, (rows, cols)), shape=(4, 4))
#
#    local_watersheds = {1: np.array([1]), 3: np.array([0, 2, 3])}
#    result_local_watersheds = util.get_local_watersheds(downslope_rivers, minimums)
#
#    for m in result_local_watersheds:
#        result_local_watersheds[m] = np.sort(result_local_watersheds[m])
#
#    assert compare_methods.compare_two_dictionaries_where_values_are_arrays(local_watersheds, result_local_watersheds)
#
#
#def test_get_local_watersheds_one_min():
#
#    minimums = np.array([5])
#
#    rows = np.array([0, 0, 0, 0, 1, 1, 1, 2, 2, 3, 3, 3, 3, 3, 4, 4, 5, 6, 6, 6, 6, 7, 7, 7, 8, 8])
#    cols = np.array([0, 1, 4, 5, 1, 4, 5, 2, 5, 3, 5, 6, 7, 8, 4, 5, 5, 5, 6, 7, 8, 5, 7, 8, 5, 8])
#    data = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
#    downslope_rivers = csr_matrix((data, (rows, cols)), shape=(9, 9))
#
#    local_watersheds = {5: np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])}
#    result_local_watersheds = util.get_local_watersheds(downslope_rivers, minimums)
#
#    for m in result_local_watersheds:
#        result_local_watersheds[m] = np.sort(result_local_watersheds[m])
#
#    assert compare_methods.compare_two_dictionaries_where_values_are_arrays(local_watersheds, result_local_watersheds)


def test_combine_minima():

    rows = 2
    cols = 2
    local_minima = np.array([1, 3])

    comb_min = [np.array([1, 3])]
    result_comb_min = util.combine_minima(local_minima, rows, cols)
    for i in range(len(result_comb_min)):
        result_comb_min[i] = sorted(result_comb_min[i])

    assert compare_methods.compare_two_lists_of_arrays(comb_min, result_comb_min)


def test_combine_minima_three_comb():

    rows = 3
    cols = 4
    local_minima = np.array([0, 3, 4, 11])

    comb_min = [np.array([0, 4]), np.array([3]), np.array([11])]
    result_comb_min = util.combine_minima(local_minima, rows, cols)
    for i in range(len(result_comb_min)):
        result_comb_min[i] = sorted(result_comb_min[i])

    assert compare_methods.compare_two_lists_of_arrays(comb_min, result_comb_min)


def test_combine_minima_advanced():

    rows = 5
    cols = 8
    local_minima = np.array([0, 6, 7, 9, 10, 11, 18, 22, 23, 27, 29, 31, 33, 39])

    comb_min = [np.array([0, 9, 10, 11, 18, 27]), np.array([6, 7]),
                np.array([22, 23, 29, 31, 39]), np.array([33])]

    result_comb_min = util.combine_minima(local_minima, rows, cols)
    for i in range(len(result_comb_min)):
        result_comb_min[i] = sorted(result_comb_min[i])

    assert compare_methods.compare_two_lists_of_arrays(comb_min, result_comb_min)


def test_get_watersheds_with_combined_minima():

    comb_min = [[0, 4], [3], [11]]
    local_watersheds = {0: np.array([0, 1]), 4: np.array([4, 5, 8, 9]),
                        3: np.array([3]), 11: np.array([2, 6, 7, 10, 11])}

    watersheds = [np.array([0, 1, 4, 5, 8, 9]), np.array([3]), np.array([2, 6, 7, 10, 11])]
    result_watersheds = util.get_watersheds_with_combined_minima(comb_min, local_watersheds)

    are_equal = compare_methods.compare_two_lists_of_arrays(watersheds, result_watersheds)

    assert are_equal


def test_get_all_boundary_pairs_in_watersheds():

    num_of_cols = 7
    num_of_rows = 7
    watersheds = [np.array([8, 9, 15, 16, 22]),
                  np.array([10, 11, 12, 17, 18, 19, 23, 24, 25, 26, 32, 33, 39, 40]),
                  np.array([29, 30, 31, 36, 37, 38])]

    result_boundary_pairs = [[np.array([9, 9, 16, 16, 16, 16, 15, 22, 22, 22, 8, 8, 8, 8, 8, 9, 9, 9, 15, 15, 15, 22, 22, 22]),
                              np.array([10, 17, 10, 17, 23, 24, 23, 23, 29, 30, 0, 1, 2, 7, 14, 1, 2, 3, 7, 14, 21, 14, 21, 28])],
                             [np.array([10, 10, 17, 17, 23, 23, 23, 23, 23, 23, 24, 24, 24, 25, 32, 32, 39, 39, 10, 10, 10, 11, 11, 11, 12, 12, 12, 12, 12, 19, 19, 19, 26, 26, 26, 33, 33, 33, 40, 40, 40, 40, 40, 39, 39, 39]),
                              np.array([9, 16, 9, 16, 15, 16, 22, 29, 30, 31, 16, 30, 31, 31, 31, 38, 31, 38, 2, 3, 4, 3, 4, 5, 4, 5, 6, 13, 20, 13, 20, 27, 20, 27, 34, 27, 34, 41, 34, 41, 48, 47, 46, 45, 46, 47])],
                             [np.array([29, 29, 30, 30, 30, 31, 31, 31, 31, 31, 38, 38, 29, 29, 29, 36, 36, 36, 36, 36, 37, 37, 37, 38, 38, 38]),
                              np.array([22, 23, 22, 23, 24, 23, 24, 25, 32, 39, 32, 39, 21, 28, 35, 28, 35, 42, 43, 44, 43, 44, 45, 44, 45, 46])]]

    boundary_pairs = util.get_all_boundary_pairs_in_watersheds(watersheds, num_of_cols, num_of_rows)

    are_equal = compare_methods.compare_list_of_lists_by_comparing_sets(boundary_pairs, result_boundary_pairs)

    assert are_equal


def test_get_boundary_pairs_in_watersheds_small():

    num_of_cols = 7
    num_of_rows = 7
    watersheds = [np.array([8, 9, 15, 16, 22]),
                  np.array([10, 11, 12, 17, 18, 19, 23, 24, 25, 26, 32, 33, 39, 40]),
                  np.array([29, 30, 31, 36, 37, 38])]

    result_boundary_pairs = [[np.array([9, 9, 16, 16, 16, 16, 15, 22, 22, 22]),
                              np.array([10, 17, 10, 17, 23, 24, 23, 23, 29, 30])],
                             [np.array([10, 10, 17, 17, 23, 23, 23, 23, 23, 23, 24, 24, 24, 25, 32, 32, 39, 39]),
                              np.array([9, 16, 9, 16, 15, 16, 22, 29, 30, 31, 16, 30, 31, 31, 31, 38, 31, 38])],
                             [np.array([29, 29, 30, 30, 30, 31, 31, 31, 31, 31, 38, 38]),
                              np.array([22, 23, 22, 23, 24, 23, 24, 25, 32, 39, 32, 39])]]

    boundary_pairs = util.get_boundary_pairs_in_watersheds(watersheds, num_of_cols, num_of_rows)

    are_equal = compare_methods.compare_list_of_lists_by_comparing_sets(boundary_pairs, result_boundary_pairs)

    assert are_equal


def test_get_boundary_pairs_in_watersheds_large():

    num_of_cols = 9
    num_of_rows = 9
    watersheds = [np.array([10, 11, 12, 13, 19, 20, 21, 22, 28, 29, 30, 37, 38, 39]),
                  np.array([14, 15, 16, 23, 24, 25, 31, 32, 33, 34, 40, 41, 42, 43, 51, 52, 60, 61, 69, 70]),
                  np.array([46, 47, 48, 49, 50, 55, 56, 57, 58, 59, 64, 65, 66, 67, 68])]

    result_boundary_pairs = [[np.array([13, 13, 22, 22, 22, 22, 21, 30, 30, 37, 37, 38, 38, 38, 39, 39, 39, 39, 39]),
                              np.array([14, 23, 14, 23, 31, 32, 31, 31, 40, 46, 47, 46, 47, 48, 31, 40, 47, 48, 49])],
                             [np.array([14, 14, 23, 23, 31, 31, 31, 31, 32, 40, 40, 40, 40, 40, 41, 41, 42, 51, 51, 60, 60, 60, 69, 69]),
                              np.array([13, 22, 13, 22, 21, 22, 30, 39, 22, 30, 39, 48, 49, 50, 49, 50, 50, 50, 59, 50, 59, 68, 59, 68])],
                             [np.array([46, 46, 47, 47, 47, 48, 48, 48, 49, 49, 49, 50, 50, 50, 50, 50, 59, 59, 59, 68, 68]),
                              np.array([37, 38, 37, 38, 39, 38, 39, 40, 39, 40, 41, 40, 41, 42, 51, 60, 51, 60, 69, 60, 69])]]

    boundary_pairs = util.get_boundary_pairs_in_watersheds(watersheds, num_of_cols, num_of_rows)

    are_equal = compare_methods.compare_list_of_lists_by_comparing_sets(boundary_pairs, result_boundary_pairs)

    assert are_equal


def test_get_possible_spill_pairs():

    heights = np.array([[4, 10, 10, 10, 10, 10, 10],
                        [10, 1, 8, 7, 7, 7, 10],
                        [10, 6, 8, 5, 5, 5, 10],
                        [10, 8, 8, 4, 2, 4, 10],
                        [10, 9, 9, 3, 3, 3, 10],
                        [10, 0, 1, 5, 5, 5, 10],
                        [10, 10, 10, 10, 10, 10, 10]])
    boundary_pairs = [[np.array([9, 9, 16, 16, 16, 16, 15, 22, 22, 22, 8, 8, 8, 8, 8, 9, 9, 9, 15, 15, 15, 22, 22, 22]),
                              np.array([10, 17, 10, 17, 23, 24, 23, 23, 29, 30, 0, 1, 2, 7, 14, 1, 2, 3, 7, 14, 21, 14, 21, 28])],
                             [np.array([10, 10, 17, 17, 23, 23, 23, 23, 23, 23, 24, 24, 24, 25, 32, 32, 39, 39, 10, 10, 10, 11, 11, 11, 12, 12, 12, 12, 12, 19, 19, 19, 26, 26, 26, 33, 33, 33, 40, 40, 40, 40, 40, 39, 39, 39]),
                              np.array([9, 16, 9, 16, 15, 16, 22, 29, 30, 31, 16, 30, 31, 31, 31, 38, 31, 38, 2, 3, 4, 3, 4, 5, 4, 5, 6, 13, 20, 13, 20, 27, 20, 27, 34, 27, 34, 41, 34, 41, 48, 47, 46, 45, 46, 47])],
                             [np.array([29, 29, 30, 30, 30, 31, 31, 31, 31, 31, 38, 38, 29, 29, 29, 36, 36, 36, 36, 36, 37, 37, 37, 38, 38, 38]),
                              np.array([22, 23, 22, 23, 24, 23, 24, 25, 32, 39, 32, 39, 21, 28, 35, 28, 35, 42, 43, 44, 43, 44, 45, 44, 45, 46])]]

    # [[np.array([9, 9, 16, 16, 16, 16, 15, 22]), np.array([10, 17, 10, 17, 23, 24, 23, 23])],
    result_min_of_max = np.array([4, 3, 3])
    result_spill_pairs = [[np.array([8]), np.array([0])],
                                   [np.array([25, 32]), np.array([31, 31])],
                                   [np.array([31, 31]), np.array([25, 32])]]
    min_of_max, spill_pairs = util.get_possible_spill_pairs(heights, boundary_pairs)

    assert compare_methods.compare_list_of_lists_by_comparing_sets(result_spill_pairs, spill_pairs) \
           and np.array_equal(result_min_of_max, min_of_max)


def test_get_steepest_spill_pair():

    heights = np.array([[4, 10, 10, 10, 10, 10, 10],
                        [10, 1, 8, 7, 7, 7, 10],
                        [10, 6, 8, 5, 5, 5, 10],
                        [10, 8, 8, 4, 2, 4, 10],
                        [10, 9, 9, 3, 3, 3, 10],
                        [10, 0, 1, 5, 5, 5, 10],
                        [10, 10, 10, 10, 10, 10, 10]])
    spill_pairs = [[np.array([8]), np.array([0])],
                   [np.array([25, 32]), np.array([31, 31])],
                   [np.array([31, 31]), np.array([25, 32])]]

    result_steepest_spill_pairs = [(8, 0), (32, 31), (31, 25)]

    steepest_spill_pairs = util.get_steepest_spill_pair(heights, spill_pairs)

    assert steepest_spill_pairs == result_steepest_spill_pairs


def test_map_nodes_to_watersheds():

    cols = 7
    rows = 7
    watersheds = [np.array([8, 9, 15, 16, 22]),
                  np.array([10, 11, 12, 17, 18, 19, 23, 24, 25, 26, 32, 33, 39, 40]),
                  np.array([29, 30, 31, 36, 37, 38])]

    result_mapping_nodes_to_watersheds = np.array([-1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 1, 1, 1, -1, -1, 0, 0, 1, 1, 1,
                                                   -1, -1, 0, 1, 1, 1, 1, -1, -1, 2, 2, 2, 1, 1, -1, -1, 2, 2, 2, 1, 1,
                                                   -1, -1, -1, -1, -1, -1, -1, -1])

    mapping_nodes_to_watersheds = util.map_nodes_to_watersheds(watersheds, rows, cols)

    assert np.array_equal(mapping_nodes_to_watersheds, result_mapping_nodes_to_watersheds)


def test_merge_watersheds_flowing_into_each_other():

