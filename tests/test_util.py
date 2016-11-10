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
    indices = np.array([0, 3, 4, 11])

    coords_rows = np.array([0, 0, 1, 2])
    coords_cols = np.array([0, 3, 0, 3])
    coords = (coords_rows, coords_cols)

    result_coords = util.map_1d_to_2d(indices, cols)

    assert np.array_equal(coords, result_coords)


def test_fill_single_cell_depressions():

    rows = 3
    cols = 3
    heights = np.array([[5, 6, 7], [12, 2, 8], [11, 10, 9]])

    filled = np.array([[5, 6, 7], [12, 5, 8], [11, 10, 9]])
    result_filled = util.fill_single_cell_depressions(heights, rows, cols)

    assert np.array_equal(filled, result_filled)


def test_fill_single_cell_depressions_advanced():

    rows = 5
    cols = 4
    heights = np.array([[1, 3, 5, 6], [3, 4, 2, 7], [5, 5, 5, 7],
                        [7, 1, 1, 7], [7, 7, 7, 7]])
    filled = np.array([[1, 3, 5, 6], [3, 4, 3, 7], [5, 5, 5, 7],
                       [7, 1, 1, 7], [7, 7, 7, 7]])

    result_filled = util.fill_single_cell_depressions(heights, rows, cols)

    assert np.array_equal(filled, result_filled)


def test_fill_single_cell_depressions_three_minima():

    rows = 5
    cols = 5
    heights = np.array([[5, 5, 5, 5, 5], [5, 0, 2, -1, 5], [5, 2, 2, 2, 5],
                        [5, 2, 0, 1, 5], [5, 5, 5, 5, 5]])
    filled = np.array([[5, 5, 5, 5, 5], [5, 2, 2, 2, 5], [5, 2, 2, 2, 5],
                       [5, 2, 1, 1, 5], [5, 5, 5, 5, 5]])

    result_filled = util.fill_single_cell_depressions(heights, rows, cols)

    assert np.array_equal(filled, result_filled)


def test_get_domain_boundary_coords():

    cols = 5
    rows = 4
    result_boundary = (np.array([0, 0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 3, 3, 3]),
                       np.array([0, 1, 2, 3, 4, 0, 4, 0, 4, 0, 1, 2, 3, 4]))

    boundary = util.get_domain_boundary_coords(cols, rows)

    assert compare_methods.compare_coordinates(boundary, result_boundary)


def test_get_domain_boundary_coords_square():

    cols = 4
    rows = 4
    result_boundary = (np.array([0, 0, 0, 0, 3, 3, 3, 3, 1, 2, 1, 2]),
                       np.array([0, 1, 2, 3, 0, 1, 2, 3, 0, 0, 3, 3]))

    boundary = util.get_domain_boundary_coords(cols, rows)

    print result_boundary

    assert compare_methods.compare_coordinates(boundary, result_boundary)


def test_get_neighbor_heights():

    cols = 4
    rows = 4
    heights = np.array([[1, 2, 3, 4],
                        [5, 6, 7, 8],
                        [9, 10, 11, 12],
                        [13, 14, 15, 16]])

    neighbors = np.empty((rows, cols, 8), dtype=object)
    neighbors[1, 1] = np.array([3, 7, 11, 10, 9, 5, 1, 2])
    neighbors[1, 2] = np.array([4, 8, 12, 11, 10, 6, 2, 3])
    neighbors[2, 1] = np.array([7, 11, 15, 14, 13, 9, 5, 6])
    neighbors[2, 2] = np.array([8, 12, 16, 15, 14, 10, 6, 7])

    result_neighbors = util.get_neighbor_heights(heights, rows, cols)

    assert np.array_equal(neighbors, result_neighbors)


def test_get_neighbor_heights_rect():

    cols = 4
    rows = 3
    heights = np.array([[0, 5, 10, 15],
                        [1, 2, 3, 4],
                        [50, 40, 30, 20]])

    neighbors = np.empty((rows, cols, 8), dtype=object)
    neighbors[1, 1] = np.array([10, 3, 30, 40, 50, 1, 0, 5])
    neighbors[1, 2] = np.array([15, 4, 20, 30, 40, 2, 5, 10])

    result_neighbors = util.get_neighbor_heights(heights, rows, cols)

    assert np.array_equal(neighbors, result_neighbors)


def test_get_neighbor_indices():

    cols = 6
    indices = np.array([7, 15, 19, 22])

    result_nbrs = np.array([[2, 8, 14, 13, 12, 6, 0, 1],
                            [10, 16, 22, 21, 20, 14, 8, 9],
                            [14, 20, 26, 25, 24, 18, 12, 13],
                            [17, 23, 29, 28, 27, 21, 15, 16]])
    nbrs = util.get_neighbor_indices(indices, cols)

    assert np.array_equal(nbrs, result_nbrs)


def test_get_derivatives():

    r = 4
    c = 4
    heights = np.array([[18, 63, 59, 3],
                        [6, 77, 59, 75],
                        [68, 65, 7, 35],
                        [11, 31, 69, 20]])
    nbr_heights = np.empty((r, c, 8), dtype=object)
    nbr_heights[1, 1] = np.array([59, 59, 7, 65, 68, 6, 18, 63])
    nbr_heights[1, 2] = np.array([3, 75, 35, 7, 65, 77, 63, 59])
    nbr_heights[2, 1] = np.array([59, 7, 69, 31, 11, 68, 6, 77])
    nbr_heights[2, 2] = np.array([75, 35, 20, 69, 31, 65, 77, 59])
    h = 10.0
    d = math.sqrt(h ** 2 + h ** 2)

    derivatives = np.array([[[None, None, None, None, None, None, None, None],
                             [None, None, None, None, None, None, None, None],
                             [None, None, None, None, None, None, None, None],
                             [None, None, None, None, None, None, None, None]],
                            [[None, None, None, None, None, None, None, None],
                             [18/d, 18/h, 70/d, 12/h, 9/d, 71/h, 59/d, 14/h],
                             [56/d, -16/h, 24/d, 52/h, -6/d, -18/h, -4/d, 0/h],
                             [None, None, None, None, None, None, None, None]],
                            [[None, None, None, None, None, None, None, None],
                             [6/d, 58/h, -4/d, 34/h, 54/d, -3/h, 59/d, -12/h],
                             [-68/d, -28/h, -13/d, -62/h, -24/d, -58/h, -70/d, -52/h],
                             [None, None, None, None, None, None, None, None]],
                            [[None, None, None, None, None, None, None, None],
                             [None, None, None, None, None, None, None, None],
                             [None, None, None, None, None, None, None, None],
                             [None, None, None, None, None, None, None, None]]], dtype=object)
    result_derivatives = util.get_derivatives(heights, nbr_heights, h)

    assert np.array_equal(derivatives, result_derivatives)


def test_get_flow_directions():

    rows = 4
    cols = 4
    step_size = 10
    heights = np.array([[18, 63, 59, 3],
                        [6, 77, 59, 75],
                        [68, 65, 7, 35],
                        [11, 31, 69, 20]])

    pos_flow_directions = np.array([[None, None, None, None],
                                    [None, 32, 8, None],
                                    [None, 2, -1, None],
                                    [None, None, None, None]])
    result_pos_flow_directions = util.get_flow_directions(heights, step_size, rows, cols)

    assert np.array_equal(pos_flow_directions, result_pos_flow_directions)


def test_get_flow_directions_advanced():

    rows = 5
    cols = 6
    step_size = 10
    heights = np.array([[5, 7, 8, 7, 6, 0],
                        [7, 2, 10, 10, 7, 6],
                        [7, 2, 4, 5, 5, 4],
                        [7, 7, 3.9, 4, 0, 0],
                        [6, 5, 4, 4, 0, 0]])

    flow_directions = np.array([[None, None, None, None, None, None],
                                [None, -1, 32, 8, 1, None],
                                [None, -1, 32, 4, 8, None],
                                [None, 128, 64, 2, -1, None],
                                [None, None, None, None, None, None]])
    result_pos_flow_directions = util.get_flow_directions(heights, step_size, rows, cols)

    assert np.array_equal(flow_directions, result_pos_flow_directions)


def test_get_flow_directions_float():

    rows = 4
    cols = 4
    step_size = 10
    heights = np.array([[399.8, 396.9, 392.7, 386.9],
                        [402.5, 399.4, 394.8, 388.8],
                        [405.4, 402.0, 396.5, 389.5],
                        [407.9, 404.8, 398.4, 389.6]])

    flow_directions = np.array([[None, None, None, None],
                                [None, 1, 2, None],
                                [None, 2, 2, None],
                                [None, None, None, None]])
    result_pos_flow_directions = util.get_flow_directions(heights, step_size, rows, cols)

    assert np.array_equal(flow_directions, result_pos_flow_directions)


def test_get_flow_direction_indices():

    rows = 5
    cols = 6
    step_size = 10
    heights = np.array([[5, 7, 8, 7, 6, 0],
                        [7, 2, 10, 10, 7, 6],
                        [7, 2, 4, 5, 5, 4],
                        [7, 7, 3.9, 4, 0, 0],
                        [6, 5, 4, 4, 0, 0]])

    result_flow_direction_indices = np.array([[None, None, None, None, None, None],
                                              [None, -1, 7, 15, -1, None],
                                              [None, -1, 13, 22, 22, None],
                                              [None, 13, 13, 22, -1, None],
                                              [None, None, None, None, None, None]])
    flow_direction_indices = util.get_flow_direction_indices(heights, step_size, rows, cols)

    assert np.array_equal(flow_direction_indices, result_flow_direction_indices)


def test_remove_out_of_boundary_flow():

    flow_directions = np.array([[-2, -2, -2, -2, -2, -2],
                                [-2, -1, 32, 8, 1, -2],
                                [-2, -1, 32, 4, 8, -2],
                                [-2, 128, 64, 2, -1, -2],
                                [-2, -2, -2, -2, -2, -2]])

    new_flow_directions = np.array([[-2, -2, -2, -2, -2, -2],
                                    [-2, -1, 32, 8, -1, -2],
                                    [-2, -1, 32, 4, 8, -2],
                                    [-2, 128, 64, 2, -1, -2],
                                    [-2, -2, -2, -2, -2, -2]])

    util.remove_out_of_boundary_flow(flow_directions)

    assert np.array_equal(flow_directions, new_flow_directions)


def test_remove_out_of_boundary_flow_advanced():

    flow_directions = np.array([[-2, -2, -2, -2, -2, -2, -2],
                                [-2, 2, 64, 4, 128, 1, -2],
                                [-2, 64, 16, 4, 32, 32, -2],
                                [-2, 32, 64, 128, 1, 2, -2],
                                [-2, 16, 16, 8, 4, 128, -2],
                                [-2, 4, 32, 16, 8, 4, -2],
                                [-2, -2, -2, -2, -2, -2, -2]])

    new_flow_directions = np.array([[-2, -2, -2, -2, -2, -2, -2],
                                    [-2, 2, -1, 4, -1, -1, -2],
                                    [-2, -1, 16, 4, 32, 32, -2],
                                    [-2, -1, 64, 128, 1, -1, -2],
                                    [-2, -1, 16, 8, 4, 128, -2],
                                    [-2, -1, 32, -1, -1, -1, -2],
                                    [-2, -2, -2, -2, -2, -2, -2]])

    util.remove_out_of_boundary_flow(flow_directions)

    assert np.array_equal(flow_directions, new_flow_directions)


def test_get_node_endpoints():

    downslope_neighbors = np.array([[None, None, None, None, None, None],
                                    [None, -1, 7, 15, -1, None],
                                    [None, -1, 13, 22, 22, None],
                                    [None, 13, 13, 22, -1, None],
                                    [None, None, None, None, None, None]])

    node_endpoints = np.array([[None, None, None, None, None, None],
                               [None, 7, 7, 22, 10, None],
                               [None, 13, 13, 22, 22, None],
                               [None, 13, 13, 22, 22, None],
                               [None, None, None, None, None, None]])

    result_node_endpoints = util.get_node_endpoints(downslope_neighbors)

    assert np.array_equal(node_endpoints, result_node_endpoints)


def test_get_local_watersheds():

    node_endpoints = np.array([[None, None, None, None, None, None],
                               [None, 7, 7, 22, 10, None],
                               [None, 13, 13, 22, 22, None],
                               [None, 13, 13, 22, 22, None],
                               [None, None, None, None, None, None]])
    result_local_watersheds = {7: np.array([7, 8]),
                               10: np.array([10]),
                               13: np.array([13, 14, 19, 20]),
                               22: np.array([9, 15, 16, 21, 22])}
    local_watersheds = util.get_local_watersheds(node_endpoints)

    assert compare_methods.compare_minima_watersheds(local_watersheds, result_local_watersheds)


def test_combine_watersheds():

    cols = 4
    node_endpoints = np.array([[None, None, None, None, None, None],
                               [None, 7, 7, 22, 10, None],
                               [None, 13, 13, 22, 22, None],
                               [None, 13, 13, 22, 22, None],
                               [None, None, None, None, None, None]])
    combined_minima = [np.array([7, 13]),
                       np.array([10]),
                       np.array([22])]
    local_watersheds = {7: np.array([7, 8]),
                        10: np.array([10]),
                        13: np.array([13, 14, 19, 20]),
                        22: np.array([9, 15, 16, 21, 22])}
    combined_watersheds = [np.array([7, 8, 13, 14, 19, 20]),
                           np.array([10]),
                           np.array([9, 15, 16, 21, 22])]

    result_combined_watersheds = util.combine_watersheds(local_watersheds, combined_minima)

    assert compare_methods.compare_watersheds(combined_watersheds, result_combined_watersheds)


def test_combine_minima():

    rows = 4
    cols = 4
    local_minima = np.array([6, 10])

    comb_min = [np.array([6, 10])]
    result_comb_min = util.combine_minima(local_minima, rows, cols)
    for i in range(len(result_comb_min)):
        result_comb_min[i] = sorted(result_comb_min[i])

    assert compare_methods.compare_two_lists_of_arrays(comb_min, result_comb_min)


def test_combine_minima_three_comb():

    rows = 5
    cols = 6
    local_minima = np.array([7, 10, 13, 22])

    comb_min = [np.array([7, 13]), np.array([10]), np.array([22])]
    result_comb_min = util.combine_minima(local_minima, rows, cols)
    for i in range(len(result_comb_min)):
        result_comb_min[i] = sorted(result_comb_min[i])

    assert compare_methods.compare_two_lists_of_arrays(comb_min, result_comb_min)


def test_combine_minima_advanced():
    # This is from when interior nodes had indexing from 0
    rows = 7
    cols = 10
    local_minima = np.array([11, 17, 18, 22, 23, 24, 33, 37, 38, 44, 46, 48, 52, 58])

    comb_min = [np.array([11, 22, 23, 24, 33, 44]), np.array([17, 18]),
                np.array([37, 38, 46, 48, 58]), np.array([52])]

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


def test_get_boundary_pairs_in_watersheds():

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

    boundary_pairs = util.get_boundary_pairs_in_watersheds(watersheds, num_of_cols, num_of_rows)

    are_equal = compare_methods.compare_list_of_lists_by_comparing_sets(boundary_pairs, result_boundary_pairs)

    assert are_equal


# def test_get_boundary_pairs_in_watersheds_exclude_domain_small():
#
#     num_of_cols = 7
#     num_of_rows = 7
#     watersheds = [np.array([8, 9, 15, 16, 22]),
#                   np.array([10, 11, 12, 17, 18, 19, 23, 24, 25, 26, 32, 33, 39, 40]),
#                   np.array([29, 30, 31, 36, 37, 38])]
#
#     result_boundary_pairs = [[np.array([9, 9, 16, 16, 16, 16, 15, 22, 22, 22]),
#                               np.array([10, 17, 10, 17, 23, 24, 23, 23, 29, 30])],
#                              [np.array([10, 10, 17, 17, 23, 23, 23, 23, 23, 23, 24, 24, 24, 25, 32, 32, 39, 39]),
#                               np.array([9, 16, 9, 16, 15, 16, 22, 29, 30, 31, 16, 30, 31, 31, 31, 38, 31, 38])],
#                              [np.array([29, 29, 30, 30, 30, 31, 31, 31, 31, 31, 38, 38]),
#                               np.array([22, 23, 22, 23, 24, 23, 24, 25, 32, 39, 32, 39])]]
#
#     boundary_pairs = util.get_boundary_pairs_in_watersheds(watersheds, num_of_cols, num_of_rows)
#
#     are_equal = compare_methods.compare_list_of_lists_by_comparing_sets(boundary_pairs, result_boundary_pairs)
#
#     assert are_equal


# def test_get_boundary_pairs_in_watersheds_exclude_domain_large():
#
#     num_of_cols = 9
#     num_of_rows = 9
#     watersheds = [np.array([10, 11, 12, 13, 19, 20, 21, 22, 28, 29, 30, 37, 38, 39]),
#                   np.array([14, 15, 16, 23, 24, 25, 31, 32, 33, 34, 40, 41, 42, 43, 51, 52, 60, 61, 69, 70]),
#                   np.array([46, 47, 48, 49, 50, 55, 56, 57, 58, 59, 64, 65, 66, 67, 68])]
#
#     result_boundary_pairs = [[np.array([13, 13, 22, 22, 22, 22, 21, 30, 30, 37, 37, 38, 38, 38, 39, 39, 39, 39, 39]),
#                               np.array([14, 23, 14, 23, 31, 32, 31, 31, 40, 46, 47, 46, 47, 48, 31, 40, 47, 48, 49])],
#                              [np.array([14, 14, 23, 23, 31, 31, 31, 31, 32, 40, 40, 40, 40, 40, 41, 41, 42, 51, 51, 60, 60, 60, 69, 69]),
#                               np.array([13, 22, 13, 22, 21, 22, 30, 39, 22, 30, 39, 48, 49, 50, 49, 50, 50, 50, 59, 50, 59, 68, 59, 68])],
#                              [np.array([46, 46, 47, 47, 47, 48, 48, 48, 49, 49, 49, 50, 50, 50, 50, 50, 59, 59, 59, 68, 68]),
#                               np.array([37, 38, 37, 38, 39, 38, 39, 40, 39, 40, 41, 40, 41, 42, 51, 60, 51, 60, 69, 60, 69])]]
#
#     boundary_pairs = util.get_all_boundary_pairs_in_watersheds(watersheds, num_of_cols, num_of_rows)
#
#     are_equal = compare_methods.compare_list_of_lists_by_comparing_sets(boundary_pairs, result_boundary_pairs)
#
#     assert are_equal


def test_get_boundary_pairs_for_specific_watersheds():

    nx = 10

    merged_watersheds = [np.array([35, 33, 34]),
                         np.array([63, 64, 65, 73, 74, 75, 83, 84, 85]),
                         np.array([38, 47, 48, 57, 58, 18, 28]),
                         np.array([11, 12, 21, 22, 31, 32])]

    result_boundary_pairs = [[np.array([33, 33, 33, 33, 33, 33, 33, 34, 34, 34, 34, 34, 34, 35, 35, 35, 35, 35, 35, 35]),
                              np.array([24, 44, 43, 42, 32, 22, 23, 25, 45, 44, 43, 23, 24, 26, 36, 46, 45, 44, 24, 25])],
                             [np.array([85, 85, 85, 85, 85, 63, 63, 63, 63, 63, 64, 64, 64, 65, 65, 65, 65, 65, 73, 73, 73, 75, 75, 75, 83,
                                        83, 83, 83, 83, 84, 84, 84]),
                              np.array([94, 95, 96, 76, 86, 54, 53, 52, 62, 72, 53, 54, 55, 54, 55, 56, 66, 76, 62, 72, 82, 66, 76, 86, 72,
                                        82, 92, 93, 94, 93, 94, 95])],
                             [np.array([18, 18, 18, 18, 18, 18, 18, 28, 28, 28, 28, 28, 28, 38, 38, 38, 38, 38, 47, 47,
                                        47, 47, 48, 48, 48, 48, 57, 57, 57, 57, 57, 58, 58, 58, 58, 58]),
                              np.array([7, 8, 9, 17, 19, 27, 29, 17, 19, 27, 29, 37, 39, 27, 29, 37, 39, 49, 36, 37, 46,
                                        56, 37, 39, 49, 59, 46, 56, 66, 67, 68, 49, 59, 67, 68, 69])],
                             [np.array([11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 21, 21, 21, 22, 22, 22, 31, 31,
                                        31, 31, 31, 32, 32, 32, 32, 32]),
                              np.array([0, 1, 2, 10, 20, 1, 2, 3, 13, 23, 10, 20, 30, 13, 23, 33, 20, 30, 40,
                                        41, 42, 23, 33, 41, 42, 43])]]

    boundary_pairs = util.get_boundary_pairs_for_specific_watersheds(merged_watersheds, nx)

    assert compare_methods.compare_list_of_lists_by_comparing_sets(boundary_pairs, result_boundary_pairs)


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

    #result_min_of_max = np.array([4, 3, 3])
    result_spill_pairs = [[np.array([8]), np.array([0])],
                          [np.array([25, 32]), np.array([31, 31])],
                          [np.array([31, 31]), np.array([25, 32])]]
    spill_pairs = util.get_possible_spill_pairs(heights, boundary_pairs)

    assert compare_methods.compare_list_of_lists_by_comparing_sets(result_spill_pairs, spill_pairs)


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

    result_steepest_spill_pairs = {(8, 0), (32, 31), (31, 25)}

    steepest_spill_pairs = util.get_steepest_spill_pair(heights, spill_pairs)

    assert steepest_spill_pairs == result_steepest_spill_pairs


def test_remove_cycles():

    rows = 8
    cols = 8
    watersheds = [np.array([9, 10, 11, 17, 18, 19, 25, 26, 27]),
                  np.array([12, 13, 14, 20, 21, 22, 28, 29, 30]),
                  np.array([33, 34, 35, 36, 37, 38, 41, 42, 43, 44, 45, 46,
                            49, 50, 51, 52, 53, 54])]
    steepest_spill_pairs = [(19, 20), (29, 37), (34, 26)]

    result_watersheds = [np.array([9, 10, 11, 17, 18, 19, 25, 26, 27,
                                   12, 13, 14, 20, 21, 22, 28, 29, 30,
                                   33, 34, 35, 36, 37, 38, 41, 42, 43, 44, 45, 46,
                                   49, 50, 51, 52, 53, 54])]
    result_steepest_spill_pairs = [(54, 55)]

    merged_watersheds, removed_spill_pairs, merged_indices = util.remove_cycles(watersheds, steepest_spill_pairs, rows, cols)

    are_equal = compare_methods.compare_watersheds(merged_watersheds, result_watersheds)

    assert are_equal


def test_remove_cycles_advanced():

    rows = 6
    cols = 8
    watersheds = [np.array([9, 10, 17]),
                  np.array([11, 12]),
                  np.array([18, 19, 20, 28]),
                  np.array([25, 26, 33, 34]),
                  np.array([27, 35]),
                  np.array([30, 38]),
                  np.array([13, 14, 21, 22]),
                  np.array([29, 36, 37])]
    steepest_spill_pairs = [(10, 11), (12, 20), (18, 26), (25, 17),
                            (27, 34), (22, 30), (38, 37), (29, 21)]

    result_watersheds = [np.array([9, 10, 17, 11, 12, 18, 19, 20, 28, 25, 26, 33, 34]),
                         np.array([13, 14, 21, 22, 29, 36, 37, 30, 38])]

    # Notice that the watershed [27, 35] is not in merged_watersheds as it was not in the cycle,
    # just spilling to a cycle. The merged watersheds are the watersheds that was part of a cycle
    merged_watersheds, removed_spill_pairs, merged_indices = util.remove_cycles(watersheds, steepest_spill_pairs, rows, cols)

    assert compare_methods.compare_watersheds(merged_watersheds, result_watersheds)


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

    cols = 7
    rows = 7
    watersheds = [np.array([8, 9, 15, 16, 22]),
                  np.array([10, 11, 12, 17, 18, 19, 23, 24, 25, 26, 32, 33, 39, 40]),
                  np.array([29, 30, 31, 36, 37, 38])]
    steepest_spill_pairs = [(8, 0), (32, 31), (31, 25)]

    result_merged_watersheds = [np.array([10, 11, 12, 17, 18, 19, 23, 24, 25, 26,
                                          32, 33, 39, 40, 29, 30, 31, 36, 37, 38])]

    merged_watersheds, removed_spill_pairs, merged_indices = util.merge_watersheds_flowing_into_each_other(watersheds, steepest_spill_pairs, rows, cols)

    assert compare_methods.compare_watersheds(merged_watersheds, result_merged_watersheds)


def test_combine_watersheds_spilling_into_each_other():

    heights = np.array([[10, 10, 10, 10, 10, 10, 10, 10],
                        [10, 4, 5, 4, 15, 3, 9, 2],
                        [10, 10, 10, 10, 10, 10, 10, 10]])
    watersheds = [np.array([9, 10]),
                  np.array([11, 12]),
                  np.array([13, 14])]
    result_watersheds = [np.array([9, 10, 11, 12]),
                         np.array([13, 14])]

    watersheds, steepest_spill_pairs = util.combine_watersheds_spilling_into_each_other(watersheds, heights)

    assert compare_methods.compare_watersheds(watersheds, result_watersheds)


def test_combine_watersheds_spilling_into_each_other_simple():

    heights = np.array([[3, 3, 3, 3, 3, 3, 3, 3, 3],
                        [3, 1, 1, 1, 2, 1, 1, 1, 3],
                        [3, 1, 0, 1, 2, 1, 0, 1, 3],
                        [3, 1, 1, 1, 2, 1, 1, 1, 3],
                        [3, 3, 3, 3, 3, 3, 3, 3, 3]])
    watersheds = [np.array([10, 11, 12, 19, 20, 21, 28, 29, 30]),
                  np.array([13, 14, 15, 16, 22, 23, 24, 25, 31, 32, 33, 34])]
    result_watersheds = [np.array([10, 11, 12, 19, 20, 21, 28, 29, 30,
                                   13, 14, 15, 16, 22, 23, 24, 25, 31, 32, 33, 34])]

    watersheds, steepest_spill_pairs = util.combine_watersheds_spilling_into_each_other(watersheds, heights)

    assert compare_methods.compare_watersheds(watersheds, result_watersheds)


def test_get_spill_heights():

    heights = np.array([[4, 10, 10, 10, 10, 10, 10],
                        [10, 1, 8, 7, 7, 7, 10],
                        [10, 6, 8, 5, 5, 5, 10],
                        [10, 8, 8, 4, 2, 4, 10],
                        [10, 9, 9, 3, 3, 3, 10],
                        [10, 0, 1, 5, 5, 5, 10],
                        [10, 10, 10, 10, 10, 10, 10]])
    watersheds = [np.array([8, 9, 15, 16, 22]),
                  np.array([10, 11, 12, 17, 18, 19, 23, 24, 25, 26, 32, 33, 39, 40, 29, 30, 31, 36, 37, 38])]
    steepest_spill_pairs = [(23, 15), (8, 0)]
    result_spill_heights = np.array([4, 8])

    spill_heights = util.get_spill_heights(watersheds, heights, steepest_spill_pairs)

    assert np.array_equal(spill_heights, result_spill_heights)


def test_get_size_of_traps():

    heights = np.array([[4, 10, 10, 10, 10, 10, 10],
                        [10, 1, 8, 7, 7, 7, 10],
                        [10, 6, 8, 5, 5, 5, 10],
                        [10, 8, 8, 4, 2, 4, 10],
                        [10, 9, 9, 3, 3, 3, 10],
                        [10, 0, 1, 5, 5, 5, 10],
                        [10, 10, 10, 10, 10, 10, 10]])
    watersheds = [np.array([8, 9, 15, 16, 22]),
                  np.array([10, 11, 12, 17, 18, 19, 23, 24, 25, 26, 32, 33, 39, 40, 29, 30, 31, 36, 37, 38])]
    spill_heights = [4, 8]
    result_size_of_traps = np.array([1, 18])

    size_of_traps = util.get_size_of_traps(watersheds, heights, spill_heights)

    assert np.array_equal(size_of_traps, result_size_of_traps)


def test_remove_watersheds_below_threshold():

    #total_nodes = 60
    watersheds = [np.array([11, 12]),
                  np.array([13, 14, 15, 16]),
                  np.array([17, 18, 27]),
                  np.array([21, 22, 23, 32, 33]),
                  np.array([24, 25, 26, 34, 35, 36]),
                  np.array([28, 37, 38]),
                  np.array([31, 41, 42]),
                  np.array([43, 44, 45, 46]),
                  np.array([47, 48])]
    rows = np.array([0, 1, 2, 3, 4, 5, 7])
    cols = np.array([3, 4, 4, 1, 7, 8, 6])
    data = np.array([1, 1, 1, 1, 1, 1, 1])
    conn_mat = csr_matrix((data, (rows, cols)), shape=(9, 9))

    # Arbitrary numbers, want 0, 8 and 4 gone
    size_of_traps = np.array([2, 3, 3, 5, 1, 3, 3, 4, 1])
    threshold_size = 3

    new_rows = np.array([0, 1, 2, 5])
    new_cols = np.array([5, 5, 0, 4])
    new_data = np.array([1, 1, 1, 1])
    result_conn_mat = csr_matrix((new_data, (new_rows, new_cols)), shape=(6, 6))
    result_watersheds = [np.array([13, 14, 15, 16]),
                         np.array([17, 18, 27]),
                         np.array([11, 12, 21, 22, 23, 32, 33]),
                         np.array([28, 37, 38]),
                         np.array([31, 41, 42]),
                         np.array([43, 44, 45, 46, 24, 25, 26, 34, 35, 36])]
    # Notice that 47 and 48 were removed in the result_watersheds
    new_conn_mat, watersheds = util.remove_watersheds_below_threshold(watersheds, conn_mat,
                                                                      size_of_traps, threshold_size)

    assert np.array_equal(new_conn_mat.todense(), result_conn_mat.todense()) \
           and compare_methods.compare_watersheds(watersheds, result_watersheds)


def test_merge_watersheds():

    cols = 10
    rows = 11
    watersheds = [np.array([11, 12, 21, 22, 31, 32]),
                  np.array([13, 14, 15, 23, 24, 25]),
                  np.array([16, 17, 26, 27, 36, 37]),
                  np.array([33, 34, 35]),
                  np.array([18, 28, 38, 47, 48, 57, 58]),
                  np.array([41, 51, 61, 62]),
                  np.array([42, 43, 44, 45, 52, 53, 54, 55]),
                  np.array([46, 56, 66, 67, 68]),
                  np.array([63, 64, 65, 73, 74, 75, 83, 84, 85]),
                  np.array([71, 72, 81, 82, 91, 92, 93]),
                  np.array([76, 77, 78, 86, 87]),
                  np.array([88, 94, 95, 96, 97, 98])]
    steepest = [(32, 43), (25, 35), (27, 28), (35, 36), (57, 56), (51, 52), (54, 64), (67, 78),
                (83, 93), (92, 102), (78, 79), (98, 87)]

    result_merged_watersheds = [np.array([11, 12, 21, 22, 31, 32, 42, 43, 44, 45, 52, 53, 54, 55, 41, 51, 61, 62,
                                          63, 64, 65, 73, 74, 75, 83, 84, 85, 71, 72, 81, 82, 91, 92, 93]),
                                np.array([13, 14, 15, 23, 24, 25, 33, 34, 35, 16, 17, 26, 27, 36, 37, 18, 28, 38, 47,
                                          48, 57, 58, 46, 56, 66, 67, 68, 76, 77, 78, 86, 87, 88, 94, 95, 96, 97, 98])]

    merged_watersheds = util.merge_watersheds(watersheds, steepest, cols, rows)

    assert compare_methods.compare_watersheds(merged_watersheds, result_merged_watersheds)


def test_merge_watersheds_modified():

    cols = 10
    rows = 11
    watersheds = [np.array([11, 12, 21, 22, 31, 32]),
                  np.array([13, 14, 15, 23, 24, 25]),
                  np.array([16, 17, 26, 27, 36, 37]),
                  np.array([33, 34, 35]),
                  np.array([18, 28, 38, 47, 48, 57, 58]),
                  np.array([41, 51, 61, 62]),
                  np.array([42, 43, 44, 45, 52, 53, 54, 55]),
                  np.array([46, 56, 66, 67, 68]),
                  np.array([63, 64, 65, 73, 74, 75, 83, 84, 85]),
                  np.array([71, 72, 81, 82, 91, 92, 93]),
                  np.array([76, 77, 78, 86, 87]),
                  np.array([88, 94, 95, 96, 97, 98])]
    steepest = [(32, 43), (25, 35), (27, 28), (35, 36), (57, 56), (51, 52), (54, 64), (67, 78),
                (83, 93), (92, 102), (78, 79), (98, 108)]

    result_merged_watersheds = [np.array([11, 12, 21, 22, 31, 32, 42, 43, 44, 45, 52, 53, 54, 55, 41, 51, 61, 62,
                                          63, 64, 65, 73, 74, 75, 83, 84, 85, 71, 72, 81, 82, 91, 92, 93]),
                                np.array([13, 14, 15, 23, 24, 25, 33, 34, 35, 16, 17, 26, 27, 36, 37, 18, 28, 38, 47,
                                          48, 57, 58, 46, 56, 66, 67, 68, 76, 77, 78, 86, 87]),
                                np.array([88, 94, 95, 96, 97, 98])]

    merged_watersheds = util.merge_watersheds(watersheds, steepest, cols, rows)

    assert compare_methods.compare_watersheds(merged_watersheds, result_merged_watersheds)


def test_create_watershed_conn_matrix():

    rows = 11
    cols = 10
    nr_of_watersheds = 12
    steepest_spill_pairs = [(32, 43), (25, 35), (27, 28), (35, 36), (57, 56),
                            (51, 52), (54, 64), (67, 78), (83, 93), (92, 102),
                            (78, 79), (98, 87)]
    watersheds = [np.array([11, 12, 21, 22, 31, 32]),
                  np.array([13, 14, 15, 23, 24, 25]),
                  np.array([16, 17, 26, 27, 36, 37]),
                  np.array([33, 34, 35]),
                  np.array([18, 28, 38, 47, 48, 57, 58]),
                  np.array([41, 51, 61, 62]),
                  np.array([42, 43, 44, 45, 52, 53, 54, 55]),
                  np.array([46, 56, 66, 67, 68]),
                  np.array([63, 64, 65, 73, 74, 75, 83, 84, 85]),
                  np.array([71, 72, 81, 82, 91, 92, 93]),
                  np.array([76, 77, 78, 86, 87]),
                  np.array([88, 94, 95, 96, 97, 98])]
    row_indices = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 11])
    col_indices = np.array([6, 3, 4, 2, 7, 6, 8, 10, 9, 10])
    data = np.ones(len(row_indices))

    result_conn_mat = csr_matrix((data, (row_indices, col_indices)), shape=(nr_of_watersheds, nr_of_watersheds))

    conn_mat = util.create_watershed_conn_matrix(watersheds, steepest_spill_pairs, rows, cols)

    assert np.array_equal(conn_mat.todense(), result_conn_mat.todense())


def test_remove_ix_from_conn_mat_no_upslope():

    rows = np.array([0, 1, 2, 3, 4, 5, 7])
    cols = np.array([3, 4, 4, 1, 7, 8, 6])
    data = np.array([1, 1, 1, 1, 1, 1, 1])
    conn_mat = csr_matrix((data, (rows, cols)), shape=(9, 9))
    remove_ix = 0

    rows_new = np.array([0, 1, 2, 3, 4, 6])
    cols_new = np.array([3, 3, 0, 6, 7, 5])
    data_new = np.array([1, 1, 1, 1, 1, 1])
    result_conn_mat = csr_matrix((data_new, (rows_new, cols_new)), shape=(8, 8))

    new_conn_mat = util.remove_ix_from_conn_mat(conn_mat, remove_ix)

    assert np.array_equal(new_conn_mat.todense(), result_conn_mat.todense())


def test_remove_ix_from_conn_mat_no_downslope():

    rows = np.array([0, 1, 2, 3, 4, 5, 7])
    cols = np.array([3, 4, 4, 1, 7, 8, 6])
    data = np.array([1, 1, 1, 1, 1, 1, 1])
    conn_mat = csr_matrix((data, (rows, cols)), shape=(9, 9))
    remove_ix = 8

    rows_new = np.array([0, 1, 2, 3, 4, 7])
    cols_new = np.array([3, 4, 4, 1, 7, 6])
    data_new = np.array([1, 1, 1, 1, 1, 1])
    result_conn_mat = csr_matrix((data_new, (rows_new, cols_new)), shape=(8, 8))

    new_conn_mat = util.remove_ix_from_conn_mat(conn_mat, remove_ix)

    assert np.array_equal(new_conn_mat.todense(), result_conn_mat.todense())


def test_remove_ix_from_conn_mat_both_upslope_and_downslope():

    rows = np.array([0, 1, 2, 3, 4, 5, 7])
    cols = np.array([3, 4, 4, 1, 7, 8, 6])
    data = np.array([1, 1, 1, 1, 1, 1, 1])
    conn_mat = csr_matrix((data, (rows, cols)), shape=(9, 9))
    remove_ix = 4

    rows_new = np.array([0, 1, 2, 3, 4, 6])
    cols_new = np.array([3, 6, 6, 1, 7, 5])
    data_new = np.array([1, 1, 1, 1, 1, 1])
    result_conn_mat = csr_matrix((data_new, (rows_new, cols_new)), shape=(8, 8))

    new_conn_mat = util.remove_ix_from_conn_mat(conn_mat, remove_ix)

    assert np.array_equal(new_conn_mat.todense(), result_conn_mat.todense())
