from lib import util
import numpy as np
import math

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


def test_create_nbr_connectivity_matrix():

    flow_directions = np.array([[8, -1],
                                [2, -1]])

    conn_mat = np.array([[1, 0, 1, 0],
                         [0, 1, 0, 0],
                         [0, 0, 1, 1],
                         [0, 0, 0, 1]])

    result_conn_mat = util.create_nbr_connectivity_matrix(flow_directions, 2, 2)

    assert np.array_equal(conn_mat, result_conn_mat.todense())


def test_create_nbr_connectivity_matrix_square():

    flow_directions = np.array([[2, 8, 8],
                                [8, 2, -1],
                                [2, 2, 128]])

    conn_mat = np.array([[1, 1, 0, 0, 0, 0, 0, 0, 0],
                         [0, 1, 0, 0, 1, 0, 0, 0, 0],
                         [0, 0, 1, 0, 0, 1, 0, 0, 0],
                         [0, 0, 0, 1, 0, 0, 1, 0, 0],
                         [0, 0, 0, 0, 1, 1, 0, 0, 0],
                         [0, 0, 0, 0, 0, 1, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 1, 1, 0],
                         [0, 0, 0, 0, 0, 0, 0, 1, 1],
                         [0, 0, 0, 0, 0, 1, 0, 0, 1]])

    result_conn_mat = util.create_nbr_connectivity_matrix(flow_directions, 3, 3)

    assert np.array_equal(conn_mat, result_conn_mat.todense())


def test_create_nbr_connectivity_matrix_advanced():

    flow_directions = np.array([[-1, 32, 8, 1],
                                [-1, 32, 4, 8],
                                [128, 64, 2, -1]])

    conn_mat = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                         [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
                         [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
                         [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
                         [0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])

    result_conn_mat = util.create_nbr_connectivity_matrix(flow_directions, 4, 3)

    assert np.array_equal(conn_mat, result_conn_mat.todense())


def test_connect_all_nodes():

    