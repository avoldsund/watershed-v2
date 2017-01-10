from lib import util, compare_methods, river_analysis
import numpy as np
from scipy.sparse import csr_matrix


def test_get_upslope_watersheds():

    rows = np.array([0, 1, 2, 3, 4, 5, 8])
    cols = np.array([3, 4, 4, 1, 8, 6, 7])
    data = np.array([1, 1, 1, 1, 1, 1, 1])
    conn_mat = csr_matrix((data, (rows, cols)), shape=(9, 9))
    w_nr = 8
    result_upslope_watersheds = [8, 4, 1, 2, 3, 0]

    upslope_watersheds, node_levels = river_analysis.get_upslope_watersheds(conn_mat, w_nr)

    assert sorted(upslope_watersheds) == sorted(result_upslope_watersheds)


def test_get_upslope_watersheds_no_upslope():

    rows = np.array([0, 1, 2, 3, 4, 5, 8])
    cols = np.array([3, 4, 4, 1, 8, 6, 7])
    data = np.array([1, 1, 1, 1, 1, 1, 1])
    conn_mat = csr_matrix((data, (rows, cols)), shape=(9, 9))
    w_nr = 5
    result_upslope_watersheds = [5]

    # Note that node_levels is None if there is no upslope
    upslope_watersheds, node_levels = river_analysis.get_upslope_watersheds(conn_mat, w_nr)

    assert sorted(upslope_watersheds) == sorted(result_upslope_watersheds)


def test_get_downslope_watersheds():

    rows = np.array([0, 1, 2, 3, 4, 5, 8])
    cols = np.array([3, 4, 4, 1, 8, 6, 7])
    data = np.array([1, 1, 1, 1, 1, 1, 1])
    conn_mat = csr_matrix((data, (rows, cols)), shape=(9, 9))
    w_nr = 0
    result_downslope_watersheds = [0, 3, 1, 4, 8, 7]

    downslope_watersheds = river_analysis.get_downslope_watersheds(conn_mat, w_nr)

    assert sorted(downslope_watersheds) == sorted(result_downslope_watersheds)


def test_get_downslope_watersheds_no_downslope():

    rows = np.array([0, 1, 2, 3, 4, 5, 8])
    cols = np.array([3, 4, 4, 1, 8, 6, 7])
    data = np.array([1, 1, 1, 1, 1, 1, 1])
    conn_mat = csr_matrix((data, (rows, cols)), shape=(9, 9))
    w_nr = 6
    result_downslope_watersheds = [6]

    downslope_watersheds = river_analysis.get_downslope_watersheds(conn_mat, w_nr)

    assert sorted(downslope_watersheds) == sorted(result_downslope_watersheds)


def test_get_rivers():

    heights = np.array([[10, 10, 10, 10, 10, 10, 10, 10],
                        [10, 6, 5, 6, 10, 1, 0, 1.5],
                        [10, 8, 7, 8, 10, 1, 1, 10],
                        [10, 7, 7, 7, 10, 1.5, 1.5, 10],
                        [10, 6, 6, 6, 10, 2, 2, 10],
                        [10, 5, 4, 5, 10, 2.5, 2.5, 10],
                        [10, 4, 2, 4, 4, 3, 3, 10],
                        [10, 10, 10, 10, 10, 10, 10, 10]])
    watersheds = [np.array([9, 10, 11, 17, 18, 19]),
                  np.array([25, 26, 27, 33, 34, 35, 41, 42, 43, 49, 50, 51]),
                  np.array([12, 13, 14, 20, 21, 22, 28, 29, 30, 36, 37, 38,
                            44, 45, 46, 52, 53, 54])]
    traps = [np.array([9, 10, 11, 18]),
             np.array([42, 49, 50, 51]),
             np.array([13, 14, 21, 22, 29, 30])]
    spill_heights = np.array([7, 4, 1.5])
    flow_direction_indices = np.array([[None, None, None, None, None, None, None, None],
                                       [None, 10, -1, 10, 13, 14, -1, None],
                                       [None, 10, 10, 10, 21, 14, 14, None],
                                       [None, 33, 34, 35, 29, 21, 22, None],
                                       [None, 42, 42, 42, 37, 29, 30, None],
                                       [None, 50, 50, 50, 45, 37, 38, None],
                                       [None, 50, -1, 50, 45, 45, 46, None],
                                       [None, None, None, None, None, None, None, None]])

    steepest_spill_pairs = [(18, 26), (51, 52), (14, -1)]
    result_rivers = np.array([26, 34, 52, 45, 37])

    new_watersheds = list(watersheds)
    rivers = river_analysis.get_rivers(watersheds, new_watersheds, steepest_spill_pairs,
                                       traps, flow_direction_indices, heights)
    rivers = np.concatenate(rivers)

    assert np.array_equal(rivers, result_rivers)


def test_get_river_in_trap():

    cols = 6
    trap = np.array([8, 13, 14, 15, 20, 21, 22, 25, 26, 27, 32, 33, 34])
    start_of_crossing = 13
    end_of_crossing = 34
    result_river = np.array([13, 20, 27, 34])

    river = river_analysis.get_river_in_trap(trap, start_of_crossing, end_of_crossing, cols)

    assert np.array_equal(river, result_river)


def test_get_river_in_trap_advanced():

    cols = 9
    trap = np.array([11, 12, 13, 14, 15, 20, 21, 23, 24, 25, 28, 29,
                     30, 32, 33, 34, 37, 38, 39, 41, 42, 43, 46, 47,
                     50, 51])
    start_of_crossing = 46
    end_of_crossing = 51
    result_river = np.array([46, 37, 29, 21, 13, 23, 32, 41, 51])

    river = river_analysis.get_river_in_trap(trap, start_of_crossing, end_of_crossing, cols)
    print river
    assert np.array_equal(river, result_river)


def test_get_river_in_trap_advanced_2():

    cols = 9
    trap = np.array([11, 12, 13, 14, 15, 20, 21, 23, 24, 25, 28, 29,
                     30, 32, 33, 34, 37, 38, 39, 41, 42, 43, 46, 47,
                     50, 51])
    start_of_crossing = 47
    end_of_crossing = 15
    result_river = np.array([47, 38, 29, 21, 13, 14, 15])

    river = river_analysis.get_river_in_trap(trap, start_of_crossing, end_of_crossing, cols)

    assert np.array_equal(river, result_river)


def test_calculate_flow_origins():
    # Note that trap 36 is flowing to 18, and not to 9

    rows = 6
    cols = 6
    row = np.array([36, 37, 19, 20, 21, 14, 15, 22, 28, 9, 38])
    col = np.array([18, 22, 38, 38, 38, 15, 37, 28, 38, 37, 31])
    data = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    traps = [np.array([7, 8, 13]), np.array([10, 16]), np.array([25, 26, 27])]

    nr_of_traps = 3
    nr_of_nodes = rows * cols
    size_with_trap_nodes = nr_of_traps + nr_of_nodes

    expanded_conn_mat = csr_matrix((data, (row, col)), shape=(size_with_trap_nodes, size_with_trap_nodes))

    start_nodes = river_analysis.calculate_flow_origins(expanded_conn_mat, traps, rows, cols)
    print start_nodes
    result_start_nodes = np.array([9, 14, 19, 20, 21, 36])

    assert np.array_equal(start_nodes, result_start_nodes)


"""
def test_calculate_nr_of_upslope_cells_river():

    depressionless_heights = np.array([[5, 5, 5],
                                       [5, 4, 5],
                                       [5, 3, 5],
                                       [5, 2, 5],
                                       [5, 1, 5],
                                       [5, 0, 5]])
    step_size = 10

    row = np.array([4, 7, 10])
    col = np.array([7, 10, 13])
    data = np.array([1, 1, 1])
    node_conn_mat = csr_matrix((data, (row, col)), shape=(18, 18))
    steepest_spill_pairs = []

    result_flow_acc = np.array([[None, None, None],
                                [None, 1, None],
                                [None, 2, None],
                                [None, 3, None],
                                [None, 4, None],
                                [None, None, None]])

    flow_acc = river_analysis.calculate_nr_of_upslope_cells(depressionless_heights, steepest_spill_pairs,
                                                            node_conn_mat, step_size)

    assert np.array_equal(flow_acc, result_flow_acc)


def test_calculate_nr_of_upslope_cells():

    depressionless_heights = np.array([[5, 5, 5],
                                       [5, 3, 5],
                                       [5, 3, 5],
                                       [5, 2, 5],
                                       [5, 2, 5],
                                       [5, 1, 5]])

    row = np.array([7])
    col = np.array([10])
    data = np.array([1])

    node_conn_mat = csr_matrix((data, (row, col)), shape=(18, 18))

    step_size = 10
    traps = [np.array([10, 13]), np.array([4])]
    steepest_spill_pairs = [(13, 16), (4, 7)]

    result_flow_acc = np.array([[None, None, None],
                                [None, 1, None],
                                [None, 2, None],
                                [None, 4, None],
                                [None, 4, None],
                                [None, None, None]])

    flow_acc = river_analysis.calculate_nr_of_upslope_cells(depressionless_heights, steepest_spill_pairs, node_conn_mat, step_size)

    assert np.array_equal(flow_acc, result_flow_acc)


def test_calculate_nr_of_upslope_cells_larger():

    depressionless_heights = np.array([[5, 5, 5, 5, 5],
                                       [5, 5, 5, 5, 5],
                                       [5, 3, 2.5, 2.7, 5],
                                       [5, 4, 2.3, 2.4, 5],
                                       [5, 5, 2.3, 5, 5]])

    row = np.array([6, 7, 8, 11, 12, 13, 16, 18])
    col = np.array([11, 12, 13, 12, 17, 18, 17, 17])
    data = np.array([1, 1, 1, 1, 1, 1, 1, 1])

    node_conn_mat = csr_matrix((data, (row, col)), shape=(25, 25))

    step_size = 10
    traps = [np.array([10, 13]), np.array([4])]
    spill_points = [(13, 16), (4, 7)]
    steepest_spill_pairs = []

    result_flow_acc = np.array([[None, None, None, None, None],
                                [None, 1, 1, 1, None],
                                [None, 2, 4, 2, None],
                                [None, 1, 9, 3, None],
                                [None, None, None, None, None]])

    flow_acc = river_analysis.calculate_nr_of_upslope_cells(depressionless_heights, steepest_spill_pairs, node_conn_mat, step_size)

    assert np.array_equal(flow_acc, result_flow_acc)

"""


def test_calculate_nr_of_upslope_cells_three_traps():

    depressionless_heights = np.array([[10, 10, 10, 10, 10, 10],
                                       [10, 9, 9, 9, 7, 10],
                                       [10, 9, 10, 9, 7, 10],
                                       [10, 10, 10, 10, 7, 10],
                                       [10, 4, 4, 4, 4.5, 10],
                                       [10, 4, 10, 10, 10, 10]])

    row = np.array([9, 14, 15, 22, 28, 19, 20, 21])
    col = np.array([10, 15, 16, 28, 27, 25, 26, 27])
    data = np.array([1, 1, 1, 1, 1, 1, 1, 1])
    total_nodes = 36

    rows = 6
    cols = 6

    node_conn_mat = csr_matrix((data, (row, col)), shape=(total_nodes, total_nodes))
    steepest_spill_pairs = [(8, 9), (16, 22), (26, 31)]

    traps = [np.array([7, 8, 13]), np.array([10, 16]), np.array([25, 26, 27])]

    result_flow_acc = np.array([[0, 0, 0, 0, 0, 0],
                                [0, 3, 3, 4, 8, 0],
                                [0, 3, 1, 2, 8, 0],
                                [0, 1, 1, 1, 9, 0],
                                [0, 16, 16, 16, 10, 0],
                                [0, 0, 0, 0, 0, 0]])

    flow_acc = river_analysis.calculate_nr_of_upslope_cells(node_conn_mat, rows, cols, traps, steepest_spill_pairs)

    assert np.array_equal(flow_acc, result_flow_acc)


def test_calculate_nr_of_upslope_cells_three_traps_diverging():

    depressionless_heights = np.array([[10, 10, 10, 10, 10, 10],
                                       [10, 9, 9, 9, 7, 10],
                                       [10, 9, 10, 9, 7, 10],
                                       [10, 10, 10, 10, 7, 10],
                                       [10, 4, 4, 4, 4.5, 10],
                                       [10, 4, 10, 10, 10, 10]])

    row = np.array([9, 14, 15, 22, 28, 19, 20, 21])
    col = np.array([10, 15, 16, 28, 27, 25, 26, 27])
    data = np.array([1, 1, 1, 1, 1, 1, 1, 1])
    total_nodes = 36

    rows = 6
    cols = 6

    node_conn_mat = csr_matrix((data, (row, col)), shape=(total_nodes, total_nodes))
    steepest_spill_pairs = [(8, 9), (16, 22), (26, 31)]

    traps = [np.array([7, 8, 13]), np.array([10, 16]), np.array([25, 26, 27])]

    result_flow_acc = np.array([[0, 0, 0, 0, 0, 0],
                                [0, 3, 3, 4, 8, 0],
                                [0, 3, 1, 2, 8, 0],
                                [0, 1, 1, 1, 9, 0],
                                [0, 16, 16, 16, 10, 0],
                                [0, 0, 0, 0, 0, 0]])

    flow_acc = river_analysis.calculate_nr_of_upslope_cells(node_conn_mat, rows, cols, traps, steepest_spill_pairs)

    assert np.array_equal(flow_acc, result_flow_acc)


def test_calculate_nr_of_upslope_cells_one_trap():

    depressionless_heights = np.array([[10, 10, 10, 10, 10],
                                       [10, 5, 5, 5, 10],
                                       [10, 5, 1, 5, 10],
                                       [10, 10, 1, 10, 10]])
    rows = 4
    cols = 5

    row = np.array([6, 7, 8, 11, 13])
    col = np.array([12, 12, 12, 12, 12])
    data = np.array([1, 1, 1, 1, 1])
    total_nodes = 20

    node_conn_mat = csr_matrix((data, (row, col)), shape=(total_nodes, total_nodes))
    steepest_spill_pairs = [(12, 17)]

    traps = [np.array([12])]

    result_flow_acc = np.array([[0, 0, 0, 0, 0],
                                [0, 1, 1, 1, 0],
                                [0, 1, 6, 1, 0],
                                [0, 0, 0, 0, 0]])

    flow_acc = river_analysis.calculate_nr_of_upslope_cells(node_conn_mat, rows, cols, traps, steepest_spill_pairs)

    assert np.array_equal(flow_acc, result_flow_acc)


def test_expand_conn_mat_basic():

    row = np.array([1, 2])
    col = np.array([2, 3])
    data = np.array([1, 1])
    total_nodes = 4
    conn_mat = csr_matrix((data, (row, col)), shape=(total_nodes, total_nodes))

    nr_of_traps = 2
    result_expanded_conn_mat = csr_matrix((data, (row, col)), shape=(total_nodes + nr_of_traps,
                                                                     total_nodes + nr_of_traps))

    expanded_conn_mat = river_analysis.expand_conn_mat(conn_mat, nr_of_traps)

    assert np.array_equal(expanded_conn_mat.todense(), result_expanded_conn_mat.todense())


def test_expand_conn_mat():

    row = np.array([9, 14, 15, 22, 28, 19, 20, 21])
    col = np.array([10, 15, 16, 28, 27, 25, 26, 27])
    data = np.array([1, 1, 1, 1, 1, 1, 1, 1])
    total_nodes = 36
    conn_mat = csr_matrix((data, (row, col)), shape=(total_nodes, total_nodes))

    traps = 3
    result_node_conn_mat = csr_matrix((data, (row, col)), shape=(total_nodes + traps, total_nodes + traps))

    node_conn_mat = river_analysis.expand_conn_mat(conn_mat, traps)

    assert np.array_equal(node_conn_mat.todense(), result_node_conn_mat.todense())


def test_reroute_trap_connections():

    # NB: Flow to boundary has been removed for this method
    # Input to function
    row = np.array([9, 14, 15, 22, 28, 19, 20, 21])
    col = np.array([10, 15, 16, 28, 27, 25, 26, 27])
    data = np.array([1, 1, 1, 1, 1, 1, 1, 1])
    nr_of_traps = 3
    nr_of_nodes = 36
    rows = 6
    cols = 6
    size_with_trap_nodes = nr_of_traps + nr_of_nodes
    conn_mat = csr_matrix((data, (row, col)), shape=(size_with_trap_nodes, size_with_trap_nodes))
    steepest_spill_pairs = [(8, 9), (16, 22), (26, 31)]
    traps = [np.array([7, 8, 13]), np.array([10, 16]), np.array([25, 26, 27])]

    # Result
    row = np.array([36, 37, 19, 20, 21, 14, 15, 22, 28, 9])
    col = np.array([9, 22, 38, 38, 38, 15, 37, 28, 38, 37])
    data = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    result_conn_mat = csr_matrix((data, (row, col)), shape=(size_with_trap_nodes, size_with_trap_nodes))

    new_conn_mat = river_analysis.reroute_trap_connections(conn_mat, rows, cols, traps, steepest_spill_pairs)

    assert np.array_equal(new_conn_mat.todense(), result_conn_mat.todense())


def test_reroute_trap_connections_two_directions():

    # Input to function
    row = np.array([9, 14, 15, 22, 28, 19, 20, 21])
    col = np.array([10, 15, 16, 28, 27, 25, 26, 27])
    data = np.array([1, 1, 1, 1, 1, 1, 1, 1])
    nr_of_traps = 3
    nr_of_nodes = 36
    rows = 6
    cols = 6
    size_with_trap_nodes = nr_of_traps + nr_of_nodes
    conn_mat = csr_matrix((data, (row, col)), shape=(size_with_trap_nodes, size_with_trap_nodes))
    steepest_spill_pairs = [(13, 18), (16, 22), (26, 31)]
    traps = [np.array([7, 8, 13]), np.array([10, 16]), np.array([25, 26, 27])]

    # Result
    row = np.array([37, 19, 20, 21, 14, 15, 22, 28, 9])
    col = np.array([22, 38, 38, 38, 15, 37, 28, 38, 37])
    data = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1])
    result_conn_mat = csr_matrix((data, (row, col)), shape=(size_with_trap_nodes, size_with_trap_nodes))

    new_conn_mat = river_analysis.reroute_trap_connections(conn_mat, rows, cols, traps, steepest_spill_pairs)

    assert np.array_equal(new_conn_mat.todense(), result_conn_mat.todense())


def test_assign_initial_flow_acc():

    rows = 6
    cols = 6
    traps = [np.array([7, 8, 13]), np.array([10, 16]), np.array([25, 26, 27])]
    start_nodes = np.array([9, 14, 19, 20, 21, 36])

    result_initial_flow_acc = np.array([0, 0, 0, 0, 0, 0,
                                        0, 0, 0, 1, 0, 0,
                                        0, 0, 1, 0, 0, 0,
                                        0, 1, 1, 1, 0, 0,
                                        0, 0, 0, 0, 0, 0,
                                        0, 0, 0, 0, 0, 0,
                                        3, 0, 0])
    result_one_or_trap_size = np.array([1, 1, 1, 1, 1, 1,
                                        1, 1, 1, 1, 1, 1,
                                        1, 1, 1, 1, 1, 1,
                                        1, 1, 1, 1, 1, 1,
                                        1, 1, 1, 1, 1, 1,
                                        1, 1, 1, 1, 1, 1,
                                        3, 2, 3])

    initial_flow_acc, one_or_trap_size = river_analysis.assign_initial_flow_acc(traps, start_nodes, rows, cols)

    assert np.array_equal(initial_flow_acc, result_initial_flow_acc) and \
           np.array_equal(one_or_trap_size, result_one_or_trap_size)


def test_get_trap_boundary():

    nx = 8
    ny = 7
    traps = [np.array([9, 10, 17]),
             np.array([12, 13, 14, 20, 21, 22, 28, 29, 30]),
             np.array([26, 34]),
             np.array([44])]

    result_traps_boundary = [np.array([9, 10, 17]),
                             np.array([12, 13, 14, 20, 22, 28, 29, 30]),
                             np.array([26, 34]),
                             np.array([44])]

    traps_boundaries = river_analysis.get_traps_boundaries(traps, nx, ny)

    assert compare_methods.compare_two_lists_of_arrays(traps_boundaries, result_traps_boundary)


def test_get_trap_boundary_advanced():

    nx = 87
    ny = 87
    traps = [np.array([6774, 6861, 6862, 6948, 6949, 6950, 7035, 7036, 7037, 7038,
                       7123, 7124, 7125, 7126, 7210, 7211, 7212, 7213, 7297, 7298,
                       7299, 7300, 7386, 7387, 7473, 7474])]

    result_traps_boundary = [np.array([6774, 6861, 6862, 6948, 6949, 6950, 7035, 7036, 7037, 7038,
                                       7123, 7125, 7126, 7210, 7213, 7297, 7298,
                                       7299, 7300, 7386, 7387, 7473, 7474])]

    print 'result_traps_boundary: ', result_traps_boundary
    traps_boundaries = river_analysis.get_traps_boundaries(traps, nx, ny)
    print 'traps_boundaries: ', traps_boundaries

    assert compare_methods.compare_two_lists_of_arrays(traps_boundaries, result_traps_boundary)
