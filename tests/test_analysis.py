from lib import util, compare_methods, analysis
import numpy as np
from scipy.sparse import csr_matrix


def test_get_upslope_watersheds():

    rows = np.array([0, 1, 2, 3, 4, 5, 8])
    cols = np.array([3, 4, 4, 1, 8, 6, 7])
    data = np.array([1, 1, 1, 1, 1, 1, 1])
    conn_mat = csr_matrix((data, (rows, cols)), shape=(9, 9))
    w_nr = 8
    result_upslope_watersheds = [8, 4, 1, 2, 3, 0]

    upslope_watersheds, node_levels = analysis.get_upslope_watersheds(conn_mat, w_nr)

    assert sorted(upslope_watersheds) == sorted(result_upslope_watersheds)


def test_get_upslope_watersheds_no_upslope():

    rows = np.array([0, 1, 2, 3, 4, 5, 8])
    cols = np.array([3, 4, 4, 1, 8, 6, 7])
    data = np.array([1, 1, 1, 1, 1, 1, 1])
    conn_mat = csr_matrix((data, (rows, cols)), shape=(9, 9))
    w_nr = 5
    result_upslope_watersheds = [5]

    upslope_watersheds, node_levels = analysis.get_upslope_watersheds(conn_mat, w_nr)

    assert sorted(upslope_watersheds) == sorted(result_upslope_watersheds)


def test_get_downslope_watersheds():

    rows = np.array([0, 1, 2, 3, 4, 5, 8])
    cols = np.array([3, 4, 4, 1, 8, 6, 7])
    data = np.array([1, 1, 1, 1, 1, 1, 1])
    conn_mat = csr_matrix((data, (rows, cols)), shape=(9, 9))
    w_nr = 0
    result_downslope_watersheds = [0, 3, 1, 4, 8, 7]

    downslope_watersheds = analysis.get_downslope_watersheds(conn_mat, w_nr)

    assert sorted(downslope_watersheds) == sorted(result_downslope_watersheds)


def test_get_downslope_watersheds_no_downslope():

    rows = np.array([0, 1, 2, 3, 4, 5, 8])
    cols = np.array([3, 4, 4, 1, 8, 6, 7])
    data = np.array([1, 1, 1, 1, 1, 1, 1])
    conn_mat = csr_matrix((data, (rows, cols)), shape=(9, 9))
    w_nr = 6
    result_downslope_watersheds = [6]

    downslope_watersheds = analysis.get_downslope_watersheds(conn_mat, w_nr)

    assert sorted(downslope_watersheds) == sorted(result_downslope_watersheds)


def test_get_rivers_between_spill_points():

    heights = np.array([[10, 10, 10, 10, 10, 10, 10, 10],
                        [10, 6, 5, 6, 10, 1, 0, 0],
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
    spill_heights = np.array([7, 4, 1.5])
    flow_direction_indices = np.array([[None, None, None, None, None, None, None, None],
                                       [None, 10, -1, 10, 13, 14, -1, None],
                                       [None, 10, 10, 10, 21, 14, 14, None],
                                       [None, 33, 34, 35, 29, 21, 22, None],
                                       [None, 42, 42, 42, 37, 29, 30, None],
                                       [None, 50, 50, 50, 45, 37, 38, None],
                                       [None, 50, -1, 50, 53, 45, 46, None],
                                       [None, None, None, None, None, None, None, None]])

    steepest_spill_pairs = [(18, 26), (51, 52), (14, -1)]
    result_rivers = np.array([26, 34, 52, 53, 45, 37])

    rivers = analysis.get_rivers_between_spill_points(watersheds, heights, steepest_spill_pairs,
                                                      spill_heights, flow_direction_indices)

    assert np.array_equal(rivers, result_rivers)


def test_get_river_in_trap():

    cols = 6
    trap = np.array([8, 13, 14, 15, 20, 21, 22, 25, 26, 27, 32, 33, 34])
    start_of_crossing = 13
    end_of_crossing = 34
    result_river = np.array([13, 20, 27, 34])

    river = analysis.get_river_in_trap(trap, start_of_crossing, end_of_crossing, cols)

    assert np.array_equal(river, result_river)


def test_get_river_in_trap_advanced():

    cols = 9
    trap = np.array([11, 12, 13, 14, 15, 20, 21, 23, 24, 25, 28, 29,
                     30, 32, 33, 34, 37, 38, 39, 41, 42, 43, 46, 47,
                     50, 51])
    start_of_crossing = 46
    end_of_crossing = 51
    result_river = np.array([46, 37, 29, 21, 13, 23, 32, 41, 51])

    river = analysis.get_river_in_trap(trap, start_of_crossing, end_of_crossing, cols)
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

    river = analysis.get_river_in_trap(trap, start_of_crossing, end_of_crossing, cols)

    assert np.array_equal(river, result_river)