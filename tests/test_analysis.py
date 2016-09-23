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

    upslope_watersheds = analysis.get_upslope_watersheds(conn_mat, w_nr)

    assert sorted(upslope_watersheds) == sorted(result_upslope_watersheds)


def test_get_upslope_watersheds_no_upslope():

    rows = np.array([0, 1, 2, 3, 4, 5, 8])
    cols = np.array([3, 4, 4, 1, 8, 6, 7])
    data = np.array([1, 1, 1, 1, 1, 1, 1])
    conn_mat = csr_matrix((data, (rows, cols)), shape=(9, 9))
    w_nr = 5
    result_upslope_watersheds = [5]

    upslope_watersheds = analysis.get_upslope_watersheds(conn_mat, w_nr)

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
