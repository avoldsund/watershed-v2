from lib import compare_methods
import numpy as np


def test_compare_two_dictionaries_where_values_are_arrays_one():
    #  Dictionaries have different lengths

    d1 = {0: np.array([1, 2, 3]), 1: np.array([4, 5, 6])}
    d2 = {0: np.array([1, 2, 3])}

    are_equal = compare_methods.compare_two_dictionaries_where_values_are_arrays(d1, d2)

    assert are_equal is False


def test_compare_two_dictionaries_where_values_are_arrays_two():
    # Dictionaries have different keys

    d1 = {0: np.array([1, 2, 3])}
    d2 = {1: np.array([1, 2, 3])}

    are_equal = compare_methods.compare_two_dictionaries_where_values_are_arrays(d1, d2)

    assert are_equal is False


def test_compare_two_dictionaries_where_values_are_arrays_three():
    # The values are in different orders

    d1 = {0: np.array([1, 2, 3])}
    d2 = {1: np.array([1, 3, 2])}

    are_equal = compare_methods.compare_two_dictionaries_where_values_are_arrays(d1, d2)

    assert are_equal is False


def test_compare_two_dictionaries_where_values_are_arrays_four():
    # The dictionaries are equal

    d1 = {0: np.array([1, 2, 3]), 1: np.array([4, 5, 6])}
    d2 = {0: np.array([1, 2, 3]), 1: np.array([4, 5, 6])}

    are_equal = compare_methods.compare_two_dictionaries_where_values_are_arrays(d1, d2)

    assert are_equal


def test_compare_two_lists_of_arrays_one():
    # The lists have different lengths

    l1 = [np.array([1, 2, 3]), np.array([4, 5, 6])]
    l2 = [np.array([1, 2, 3])]

    are_equal = compare_methods.compare_two_lists_of_arrays(l1, l2)

    assert are_equal is False


def test_compare_two_lists_of_arrays_two():
    # The lists have arrays in a different order

    l1 = [np.array([1, 2, 3]), np.array([4, 5, 6])]
    l2 = [np.array([4, 5, 6]), np.array([1, 2, 3])]

    are_equal = compare_methods.compare_two_lists_of_arrays(l1, l2)

    assert are_equal is False


def test_compare_two_lists_of_arrays_three():
    # The lists have arrays with different orderings

    l1 = [np.array([1, 2, 3]), np.array([4, 5, 6])]
    l2 = [np.array([1, 2, 3]), np.array([6, 5, 4])]

    are_equal = compare_methods.compare_two_lists_of_arrays(l1, l2)

    assert are_equal is False


def test_compare_two_lists_of_arrays_four():
    # The lists have arrays of different length

    l1 = [np.array([1, 2, 3]), np.array([4, 5, 6, 7])]
    l2 = [np.array([1, 2, 3]), np.array([4, 5, 6])]

    are_equal = compare_methods.compare_two_lists_of_arrays(l1, l2)

    assert are_equal is False


def test_compare_two_lists_of_arrays_five():
    # The lists are equal

    l1 = [[np.array([1, 2, 3]), np.array([4, 5, 6])]]
    l2 = [[np.array([1, 2, 3]), np.array([4, 5, 6])]]

    are_equal = compare_methods.compare_list_of_lists_by_comparing_sets(l1, l2)

    assert are_equal is True


def test_compare_list_of_lists_by_comparing_sets_one():
    # Different length of the lists

    l1 = [[np.array([1, 2]), np.array([3, 4])],
          [np.array([5, 6]), np.array([7, 8])]]
    l2 = [[np.array([1, 2]), np.array([3, 4])]]

    are_equal = compare_methods.compare_list_of_lists_by_comparing_sets(l1, l2)

    assert are_equal is False


def test_compare_list_of_lists_by_comparing_sets_two():
    # More elements in a list in one of the lists

    l1 = [[np.array([1, 2]), np.array([3, 4])],
          [np.array([5, 6]), np.array([7, 8])]]
    l2 = [[np.array([1, 2]), np.array([3, 4])],
          [np.array([1, 2, 3]), np.array([3, 4, 5])]]

    are_equal = compare_methods.compare_list_of_lists_by_comparing_sets(l1, l2)

    assert are_equal is False


def test_compare_list_of_lists_by_comparing_sets_three():
    # Different lists in the lists

    l1 = [[np.array([1, 2]), np.array([3, 4])]]
    l2 = [[np.array([1, 2]), np.array([4, 3])]]

    are_equal = compare_methods.compare_list_of_lists_by_comparing_sets(l1, l2)

    assert are_equal is False


def test_compare_list_of_lists_by_comparing_sets_four():
    # The lists of lists are equal

    l1 = [[np.array([1, 2]), np.array([3, 4])],
          [np.array([5, 6]), np.array([7, 8])]]
    l2 = [[np.array([1, 2]), np.array([3, 4])],
          [np.array([5, 6]), np.array([7, 8])]]

    are_equal = compare_methods.compare_list_of_lists_by_comparing_sets(l1, l2)

    assert are_equal is True


def test_compare_watersheds_one():
    # Not the same amount of watersheds

    w1 = [np.array([0, 1]), np.array([2, 3])]
    w2 = [np.array([0, 1])]

    are_equal = compare_methods.compare_watersheds(w1, w2)

    assert are_equal is False


def test_compare_watersheds_two():
    # Different order

    w1 = [np.array([0, 1]), np.array([2, 3])]
    w2 = [np.array([2, 3]), np.array([0, 1])]

    are_equal = compare_methods.compare_watersheds(w1, w2)

    assert are_equal is True


def test_compare_watersheds_three():
    # Different elements in the watersheds

    w1 = [np.array([0, 1]), np.array([3, 4])]
    w2 = [np.array([0, 1]), np.array([2, 3])]

    are_equal = compare_methods.compare_watersheds(w1, w2)

    assert are_equal is False


def test_compare_watersheds_four():
    # Different sizes of the watersheds

    w1 = [np.array([0, 1]), np.array([3, 4])]
    w2 = [np.array([0, 1]), np.array([3, 4, 5])]

    are_equal = compare_methods.compare_watersheds(w1, w2)

    assert are_equal is False


def test_compare_watersheds_five():
    # Watersheds are equal

    w1 = [np.array([0, 1]), np.array([2, 3])]
    w2 = [np.array([0, 1]), np.array([2, 3])]

    are_equal = compare_methods.compare_watersheds(w1, w2)

    assert are_equal is True


def test_compare_coordinates_one():
    # Different number of coords

    c1 = (np.array([0, 1]), np.array([2, 3]))
    c2 = (np.array([0, 1, 2]), np.array([2, 3, 4]))

    are_equal = compare_methods.compare_coordinates(c1, c2)

    assert are_equal is False


def test_compare_coordinates_two():
    # Different ordering of coordinates

    c1 = (np.array([0, 1]), np.array([2, 3]))
    c2 = (np.array([1, 0]), np.array([3, 2]))

    are_equal = compare_methods.compare_coordinates(c1, c2)

    assert are_equal is True


def test_compare_coordinates_three():
    # Different lengths of one of the arrays

    c1 = (np.array([0, 1]), np.array([2, 3]))
    c2 = (np.array([0, 1, 2]), np.array([2, 3]))

    are_equal = compare_methods.compare_coordinates(c1, c2)

    assert are_equal is False


def test_compare_coordinates_four():
    # Coordinates are identical

    c1 = (np.array([0, 1]), np.array([2, 3]))
    c2 = (np.array([0, 1]), np.array([2, 3]))

    are_equal = compare_methods.compare_coordinates(c1, c2)

    assert are_equal is True


def test_compare_minima_watersheds_one():
    # Different amounts of minima

    d1 = {0: np.array([0, 1, 2])}
    d2 = {0: np.array([0, 1, 2]), 1: np.array([3, 4, 5])}

    are_equal = compare_methods.compare_minima_watersheds(d1, d2)

    assert are_equal is False


def test_compare_minima_watersheds_two():
    # Different nodes

    d1 = {0: np.array([0, 1, 2])}
    d2 = {0: np.array([3, 4, 5])}

    are_equal = compare_methods.compare_minima_watersheds(d1, d2)

    assert are_equal is False


def test_compare_minima_watersheds_three():
    # Same nodes, different minimum

    d1 = {0: np.array([0, 1, 2])}
    d2 = {1: np.array([0, 1, 2])}

    are_equal = compare_methods.compare_minima_watersheds(d1, d2)

    assert are_equal is False


def test_compare_minima_watersheds_four():
    # They are equal

    d1 = {0: np.array([0, 1, 2]), 1: np.array([3, 4, 5])}
    d2 = {0: np.array([0, 1, 2]), 1: np.array([3, 4, 5])}

    are_equal = compare_methods.compare_minima_watersheds(d1, d2)

    assert are_equal is True
