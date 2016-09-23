import numpy as np


def compare_two_dictionaries_where_values_are_arrays(d1, d2):
    """
    Dictionary with number as key, value is an array.
    :param d1: Dictionary 1.
    :param d2: Dictionary 2.
    :return: True if d1 and d2 are equal, False if not.
    """

    if len(d1) != len(d2):  # Test if the number of keys are not equal
        return False
    else:
        if set(d1) != set(d2):  # Test if the keys are not equal
            return False
        else:
            for key in d1:  # Test if all values in d1 and d2 are equal
                values_are_equal = np.array_equal(d1[key], d2[key])
                if not values_are_equal:
                    return False

    return True


def compare_two_lists_of_arrays(l1, l2):
    """
    List with arrays as elements. Both the arrays, and their elements, must be in the same order.
    Returns True if the two lists have equal elements, and their ordering of them is equal.
    :param l1: List of arrays 1.
    :param l2: List of arrays 2.
    :return: True if the lists are equal, False if not.
    """

    if len(l1) != len(l2):  # Test if the length of the lists are equal
        return False
    else:
        for i in range(len(l1)):  # The arrays must have the same ordering
            arrays_are_equal = np.array_equal(l1[i], l2[i])
            if not arrays_are_equal:
                return False

    return True


def compare_list_of_lists_by_comparing_sets(l1, l2):
    """
    List of lists as elements. The lists must have the same number of lists, and have the same pairs
    when pairing the two arrays in each list.
    Returns True if the lists have the same length and the same pairs.
    :param l1: List of lists with two arrays 1.
    :param l2: List of lists with two arrays 1.
    :return: True if the lists are equal, False if not.
    """

    if len(l1) != len(l2):  # Check if the number of elements is equal
        return False
    else:
        for i in range(len(l1)):  # Convert to sets, check if elements are equal
            l1_set = set(zip(l1[i][0], l1[i][1]))
            l2_set = set(zip(l2[i][0], l2[i][1]))
            if l1_set != l2_set:
                return False

    return True


def compare_watersheds(w1, w2):

    if len(w1) != len(w2):  # Different number of watersheds
        return False
    else:
        w1_set = [set(el) for el in w1]
        w2_set = [set(el) for el in w2]
        for i in range(len(w2)):
            if w2_set[i] not in w1_set:
                return False

    return True


def compare_coordinates(c1, c2):

    if len(c1[0]) != len(c1[1]) or len(c2[0]) != len(c2[1]):
        return False

    c1_set = set(zip(c1[0], c1[1]))
    c2_set = set(zip(c2[0], c2[1]))

    if c1_set == c2_set:
        return True
    else:
        return False


def compare_minima_watersheds(d1, d2):

    if len(d1) != len(d2):
        return False

    for min_index in sorted(d1):
        if min_index in d2:
            ws_1 = np.sort(d1[min_index])
            ws_2 = np.sort(d2[min_index])
        else:
            return False
        if not np.array_equal(ws_1, ws_2):
            return False

    return True


def compare_boundary_pairs(b1, b2):

    if len(b1) != len(b2):  # Boundary pairs for diff number of watersheds
        return False

    for i in range(len(b1)):
        b1_set = set(zip(b1[i][0], b1[i][1]))
        b2_set = set(zip(b2[i][0], b2[i][1]))
        if b1_set != b2_set:
            return False
