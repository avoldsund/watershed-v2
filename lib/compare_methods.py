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
