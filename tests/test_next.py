
import pytest

from sorted_nearest.src.sorted_nearest import nearest_next

import numpy as np


def test_simple_sorted_on_ends():

    l_s = np.array([3, 4, 7, 8])
    l_e = np.array([4, 5, 9, 11])

    r_s = np.array([1, 5, 8])
    r_e = np.array([2, 6, 10])

    idx_r, dist = nearest_next(l_s, l_e, r_s, r_e)

    idx_r, dist = list(idx_r), list(dist)

    print(idx_r)
    print(dist)

    assert idx_r == [1, 1, 2, 2]
    assert dist == [1, 0, 0, 0]


def test_simple2():

    l_s = np.array([0, 1, 3, 4, 7, 8])
    l_e = np.array([1, 2 ,4, 5, 9, 11])

    r_s = np.array([1, 5, 8])
    r_e = np.array([2, 6, 10])

    idx_r, dist = nearest_next(l_s, l_e, r_s, r_e)

    idx_r, dist = list(idx_r), list(dist)

    print(idx_r)
    print(dist)

    assert idx_r == [0, 0, 1, 1, 2, 2]
    assert dist == [0, 0, 1, 0, 0, 0]



def test_simple_some_missing_next():

    l_s = np.array([3, 4, 7, 800, 800])
    l_e = np.array([4, 5, 9, 811, 812])

    r_s = np.array([1, 5, 18])
    r_e = np.array([2, 6, 19])

    idx_r, dist = nearest_next(l_s, l_e, r_s, r_e)

    idx_r, dist = list(idx_r), list(dist)

    assert idx_r == [1, 1, 2, -1, -1]
    assert dist == [1, 0, 9, -1, -1]
