import pytest

from sorted_nearest.src.sorted_nearest import nearest

import numpy as np

def test_simple():

    l_s = np.array([3, 4, 7, 8])
    l_e = np.array([4, 5, 9, 11])

    r_s = np.array([1, 5, 8])
    r_e = np.array([2, 6, 10])

    idx_l, idx_r, dist = nearest(l_s, l_e, r_s, r_e)

    idx_l, idx_r, dist = list(idx_l), list(idx_r), list(dist)

    assert idx_l == [0, 1, 2, 3]
    assert idx_r == [1, 1, 2, 2]
    assert dist == [1, 0, 0, 0]


def test_simple_start_overlaps():

    l_s = np.array([0, 1, 3, 4, 7, 8])
    l_e = np.array([1, 2 ,4, 5, 9, 11])

    r_s = np.array([1, 5, 8])
    r_e = np.array([2, 6, 10])

    idx_l, idx_r, dist = nearest(l_s, l_e, r_s, r_e)

    idx_l, idx_r, dist = list(idx_l), list(idx_r), list(dist)

    assert idx_r == [0, 0, 1, 1, 2, 2]
    assert idx_l == [0, 1, 2, 3, 4, 5]
    assert dist == [0, 0, 1, 0, 0, 0]


def test_simple_duplicates_left():

    l_s = np.array([3, 3, 3, 4, 4, 4, 7, 8])
    l_e = np.array([4, 4, 4, 5, 5, 5, 9, 11])

    r_s = np.array([1, 5, 8])
    r_e = np.array([2, 6, 10])

    idx_l, idx_r, dist = nearest(l_s, l_e, r_s, r_e)

    idx_l, idx_r, dist = list(idx_l), list(idx_r), list(dist)

    assert idx_r == [1, 1, 1, 1, 1, 1, 2, 2]
    assert idx_l == [0, 1, 2, 3, 4, 5, 6, 7]
    assert dist == [1, 1, 1, 0, 0, 0, 0, 0]



def test_simple_duplicates_right():

    l_s = np.array([0, 1, 3, 4, 7, 8])
    l_e = np.array([1, 2, 4, 5, 9, 11])

    r_s = np.array([1, 1, 5, 5, 8])
    r_e = np.array([2, 2, 6, 6, 10])

    idx_l, idx_r, dist = nearest(l_s, l_e, r_s, r_e)

    idx_l, idx_r, dist = list(idx_l), list(idx_r), list(dist)

    assert idx_r == [0, 0, 3, 3, 4, 4]
    assert idx_l == [0, 1, 2, 3, 4, 5]
    assert dist == [0, 0, 1, 0, 0, 0]
