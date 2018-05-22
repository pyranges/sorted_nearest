import pytest

from sorted_nearest.src.sorted_nearest import nearest

import numpy as np

def test_simple():

    l_s = np.array([3, 4, 7, 8])
    l_e = np.array([4, 5, 9, 11])

    r_s = np.array([1, 5, 8])
    r_e = np.array([2, 6, 10])

    idx_r, dist = nearest(l_s, l_e, r_s, r_e)

    idx_r, dist = list(idx_r), list(dist)

    assert idx_r == [1, 1, 2, 2]
    assert dist == [1, 0, 0, 0]


def test_simple_start_overlaps():

    l_s = np.array([0, 1, 3, 4, 7, 8])
    l_e = np.array([1, 2 ,4, 5, 9, 11])

    r_s = np.array([1, 5, 8])
    r_e = np.array([2, 6, 10])

    idx_r, dist = nearest(l_s, l_e, r_s, r_e)

    idx_r, dist = list(idx_r), list(dist)

    assert idx_r == [0, 0, 1, 1, 2, 2]
    assert dist == [0, 0, 1, 0, 0, 0]


def test_simple_duplicates_left():

    l_s = np.array([3, 3, 3, 4, 4, 4, 7, 8])
    l_e = np.array([4, 4, 4, 5, 5, 5, 9, 11])

    r_s = np.array([1, 5, 8])
    r_e = np.array([2, 6, 10])

    idx_r, dist = nearest(l_s, l_e, r_s, r_e)

    idx_r, dist = list(idx_r), list(dist)

    assert idx_r == [1, 1, 1, 1, 1, 1, 2, 2]
    assert dist == [1, 1, 1, 0, 0, 0, 0, 0]



def test_simple_duplicates_right():

    l_s = np.array([0, 1, 3, 4, 7, 8])
    l_e = np.array([1, 2, 4, 5, 9, 11])

    r_s = np.array([1, 1, 5, 5, 8])
    r_e = np.array([2, 2, 6, 6, 10])

    idx_r, dist = nearest(l_s, l_e, r_s, r_e)

    idx_r, dist = list(idx_r), list(dist)

    assert idx_r == [0, 0, 3, 3, 4, 4]
    assert dist == [0, 0, 1, 0, 0, 0]


def test_simple_duplicates_both():

    l_s = np.array([0, 0, 1, 3, 3, 4, 7, 8])
    l_e = np.array([1, 1, 2, 4, 4, 5, 9, 11])

    r_s = np.array([1, 1, 5, 5, 8])
    r_e = np.array([2, 2, 6, 6, 10])

    idx_r, dist = nearest(l_s, l_e, r_s, r_e)

    idx_r, dist = list(idx_r), list(dist)

    assert idx_r == [0, 0, 0, 3, 3, 3, 4, 4]
    assert dist == [0, 0, 0, 1, 1, 0, 0, 0]


def test_simple_nonoverlapping_last():

    l_s = np.array([3, 4, 7, 8])
    l_e = np.array([4, 5, 9, 11])

    r_s = np.array([1, 5, 18])
    r_e = np.array([2, 6, 19])

    idx_r, dist = nearest(l_s, l_e, r_s, r_e)

    idx_r, dist = list(idx_r), list(dist)

    assert idx_r == [1, 1, 1, 1]
    assert dist == [1, 0, 1, 2]



def test_simple_nonoverlapping_last_long_distance():

    l_s = np.array([3, 4, 7, 800])
    l_e = np.array([4, 5, 9, 811])

    r_s = np.array([1, 5, 18])
    r_e = np.array([2, 6, 19])

    idx_r, dist = nearest(l_s, l_e, r_s, r_e)

    idx_r, dist = list(idx_r), list(dist)

    assert idx_r == [1, 1, 1, 2]
    assert dist == [1, 0, 1, 781]


def test_simple_nonoverlapping_last_long_distance_several():

    l_s = np.array([3, 4, 7, 800, 800])
    l_e = np.array([4, 5, 9, 811, 812])

    r_s = np.array([1, 5, 18])
    r_e = np.array([2, 6, 19])

    idx_r, dist = nearest(l_s, l_e, r_s, r_e)

    idx_r, dist = list(idx_r), list(dist)

    assert idx_r == [1, 1, 1, 2, 2]
    assert dist == [1, 0, 1, 781, 781]



def test_simple_nonoverlapping_last_right_long_distance():

    l_s = np.array([3, 4, 7, 800, 800])
    l_e = np.array([4, 5, 9, 811, 812])

    r_s = np.array([1, 5, 8818])
    r_e = np.array([2, 6, 8819])

    idx_r, dist = nearest(l_s, l_e, r_s, r_e)

    idx_r, dist = list(idx_r), list(dist)

    assert idx_r == [1, 1, 1, 1, 1]
    assert dist == [1, 0, 1, 794, 794]
