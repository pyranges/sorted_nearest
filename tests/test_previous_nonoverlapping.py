import pytest

from sorted_nearest.src.sorted_nearest import nearest_previous_nonoverlapping

import numpy as np


def test_simple_sorted_on_ends():

    l_s = np.array([3, 4, 7, 8])

    r_e = np.array([2, 6, 10])

    idx_r, dist = nearest_previous_nonoverlapping(l_s, r_e)

    idx_r, dist = list(idx_r), list(dist)

    print(idx_r)
    print(dist)

    assert idx_r == [0, 0, 1, 1]
    assert dist == [1, 2, 1, 2]



def test_simple_not_sorted_on_ends():

    l_s = np.array([3, 4, 7, 7])

    r_e = np.array([2, 6, 10])

    idx_r, dist = nearest_previous_nonoverlapping(l_s, r_e)

    idx_r, dist = list(idx_r), list(dist)

    print(idx_r)
    print(dist)

    assert idx_r == [0, 0, 1, 1]
    assert dist == [1, 2, 1, 1]


def test_simple_some_missing_previous():

    l_s = np.array([3, 4, 7, 8])

    r_e = np.array([5, 8, 11])

    idx_r, dist = nearest_previous_nonoverlapping(l_s, r_e)

    idx_r, dist = list(idx_r), list(dist)

    print(idx_r)
    print(dist)

    assert idx_r == [-1, -1, 0, 0]
    assert dist == [-1, -1, 2, 3]


def test_simple_some_missing_previous_only_one_r():

    l_s = np.array([3, 4, 7, 8])

    r_e = np.array([5])

    idx_r, dist = nearest_previous_nonoverlapping(l_s, r_e)

    idx_r, dist = list(idx_r), list(dist)

    print(idx_r)
    print(dist)

    assert idx_r == [-1, -1, 0, 0]
    assert dist == [-1, -1, 2, 3]

def test_simple_previous_only_one_r():

    l_s = np.array([7, 8, 10, 12])

    r_e = np.array([5])

    idx_r, dist = nearest_previous_nonoverlapping(l_s, r_e)

    idx_r, dist = list(idx_r), list(dist)

    print(idx_r)
    print(dist)

    assert idx_r == [0, 0, 0, 0]
    assert dist == [2, 3, 5, 7]


def test_simple_previous_only_one_l():

    l_s = np.array([7])

    r_e = np.array([5, 6, 10, 12])

    idx_r, dist = nearest_previous_nonoverlapping(l_s, r_e)

    idx_r, dist = list(idx_r), list(dist)

    print(idx_r)
    print(dist)

    assert idx_r == [1]
    assert dist == [1]

# def test_simple_start_overlaps():

#     l_s = np.array([0, 1, 3, 4, 7, 8])
#     l_e = np.array([1, 2 ,4, 5, 9, 11])

#     r_s = np.array([1, 5, 8])
#     r_e = np.array([2, 6, 10])

#     idx_r, dist = nearest(l_s, l_e, r_s, r_e)

#     idx_r, dist = list(idx_r), list(dist)

#     assert idx_r == [0, 0, 1, 1, 2, 2]
#     assert dist == [0, 0, 1, 0, 0, 0]


# def test_simple_duplicates_left():

#     l_s = np.array([3, 3, 3, 4, 4, 4, 7, 8])
#     l_e = np.array([4, 4, 4, 5, 5, 5, 9, 11])

#     r_s = np.array([1, 5, 8])
#     r_e = np.array([2, 6, 10])

#     idx_r, dist = nearest(l_s, l_e, r_s, r_e)

#     idx_r, dist = list(idx_r), list(dist)

#     assert idx_r == [1, 1, 1, 1, 1, 1, 2, 2]
#     assert dist == [1, 1, 1, 0, 0, 0, 0, 0]



# def test_simple_duplicates_right():

#     l_s = np.array([0, 1, 3, 4, 7, 8])
#     l_e = np.array([1, 2, 4, 5, 9, 11])

#     r_s = np.array([1, 1, 5, 5, 8])
#     r_e = np.array([2, 2, 6, 6, 10])

#     idx_r, dist = nearest(l_s, l_e, r_s, r_e)

#     idx_r, dist = list(idx_r), list(dist)

#     assert idx_r == [0, 0, 3, 3, 4, 4]
#     assert dist == [0, 0, 1, 0, 0, 0]


# def test_simple_duplicates_both():

#     l_s = np.array([0, 0, 1, 3, 3, 4, 7, 8])
#     l_e = np.array([1, 1, 2, 4, 4, 5, 9, 11])

#     r_s = np.array([1, 1, 5, 5, 8])
#     r_e = np.array([2, 2, 6, 6, 10])

#     idx_r, dist = nearest(l_s, l_e, r_s, r_e)

#     idx_r, dist = list(idx_r), list(dist)

#     assert idx_r == [0, 0, 0, 3, 3, 3, 4, 4]
#     assert dist == [0, 0, 0, 1, 1, 0, 0, 0]


# def test_simple_nonoverlapping_last():

#     l_s = np.array([3, 4, 7, 8])
#     l_e = np.array([4, 5, 9, 11])

#     r_s = np.array([1, 5, 18])
#     r_e = np.array([2, 6, 19])

#     idx_r, dist = nearest(l_s, l_e, r_s, r_e)

#     idx_r, dist = list(idx_r), list(dist)

#     assert idx_r == [1, 1, 1, 1]
#     assert dist == [1, 0, 1, 2]



# def test_simple_nonoverlapping_last_long_distance():

#     l_s = np.array([3, 4, 7, 800])
#     l_e = np.array([4, 5, 9, 811])

#     r_s = np.array([1, 5, 18])
#     r_e = np.array([2, 6, 19])

#     idx_r, dist = nearest(l_s, l_e, r_s, r_e)

#     idx_r, dist = list(idx_r), list(dist)

#     assert idx_r == [1, 1, 1, 2]
#     assert dist == [1, 0, 1, 781]


# def test_simple_nonoverlapping_last_long_distance_several():

#     l_s = np.array([3, 4, 7, 800, 800])
#     l_e = np.array([4, 5, 9, 811, 812])

#     r_s = np.array([1, 5, 18])
#     r_e = np.array([2, 6, 19])

#     idx_r, dist = nearest(l_s, l_e, r_s, r_e)

#     idx_r, dist = list(idx_r), list(dist)

#     assert idx_r == [1, 1, 1, 2, 2]
#     assert dist == [1, 0, 1, 781, 781]



# def test_simple_nonoverlapping_last_right_long_distance():

#     l_s = np.array([3, 4, 7, 800, 800])
#     l_e = np.array([4, 5, 9, 811, 812])

#     r_s = np.array([1, 5, 8818])
#     r_e = np.array([2, 6, 8819])

#     idx_r, dist = nearest(l_s, l_e, r_s, r_e)

#     idx_r, dist = list(idx_r), list(dist)

#     assert idx_r == [1, 1, 1, 1, 1]
#     assert dist == [1, 0, 1, 794, 794]
