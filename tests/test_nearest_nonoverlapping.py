import pytest

from sorted_nearest.src.sorted_nearest import nearest_nonoverlapping

import numpy as np


def test_simple():

    l_s = np.array([3, 4, 7, 8])
    l_e = np.array([4, 5, 9, 11])

    r_s = np.array([1, 5, 8])
    r_e = np.array([2, 6, 10])

    idx_r, dist = nearest_nonoverlapping(l_s, l_e, r_s, r_e)

    idx_r, dist = list(idx_r), list(dist)

    assert idx_r == [1, 0, 1, 1]
    assert dist == [1, 2, 1, 2]



def test_simple_nonoverlapping_last_right_long_distance():

    l_s = np.array([3, 4, 7, 800, 800])
    l_e = np.array([4, 5, 9, 811, 812])

    r_s = np.array([1, 5, 8818])
    r_e = np.array([2, 6, 8819])

    idx_r, dist = nearest_nonoverlapping(l_s, l_e, r_s, r_e)

    idx_r, dist = list(idx_r), list(dist)

    assert idx_r == [1, 0, 1, 1, 1]
    assert dist == [1, 2, 1, 794, 794]
