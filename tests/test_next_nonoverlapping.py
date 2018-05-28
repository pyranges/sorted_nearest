
import pytest

from sorted_nearest.src.sorted_nearest import nearest_next_nonoverlapping

import numpy as np


def test_simple_sorted_on_ends():

    l_e = np.array([3, 4, 7, 8])

    r_s = np.array([2, 6, 10])

    idx_r, dist = nearest_next_nonoverlapping(l_e, r_s)

    idx_r, dist = list(idx_r), list(dist)

    print(idx_r)
    print(dist)

    assert idx_r == [1, 1, 2, 2]
    assert dist == [3, 2, 3, 2]


def test_simple_some_missing_next():

    l_e = np.array([3, 4, 7, 18])

    r_s = np.array([5, 8, 11])

    idx_r, dist = nearest_next_nonoverlapping(l_e, r_s)

    idx_r, dist = list(idx_r), list(dist)

    print(idx_r)
    print(dist)

    assert idx_r == [0, 0, 1, -1]
    assert dist == [2, 1, 1, -1]
