import numpy as np

from sorted_nearest import get_all_ties, get_different_ties, nearest_next_nonoverlapping


def strided(values, dtype=np.int64):
    # A column of a row-major 2D array, as pandas >= 3.1 can return from Series.values.
    block = np.zeros((len(values), 3), dtype=dtype)
    block[:, 1] = values
    column = block[:, 1]
    assert not column.flags.c_contiguous
    return column


def test_ties_accept_strided_input():
    lx = np.arange(6, dtype=np.int64)
    ids = [0, 0, 1, 1, 2, 2]
    dist = [1, 1, 2, 3, 0, 0]
    for func in (get_all_ties, get_different_ties):
        expected = func(lx, np.array(ids, dtype=np.int64), np.array(dist, dtype=np.int64), 1)
        assert list(func(lx, strided(ids), strided(dist), 1)) == list(expected)


def test_nearest_accepts_strided_input():
    l_e = [2, 8]
    r_s = [3, 9]
    r_idx = [10, 11]
    contiguous = nearest_next_nonoverlapping(*(np.array(a, dtype=np.int64) for a in (l_e, r_s, r_idx)))
    result = nearest_next_nonoverlapping(strided(l_e), strided(r_s), strided(r_idx))
    for got, want in zip(result, contiguous):
        assert list(got) == list(want)
