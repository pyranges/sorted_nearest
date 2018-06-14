import pytest

from sorted_nearest.src.sorted_nearest import nearest
from hypothesis import given

from conftest import dfs

import numpy as np

@given(dfs, dfs)
def test_simple(df1, df2):

    l_s = df1.Start.values
    l_e = df1.End.values

    r_s = df2.Start.values
    r_e = df2.End.values

    idx_r, dist = nearest(l_s, l_e, r_s, r_e)

    idx_r, dist = list(idx_r), list(dist)
