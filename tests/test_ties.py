import pytest

import pandas as pd
import numpy as np 

from io import StringIO

from sorted_nearest import get_all_ties, get_different_ties


@pytest.fixture
def data():
    c = """ids dist
1 1
1 1
1 2
1 3
0 5000
0 5000
0 5000
2 100
2 110
2 110
2 111
3 111
3 111
3 111
3 112
3 112
3 113
4 112
4 113
4 113
4 113
4 113
4 113
4 113
4 150"""

    df = pd.read_table(StringIO(c), header=0, sep="\s+")

    return df

def test_get_all_ties(data):

    df = data

    print(df)

    k = 2
    result = get_all_ties(df.index.values, df.ids.values, df.dist.values, k)
    print(df.reindex(result))
    print(result)

    expected = [0, 1, 4, 5, 6, 7, 8, 9, 11, 12, 13, 17, 18, 19, 20, 21, 22, 23]
    assert list(result) == expected

def test_get_different_ties(data):

    df = data

    k = 2
    result = get_different_ties(df.index.values, df.ids.values, df.dist.values, k)
    print(df.reindex(result))
    print(result)
    assert list(result) == [0, 1, 2, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 17, 18, 19, 20, 21, 22, 23]
