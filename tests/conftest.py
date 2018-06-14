from hypothesis import given
from hypothesis.extra.pandas import data_frames, columns, range_indexes
import hypothesis.strategies as st


positions = st.integers(min_value=0, max_value=int(1e7))


def mysort(pos1, pos2):

    if pos1 > pos2:
        return pos2, pos1
    elif pos2 > pos1:
        return pos1, pos2
    else:
        return pos1, pos2 + 1


dfs = data_frames(columns=columns("Start End".split(), dtype=int),
                  rows=st.tuples(positions, positions).map(mysort))
