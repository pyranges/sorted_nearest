from sorted_nearest.src.sorted_nearest import nearest

import pandas as pd
import numpy as np

l_s = np.array([3, 4, 7, 8])
l_e = np.array([4, 5, 9, 11])

r_s = np.array([1, 5, 8])
r_e = np.array([2, 6, 10])

idx_l, idx_r, dist = nearest(l_s, l_e, r_s, r_e)

print(idx_l, idx_r, dist)
