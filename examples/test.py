starts = [1, 5]
import numpy as np

starts = np.array([1, 5, 13], dtype=int)
ends = np.array([3, 10, 14], dtype=int)
from sorted_nearest.src.sorted_nearest import find_clusters64

print(starts, ends)
for slack in [1, 2, 3]:
    cstarts, cends = find_clusters64(starts, ends, slack=slack)
    print(slack)
    print(cstarts, cends)
