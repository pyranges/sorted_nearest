cimport sorted_nearest.src.csorted_nearest as cn

cimport cython

import numpy as np

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef nearest(long [::1] l_s, long [::1] l_e, long [::1] r_s, long [::1] r_e):

    cdef int _continue = 1
    cdef int new_diff
    cdef int old_diff = cn.INT_MAX
    cdef int j = 0
    cdef int i = 0
    cdef int ZERO = 0
    cdef int length_l = len(l_s)
    cdef int length_r_minus_one = len(r_s) - 1

    output_arr_ridx = np.zeros(length_l, dtype=np.long)
    output_arr_dist = np.zeros(length_l, dtype=np.long)
    cdef long [::1] output_ridx
    cdef long [::1] output_dist

    output_ridx = output_arr_ridx
    output_dist = output_arr_dist

    while _continue:
        # print("while j == 0")
        if l_e[i] < r_s[0]:
            # print("  l_e[i] < r_s[0]")
            old_diff = r_s[0] - l_e[i]
            _continue = 0
        elif l_s[i] > r_e[0]:
            # print("  l_s[i] > r_e[0]")
            old_diff = l_s[i] - r_e[0]
            _continue = 0
        else: # overlapping
            # print("  overlapping")
            output_ridx[i] = j
            output_dist[i] = 0
            i += 1

    while i < length_l and j < length_r_minus_one:

        # print("i is", i, "and j + 1 is", j + 1)
        # print("left start, end: ", l_s[i], l_e[i])
        # print("right start, end: ", r_s[j + 1], r_e[j + 1])

        if l_e[i] < r_s[j + 1]:
            # print("l_e[i] < r_s[j + 1]")
            new_diff = r_s[j + 1] - l_e[i]
            # print("  New diff:", new_diff, "old diff:", old_diff)
            if new_diff > old_diff:
                # print("  Pushing", i, j, old_diff)
                output_ridx[i] = j
                output_dist[i] = old_diff
                old_diff = cn.INT_MAX
                j -= 1
                i += 1
            else:
                j += 1
                old_diff = new_diff

        elif l_s[i] > r_e[j + 1]:
            # print("l_s[i] > r_e[j + 1]")
            new_diff = l_s[i] - r_e[j + 1]
            # print("  New diff:", new_diff, "old diff:", old_diff)
            if new_diff > old_diff:
                # print("  Pushing", i, j, old_diff)
                output_ridx[i] = j
                output_dist[i] = old_diff
                old_diff = cn.INT_MAX
                j -= 1
                i += 1
            else:
                j += 1
                old_diff = new_diff
        else: # overlapping
            # print("else")
            # print("  Pushing", i, j, 0)
            j += 1
            output_ridx[i] = j
            output_dist[i] = 0
            old_diff = cn.INT_MAX
            j -= 1
            i += 1


    cdef int second_last = len(r_s) - 2
    cdef int last = len(r_s) - 1
    cdef int sl_dist, l_dist
    while i < length_l:
        # print("we are here")
        if l_e[i] < r_s[second_last]:
            # print("l_e[i] < r_s[second_last]")
            # print(l_e[i], r_s[second_last])
            sl_dist = r_s[second_last] - l_e[i]
        elif l_s[i] > r_e[second_last]:
            # print("l_s[i] > r_e[second_last]")
            # print(l_s[i], r_e[second_last])
            sl_dist = l_s[i] - r_e[second_last]
        else:
            sl_dist = 0

        if l_e[i] < r_s[last]:
            # print("l_e[i] < r_s[last]")
            # print(l_e[i], r_s[last])
            l_dist = r_s[last] - l_e[i]
        elif l_s[i] > r_e[last]:
            # print("l_s[i] > r_e[last]")
            # print(l_s[i], r_e[last])
            l_dist = l_s[i] - r_e[last]
        else:
            # print("else ldist=0")
            l_dist = 0

        # print("sl_dist", sl_dist, "l_dist", l_dist)
        if sl_dist < l_dist:
            # print(" Pushing", second_last, sl_dist)
            output_ridx[i] = second_last
            output_dist[i] = sl_dist
        else:
            # print(" Pushing", last, l_dist)
            output_ridx[i] = last
            output_dist[i] = l_dist

        i += 1

    return output_arr_ridx, output_arr_dist
