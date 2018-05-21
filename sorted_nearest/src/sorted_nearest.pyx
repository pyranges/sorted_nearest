cimport sorted_nearest.src.csorted_nearest as cn

cimport cython

import numpy as np

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef nearest(long [::1] l_s, long [::1] l_e, long [::1] r_s, long [::1] r_e):

    cdef int _continue = 1
    cdef int new_diff
    cdef int old_diff = cn.INT_MAX
    cdef unsigned int j = 0
    cdef unsigned int i = 0
    cdef int ZERO = 0

    cdef cn.UT_array *idx_left
    cn.utarray_new(idx_left, &(cn.ut_int_icd))

    cdef cn.UT_array *idx_right
    cn.utarray_new(idx_right, &(cn.ut_int_icd))

    cdef cn.UT_array *dist
    cn.utarray_new(dist, &(cn.ut_int_icd))

    while _continue:
        print("while j == 0")
        if l_e[i] < r_s[0]:
            print("  l_e[i] < r_s[0]")
            old_diff = r_s[0] - l_e[i]
            _continue = 0
        elif l_s[i] > r_e[0]:
            print("  l_s[i] > r_e[0]")
            old_diff = l_s[i] - r_e[0]
            _continue = 0
        else: # overlapping
            print("  overlapping")
            cn.utarray_push_back(idx_left, &(i))
            cn.utarray_push_back(idx_right, &(j)) # should always be zero in this loop
            cn.utarray_push_back(dist, &(ZERO))
            i += 1

    while i < len(l_s) and j < (len(r_s) - 1):

        print("i is", i, "and j + 1 is", j + 1)
        print("left start, end: ", l_s[i], l_e[i])
        print("right start, end: ", r_s[j + 1], r_e[j + 1])

        if l_e[i] < r_s[j + 1]:
            print("l_e[i] < r_s[j + 1]")
            new_diff = r_s[j + 1] - l_e[i]
            print("  New diff:", new_diff, "old diff:", old_diff)
            if new_diff > old_diff:
                print("  Pushing", i, j, old_diff)
                cn.utarray_push_back(idx_left, &(i))
                cn.utarray_push_back(idx_right, &(j))
                cn.utarray_push_back(dist, &(old_diff))
                old_diff = cn.INT_MAX
                j -= 1
                i += 1
            else:
                j += 1
                old_diff = new_diff

        elif l_s[i] > r_e[j + 1]:
            print("l_s[i] > r_e[j + 1]")
            new_diff = l_s[i] - r_e[j + 1]
            print("  New diff:", new_diff, "old diff:", old_diff)
            if new_diff > old_diff:
                print("  Pushing", i, j, old_diff)
                cn.utarray_push_back(idx_left, &(i))
                cn.utarray_push_back(idx_right, &(j))
                cn.utarray_push_back(dist, &(old_diff))
                old_diff = cn.INT_MAX
                j -= 1
                i += 1
            else:
                j += 1
                old_diff = new_diff
        else: # overlapping
            print("else")
            print("  Pushing", i, j, 0)
            cn.utarray_push_back(idx_left, &(i))
            j += 1
            cn.utarray_push_back(idx_right, &(j))
            cn.utarray_push_back(dist, &(ZERO))
            old_diff = cn.INT_MAX
            j -= 1
            i += 1




    cdef int *arr_left_idx
    cdef int *arr_right_idx
    cdef int *arr_dist

    length = cn.utarray_len(idx_left)

    arr_left_idx = cn.utarray_eltptr(idx_left, 0)
    arr_right_idx = cn.utarray_eltptr(idx_right, 0)
    arr_dist = cn.utarray_eltptr(dist, 0)

    output_arr_lidx = np.zeros(length, dtype=np.long)
    output_arr_ridx = np.zeros(length, dtype=np.long)
    output_arr_dist = np.zeros(length, dtype=np.long)
    cdef long [::1] output_lidx
    cdef long [::1] output_ridx
    cdef long [::1] output_dist

    output_lidx = output_arr_lidx
    output_ridx = output_arr_ridx
    output_dist = output_arr_dist

    i = 0
    for i in range(length):
        output_arr_lidx[i] = arr_left_idx[i]
        output_arr_ridx[i] = arr_right_idx[i]
        output_arr_dist[i] = arr_dist[i]

    cn.utarray_free(idx_left)
    cn.utarray_free(idx_right)
    cn.utarray_free(dist)

    return output_arr_lidx, output_arr_ridx, output_arr_dist


    # while i < len(l_s):

    #     old_diff =

    #     if l_e[i] < r_s[j + 1]:
    #         new_diff = r_s[j + 1] - l_e[i]
    #         if new_diff > old_diff:
    #             cn.utarray_push_back(idx_left, &(i))
    #             cn.utarray_push_back(idx_right, &(j))
    #             cn.utarray_push_back(dist, &(old_diff))
    #             old_diff = cn.INT_MAX
    #             j -= 1
    #             i += 1
    #         else:
    #             j += 1
    #             old_diff = new_diff

    #     elif l_s[i] > r_e[j + 1]:
    #         new_diff = l_s[i] - r_e[j + 1]
    #         if new_diff > old_diff:
    #             cn.utarray_push_back(idx_left, &(i))
    #             cn.utarray_push_back(idx_right, &(j))
    #             cn.utarray_push_back(dist, &(old_diff))
    #             old_diff = cn.INT_MAX
    #             j -= 1
    #             i += 1
    #         else:
    #             j += 1
    #             old_diff = new_diff
    #     else: # overlapping
    #         cn.utarray_push_back(idx_left, &(i))
    #         cn.utarray_push_back(idx_right, &(j))
    #         cn.utarray_push_back(dist, &(ZERO))
    #         old_diff = cn.INT_MAX
    #         j -= 1
    #         i += 1

    # # if there are i-s left, means that they are either closest to j - 1 or j
    # # does not need to be very efficient
    # # while i < len(l_s):

    # #     # know that i is closer to j-1 than j due to sort
    # #     if l_e[i] < r_s[j - 1]:
    # #         new_diff = r_s[j - 1] - l_e[i]
    # #         cn.utarray_push_back(idx_left, &(i))
    # #         cn.utarray_push_back(idx_right, &(j - 1))
    # #         cn.utarray_push_back(dist, &(new_diff))
    # #         i += 1
    # #     elif l_s[i] > r_e[j - 1]:
    # #         if l_s[i] > r_e[j]:
    # #             new_diff = l_s[i] - r_e[j]
    # #             cn.utarray_push_back(idx_left, &(i))
    # #             cn.utarray_push_back(idx_right, &(j))
    # #             cn.utarray_push_back(dist, &(new_diff))
    # #         elif r_s[j] > l_e[i]:

    # #         i += 1



    # #     # starts behind the backmost so j is the closest
    # #     elif l_s[i] > r_e[j]:
