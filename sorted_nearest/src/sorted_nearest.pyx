cimport sorted_nearest.src.csorted_nearest as cn

cimport cython

import numpy as np

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef nearest_next_nonoverlapping(long [::1] l_e, long [::1] r_s, long [::1] r_idx):

    cdef int j = 0
    cdef int i = 0

    cdef int len_l = len(l_e)
    cdef int len_r = len(r_s)

    arr_ridx = np.ones(len_l, dtype=np.long) * -1
    arr_dist = np.ones(len_l, dtype=np.long) * -1
    cdef long [::1] ridx
    cdef long [::1] dist

    ridx = arr_ridx
    dist = arr_dist

    # print("l_e", list(l_e))
    # print("r_s", list(r_s))
    while i < len_l and j < len_r:
        # print("l_e[i]", l_e[i])
        # print("r_s[j]", r_s[j])
        if l_e[i] >= r_s[j]:
            # print("in if")
            j += 1
        else:
            # print("in else")
            dist[i] = r_s[j] - l_e[i]
            ridx[i] = r_idx[j]
            i += 1
        # print("dist", list(dist))
        # print("ridx", list(ridx))

    return arr_ridx, arr_dist



@cython.boundscheck(False)
@cython.wraparound(False)
cpdef nearest_previous_nonoverlapping(long [::1] l_s, long [::1] r_e, long [::1] r_idx):

    cdef int len_l = len(l_s)
    cdef int len_r = len(r_e)

    cdef int j = len_r - 1
    cdef int i = len_l - 1

    arr_ridx = np.ones(len_l, dtype=np.long) * -1
    arr_dist = np.ones(len_l, dtype=np.long) * -1
    cdef long [::1] ridx
    cdef long [::1] dist

    ridx = arr_ridx
    dist = arr_dist

    while -1 < i and -1 < j:
        if l_s[i] <= r_e[j]:
            # print("in if")
            j -= 1
        else:
            # print("in else")
            # print("l_s[i]", l_s[i])
            # print("r_e[j]", r_e[j])
            dist[i] = l_s[i] - r_e[j]
            ridx[i] = r_idx[j]
            i -= 1
        # print("dist", list(dist))
        # print("ridx", list(ridx))

    # print("final dist", list(dist))
    # print("final ridx", list(ridx))

    return arr_ridx, arr_dist


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef nearest_nonoverlapping(long [::1] prev_ridx, long [::1] prev_dist,
                             long [::1] next_ridx, long [::1] next_dist):

    cdef int i = 0

    cdef int length = len(prev_ridx)

    output_arr_ridx = np.ones(length, dtype=np.long) * -1
    output_arr_dist = np.ones(length, dtype=np.long) * -1

    cdef long [::1] output_ridx
    cdef long [::1] output_dist

    output_ridx = output_arr_ridx
    output_dist = output_arr_dist

    for i in range(length):
        if next_ridx[i] == -1 and prev_ridx[i] > -1:
            output_ridx[i] = prev_ridx[i]
            output_dist[i] = prev_dist[i]
        elif prev_ridx[i] > -1 and next_ridx[i] > -1 and prev_dist[i] <= next_dist[i]:
            output_ridx[i] = prev_ridx[i]
            output_dist[i] = prev_dist[i]
        elif next_dist[i] > -1:
            output_ridx[i] = next_ridx[i]
            output_dist[i] = next_dist[i]


    return output_arr_ridx, output_arr_dist


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef merge_sort_overlapping_and_nearest(long [::1] o_lidx, long [::1] o_ridx,
                                         long [::1] n_lidx, long [::1] n_ridx, long[::1] n_dist):

    cpdef int nc = 0
    cpdef int oc = 0
    cpdef int i = 0
    cpdef int nlen = len(n_lidx)
    cpdef int olen = len(o_lidx)
    cpdef int total_len = nlen + olen

    output_arr_ridx = np.ones(total_len, dtype=np.long) * -1
    output_arr_dist = np.zeros(total_len, dtype=np.long)

    cdef long [::1] output_ridx
    cdef long [::1] output_dist

    output_ridx = output_arr_ridx
    output_dist = output_arr_dist

    while nc < nlen and oc < olen:

        if n_lidx[nc] == i:
            output_ridx[i] = n_ridx[nc]
            output_dist[i] = n_dist[nc]
            nc += 1
        elif o_lidx[oc] == i:
            output_ridx[i] = o_ridx[oc]
            oc += 1

        i += 1

    while nc < nlen:
        output_ridx[i] = n_ridx[nc]
        output_dist[i] = n_dist[nc]
        nc += 1
        i += 1

    while oc < olen:
        output_ridx[i] = o_ridx[oc]
        oc += 1
        i += 1

    return output_arr_ridx, output_arr_dist
