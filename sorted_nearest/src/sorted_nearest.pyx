from libc.stdint cimport int32_t

cimport cython

import numpy as np




def find_clusters(starts, ends, slack):

    if starts.dtype == np.long:
        return find_clusters64(starts, ends, slack)
    elif starts.dtype == np.int32:
        return find_clusters32(starts, ends, slack)
    else:
        raise Exception("Starts/Ends not int64 or int32: " + str(starts.dtype))




def nearest_nonoverlapping(prev_ridx, prev_dist,
                           next_ridx, next_dist):

    if prev_dist.dtype == np.int32:
        return nearest_nonoverlapping32(prev_ridx, prev_dist,
                                        next_ridx, next_dist)
    elif prev_dist.dtype == np.int64:
        return nearest_nonoverlapping64(prev_ridx, prev_dist,
                                        next_ridx, next_dist)
    else:
        raise Exception("Starts/Ends not int64 or int32: " + str(prev_dist.dtype))

def nearest_previous_nonoverlapping(ls, re, ridx):

    if ls.dtype == np.int32:
        return nearest_previous_nonoverlapping32(ls, re, ridx)
    elif ls.dtype == np.int64:
        return nearest_previous_nonoverlapping64(ls, re, ridx)
    else:
        raise Exception("Starts/Ends not int64 or int32: " + str(ls.dtype))


def nearest_next_nonoverlapping(le, rs, ridx):

    if le.dtype == np.int32:
        return nearest_next_nonoverlapping32(le, rs, ridx)
    elif le.dtype == np.int64:
        return nearest_next_nonoverlapping64(le, rs, ridx)
    else:
        raise Exception("Starts/Ends not int64 or int32: " + str(le.dtype))


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cpdef nearest_next_nonoverlapping64(const long [::1] l_e, const long [::1] r_s, const long [::1] r_idx):

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
@cython.initializedcheck(False)
cpdef nearest_next_nonoverlapping32(const int32_t [::1] l_e, const int32_t [::1] r_s, const long [::1] r_idx):

    cdef int j = 0
    cdef int i = 0

    cdef int len_l = len(l_e)
    cdef int len_r = len(r_s)

    arr_ridx = np.ones(len_l, dtype=np.long) * -1
    arr_dist = np.ones(len_l, dtype=np.int32) * -1
    cdef long [::1] ridx
    cdef int32_t [::1] dist

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
@cython.initializedcheck(False)
cpdef nearest_previous_nonoverlapping64(const long [::1] l_s, const long [::1] r_e, const long [::1] r_idx):

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
@cython.initializedcheck(False)
cpdef nearest_previous_nonoverlapping32(const int32_t [::1] l_s, const int32_t [::1] r_e, const long [::1] r_idx):

    cdef int len_l = len(l_s)
    cdef int len_r = len(r_e)

    cdef int j = len_r - 1
    cdef int i = len_l - 1

    arr_ridx = np.ones(len_l, dtype=np.long) * -1
    arr_dist = np.ones(len_l, dtype=np.int32) * -1
    cdef long [::1] ridx
    cdef int32_t [::1] dist

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
@cython.initializedcheck(False)
cpdef nearest_nonoverlapping64(const long [::1] prev_ridx, const long [::1] prev_dist,
                               const long [::1] next_ridx, const long [::1] next_dist):

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
@cython.initializedcheck(False)
cpdef nearest_nonoverlapping32(const long [::1] prev_ridx, const int32_t [::1] prev_dist,
                               const long [::1] next_ridx, const int32_t [::1] next_dist):

    cdef int i = 0

    cdef int length = len(prev_ridx)

    output_arr_ridx = np.ones(length, dtype=np.long) * -1
    output_arr_dist = np.ones(length, dtype=np.int32) * -1

    cdef long [::1] output_ridx
    cdef int32_t [::1] output_dist

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
@cython.initializedcheck(False)
cpdef find_clusters64(long [::1] starts, long [::1] ends, int slack):

    cpdef int min_start = starts[0]
    cpdef int max_end = ends[0]
    cpdef int i = 0
    cpdef int n_clusters = 0
    cpdef int length = len(starts)

    output_arr_start = np.ones(length, dtype=np.long) * -1
    output_arr_end = np.zeros(length, dtype=np.long) * -1

    cdef long [::1] output_start
    cdef long [::1] output_end

    output_start = output_arr_start
    output_end = output_arr_end

    for i in range(length):
        if not (starts[i] - slack) <= max_end:
            output_start[n_clusters] = min_start
            output_end[n_clusters] = max_end
            min_start = starts[i]
            max_end = ends[i]
            n_clusters += 1
        else:
            if ends[i] > max_end:
                max_end = ends[i]

    if output_arr_start[n_clusters] != min_start:
        output_arr_start[n_clusters] = min_start
        output_arr_end[n_clusters] = max_end
        n_clusters += 1

    return output_arr_start[:n_clusters], output_arr_end[:n_clusters]


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cpdef find_clusters32(int32_t [::1] starts, int32_t [::1] ends, int slack):

    cpdef int min_start = starts[0]
    cpdef int max_end = ends[0]
    cpdef int i = 0
    cpdef int n_clusters = 0
    cpdef int length = len(starts)

    output_arr_start = np.ones(length, dtype=np.int32) * -1
    output_arr_end = np.zeros(length, dtype=np.int32) * -1

    cdef int32_t [::1] output_start
    cdef int32_t [::1] output_end

    output_start = output_arr_start
    output_end = output_arr_end

    for i in range(length):
        if not (starts[i] - slack) <= max_end:
            output_start[n_clusters] = min_start
            output_end[n_clusters] = max_end
            min_start = starts[i]
            max_end = ends[i]
            n_clusters += 1
        else:
            if ends[i] > max_end:
                max_end = ends[i]

    if output_arr_start[n_clusters] != min_start:
        output_arr_start[n_clusters] = min_start
        output_arr_end[n_clusters] = max_end
        n_clusters += 1

    return output_arr_start[:n_clusters], output_arr_end[:n_clusters]




def makewindows(indexes, starts, ends, window_size, tile=False):

    _starts = starts - (starts % window_size)
    _ends = ends - (ends % window_size) + window_size

    if tile:
        starts, ends = _starts, _ends

    max_n_windows = int((np.sum(_ends - _starts + window_size) / window_size))

    if starts.dtype == np.long:
        return makewindows64(indexes, starts, ends, max_n_windows, window_size)
    elif starts.dtype == np.int32:
        return makewindows32(indexes, starts, ends, max_n_windows, window_size)
    else:
        raise Exception("Starts/Ends not int64 or int32: " + str(starts.dtype))


@cython.boundscheck(True)
@cython.wraparound(True)
@cython.initializedcheck(True)
cpdef makewindows32(long [::1] indexes, int32_t [::1] starts, int32_t [::1] ends, max_n_windows, int window_size):

    cdef:
        int nfound = 0
        int i = 0
        int length = len(starts)
        int start, end

    output_arr_indexes = np.ones(max_n_windows, dtype=long) * -1
    output_arr_start = np.ones(max_n_windows, dtype=np.int32) * -1
    output_arr_end = np.ones(max_n_windows, dtype=np.int32) * -1

    cdef long [::1] output_indexes
    cdef int32_t [::1] output_start
    cdef int32_t [::1] output_end

    output_indexes = output_arr_indexes
    output_start = output_arr_start
    output_end = output_arr_end

    for i in range(length):

        start = starts[i]
        end = ends[i]

        if start + window_size > end:
            output_indexes[nfound] = indexes[i]
            output_start[nfound] = start
            output_end[nfound] = end
            nfound += 1
            continue


        while start + window_size <= end:
            output_indexes[nfound] = indexes[i]
            output_start[nfound] = start
            start += window_size
            output_end[nfound] = start
            nfound += 1

        if start > end:
            output_indexes[nfound] = indexes[i]
            output_start[nfound] = start
            output_end[nfound] = end
            nfound += 1

    return output_arr_indexes[:nfound], output_arr_start[:nfound], output_arr_end[:nfound]



@cython.boundscheck(True)
@cython.wraparound(True)
@cython.initializedcheck(True)
cpdef makewindows64(long [::1] indexes, long [::1] starts, long [::1] ends, max_n_windows, int window_size):

    cdef:
        int nfound = 0
        int i = 0
        int length = len(starts)
        int start, end

    output_arr_indexes = np.ones(max_n_windows, dtype=long) * -1
    output_arr_start = np.ones(max_n_windows, dtype=long) * -1
    output_arr_end = np.ones(max_n_windows, dtype=long) * -1

    cdef long [::1] output_indexes
    cdef long [::1] output_start
    cdef long [::1] output_end

    output_indexes = output_arr_indexes
    output_start = output_arr_start
    output_end = output_arr_end

    for i in range(length):

        # print(i)
        start = starts[i]
        end = ends[i]

        if start + window_size > end:
            output_indexes[nfound] = indexes[i]
            output_start[nfound] = start
            output_end[nfound] = end
            nfound += 1
            continue


        while start + window_size <= end:
            output_indexes[nfound] = indexes[i]
            output_start[nfound] = start
            start += window_size
            output_end[nfound] = start
            nfound += 1

        if start < end:
            output_indexes[nfound] = indexes[i]
            output_start[nfound] = start
            output_end[nfound] = end
            nfound += 1

    return output_arr_indexes[:nfound], output_arr_start[:nfound], output_arr_end[:nfound]
