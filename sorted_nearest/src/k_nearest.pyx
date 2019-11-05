from libc.stdint cimport int32_t

cimport cython

import numpy as np


def nearest_previous_nonoverlapping(ls, re, lidx, ridx, k, ties=None):

    if ls.dtype == np.int32 and not ties:
        return nearest_previous_nonoverlapping32_all(ls, re, lidx, ridx, k)
    elif ls.dtype == np.int32 and ties == "first":
        return nearest_previous_nonoverlapping32_first(ls, re, lidx, ridx, k)
    elif ls.dtype == np.int32 and ties == "different":
        res = nearest_previous_nonoverlapping32_k_distances(ls, re, lidx, ridx, k)
        return res
    else:
        raise Exception("Starts/Ends not int64 or int32: " + str(ls.dtype))

def nearest_next_nonoverlapping(le, rs, lidx, ridx, k, ties=None):

    if le.dtype == np.int32 and not ties:
        return nearest_next_nonoverlapping32_all(le, rs, lidx, ridx, k)
    elif le.dtype == np.int32 and ties == "first":
        return nearest_next_nonoverlapping32_first(le, rs, lidx, ridx, k)
    elif le.dtype == np.int32 and ties == "different":
        res = nearest_next_nonoverlapping32_k_distances(le, rs, lidx, ridx, k)
        return res
    else:
        raise Exception("Starts/Ends not int64 or int32: " + str(le.dtype))



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cpdef nearest_previous_nonoverlapping32_all(const int32_t [::1] l_s, const int32_t [::1] r_e,
                                            const long [::1] l_idx, const long [::1] r_idx, const long [::1] k):

    cdef int len_l = len(l_s)
    # cdef int len_rx = len(r_idx)
    cdef int len_r = len(r_e)

    cdef int j = 0 # counter for r_e
    cdef int i = 0 # counter for l_s
    cdef int h = 0

    cdef int outarr_length = len_l
    cdef int nfound = 0
    cdef int nfound_local = 0
    cdef int last = -1
    cdef int _dist = -1

    arr_lidx = np.ones(len_l, dtype=np.long) * -1
    arr_ridx = np.ones(len_l, dtype=np.long) * -1
    arr_dist = np.ones(len_l, dtype=np.int32) * -1

    cdef long [::1] lidx
    cdef long [::1] ridx
    cdef int32_t [::1] dist

    lidx = arr_lidx
    ridx = arr_ridx
    dist = arr_dist

    while i < len_l and j < len_r:

        h = r_idx[j]
        nfound_local = 0
        j += 1
        last = -1

        while h >= 0 and nfound_local < k[i]:

            if nfound == outarr_length - 1:

                outarr_length = outarr_length * 2
                arr_lidx = np.resize(arr_lidx, outarr_length)
                lidx = arr_lidx
                arr_ridx = np.resize(arr_ridx, outarr_length)
                ridx = arr_ridx
                arr_dist = np.resize(arr_dist, outarr_length)
                dist = arr_dist

            ridx[nfound] = h
            lidx[nfound] = l_idx[i]
            dist[nfound] = l_s[i] - r_e[h]

            nfound += 1

            if r_e[h] != last:
                nfound_local += 1

            last = r_e[h]
            h -= 1

        i += 1

    return arr_lidx[:nfound], arr_ridx[:nfound], arr_dist[:nfound]


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cpdef nearest_next_nonoverlapping32_all(const int32_t [::1] l_e, const int32_t [::1] r_s,
                                        const long [::1] l_idx, const long [::1] r_idx, const long [::1] k):

    cdef int len_l = len(l_e)
    cdef int len_rx = len(r_idx)
    cdef int len_r = len(r_s)

    cdef int j = 0 
    cdef int i = 0
    cdef int h = 0

    cdef int outarr_length = len_l
    cdef int nfound = 0
    cdef int nfound_local = 0
    cdef int last = -1
    cdef int _dist = -1

    arr_lidx = np.ones(len_l, dtype=np.long) * -1
    arr_ridx = np.ones(len_l, dtype=np.long) * -1
    arr_dist = np.ones(len_l, dtype=np.int32) * -1

    cdef long [::1] lidx
    cdef long [::1] ridx
    cdef int32_t [::1] dist

    lidx = arr_lidx
    ridx = arr_ridx
    dist = arr_dist

    while i < len_l and j < len_rx:

        h = r_idx[j]
        nfound_local = 0
        j += 1
        last = -1

        while h < len_r and nfound_local < k[i]:

            if nfound == outarr_length - 1:

                outarr_length = outarr_length * 2
                arr_lidx = np.resize(arr_lidx, outarr_length)
                lidx = arr_lidx
                arr_ridx = np.resize(arr_ridx, outarr_length)
                ridx = arr_ridx
                arr_dist = np.resize(arr_dist, outarr_length)
                dist = arr_dist

            ridx[nfound] = h
            lidx[nfound] = l_idx[i]
            dist[nfound] = r_s[h] - l_e[i]

            nfound += 1

            if r_s[h] != last:
                nfound_local += 1

            last = r_s[h]
            h += 1

        i += 1

    return arr_lidx[:nfound], arr_ridx[:nfound], arr_dist[:nfound]



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cpdef nearest_previous_nonoverlapping32_k_distances(const int32_t [::1] l_s, const int32_t [::1] r_e,
                                                    const long [::1] l_idx, const long [::1] r_idx, const long [::1] k):

    cdef int len_l = len(l_s)
    # cdef int len_rx = len(r_idx)
    cdef int len_r = len(r_e)

    cdef int j = 0 # counter for r_e
    cdef int i = 0 # counter for l_s
    cdef int h = 0

    cdef int outarr_length = len_l
    cdef int nfound = 0
    cdef int nfound_local = 0
    cdef int last = -1
    cdef int last_dist = -1
    cdef int curr_dist = -1

    arr_lidx = np.ones(len_l, dtype=np.long) * -1
    arr_ridx = np.ones(len_l, dtype=np.long) * -1
    arr_dist = np.ones(len_l, dtype=np.int32) * -1

    cdef long [::1] lidx
    cdef long [::1] ridx
    cdef int32_t [::1] dist

    lidx = arr_lidx
    ridx = arr_ridx
    dist = arr_dist

    while i < len_l and j < len_r:

        h = r_idx[j]
        nfound_local = 0
        j += 1
        last_dist = -1

        while h >= 0 and nfound_local < k[i]:

            if nfound == outarr_length - 1:

                outarr_length = outarr_length * 2
                arr_lidx = np.resize(arr_lidx, outarr_length)
                lidx = arr_lidx
                arr_ridx = np.resize(arr_ridx, outarr_length)
                ridx = arr_ridx
                arr_dist = np.resize(arr_dist, outarr_length)
                dist = arr_dist

            curr_dist = l_s[i] - r_e[h]
            ridx[nfound] = h
            lidx[nfound] = l_idx[i]
            dist[nfound] = curr_dist

            nfound += 1

            if curr_dist != last_dist:
                nfound_local += 1

            curr_dist = last_dist

            last = r_e[h]
            h -= 1

        i += 1

    return arr_lidx[:nfound], arr_ridx[:nfound], arr_dist[:nfound]


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cpdef nearest_next_nonoverlapping32_k_distances(const int32_t [::1] l_e, const int32_t [::1] r_s,
                                        const long [::1] l_idx, const long [::1] r_idx, const long [::1] k):

    cdef int len_l = len(l_e)
    cdef int len_rx = len(r_idx)
    cdef int len_r = len(r_s)

    cdef int j = 0
    cdef int i = 0
    cdef int h = 0

    cdef int outarr_length = len_l
    cdef int nfound = 0
    cdef int nfound_local = 0
    cdef int last = -1
    cdef int curr_dist = -1
    cdef int last_dist = -1


    arr_lidx = np.ones(len_l, dtype=np.long) * -1
    arr_ridx = np.ones(len_l, dtype=np.long) * -1
    arr_dist = np.ones(len_l, dtype=np.int32) * -1

    cdef long [::1] lidx
    cdef long [::1] ridx
    cdef int32_t [::1] dist

    lidx = arr_lidx
    ridx = arr_ridx
    dist = arr_dist

    while i < len_l and j < len_rx:

        h = r_idx[j]
        nfound_local = 0
        j += 1
        last = -1

        while h < len_r and nfound_local < k[i]:

            if nfound == outarr_length - 1:

                outarr_length = outarr_length * 2
                arr_lidx = np.resize(arr_lidx, outarr_length)
                lidx = arr_lidx
                arr_ridx = np.resize(arr_ridx, outarr_length)
                ridx = arr_ridx
                arr_dist = np.resize(arr_dist, outarr_length)
                dist = arr_dist

            curr_dist = r_s[h] - l_e[i]

            ridx[nfound] = h
            lidx[nfound] = l_idx[i]
            dist[nfound] = curr_dist

            nfound += 1

            if curr_dist != last_dist:
                nfound_local += 1

            last_dist = curr_dist

            last = r_s[h]
            h += 1

        i += 1

    return arr_lidx[:nfound], arr_ridx[:nfound], arr_dist[:nfound]


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cpdef nearest_next_nonoverlapping32_first(const int32_t [::1] l_e, const int32_t [::1] r_s,
                                        const long [::1] l_idx, const long [::1] r_idx, const long [::1] k):

    # print("nearest_next_nonoverlapping first")

    cdef int len_l = len(l_e)
    cdef int len_rx = len(r_idx)
    cdef int len_r = len(r_s)

    cdef int j = 0
    cdef int i = 0
    cdef int h = 0

    cdef int outarr_length = len_l
    cdef int nfound = 0
    cdef int nfound_local = 0
    cdef int last = -1
    cdef int curr_dist = -1
    cdef int last_dist = -1


    arr_lidx = np.ones(len_l, dtype=np.long) * -1
    arr_ridx = np.ones(len_l, dtype=np.long) * -1
    arr_dist = np.ones(len_l, dtype=np.int32) * -1

    cdef long [::1] lidx
    cdef long [::1] ridx
    cdef int32_t [::1] dist

    lidx = arr_lidx
    ridx = arr_ridx
    dist = arr_dist

    # print("i", i)
    # print("len_l", len_l)
    # print("len_rx", len_rx)
    # print("j", j)
    while i < len_l and j < len_rx:

        # print("nearest next first i", i)

        h = r_idx[j]
        nfound_local = 0
        j += 1
        last = -1

        while h < len_r and nfound_local < k[i]:

            if nfound == outarr_length - 1:

                outarr_length = outarr_length * 2
                arr_lidx = np.resize(arr_lidx, outarr_length)
                lidx = arr_lidx
                arr_ridx = np.resize(arr_ridx, outarr_length)
                ridx = arr_ridx
                arr_dist = np.resize(arr_dist, outarr_length)
                dist = arr_dist

            curr_dist = r_s[h] - l_e[i]

            if curr_dist != last_dist:
                ridx[nfound] = h
                lidx[nfound] = l_idx[i]
                dist[nfound] = curr_dist

                nfound += 1

                nfound_local += 1

                last_dist = curr_dist

            last = r_s[h]
            h += 1

        i += 1

    return arr_lidx[:nfound], arr_ridx[:nfound], arr_dist[:nfound]


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cpdef nearest_previous_nonoverlapping32_first(const int32_t [::1] l_s, const int32_t [::1] r_e,
                                              const long [::1] l_idx, const long [::1] r_idx,
                                              const long [::1] k):

    cdef int len_l = len(l_s)
    # cdef int len_rx = len(r_idx)
    cdef int len_r = len(r_e)

    cdef int j = 0 # counter for r_e
    cdef int i = 0 # counter for l_s
    cdef int h = 0

    cdef int outarr_length = len_l
    cdef int nfound = 0
    cdef int nfound_local = 0
    cdef int last = -1
    cdef int last_dist = -1
    cdef int curr_dist = -1

    arr_lidx = np.ones(len_l, dtype=np.long) * -1
    arr_ridx = np.ones(len_l, dtype=np.long) * -1
    arr_dist = np.ones(len_l, dtype=np.int32) * -1

    cdef long [::1] lidx
    cdef long [::1] ridx
    cdef int32_t [::1] dist

    lidx = arr_lidx
    ridx = arr_ridx
    dist = arr_dist

    while i < len_l and j < len_r:

        h = r_idx[j]
        nfound_local = 0
        j += 1
        last_dist = -1

        while h >= 0 and nfound_local < k[i]:

            if nfound == outarr_length - 1:

                outarr_length = outarr_length * 2
                arr_lidx = np.resize(arr_lidx, outarr_length)
                lidx = arr_lidx
                arr_ridx = np.resize(arr_ridx, outarr_length)
                ridx = arr_ridx
                arr_dist = np.resize(arr_dist, outarr_length)
                dist = arr_dist

            curr_dist = l_s[i] - r_e[h]

            if curr_dist != last_dist:
                ridx[nfound] = h
                lidx[nfound] = l_idx[i]
                dist[nfound] = curr_dist
                last_dist = curr_dist

                nfound += 1

                nfound_local += 1

            last = r_e[h]
            h -= 1

        i += 1

    return arr_lidx[:nfound], arr_ridx[:nfound], arr_dist[:nfound]
