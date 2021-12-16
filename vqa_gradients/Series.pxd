# cython: annotation_typing = True
# cython: language_level = 3
import cython

cdef class Series:
    cdef public double[:] x
    cdef public double[:] y
    cdef public int R

    @cython.locals(x=cython.double[:],
                   R=cython.int,
                   series=cython.double[:,:],
                   i=cython.int,
                   j=cython.int)
    cpdef cython.double[:,:] __generate_fourier_term_matrix(self)
