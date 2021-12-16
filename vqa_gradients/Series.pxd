# cython: annotation_typing = True
# cython: language_level = 3
# cython: boundscheck=True
import cython

cdef class Series:
    cdef public double[:] x
    cdef public complex[:] y
    cdef public int R
    def public series

    @cython.locals(x=double[:],
                   R=int,
                   series=double[:,:],
                   i=int,
                   j=int)
    cpdef double[:,:] __generate_fourier_term_matrix(self)

