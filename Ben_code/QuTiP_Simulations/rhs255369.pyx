# This file is generated automatically by QuTiP.
# (C) 2011 and later, QuSTaR

import numpy as np
cimport numpy as np
cimport cython

from qutip.cy.spmatfuncs cimport spmvpy
from qutip.cy.interpolate cimport interp, zinterp
from qutip.cy.math cimport erf
cdef double pi = 3.14159265358979323

include '/home/ben/anaconda3/lib/python3.6/site-packages/qutip/cy/complex_math.pxi'



@cython.boundscheck(False)
@cython.wraparound(False)
def cy_td_ode_rhs(
        double t,
        complex[::1] vec,
        complex[::1] data0,int[::1] idx0,int[::1] ptr0,
        complex[::1] data1,int[::1] idx1,int[::1] ptr1,
        complex[::1] data2,int[::1] idx2,int[::1] ptr2,
        np.ndarray[np.complex128_t, ndim=1] _td_array_0,
        np.ndarray[np.complex128_t, ndim=1] _td_array_1):
    
    cdef size_t row
    cdef unsigned int num_rows = vec.shape[0]
    cdef np.ndarray[complex, ndim=1, mode='c'] out = np.zeros((num_rows),dtype=np.complex)
     
    spmvpy(&data0[0], &idx0[0], &ptr0[0], &vec[0], (0 if (t > 2932.153143) else _td_array_0[int(round(3500 * (t/2932.153143)))]), &out[0], num_rows)
    spmvpy(&data1[0], &idx1[0], &ptr1[0], &vec[0], (0 if (t > 2932.153143) else _td_array_1[int(round(3500 * (t/2932.153143)))]), &out[0], num_rows)
    spmvpy(&data2[0], &idx2[0], &ptr2[0], &vec[0], 1.0, &out[0], num_rows)
    return out
