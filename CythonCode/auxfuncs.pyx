"""
Auxiliary funtions:
    getValues - Compute zeta, phi, and their dervivatives from the vector of unknowns

"""
import numpy as np
cimport numpy as np
cimport cython

DTYPE=np.float64
ctypedef np.float64_t DTYPE_t

@cython.boundscheck(False) #turn off boundary checking for function
def reshapingUnknowns(np.ndarray[DTYPE_t, ndim=2] u,int M,int N):
    # get phi1, phix, zeta1, zetax from vector of unknowns
    cdef np.ndarray[DTYPE_t, ndim=2] allPhi
    cdef np.ndarray[DTYPE_t, ndim=2] allZet
    cdef np.ndarray[np.int32_t, ndim=1] indx
    cdef np.ndarray[DTYPE_t, ndim=2] phi1
    cdef np.ndarray[DTYPE_t, ndim=2] zet1
    cdef np.ndarray[DTYPE_t, ndim=2] phix
    cdef np.ndarray[DTYPE_t, ndim=2] zetx

    allPhi = u[:M*(N+1)] # all values of phi1 and phix
    allZet = u[M+N*M:] # all values of zeta1 and zetax
    indx = (N+1)*np.arange(M) # indices of phi1 and zeta1
    phi1 = allPhi[indx] # extract values of phi1
    zet1 = allZet[indx] # extract values of zeta1
    phix = np.delete(allPhi,indx).reshape(M,N) # extract phix as an MxN matrix
    zetx = np.delete(allZet,indx).reshape(M,N) # extract zetax as an MXN matrix
    return phi1, phix, zet1, zetx

@cython.boundscheck(False) #turn off boundary checking for function
def allVals(np.ndarray[DTYPE_t, ndim=2] u1,np.ndarray[DTYPE_t, ndim=2] ux,float dx,int M,int N):
    cdef np.ndarray[DTYPE_t, ndim=2] v1
    cdef np.ndarray[DTYPE_t, ndim=2] v
    cdef int ii

    v1 = u1.reshape(M,1)
    v  = v1 + np.zeros((N),dtype=DTYPE)
    for ii in range(N-1):
        v[:,ii+1] = v[:,ii] + dx/2 * (ux[:,ii+1] + ux[:,ii])
    return v

@cython.boundscheck(False) #turn off boundary checking for function
def xDerivs(np.ndarray[DTYPE_t, ndim=2] v,float dx):
    # second order forward differentiation
    cdef np.ndarray[DTYPE_t, ndim=1] dv

    dv = -(v[:,2] - 4.*v[:,1] + 3.*v[:,0])/(2.*dx)
    return dv




