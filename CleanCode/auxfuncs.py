"""
Auxiliary funtions:
    getValues - Compute zeta, phi, and their dervivatives from the vector of unknowns

"""
import numpy as np

def reshapingUnknowns(u,M,N):
    # get phi1, phix, zeta1, zetax from vector of unknowns
    allPhi = u[:M*(N+1)] # all values of phi1 and phix
    allZet = u[M+N*M:] # all values of zeta1 and zetax
    indx = (N+1)*np.arange(M) # indices of phi1 and zeta1
    phi1 = allPhi[indx] # extract values of phi1
    zet1 = allZet[indx] # extract values of zeta1
    phix = np.delete(allPhi,indx).reshape(M,N) # extract phix as an MxN matrix
    zetx = np.delete(allZet,indx).reshape(M,N) # extract zetax as an MXN matrix
    return phi1, phix, zet1, zetx

def allVals(u1,ux,dx,M,N):
    v1 = u1.reshape(M,1)
    v  = v1 + np.zeros((N,))
    for ii in range(N-1):
        v[:,ii+1] = v[:,ii] + dx/2 * (ux[:,ii+1] + ux[:,ii])
    return v

def xDerivs(v,dx):
    # second order forward differentiation
    dv = -(v[:,2] - 4.*v[:,1] + 3.*v[:,0])/(2.*dx)
    return dv




