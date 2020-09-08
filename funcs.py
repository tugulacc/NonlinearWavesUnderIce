# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 04:35:12 2020

@author: Claudia
"""

import numpy as np
#import scipy as sc


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
    # using trapezoidal rule integration
    # compute points on the domain upstream of the boundary
    # WARNING: this function may not give the right results
    v1 = u1.reshape(M,1)
    v  = v1 + np.zeros((N,))
    indx = np.arange(1,N)
    for ii in indx:
            v[:,ii] += (.5*ux[:,0] + ux[:,1:ii].sum(axis=1) + .5*ux[:,ii])*dx
    return v


def halfMesh(v,N):
    # use 2 point interpolation to get half mesh points in x direction
    vHalf = (v[:,1:N] + v[:,0:N-1])/2.
    return vHalf

def yDerivs(v,y,M,N):
    dv = np.zeros((M,N))
    dv[0,:] = (v[1,:] - v[0,:])/y[1]
    return dv

def xDerivs(v,dx):
    # second order forward differentiation
    dv = -(v[:,2] - 4.*v[:,1] + 3.*v[:,0])/(2.*dx)
    return dv

def guessUnknowns(phi1, phix, zeta1, zetax, M,N):
    phis = np.hstack((phi1.reshape(M,1),phix)).reshape(M*(N+1),1)
    zets = np.hstack((zeta1.reshape(M,1),zetax)).reshape(M*(N+1),1)
    u = np.vstack((phis,zets))
    return u

def allVals2(u1,ux,dx,M,N):
    v1 = u1.reshape(M,1)
    v  = v1 + np.zeros((N,))
    for ii in range(N-1):
        v[:,ii+1] = v[:,ii] + dx/2 * (ux[:,ii+1] + ux[:,ii])
    return v
    '''
    for i=1:N-1
    zeta(i+1,:) = zeta(i,:)+deltaX/2 * (zetaX(i+1,:)+zetaX(i,:));
    phi(i+1,:) = phi(i,:)+deltaX/2 * (phiX(i+1,:)+phiX(i,:));
    '''