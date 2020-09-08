# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#import scipy as sc
import numpy as np
from scipy.sparse import block_diag

def TridiMatrix(N,M,dx,x1,n):
    E34 = np.zeros((2,N+1))
    E34[0,[0,1]]=[n,x1]
    E34[1,[1,2]]=[n-x1/dx,x1/dx]

    dphix = (np.eye(N-1,N,k=0) + np.eye(N-1,N,k=1))/2.
    dphi = np.zeros((N-1,1))
    block = np.vstack((E34, np.hstack((dphi,dphix))))
    A = block_diag((block,)*M).toarray()
    if A.size != ((N+1)*M)**2:
        print(f"shape of matrix: {A.shape}")
    return A

def BlockMatrix(N,M,dx,const):
    T = np.tri(N-1)
    T += 2.*np.tril(T,-1) + np.tril(T,-2)
    v = np.ones((N-1,1))
    v[1:,:] *= 2.
    block = const*np.hstack((np.ones((N-1,1)),dx/4.*np.hstack((v,T))))
    block = np.vstack((np.zeros((2,(N+1))),block))
    return block_diag((block,)*M).toarray()

def integ(x,y,xH):
    # analytical solution 
    F1 = lambda s, t: t*(np.log(s+np.sqrt(s**2+t**2))+np.log(2))
    F2 = lambda s, t: s*(np.log(t+np.sqrt(s**2+t**2))+np.log(2))
    EVL= lambda f, t: f(sN, t) - f(s1, t)
    # xH = (x[1:]+x[:-1])/2.    
    slim = [[x[0]],[x[-1]]] - xH
    tlim = y[[0,-1],:][:,None] - y

    s1, sN = (slim[0], slim[1])
    t1, tM = (tlim[0], tlim[1])


    I = EVL(F2,tM) - EVL(F2,t1)
    # define masks
    mt1 = t1!=0.
    mtM = tM!=0.

    I[np.ravel(mt1),:] -= EVL(F1, t1[mt1][:,None])
    I[np.ravel(mtM),:] += EVL(F1, tM[mtM][:,None])

    I += EVL(F1,tM+2*y) + EVL(F2,tM+2*y) - EVL(F2,t1+2*y)

    mt0 = t1+2*y!=0.
    I[np.ravel(mt0),:] -= EVL(F1, (t1+2*y)[mt0][:,None])
    
    return I

def denseMat(x,y,dx,dy,N,M,n):

    xH = (x[1:]+x[0:-1])/2.
    I = integ(x,y,xH)
    
    all_K3 = None
    
    for jj in range(M):
        a = np.zeros((2,N*M+M))
        zind = jj*(N+1)
        a[0,[zind,zind+1]]=[n,x[0]]
        a[1,[zind+1,zind+2]]=[n-x[0]/dx,x[0]/dx]
        for ii in range(N-1):
 
            # calculate K3(x,y;x*,y*)
            denom = lambda c: 1./np.sqrt((x-xH[ii])**2+(y-c*y[jj])**2)
            K3 = denom(1.) + denom(-1.)
            #print(f"(k={ii},l={jj})\n {I[jj,ii]}")
            
            # Apply weighting function
            K3[:,::N-1] /= 2. 
            K3[::M-1,:] /= 2.
            K3 *= -dx*dy
            # Calculate sum
            Sum = -K3.sum()
            
            #print(f"contition satisfied: {k3[jj,ii:ii+2]}\n")
            K3[jj, ii:ii+2] += (Sum - I[jj,ii])/2.
            
            # combine
            K3_new = np.hstack((np.zeros((M,1)),K3)).reshape((1,(N+1)*M))
            if ii % (N-1) == 0:
                K3_new = np.vstack((a,K3_new))
            
            if all_K3 is None:
                all_K3 = K3_new
            else:
                all_K3 = np.vstack((all_K3,K3_new))
                
    return all_K3

def Jacobian(x,y,dx,dy,N,M,F,n):
    A = TridiMatrix(N,M,dx,x[0],n)
    B = BlockMatrix(N,M,dx,F**(-2.))
    C = BlockMatrix(N,M,dx,2.*np.pi)
    D = denseMat(x,y,dx,dy,N,M,n)
    J = np.block([[A,B],[C,D]])
    return J