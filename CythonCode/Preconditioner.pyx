"""
This is where I compute the Jacobian of the linear problem
used to generate the preconditioner
"""
import numpy as np
cimport numpy as np
cimport cython
import scipy as sc
from scipy.sparse import block_diag
import icefuncs # for ice preconditioner

DTYPE=np.float64
ctypedef np.float64_t DTYPE_t

@cython.boundscheck(False) #turn off boundary checking for function
def TridiMatrix(int N,int M,float dx,float x1,float n):
    cdef np.ndarray[DTYPE_t, ndim=2] dphix
    cdef np.ndarray[DTYPE_t, ndim=2] dphi
    cdef np.ndarray[DTYPE_t, ndim=2] block
    E34 = np.zeros((2,N+1),dtype=DTYPE)
    E34[0,[0,1]]=[n,x1]
    E34[1,[1,2]]=[n-x1/dx,x1/dx]

    dphix = (np.eye(N-1,N,k=0) + np.eye(N-1,N,k=1))/2.
    dphi = np.zeros((N-1,1),dtype=DTYPE)
    block = np.vstack((E34, np.hstack((dphi,dphix))))
    A = block_diag((block,)*M).toarray()
    if A.size != ((N+1)*M)**2:
        print(f"shape of matrix: {A.shape}")
    return A

@cython.boundscheck(False) #turn off boundary checking for function
def BlockMatrix(int N,int M,float dx,float constant):
    cdef np.ndarray[DTYPE_t, ndim=2] T
    cdef np.ndarray[DTYPE_t, ndim=2] v
    cdef np.ndarray[DTYPE_t, ndim=2] block
    T = np.tri(N-1)
    T += 2.*np.tril(T,-1) + np.tril(T,-2)
    v = np.ones((N-1,1),dtype=DTYPE)
    v[1:,:] *= 2.
    block = constant*np.hstack((np.ones((N-1,1),dtype=DTYPE),dx/4.*np.hstack((v,T))))
    block = np.vstack((np.zeros((2,(N+1)),dtype=DTYPE),block))
    return block_diag((block,)*M).toarray()

@cython.boundscheck(False) #turn off boundary checking for function
def integ(np.ndarray[DTYPE_t, ndim=1] x,np.ndarray[DTYPE_t, ndim=2] y,np.ndarray[DTYPE_t, ndim=1] xH):
    cdef np.ndarray[DTYPE_t, ndim=2] slim
    cdef np.ndarray[DTYPE_t, ndim=3] tlim
    cdef np.ndarray[DTYPE_t, ndim=2] t1
    cdef np.ndarray[DTYPE_t, ndim=2] tM

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

@cython.boundscheck(False) #turn off boundary checking for function
def denseMat(x,y,float dx,float dy,int N,int M,float n):
    cdef np.ndarray[DTYPE_t, ndim=2] I
    cdef int jj
    cdef int ii
    cdef int zind
    cdef float Sum
    cdef np.ndarray[DTYPE_t, ndim=2] a
    cdef np.ndarray[DTYPE_t, ndim=2] K3
    cdef np.ndarray[DTYPE_t, ndim=2] K3_new
    cdef np.ndarray[DTYPE_t, ndim=2] all_K3 
    xH = (x[1:]+x[0:-1])/2.
    I = integ(x,y,xH)
    all_K3 = None
    for jj in range(M):
        a = np.zeros((2,N*M+M),dtype=DTYPE)
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


@cython.boundscheck(False) #turn off boundary checking for function
def IceJacobian(int M,int N,float deltaX,float deltaY,float x0,float Fr,float n,float beta):
    # Defining the domain
    cdef np.ndarray[DTYPE_t, ndim=1] x
    cdef np.ndarray[DTYPE_t, ndim=2] y
    cdef np.ndarray[DTYPE_t, ndim=2] A
    cdef np.ndarray[DTYPE_t, ndim=2] B
    cdef np.ndarray[DTYPE_t, ndim=2] C
    cdef np.ndarray[DTYPE_t, ndim=2] D
    cdef np.ndarray[DTYPE_t, ndim=2] J
    x = deltaX*sc.r_[:N] + x0
    y = deltaY*sc.c_[:M]
    

    A = TridiMatrix(N,M,deltaX,x0,n)
    B = (BlockMatrix(N,M,deltaX,Fr**(-2.))
         + icefuncs.jacobian_bilaplacian(N,M,deltaX,deltaY,x0,beta))
    C = BlockMatrix(N,M,deltaX,2.*np.pi)
    D = denseMat(x,y,deltaX,deltaY,N,M,n)
    J = np.block([[A,B],[C,D]])
    return J



