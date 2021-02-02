"""
icefuncs.py - functions to generate the flexural term for ice
"""


import numpy as np
import auxfuncs

def Bilaplacian(zeta,dx,dy):
    ''' Calculate the bilaplacian operator
    '''
    pad_upstream = np.zeros((zeta.shape[0],2)) # zeta=0
    pad_downstream = zeta[:,-2:] # zeta_x = 0
    vx = np.hstack((pad_upstream,zeta,pad_downstream))
    vy = np.pad(zeta,((2,0),(0,0)),'reflect')
    v = np.pad(vx,((2,0),(0,0)),'reflect')
    
    D_xxxx = (vx[:,:-4]-4*vx[:,1:-3]+6*vx[:,2:-2]-4*vx[:,3:-1]+vx[:,4:])/(dx**4) # centered difference
    D_yyyy = (vy[:-4,:]-4*vy[1:-3,:]+6*vy[2:-2,:]-4*vy[3:-1,:]+vy[4:,:])/(dy**4) # centered difference (interior)
    D_yyyy = np.pad(D_yyyy,((0,2),(0,0)),'edge') # one-sided difference at lateral boundary
    
    D_xxyy = (v[4:,4:]-2*v[4:,2:-2]+v[4:,:-4]-2*(v[2:-2,4:]-2*v[2:-2,2:-2]+v[2:-2,:-4])+v[:-4,4:]-2*v[:-4,2:-2]+v[:-4,:-4])/(16*(dx**2)*(dy**2))
    D_xxyy = np.pad(D_xxyy,((0,2),(0,0)),'edge')
    
    
    P_flex = D_xxxx + 2*D_xxyy + D_yyyy
    
    return P_flex

''' 
Here is where I try to calculate the contribution for the ice preconditioner 
'''
def funcs_bilaplacian(u,N,M,dx,dy,x1,beta):
    indx = (N+1)*np.arange(M) # indices of zeta1
    zeta1 = u[indx] # extract values of zeta1
    zetax = np.delete(u,indx).reshape(M,N) # extract zetax as an MXN matrix
    zeta = auxfuncs.allVals(zeta1,zetax,dx,M,N)
    Pflex = Bilaplacian(zeta,dx,dy)
    PflexHalf = (Pflex[:,1:N] + Pflex[:,0:N-1])/2.
    flxtrm = beta*PflexHalf
    
    bc = np.zeros((M,1))
    E1 = np.hstack((bc,bc,flxtrm)).reshape(M*(N+1),1)
    return E1[:,0]

def jacobian_bilaplacian(N,M,dx,dy,x1,beta):
    '''
    Function returns contribution to the Jacobian from the bilaplacian
    Calculate jacobian numerically using centered finite differences
    '''
    shift = 1e-10 # step size for centered finite difference calculation
    neq = M*(N+1) # number of equations for Jacobian submatrix
    J = np.zeros((neq,neq)) # Initialize
    # initial guess for zeta: flat surface
    uInit= np.zeros((M*(N+1),1))
    for i in range(neq):
        y1 = uInit.copy()
        y2 = uInit.copy()
        y1[i]+= shift
        y2[i]-= shift
        f1 = funcs_bilaplacian(y1,N,M,dx,dy,x1,beta)
        f2 = funcs_bilaplacian(y2,N,M,dx,dy,x1,beta)
        J[:,i]=(f1-f2)/(2*shift)
    return J