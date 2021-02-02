"""
BIFunction.py - The system of nonlinear equations

"""
import numpy as np
import scipy as sc
import auxfuncs
import icefuncs

def BIFunction(unknowns,M,N,deltaX,deltaY,x0,Fr,epsilon,n,Lx,beta):
    '''BIfunction is the function that needs to be minimized.
    Takes the following inputs:
    unknowns - vector of unknowns
    M,N - the number of nodes in the y and x directions respectively
    deltaX,deltaY - the distance between nodes in the x and y directions
    x0 - the smallest x value
    Fr - the Froude number
    epsilon - strength of source/sink
    n - parameter for the radiation condition 
    beta* - coefficient for ice 
    Lx - length of pressure distribution
    '''
    
    # Defining the domain
    x = deltaX*sc.r_[:N] + x0
    y = deltaY*sc.c_[:M] 
    
    # Use auxiliary function to take variables and return the mesh values
    [phi1, phiX, zeta1, zetaX] = auxfuncs.reshapingUnknowns(unknowns,M,N)
    phi = auxfuncs.allVals(phi1,phiX,deltaX,M,N)
    zeta = auxfuncs.allVals(zeta1,zetaX,deltaX,M,N)
    
    # y-derivatives 
    phiY = np.gradient(phi, axis=0)
    zetaY = np.gradient(zeta, axis=0)
    
    # second derivative computed as forward difference on first derivatives
    phiXX1 = auxfuncs.xDerivs(phiX,deltaX)
    zetaXX1 = auxfuncs.xDerivs(zetaX,deltaX)
    
    # Calculate half-mesh points using two-point interpolation
    xHalf = (x[1:]+x[:-1])/2.
    yHalf = y
    
    zetaHalf = (zeta[:,1:N] + zeta[:,0:N-1])/2.
    zetaXHalf = (zetaX[:,1:N] + zetaX[:,0:N-1])/2.
    zetaYHalf = (zetaY[:,1:N] + zetaY[:,0:N-1])/2.
    phiHalf = (phi[:,1:N] + phi[:,0:N-1])/2.
    phiXHalf = (phiX[:,1:N] + phiX[:,0:N-1])/2.
    phiYHalf = (phiY[:,1:N] + phiY[:,0:N-1])/2.
    
    ''' computing the pressure term '''
    if Lx**2. in xHalf:
        print("division by zero, bad x0 or deltaX value ")
    p = np.zeros((M,N-1))
    xInd = np.array((np.abs(xHalf)<Lx),'double')
    yInd = np.array((np.abs(yHalf)<1),'double')
    pInd = np.outer(xInd,yInd)
    p = epsilon*np.exp(Lx**2./(xHalf**2.-Lx**2.)+1./(yHalf**2.-1.))*pInd.T
    # pHalf = (p[:,1:N]+p[:,0:N-1])/2.
    
    '''
    Computing the flexural term 
    Add in D function later for variable thickness
    '''
    bilaplacian = icefuncs.Bilaplacian(zeta,deltaX,deltaY)
    bilaplacianHalf = (bilaplacian[:,1:N]+bilaplacian[:,0:N-1])/2.
    pFlex = beta*bilaplacianHalf # constant thickness
    
    # Enforce surface condition for every half point in the mesh
    Func1 = 1/2*((1+zetaXHalf**2.)*phiYHalf**2.+ (1+zetaYHalf**2.)*phiXHalf**2.-2*zetaXHalf*zetaYHalf*phiXHalf*phiYHalf)/(1+zetaXHalf**2.+zetaYHalf**2.)+zetaHalf/Fr**2.-1/2 + p + pFlex
    Func2 = np.zeros((M,N-1))
    
    # declaring all needed arrays for computing singular integral
    I1 = np.zeros((M,N-1))
    I2p = np.zeros((M,N-1))
    I2pp = np.zeros((M,N-1))
    
    
    # Calculate often used values: (x-x*), (y-y*), (y+y*)
    xDiff = x - xHalf[:,None]
    yNegDiff = y - yHalf[:,None]
    yPosDiff = y + yHalf[:,None]
    # zetaDiff = zeta - zetaHalf
    
    
    for l in range(M):
        for k in range(N-1):
            # Initialise sums
            A = 1. + zetaXHalf[l,k]**2.
            B = 2.*zetaXHalf[l,k]*zetaYHalf[l,k]
            C = 1. + zetaYHalf[l,k]**2.
            
            # Calculate complicated values for the integral
            S2denomYNeg = np.sqrt(A*xDiff[k]**2.+B*xDiff[k]*yNegDiff[l]+C*yNegDiff[l]**2.)
            S2denomYPos = np.sqrt(A*xDiff[k]**2.-B*xDiff[k]*yPosDiff[l]+C*yPosDiff[l]**2.)
            S2 = 1./S2denomYNeg + 1./S2denomYPos
            
            KdenomYNeg = np.sqrt(xDiff[k]**2.+yNegDiff[l]**2.+(zeta-zetaHalf[l,k])**2.)
            KdenomYPos = np.sqrt(xDiff[k]**2.+yPosDiff[l]**2.+(zeta-zetaHalf[l,k])**2.)
            
            K1numerYNeg = zeta-zetaHalf[l,k]-xDiff[k]*zetaX-yNegDiff[l]*zetaY
            K1numerYPos = zeta-zetaHalf[l,k]-xDiff[k]*zetaX-yPosDiff[l]*zetaY
            
            
            K1 = K1numerYNeg/KdenomYNeg**3.+K1numerYPos/KdenomYPos**3.
            K2 = 1./KdenomYNeg + 1./KdenomYPos
            
            
            I1in = (phi-phiHalf[l,k]-x+xHalf[k])*K1
            I1[l,k] = np.trapz(np.trapz(I1in,x).T,y.T)
            
            I2pin = (zetaX*K2 - zetaXHalf[l,k]*S2)
            I2p[l,k] = np.trapz(np.trapz(I2pin,x).T,y.T)
            
            I2pp1 = lambda sIn, tIn : tIn/np.sqrt(A)*np.log(2.*A*sIn+B*tIn+2.*np.sqrt(A*(A*sIn**2.+B*sIn*tIn+C*tIn**2.)))
            I2pp2 = lambda sIn, tIn : sIn/np.sqrt(C)*np.log(2.*C*tIn+B*sIn+2.*np.sqrt(C*(A*sIn**2.+B*sIn*tIn+C*tIn**2.)))
            #EVL= lambda f, t: f(sN, t) - f(s1, t)
            
            sN = xDiff[k,-1]
            tN = yNegDiff[l,-1]
            s1 = xDiff[k,0]
            t1 = yNegDiff[l,0]
            
            I2pp[l,k] = I2pp2(sN,tN) - I2pp2(sN,t1) - I2pp2(s1,tN) + I2pp2(s1,t1)
            
            if t1!=0:
                I2pp[l,k] += - I2pp1(sN,t1) + I2pp1(s1,t1)
            if tN!=0:
                I2pp[l,k] += - I2pp1(s1,tN) + I2pp1(sN,tN)
            
            tN = yPosDiff[l,-1]
            t1 = yPosDiff[l,0]
            B = -B
            
            I2pp[l,k] += I2pp1(sN,tN) - I2pp1(s1,tN) + I2pp2(sN,tN) - I2pp2(sN,t1) - I2pp2(s1,tN) + I2pp2(s1,t1)
            
            if t1!=0:
                I2pp[l,k] += - I2pp1(sN,t1) + I2pp1(s1,t1)
            
            I2pp[l,k] *=  zetaXHalf[l,k] 
            
            
            Func2[l,k] = -2.*np.pi*(phiHalf[l,k]-xHalf[k])+I1[l,k]+I2p[l,k]+I2pp[l,k]
            
    ''' Boundary conditions '''        
    Func3 = x0*(phiX[:,0]-1.)+n*(phi[:,0]-x0)
    Func4 = x0*(phiXX1)+n*(phiX[:,0]-1.)
    Func5 = x0*(zetaX[:,0])+n*(zeta[:,0])
    Func6 = x0*(zetaXX1)+n*(zetaX[:,0])
    
    ''' reordering'''
    E1 = np.hstack((Func3.reshape(M,1),Func4.reshape(M,1),Func1)).reshape(M*(N+1),1)
    E2 = np.hstack((Func5.reshape(M,1),Func6.reshape(M,1),Func2)).reshape(M*(N+1),1)
    Funcs = np.vstack((E1,E2))
    Funcs = Funcs[:,0]

    return Funcs
    


