"""
Function to compute system of equations in a particular order

"""
import numpy as np
import funcs
#import Pflex

def wake(u,x,y,dx,dy,N,M,n,F,epsi,beta):
    # computing all needed quantities
    [phi1, phix, zeta1, zetax] = funcs.reshapingUnknowns(u,M,N)
    phi = funcs.allVals2(phi1,phix,dx,M,N)
    zeta = funcs.allVals2(zeta1,zetax,dx,M,N)
    # y-derivatives 
    phiy = np.gradient(phi, axis=0)
    zetay = np.gradient(zeta, axis=0)
    # second derivative computed as forward differnce on first derivatives
    phixx1 = funcs.xDerivs(phix,dx)
    zetaxx1 = funcs.xDerivs(zetax,dx)
    
    # computing Pflex term
    '''Pflx = Pflex.Bilaplacian(zeta,dx,dy)
    PflxH = funcs.halfMesh(Pflx,N)'''
    
    # computing half-mesh points
    phiH = funcs.halfMesh(phi,N)
    phixH = funcs.halfMesh(phix,N)
    phiyH = funcs.halfMesh(phiy,N)
    zetaH = funcs.halfMesh(zeta,N) 
    zetaxH = funcs.halfMesh(zetax,N) 
    zetayH = funcs.halfMesh(zetay,N) 
    xH = (x[1:]+x[:-1])/2.
    
    # computing pressure term
    '''
    P = np.zeros((M,N-1))
    xInd = np.array((np.abs(xH)<1),'double')
    yInd = np.array((np.abs(y)<1),'double')
    Pind = np.outer(xInd,yInd)
    P = np.exp(1./(xH**2.-1.)+1./(y**2.-1.))*Pind.T
    '''
    
    Dx = x - xH[:,None]
    Dy1 = y - y[:,None]
    Dy2 = y + y[:,None]
    
    # declaring all needed arrays
    I1 = np.zeros((M,N-1))
    I2p = np.zeros((M,N-1))
    I2pp = np.zeros((M,N-1))
    eqnsInt = np.zeros((M,N-1))
    eqns=np.zeros((2*M*(N+1),1))
    

    eqnsSurf = 1./2.*((1+zetaxH**2.)*phiyH**2.+(1+zetayH**2.)*phixH**2.-2.*zetaxH*zetayH*phixH*phiyH)/(1.+zetaxH**2.+zetayH**2.) + zetaH/F**2.-1./2. #+ beta*PflxH + epsi*P
    # eqnsSurf = phixH + zetaH/F**2.-1. 
    for jj in range(M):
        for ii in range(N-1):
            A = 1. + zetaxH[jj,ii]**2.
            B = 2.*zetaxH[jj,ii]*zetayH[jj,ii]
            C = 1. + zetayH[jj,ii]**2.
            
            S2denom1 = np.sqrt(A*Dx[ii]**2. + B*Dx[ii]*Dy1[jj]+C*Dy1[jj]**2.)
            S2denom2 = np.sqrt(A*Dx[ii]**2. - B*Dx[ii]*Dy2[jj]+C*Dy2[jj]**2.)
            S2 = 1./S2denom1 + 1./S2denom2
            
            K2denom1 = np.sqrt(Dx[ii]**2.+Dy1[jj]**2. + (zeta-zetaH[jj,ii])**2.)
            K2denom2 = np.sqrt(Dx[ii]**2.+Dy2[jj]**2. + (zeta-zetaH[jj,ii])**2.)
            K1num1 = zeta - zetaH[jj,ii] - Dx[ii]*zetax - Dy1[jj]*zetay
            K1num2 = zeta - zetaH[jj,ii] - Dx[ii]*zetax - Dy2[jj]*zetay
            
            K1 = K1num1/K2denom1**3. + K1num2/K2denom2**3.
            K2 = 1./K2denom1 + 1./K2denom2
            
            I1in = (phi-phiH[jj,ii] - x + xH[ii])*K1
            I1[jj,ii] = np.trapz(np.trapz(I1in,x).T,y.T)
            
            I2pin = (zetax*K2 - zetaxH[jj,ii]*S2)
            I2p[jj,ii] = np.trapz(np.trapz(I2pin,x).T,y.T)
            
            I2pp1 = lambda sIn, tIn : tIn/np.sqrt(A)*np.log(2.*A*sIn+B*tIn+2.*np.sqrt(A*(A*sIn**2.+B*sIn*tIn+C*tIn**2.)))
            I2pp2 = lambda sIn, tIn : sIn/np.sqrt(C)*np.log(2.*C*tIn+B*sIn+2.*np.sqrt(C*(A*sIn**2.+B*sIn*tIn+C*tIn**2.)))
            #EVL= lambda f, t: f(sN, t) - f(s1, t)
            
            sN = Dx[ii,-1]
            tN = Dy1[jj,-1]
            s1 = Dx[ii,0]
            t1 = Dy1[jj,0]
            
            I2pp[jj,ii] = I2pp2(sN,tN) - I2pp2(sN,t1) - I2pp2(s1,tN) + I2pp2(s1,t1)
            
            if t1!=0:
                I2pp[jj,ii] += - I2pp1(sN,t1) + I2pp1(s1,t1)
            if tN!=0:
                I2pp[jj,ii] += - I2pp1(s1,tN) + I2pp1(sN,tN)
            
            tN = Dy2[jj,-1]
            t1 = Dy2[jj,0]
            B = -B
            
            I2pp[jj,ii] += I2pp1(sN,tN) - I2pp1(s1,tN) + I2pp2(sN,tN) - I2pp2(sN,t1) - I2pp2(s1,tN) + I2pp2(s1,t1)
            
            if t1!=0:
                I2pp[jj,ii] += - I2pp1(sN,t1) + I2pp1(s1,t1)
            
            I2pp[jj,ii] *=  zetaxH[jj,ii] 
            
            eqnsInt[jj,ii] = - 2.*np.pi*(phiH[jj,ii] - xH[ii]) - epsi/np.sqrt(xH[ii]**2. + y[jj]**2. + (zetaH[jj,ii]+1.)**2.) + I1[jj,ii] + I2p[jj,ii] + I2pp[jj,ii]
            # eqnsInt[jj,ii] = - 2.*np.pi*(phiH[jj,ii] - xH[ii]) + I1[jj,ii] + I2p[jj,ii] + I2pp[jj,ii]
            
    bc1 = x[0]*(phix[:,0]-1.)+n*(phi[:,0]-x[0])
    bc2 = x[0]*(phixx1)+n*(phix[:,0]-1.)
    bc3 = x[0]*(zetax[:,0])+n*(zeta[:,0])
    bc4 = x[0]*(zetaxx1)+n*(zetax[:,0])

    eqnsSurf = np.reshape(eqnsSurf,(M*(N-1),1))
    eqnsInt = np.reshape(eqnsInt,(M*(N-1),1))
    
    for i in range(M):
        pos = i*(N-1)
        pos1 = i*(N+1)
        pos2 = i*(N+1)+M*(N+1)
        eqns[pos1]=bc1[i]
        eqns[pos1+1]=bc2[i]
        eqns[(pos1+2):(pos1+2+N-2)]= eqnsSurf[pos:(pos+N-2)]
        eqns[pos2] = bc3[i]
        eqns[(pos2+1)] = bc4[i]
        eqns[(pos2+2):(pos2 +2 +N-2)]= eqnsInt[pos:(pos+N-2)]
    
    #phis = np.hstack((phi1.reshape(M,1),phix)).reshape(M*(N+1),1)
    
    # eqnsBound = np.vstack((bc1, bc2, bc3, bc4))
    # eqns = np.vstack((np.reshape(eqnsSurf,(M*(N-1),1)), np.reshape(eqnsInt,(M*(N-1),1)),np.reshape(eqnsBound,(4*M,1))))
    
    eqns = eqns[:,0]

    return eqns

