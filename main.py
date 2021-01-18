"""
Main function that runs everything
"""

import numpy as np
import scipy as sc
from scipy import optimize#, linalg
import funcs
# import jacfuncs
import pltfuncs
#import myWake
#import linSoln
#import wakeMcCue
import wakeMcCue
import time


# from scipy.sparse import csc_matrix
# from scipy.sparse.linalg import spilu, LinearOperator

# Declaring Constants
N, M = 200, 30#80, 40 #91, 31
dx, dy =  0.2, 0.8 #0.6, 0.6
x1 = -20.
F = 1./np.sqrt(0.7)
beta = 0.5
n, epsi = 0.05, 1.
epsiMax, amp = 1., 1.
Lx = 1.

# Defining the domain
x = dx*sc.r_[:N] + x1
y = dy*sc.c_[:M] 


# J = jacfuncs.Jacobian(x,y,dx,dy,N,M,F,n)
# Jinv = linalg.inv(J)
#pltfuncs.plot_jac(J)

# Js=csc_matrix(J, dtype=float)
# J_ilu = spilu(Js)
# len_u = 2*(N+1)*M
# M = LinearOperator(shape=(len_u,len_u),matvec=J_ilu.solve)


### INITIAL GUESS ###
phi1 = x1*np.ones((M,1))
phix = np.ones((M,N))
zeta1 = np.zeros((M,1))
zetax = np.zeros((M,N))
#[zeta1, zetax] = linSoln.linZeta(x,y,M,N,F,epsi)
# stacking the guess into one vector
uInit = funcs.guessUnknowns(phi1,phix,zeta1,zetax,M,N) 

# this is where we tried computing the Jacobian
'''
flat_sol = newMcCue.wake(uInit,x,y,dx,dy,N,M,n,F,epsi,beta)

shift = 0.001
J_nonlin = np.zeros((2*M*(N+1),2*M*(N+1)))
for jj in range(2*M*(N+1)):
    ushift = uInit
    ushift[jj] += shift
    J_nonlin[:,jj] = (newMcCue.wake(ushift,x,y,dx,dy,N,M,n,F,epsi,beta) - flat_sol)/shift
'''
start_time = time.time()

# try using the gmres solver
# uNew = optimize.newton_krylov(lambda u: wakeMcCue.wake(u,x,y,dx,N,M,n,F,epsi), uInit, method='lgmres', inner_M=Jinv, verbose=1)

uNew = optimize.fsolve(wakeMcCue.wake,uInit,args=(x,y,dx,dy,N,M,n,F,epsi,beta, Lx))

print("--- %s seconds ---" % (time.time() - start_time))

[phi1, phix, zeta1, zetax] = funcs.reshapingUnknowns(uNew,M,N)
zeta = funcs.allVals(zeta1,zetax,dx,M,N)

### Main iteration ###
'''
while epsi <= epsiMax:
    ### Calling the main function ###
    uNew = optimize.fsolve(wakeMcCue.wake,uInit,args=(x,y,dx,N,M,n,F,epsi))
    # checking error
    err  = wakeMcCue.wake(uNew,x,y,dx,N,M,n,F,epsi)
    err = np.max(err)
    ### marching forward ### <- rescale amplitude?
    epsi = amp + epsi
    ### get new guess ###
    [phi1, phix, zeta1, zetax] = funcs.reshapingUnknowns(uNew,M,N)
    zeta = funcs.allVals(zeta1,zetax,dx,M,N)
    # stacking the guess into one vector
    uInit = funcs.guessUnknowns(phi1,phix,zeta1,zetax,M,N)
    
    print(epsi)
    print(err)

'''

np.save('07_200x30p',zeta)

pltfuncs.plot_surf(x,y,zeta)

# pltfuncs.plot_jac(J_nonlin)
