"""
runCode.py - main function that runs all the code
"""

cimport numpy as np
import numpy as np
import scipy as sc
cimport cython
from scipy import optimize, linalg
import time
import auxfuncs
import icefuncs
import BIFunction

import Preconditioner


#import PlotSurface

ctypedef np.float64_t DTYPE_t

'''
Try:
    
    (deltaX,deltaY)= (0.6,0.6),(0.3,0.3) <-try these for each (N,M)
  (N,M)= (80,40),(160,80), (320,160)
    x0 = -24,   -48,        -96 <- for deltaX,Y = 0.6
    x0 = -12,   -24,        -48 <- for deltaX,Y = 0.3
* Should have 6 runs total
* Pipe parameters into file
* Use old plotting file for visualising solutions
'''
@cython.boundscheck(False) #turn off boundary checking for function
def main():
	# Defining parameters
	cdef int N = 40
	cdef int M=30
	cdef float deltaX= 0.3
	cdef float deltaY = 0.3
	cdef float x0 = -6.
	cdef float Fr = 1./np.sqrt(0.7) 
	cdef float Lx = 1.
	cdef float beta = 1. 
	cdef float n = 0.05
	cdef float epsilon = 1.

	# add deltaX,Y and x0 to file name
	fname = 'unknowns_'+str(N)+'x'+str(M) # file name

	cdef np.ndarray[DTYPE_t, ndim=2] uInit
	### Initial guess: flat surface ###
	uInit = np.vstack((np.tile(np.vstack(([x0],np.ones((N,1)))),(M,1)),np.zeros((M*(N+1),1))))
	# Get the Jacobian for ice
	start_time0 = time.time()
	cdef np.ndarray[DTYPE_t, ndim=2] J_ice
	J_ice = Preconditioner.IceJacobian(M,N,deltaX,deltaY,x0,Fr,n,beta)
	print("--- %s seconds to generate the Jacobian ---" % (time.time() - start_time0))

	#np.save(fname,J_ice)

	# Get the inverse for the preconditioner
	start_time1 = time.time()
	cdef np.ndarray[DTYPE_t, ndim=2] Jinv
	Jinv = linalg.inv(J_ice)
	print("--- %s seconds for inverse ---" % (time.time() - start_time1))

	start_time = time.time()
	uNew = optimize.newton_krylov(lambda u: BIFunction.BIFunction(u,M,N,deltaX,deltaY,x0,Fr,epsilon,n,Lx,beta), uInit, method='lgmres', inner_M=Jinv, verbose=True)
	print("--- %s seconds for solver ---" % (time.time() - start_time))


	cdef np.ndarray[DTYPE_t, ndim=2] phi1
	cdef np.ndarray[DTYPE_t, ndim=2] phix
	cdef np.ndarray[DTYPE_t, ndim=2] zeta1
	cdef np.ndarray[DTYPE_t, ndim=2] zetax

	[phi1, phix, zeta1, zetax] = auxfuncs.reshapingUnknowns(uNew,M,N)
	cdef np.ndarray[DTYPE_t, ndim=2] zeta
	zeta = auxfuncs.allVals(zeta1,zetax,deltaX,M,N)

	# np.save(fname,uNew)

	# PlotSurface.surface_full(N,M,deltaX,deltaY,x0,zeta)