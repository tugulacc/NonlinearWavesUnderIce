# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 04:06:54 2020

@author: Claudia
"""


import numpy as np
#import scipy as sc 
import numpy.ma as ma
import matplotlib.pyplot as plt
#from matplotlib import cm
#import mpl_toolkits.axes_grid1 as axes_grid1
#import jacfuncs

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

def plot_jac(J):
    # replace all zero matrix elements with 10e-8
    mj = ma.masked_values(np.abs(J),0.).filled(10.**-8)
    data = np.log10(mj)

    plt.figure(figsize=(6, 5), dpi=600)

    plt.imshow(data,cmap='binary',vmin=-8.0,vmax=1.0)
    plt.colorbar()
     
    #plt.savefig(pltname)
    plt.show()
    
def plot_jac2(J):
    # replace all zero matrix elements with 10e-8
    mj = ma.masked_less_equal(np.abs(J),10).filled(10.**-8)
    data = np.log10(mj)

    plt.figure(figsize=(6, 5), dpi=600)

    plt.imshow(data,cmap='binary')
    plt.colorbar()
     
    #plt.savefig(pltname)
    plt.show()

def plot_surf(x,y,zeta):
    fig = plt.figure(dpi=600)
    ax = fig.gca(projection='3d')
    X,Y = np.meshgrid(x,y)
    
    # Add contour plots
    #cset = ax.contour(X,Y, zeta, zdir='x', offset=np.min(x), cmap=cm.coolwarm, alpha=0.3)
    cset = ax.contourf(X,Y, zeta, zdir='z', offset=np.min(zeta), cmap=cm.coolwarm)
    
    # Surface plot
    ax.plot_surface(X,Y, zeta, rstride=8, cstride=8, alpha=0.3 )
    # Add a color bar which maps values to colors.
    fig.colorbar(cset, shrink=0.6)
    # set number of ticks for each axis
    ax.zaxis.set_major_locator(LinearLocator(3))
    ax.xaxis.set_major_locator(LinearLocator(3))
    ax.yaxis.set_major_locator(LinearLocator(3))

    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))


    ax.set_ylim(np.min(y),np.max(y))
    ax.set_xlim(np.min(x),np.max(x))

    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')
    ax.set_zlabel(r'$\zeta$')
    ax.grid(False)
    plt.tight_layout()
    plt.show()
    
def plot_zeta1d(x,zeta):
    plt.figure(dpi=600)
    plt.plot(x,zeta[0,:])
    plt.ylabel(r'$\zeta (x,0)$')
    plt.xlabel(r'$x$')
    plt.show()