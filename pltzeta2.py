"""
Function to plot zeta directly from .npy output file

Created on Jan 11 2021

@author: Claudia
"""
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import FormatStrFormatter#, LinearLocator
from matplotlib.colors import LightSource

import numpy as np
import scipy as sc
import funcs

'''
Define parameters (change to whatever you used)
'''
M, N = 30, 15
dx, dy = 0.4, 0.4
x1= -3.

# Defining the x, y domain
xData = dx*sc.r_[:N] + x1
yData = dy*sc.c_[:M] 

'''
Extracting zeta
'''
uNew = np.load('andrew_2.npy') # import data
[phi1, phix, zeta1, zetax] = funcs.reshapingUnknowns(uNew,M,N)
zet = funcs.allVals(zeta1,zetax,dx,M,N) # zeta evaluation

# extend domain via symmetry
yfull = np.vstack((np.flip(-yData[1:,:]),yData))
zfull = np.vstack((np.flip(zet[1:,:],axis=0),zet))

# Make the X, Y meshgrid
X, Y = np.meshgrid(xData, yfull)

'''
Plotting
'''
fig = plt.figure(dpi=600)
ax = fig.gca(projection='3d', proj_type='ortho')

ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# set number of ticks for each axis
#ax.zaxis.set_major_locator(LinearLocator(5))
#ax.xaxis.set_major_locator(LinearLocator(5))
#ax.yaxis.set_major_locator(LinearLocator(3))

# axes labels
ax.set_xlabel(r'$x$')  
ax.set_ylabel(r'$y$', rotation=0)  
ax.zaxis.set_rotate_label(False)
ax.set_zlabel(r'$\zeta$',rotation=0)   
# ax.grid(True)

# create light source object.
ls = LightSource(azdeg=-10, altdeg=45)
rgb = ls.shade(zfull, cmap=cm.ocean, vert_exag=1.5, blend_mode='soft')


# generate surface plot
surf = ax.plot_surface(X, Y, zfull , rstride=1, cstride=1, facecolors=rgb, alpha=.8)
# ax.view_init(elev=30., azim=-95) # change viewing angle


''' uncomment to add contour plot and color bar legend '''
# cset = ax.contourf(X, Y, zfull, zdir='z', offset=np.min(zfull), cmap=cm.ocean)
# cbar=fig.colorbar(cset, shrink=0.5)
# cbar.set_label(r"$\zeta$",rotation=0)

# plt.axis('off') # remove axes

plt.tight_layout()
plt.show()

