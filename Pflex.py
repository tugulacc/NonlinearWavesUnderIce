# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 06:25:35 2020

@author: Claudia
"""
import numpy as np

def Bilaplacian(zeta,dx,dy):
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
    
    '''
    bc1 = ((v[-1,4:]-2*v[-1,2:-2]+v[-1,:-4])-(v[-2,4:]-2*v[-2,2:-2]+v[-2,:-4])-(v[-3,4:]-2*v[-3,2:-2]+v[-3,:-4])+(v[-4,4:]-2*v[-4,2:-2]+v[-4,:-4]))/(8*(dx**2)*(dy**2))
    bc2 = ((v[-1,4:]-2*v[-1,2:-2]+v[-1,:-4])-2*(v[-2,4:]-2*v[-2,2:-2]+v[-2,:-4])+(v[-3,4:]-2*v[-3,2:-2]+v[-3,:-4]))/(4*(dx**2)*(dy**2))
    D_xxyy = np.vstack((D_xxyy,bc1,bc2))
    '''
    
    P_flex = D_xxxx + 2*D_xxyy + D_yyyy
    
    return P_flex

