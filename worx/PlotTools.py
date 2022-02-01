#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  2 20:42:40 2021

@author: ptbowen
"""
import numpy as np
from mayavi import mlab

import worx.MathObjects as math
pi=np.pi


def axisEqual3D(ax):
    extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
    sz = extents[:,1] - extents[:,0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize/2
    for ctr, dim in zip(centers, 'xyz'):
        getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)
        
        
def PlotSphere(ax,R,r0):
    (xhat,yhat,zhat)=math.CreateEuclideanBasis()
    
    u = np.linspace(0, 2*pi, 20)
    v = np.linspace(0, pi, 10)
    x = R * np.outer(np.cos(u), np.sin(v))+r0.dot(xhat)
    y = R * np.outer(np.sin(u), np.sin(v))+r0.dot(yhat)
    z = R * np.outer(np.ones(np.size(u)), np.cos(v))+r0.dot(zhat)
    
    ax.plot_surface(x, y, z, color='green',zorder=1)
    

def MayaviRect(base,u,v,color,opacity):
    (xhat,yhat,zhat)=math.CreateEuclideanBasis()
    
    p1=base
    p2=base+u
    p3=base+u+v
    p4=base+v
    
    (x,y,z)=math.ListOfVectorsToGrid([p1,p2,p3,p4])
    triangles=[[0,1,2],[0,2,3]]
    mlab.triangular_mesh(np.real(x),np.real(y),np.real(z),triangles,color=color,opacity=opacity)
    
    return None

def MayaviAxes(N,x_min,x_max,y_min,y_max,z_min,z_max):
    (xhat,yhat,zhat)=math.CreateEuclideanBasis()
    N=N+1
    ## XY plane
    x1=x_min*np.ones(N)
    x2=x_max*np.ones(N)
    y1=np.mgrid[y_min:y_max:N*1j]
    y2=y1
    z1=z_min*np.ones(np.shape(y1))
    z2=z1
    for i in range(0,len(x1)):
        mlab.plot3d([x1[i],x2[i]],[y1[i],y2[i]],[z1[i],z2[1]],tube_radius=None,line_width=1.0,color=(0,0,0))
    
    x1=np.mgrid[x_min:x_max:N*1j]
    x2=x1
    y1=y_min*np.ones(N)
    y2=y_max*np.ones(N)
    z1=z_min*np.ones(np.shape(y1))
    z2=z1
    for i in range(0,len(x1)):
        mlab.plot3d([x1[i],x2[i]],[y1[i],y2[i]],[z1[i],z2[1]],tube_radius=None,line_width=1.0,color=(0,0,0))

    base=xhat*x_min+yhat*y_min+zhat*z_min
    u=xhat*(x_max-x_min)
    v=yhat*(y_max-y_min)
    MayaviRect(base,u,v,(0.99,0.99,0.99),0.5)
    # x=[x_min,x_max,x_max,x_min]
    # y=[y_min,y_min,y_max,y_max]
    # z=z_min*np.ones(np.shape(x))
    # triangles=[[0,1,2],[0,3,2]]
    # mlab.triangular_mesh(x,y,z,triangles,color=(0.99,0.99,0.99),opacity=0.5)




    ## YZ plane
    x1=x_min*np.ones(N)
    x2=x1
    y1=np.mgrid[y_min:y_max:N*1j]
    y2=y1
    z1=z_min*np.ones(np.shape(y1))
    z2=z_max*np.ones(np.shape(y1))
    for i in range(0,len(x1)):
        mlab.plot3d([x1[i],x2[i]],[y1[i],y2[i]],[z1[i],z2[i]],tube_radius=None,line_width=1.0,color=(0,0,0))
    
    x1=x_min*np.ones(N)
    x2=x1
    y1=y_min*np.ones(np.shape(y1))
    y2=y_max*np.ones(np.shape(y1))
    z1=np.mgrid[z_min:z_max:N*1j]
    z2=z1
    for i in range(0,len(x1)):
        mlab.plot3d([x1[i],x2[i]],[y1[i],y2[i]],[z1[i],z2[i]],tube_radius=None,line_width=1.0,color=(0,0,0))

    y=[y_min,y_min,y_max,y_max]
    z=[z_min,z_max,z_max,z_min]
    x=x_min*np.ones(np.shape(y))
    triangles=[[0,1,2],[0,3,2]]
    mlab.triangular_mesh(x,y,z,triangles,color=(0.99,0.99,0.99),opacity=0.5)




    ## XZ plane
    x1=np.mgrid[x_min:x_max:N*1j]
    x2=x1
    y1=y_max*np.ones(np.shape(x1))
    y2=y_max*np.ones(np.shape(x1))
    z1=z_min*np.ones(np.shape(y1))
    z2=z_max*np.ones(np.shape(y1))
    for i in range(0,len(x1)):
        mlab.plot3d([x1[i],x2[i]],[y1[i],y2[i]],[z1[i],z2[i]],tube_radius=None,line_width=1.0,color=(0,0,0))
        
    x1=x_min*np.ones(np.shape(x1))
    x2=x_max*np.ones(np.shape(x1))
    y1=y_max*np.ones(np.shape(x1))
    y2=y_max*np.ones(np.shape(x1))
    z1=np.mgrid[z_min:z_max:N*1j]
    z2=z1
    for i in range(0,len(x1)):
        mlab.plot3d([x1[i],x2[i]],[y1[i],y2[i]],[z1[i],z2[i]],tube_radius=None,line_width=1.0,color=(0,0,0))


    x=[x_min,x_max,x_max,x_min]
    y=y_max*np.ones(np.shape(x))
    z=[z_min,z_max,z_max,z_min]
    triangles=[[0,1,2],[0,3,2]]
    mlab.triangular_mesh(x,y,z,triangles,color=(0.99,0.99,0.99),opacity=0.5)
    
    
def material_color(name):
    if name=='copper':
        return (0.71875, 0.44921875, 0.19921875)
    
    
    
    