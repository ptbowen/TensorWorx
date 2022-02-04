#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 17:15:09 2021

@author: ptbowen
"""
import worx.ElectromagneticObjects as emag
import worx.MathObjects as math
import worx.Geometries as geom
import worx.PlotTools as PlotTools
from worx.Extraction import WaveguideExtraction

import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.scimath import sqrt
from matplotlib import cm
from mayavi import mlab

# constants
pi=np.pi
Z0=376
c=2.998e8

# We are going to operate in a 3D euclidean global geometry.
(xhat,yhat,zhat)=math.CreateEuclideanBasis()
vec_space=[xhat,yhat,zhat]

# some preliminary variables
f0=np.array(9.85e9)
lmbda=c/f0
k0=2*pi/lmbda

# With the preliminaries out of the way, this tutorial walks through creating a single
# linear waveguide with an array of elements, modulating it, and finding the radiated
# power in an unperturbed feed wave approximation.


# Waveguide design
N=21
pitch=lmbda/5.5
L=(N+1)*pitch
a=14e-3
b=0.762e-3
n_wg=sqrt(3.55*(1-0.0027j))
WG=geom.Waveguide(a,b,n_wg,L)
WG.N_modes=4
HS=geom.HalfSpace()


# Individual dipole design
dip1=emag.Dipole(xhat*0)
# Tuning ranges for polarizabilities will always be presented as functions, whether
# they are interpolants or analytic.  Let's start with an analytic expression for
# a tunable Lorentzian dipole.
alpha0=(18e-3)**3           # polarizability amplitude
Q=40                        # Quality factor
m=np.linspace(0,1,100)      # Lorentzian modulation parameter 
alpha_m=(lambda f,m: (alpha0*Q)*np.sin(m*pi)*np.exp(-1j*m*pi))
alpha_e=(lambda f,m: 0)
dip1.TuningFunction_m=alpha_m
dip1.TuningFunction_e=alpha_e
dip1.nu_m=xhat
dip1.nu_e=yhat

# Dipole array design
z_pos=np.arange(pitch/2,(N+1/2)*pitch,pitch)
x_pos=np.ones(np.shape(z_pos))*1.6e-3
y_pos=np.zeros(np.shape(z_pos))
r_pos=math.GridToListOfVectors({xhat:x_pos,yhat:y_pos,zhat:z_pos})
dip_array=dip1.Array(r_pos)
for dipole in dip_array:
    WG.add_dipole(dipole)
    HS.add_dipole(dipole)

# Now, position it where you want it in space
R=math.RotTensor(-pi/2,'x',vec_space)
WG.Rotate(R)

# Define a source
dip=emag.Dipole(dip_array[10].r0)
dip.M=xhat*2
WG.Source(f0,source_type='dipole',dip=dip)

# Desired beam k-vector
theta_b=0
phi_b=0
rhat=math.SphericalBasis('r')
k_b=rhat(theta_b,phi_b)*k0

# Desired beam polarization vector
E_b=(math.SphericalBasis('phi'))(theta_b,phi_b)
#%%
# Modulate
WG.Modulate(k_b,E_b,False)

# Compute to excite dipoles with source
WG.Compute(f0)
HS.ComputeFarField(f0)

# Plot geometry
mlab.figure(figure=1,bgcolor=(1,1,1), fgcolor=(0.,0.,0.))
mlab.clf()
WG.MayaviPlot(1,plot_fields=True,phase=(0/6)*pi)
PlotTools.MayaviAxes(10,-L/3,L/3,0,L,-L/10,L/2)


# Far field plotting
directivity=HS.FarField.directivity
k=HS.FarField.k
(theta_grid,phi_grid,th_axis,ph_axis,dth,dph)=math.HalfSphereGrid(360,90)
Dir_dB=10*np.log10(np.real(directivity(k(theta_grid,phi_grid)))+1e-16) 
Dir_dB[Dir_dB<0]=0
rhat=math.SphericalBasis('r')
X=Dir_dB*(rhat.dot(xhat))(theta_grid,phi_grid)
Y=Dir_dB*(rhat.dot(yhat))(theta_grid,phi_grid)
Z=Dir_dB*(rhat.dot(zhat))(theta_grid,phi_grid)
mlab.figure(figure=2,bgcolor=(1,1,1), fgcolor=(0.,0.,0.))
mlab.clf()
mlab.mesh(X,Y,Z)
coordmax=np.max([X,Y,Z])
PlotTools.MayaviAxes(10,-coordmax,coordmax,-coordmax,coordmax,0,coordmax)







