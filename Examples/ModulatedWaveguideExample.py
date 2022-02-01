#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  8 22:29:59 2021

@author: ptbowen
"""
import worx.ElectromagneticObjects as emag
import worx.MathObjects as math
import worx.PlotTools as PlotTools
import worx.Geometries as geom

import numpy as np
from numpy.lib.scimath import sqrt
from mayavi import mlab

pi=np.pi
c=2.998e8
(xhat,yhat,zhat)=math.CreateEuclideanBasis()
vec_space=[xhat,yhat,zhat]

# Operating frequency and beam definition
f0=10.2e9
lmbda=c/f0
k=2*pi/lmbda
theta_b=90*pi/180
phi_b=30*pi/180


# Waveguide design
n_wg=sqrt(3.55*(1-0.0027j))
pitch=0.006
N=30
L=N*pitch
a=14e-3
b=0.762e-3
WG=geom.Waveguide(a,b,n_wg,L)
WG.N_modes=1
# WG.N_modes=10

# Create a half space for far-field calcs
HS=geom.HalfSpace()


# Data files for polarizability extraction
S11MagFile='../HFSS/S11mag_MikeCell_CapSweep.csv'
S11PhaseFile='../HFSS/S11phase_MikeCell_CapSweep.csv'
S21MagFile='../HFSS/S21mag_MikeCell_CapSweep.csv'
S21PhaseFile='../HFSS/S21phase_MikeCell_CapSweep.csv'
HFSS_Files={'S11Mag':S11MagFile,'S11Phase':S11PhaseFile,
            'S21Mag':S21MagFile,'S21Phase':S21PhaseFile}


# Create initial dipole.  This time it's parameterized.
r0=xhat*1.6e-3
dip=emag.Dipole(r0)
dip.extract(WG,HFSS_Files,parameterized=1)
f=np.linspace(8e9,12e9,200)
mod_range=np.linspace(0,1,5)
dip.analyze(f=f,mod=mod_range)


# Dipole array design
z_pos=np.arange(pitch/2,(N+1/2)*pitch,pitch)+1.2e-3
x_pos=np.ones(np.shape(z_pos))*dip.r0[xhat]
y_pos=np.ones(np.shape(z_pos))*dip.r0[yhat]
r_pos=math.GridToListOfVectors({xhat:x_pos,yhat:y_pos,zhat:z_pos})
dip_array=dip.Array(r_pos)
for dipole in dip_array:
    WG.add_dipole(dipole)
    HS.add_dipole(dipole)
    
# Now, position it where you want it in space
R=math.RotTensor(pi/2,'x',vec_space)
WG.Rotate(R)


# Define source (You have to do this before you can modulate)
WG.Source(f0)


# Modulate
k_b=math.SphericalBasis('r')(theta_b,phi_b)*k
E_b=math.SphericalBasis('phi')(theta_b,phi_b)
WG.Modulate(k_b,E_b,plot=1)


# Compute
WG.Compute(f0,compute_type='coupled')
HS.ComputeFarField(f0)


# Plot geometry
mlab.figure(figure=1,bgcolor=(1,1,1), fgcolor=(0.,0.,0.))
mlab.clf()
WG.MayaviPlot(1,plot_fields=True,field_clip=30,phase=1.08*pi)
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
