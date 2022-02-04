
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  8 22:29:59 2021

@author: ptbowen
"""
import worx.ElectromagneticObjects as emag
import worx.MathObjects as math
import worx.Geometries as geom
import worx.PlotTools as PlotTools

import numpy as np
from numpy.lib.scimath import sqrt
from mayavi import mlab
import copy

pi=np.pi
c=2.998e8
(xhat,yhat,zhat)=math.CreateEuclideanBasis()
vec_space=[xhat,yhat,zhat]

# Operating frequency and beam definition
f0=9.8e9
lmbda=c/f0
k=2*pi/lmbda
theta_b=90*pi/180
phi_b=0.1*pi/180
k_b=math.SphericalBasis('r')(theta_b,phi_b)*k
E_b=math.SphericalBasis('theta')(theta_b,phi_b)

# Waveguide design
n_wg=sqrt(3.55*(1-0.0027j))
pitch=0.008
N=5
L=N*pitch
a=14e-3
b=0.762e-3
WG=geom.Waveguide(a,b,n_wg,L)
WG.N_modes=1

# Create a half space for far-field calcs
HS=geom.HalfSpace()

# Data files for polarizability extraction
S11MagFile='./HFSS/S11mag_MikeCell_CapSweep.csv'
S11PhaseFile='./HFSS/S11phase_MikeCell_CapSweep.csv'
S21MagFile='./HFSS/S21mag_MikeCell_CapSweep.csv'
S21PhaseFile='./HFSS/S21phase_MikeCell_CapSweep.csv'
HFSS_Files={'S11Mag':S11MagFile,'S11Phase':S11PhaseFile,
            'S21Mag':S21MagFile,'S21Phase':S21PhaseFile}

# Create initial dipole.  This time it's parameterized.
r0=xhat*1.6e-3
dip=emag.Dipole(r0)
dip.extract(WG,HFSS_Files,parameterized=1)
f=np.linspace(8e9,12e9,200)
mod_range=np.linspace(0,1,5)
# dip.analyze(f=f,mod=mod_range)

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
R1=math.RotTensor(pi/2,'x',vec_space)
R2=math.RotTensor(-pi/2,'z',vec_space)
WG.Rotate(R1)
WG.Rotate(R2)
WG.Translate(yhat*(a/2))

# Define source and compute unperturbed feed wave
WG.Source(f0)
WG.Modulate(k_b,E_b)
# for dipole in WG.DipoleList: dipole.tune(f0,0.5)
WG.Compute(f0,compute_type='coupled')
HS.ComputeFarField(f0)

# Plot
mlab.figure(figure=1,bgcolor=(1,1,1), fgcolor=(0.,0.,0.))
mlab.clf()
WG.MayaviPlot(1,plot_fields=True,field_clip=30)
PlotTools.MayaviAxes(10,-L/2,L/2,0,1.5*L,-L/10,L/2)


# Find total far-field and plot
directivity=HS.FarField.directivity
k_ff=HS.FarField.k
(theta_grid,phi_grid,th_axis,ph_axis,dth,dph)=math.HalfSphereGrid(HS.FarField.Nth,HS.FarField.Nph)
Dir_dB=10*np.log10(np.real(directivity(k_ff(theta_grid,phi_grid)))+1e-16) 
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


#%% Make a waveguide array
# First, make a fresh half space so the old waveguide's dipoles aren't included.
HS=geom.HalfSpace()

# Copy waveguide to make array
WG_spacing=16e-3
N_wg=8
WG_Array=[0 for i in range(0,N_wg)]
for i in range(0,N_wg):
    angle=i*2*pi/N_wg
    translate_vec=(xhat*np.cos(angle)+yhat*np.sin(angle))*WG_spacing
    R=math.RotTensor(i*2*pi/N_wg,'z',vec_space)
    WG_Array[i]=copy.deepcopy(WG)
    WG_Array[i].Rotate(R)
    WG_Array[i].Translate(translate_vec)
    # You have to make sure all the dipoles are also added to the half space for the far field calculation.
    for dipole in WG_Array[i].DipoleList:
        dipole.tune(f0,0.3)
        HS.add_dipole(dipole)

# Plot geometry
mlab.figure(figure=1,bgcolor=(1,1,1), fgcolor=(0.,0.,0.))
mlab.clf()
for wg in WG_Array:
    wg.MayaviPlot(1,plot_fields=False,field_clip=30)
PlotTools.MayaviAxes(10,-1.5*L,1.5*L,-1.5*L,1.5*L,-L/10,L/2)


#%% Modulate and recompute

for wg in WG_Array:
    wg.Source(f0)
    # for dipole in wg.DipoleList: dipole.tune(f0,0.8)
    wg.Modulate(k_b,E_b)
    wg.Compute(f0,compute_type='coupled')


# Plot near fields
mlab.figure(figure=1,bgcolor=(1,1,1), fgcolor=(0.,0.,0.))
mlab.clf()
for wg in WG_Array:
    wg.MayaviPlot(1,plot_fields=True,field_clip=30)
PlotTools.MayaviAxes(10,-1.5*L,1.5*L,-1.5*L,1.5*L,-L/10,L/2)



#%% Find total far-field and plot
HS.ComputeFarField(f0)

# Far field plotting
directivity=HS.FarField.directivity
k_ff=HS.FarField.k
(theta_grid,phi_grid,th_axis,ph_axis,dth,dph)=math.HalfSphereGrid(HS.FarField.Nth,HS.FarField.Nph)
Dir_dB=10*np.log10(np.real(directivity(k_ff(theta_grid,phi_grid)))+1e-16) 
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

print(HS.FarField.Beams)


#%%
# Define k-vector
rhat=math.SphericalBasis('r')
HS.FarField.k=rhat*(2*pi*f/c)

# Find maximum dimension of aperture to find ideal angular resolution for integration
dist=np.zeros([len(HS.DipoleList),len(HS.DipoleList)])
for i, dip1 in enumerate(HS.DipoleList):
    for j, dip2 in enumerate(HS.DipoleList):
        dist[i,j]=(dip1.r0-dip2.r0).norm()
L=np.max(dist)

k0=2*pi*f0/c
dk=2*pi/L
[kx,ky]=np.mgrid[-k0:k0+dk:dk,-k0:k0+dk:dk]
k_grid=xhat*kx+yhat*ky+zhat*sqrt(k0**2-kx**2-ky**2)
dir_grid=directivity(k_grid)
dir_grid[k_grid.norm()>k0]=0
k_grid[zhat]=np.real(k_grid[zhat])

(r,theta,phi)=math.VectorToSphericalCoordinates()
Beams_Theta=np.squeeze(theta(k_grid)[0,0])*180/pi
Beams_Phi=np.squeeze(phi(k_grid)[0,0])*180/pi


