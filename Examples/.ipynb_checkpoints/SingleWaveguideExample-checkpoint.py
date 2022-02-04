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
import worx.HFSS_IO as HFSS_IO

import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.scimath import sqrt
from mayavi import mlab


# constants
pi=np.pi
Z0=376
c=2.998e8

# We are going to operate in a 3D euclidean global geometry.
(xhat,yhat,zhat)=math.CreateEuclideanBasis()
vec_space=[xhat,yhat,zhat]

# Waveguide design
f0=9.8e9
lmbda=c/f0
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
S11MagFile='../HFSS/S11mag_MikeCell_DS_1p1.csv'
S11PhaseFile='../HFSS/S11phase_MikeCell_DS_1p1.csv'
S21MagFile='../HFSS/S21mag_MikeCell_DS_1p1.csv'
S21PhaseFile='../HFSS/S21phase_MikeCell_DS_1p1.csv'
HFSS_Files={'S11Mag':S11MagFile,'S11Phase':S11PhaseFile,
            'S21Mag':S21MagFile,'S21Phase':S21PhaseFile}


# Individual dipole setup for sim
r0=xhat*1.6e-3+yhat*0+zhat*L/2
dip=emag.Dipole(r0)
dip.extract(WG,HFSS_Files)


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
WG.Translate(xhat*-a/2)


# Everything's set up, so just compute at the particular frequency.
for dipole in WG.DipoleList: dipole.tune(f0)
WG.Source(f0)
WG.Compute(f0,compute_type='coupled')
HS.ComputeFarField(f0)

# Plot geometry
mlab.figure(figure=3,bgcolor=(1,1,1), fgcolor=(0.,0.,0.))
mlab.clf()
WG.MayaviPlot(1,plot_fields=True,field_clip=30)
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

#%% Let's do the same study but over a frequency sweep.

f=np.linspace(8e9,12e9,100)
# allocate quantities we want to track
S11=np.zeros(np.shape(f),dtype=complex)
S21=np.zeros(np.shape(f),dtype=complex)
Prad=np.zeros(np.shape(f),dtype=complex)
WG.N_modes=2

# Frequency loop
for i, f0 in enumerate(f):
    
    # Update polarizabilities
    for dip in WG.DipoleList: dip.tune(f0)
    
    # Update source
    WG.Source(f0)
    
    # Compute
    WG.Compute(f0,compute_type='coupled')
    HS.ComputeFarField(f0)
    
    S11[i]=WG.S11
    S21[i]=WG.S21
    Prad[i]=HS.FarField.Prad


# Import HFSS S-params for comparison
S11MagFile='../HFSS/S11mag_MikeCell_5CellArray.csv'
S11PhaseFile='../HFSS/S11phase_MikeCell_5CellArray.csv'
S21MagFile='../HFSS/S21mag_MikeCell_5CellArray.csv'
S21PhaseFile='../HFSS/S21phase_MikeCell_5CellArray.csv'
Prad_file='../HFSS/Prad_MikeCell_5CellArray.csv'
HFSS_Files={'S11Mag':S11MagFile,'S11Phase':S11PhaseFile,
            'S21Mag':S21MagFile,'S21Phase':S21PhaseFile}


(S11_hfss_data,S21_hfss_data,f_hfss)=HFSS_IO.Import_SParams(HFSS_Files)

(Prad_hfss_data,f_hfss_prad)=HFSS_IO.ImportHFSS_Data(Prad_file)
S11_hfss=np.interp(f,f_hfss,S11_hfss_data)
S21_hfss=np.interp(f,f_hfss,S21_hfss_data)
Prad_hfss=np.interp(f,f_hfss_prad,Prad_hfss_data)


# Plotting
plt.close(0)
plt.figure(0)
plt.plot(f,np.abs(S11),label='Mag S11 GoatWorx')
plt.plot(f,np.abs(S21),label='Mag S21 GoatWorx')
plt.plot(f,np.abs(S11_hfss),label='Mag S11 HFSS')
plt.plot(f,np.abs(S21_hfss),label='Mag S21 HFSS')
plt.ylabel('Mag S-param')
plt.xlabel('Frequency [Hz]')
plt.grid()
plt.legend()

plt.close(1)
plt.figure(1)
plt.plot(f,np.angle(S11)*180/pi,label='Angle S11 GoatWorx')
plt.plot(f,np.angle(S21)*180/pi,label='Angle S21 GoatWorx')
plt.plot(f,np.angle(S11_hfss)*180/pi,label='Angle S11 HFSS')
plt.plot(f,np.angle(S21_hfss)*180/pi,label='Angle S21 HFSS')
plt.legend()
plt.grid()
plt.ylabel('Phase [deg]')
plt.xlabel('Frequency [Hz]')

plt.close(2)
plt.figure(2)
plt.plot(f,np.real(Prad),label='GoatWorx')
plt.plot(f,Prad_hfss,label='HFSS')
plt.grid()
plt.title('Radiated Power')
plt.ylabel('Power [W]')
plt.xlabel('Frequency [Hz]')


