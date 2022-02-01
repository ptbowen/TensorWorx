#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 14:11:19 2021

@author: ptbowen
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 17:15:09 2021

@author: ptbowen
"""
import worx.ElectromagneticObjects as emag
import worx.MathObjects as math
import worx.Geometries as geom
from worx.HFSS_IO import Import_SParams
import worx.PlotTools as PlotTools

import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.scimath import sqrt
from mayavi import mlab
pi=np.pi

# We are going to operate in a 3D euclidean global geometry.
(xhat,yhat,zhat)=math.CreateEuclideanBasis()
vec_space=[xhat,yhat,zhat]

# Waveguide design
L=50e-3
a=14e-3
b=0.762e-3
n_wg=sqrt(3.55*(1-0.0027j))
WG=geom.Waveguide(a,b,n_wg,L)
WG.N_modes=2

# Half-space setup
HS=geom.HalfSpace()

# Data files for polarizability extraction
S11MagFile='../HFSS/S11mag_MikeCell_DS_1p1.csv'
S11PhaseFile='../HFSS/S11phase_MikeCell_DS_1p1.csv'
S21MagFile='../HFSS/S21mag_MikeCell_DS_1p1.csv'
S21PhaseFile='../HFSS/S21phase_MikeCell_DS_1p1.csv'
HFSS_Files={'S11Mag':S11MagFile,'S11Phase':S11PhaseFile,
            'S21Mag':S21MagFile,'S21Phase':S21PhaseFile}

# Create dipole
r0=xhat*1.6e-3+yhat*0+zhat*L/2
dip=emag.Dipole(r0)
dip.extract(WG,HFSS_Files)

# Add dipole to geometries
WG.add_dipole(dip)
HS.add_dipole(dip)

# Now, position it where you want it in space
R=math.RotTensor(pi/2,'x',vec_space)
WG.Rotate(R)

# Plot geometry
mlab.figure(figure=1,bgcolor=(1,1,1), fgcolor=(0.,0.,0.))
mlab.clf()
WG.MayaviPlot(1,plot_fields=False)
PlotTools.MayaviAxes(10,-L/3,L/3,0,L,-L/10,L/2)


#%%# First, let's define a frequency range and see if we can recompute the same S-parameters
# as were used to extract the polarizability over that frequency range.  We will also
# compute the radiated power to compare with HFSS.
f=np.linspace(9e9,11e9,100)
S11=np.zeros(np.shape(f),dtype=complex)
S21=np.zeros(np.shape(f),dtype=complex)
Prad=np.zeros(np.shape(f),dtype=complex)

for i, f0 in enumerate(f):

    # tune dipole 
    dip.tune(f0)
    
    # Define source
    WG.Source(f0,source_type='mode')
    
    # Compute to excite dipoles with source
    WG.Compute(f0,compute_type='single_scattering',DeEmbedPort1=L/2,DeEmbedPort2=L/2)
    HS.ComputeFarField(f0)
    
    # Calculate far field
    Prad[i]=HS.FarField.Prad
    S11[i]=WG.S11
    S21[i]=WG.S21


# Load HFSS data for comparison
(S11_hfss_data,S21_hfss_data,f_hfss)=Import_SParams(HFSS_Files)
S11_hfss=np.interp(f,f_hfss,S11_hfss_data)
S21_hfss=np.interp(f,f_hfss,S21_hfss_data)

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
plt.plot(f,np.real(Prad))
plt.grid()
plt.ylabel('Power [W]')
plt.xlabel('Frequency [Hz]')



#%% Single Frequency analysis

# Next, let's pick one particular frequency and plot the fields
f0=9.85e9

# Choose the precision of the reconstruction by changing the number of modes to include.
WG.N_modes=100

# Everything's already set up, so just compute at the particular frequency.
dip.tune(f0)
WG.Source(f0)
WG.Compute(f0,compute_type='single_scattering',DeEmbedPort1=L/2,DeEmbedPort2=L/2)
HS.ComputeFarField(f0)

# Plot geometry
mlab.figure(figure=1,bgcolor=(1,1,1), fgcolor=(0.,0.,0.))
mlab.clf()
WG.MayaviPlot(1,plot_fields=True,phase=0*pi/6,field_clip=30)
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

