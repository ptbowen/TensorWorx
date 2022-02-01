#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 14:10:34 2021

@author: ptbowen
"""

import worx.ElectromagneticObjects as emag
import worx.MathObjects as math
import worx.Geometries as geom
import worx.PlotTools as PlotTools

import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.scimath import sqrt
from mayavi import mlab

pi=np.pi
Z0=376
c=2.998e8

# We are going to operate in a 3D euclidean global geometry.
(xhat,yhat,zhat)=math.CreateEuclideanBasis()
vec_space=[xhat,yhat,zhat]

# In spyder, run the commands %gui qt and %matplotlib in the console before running the rest of this code to allow 
# interactive plots.

#%%  Waveguide modes
# Waveguide geometry
L=30e-3
a=14e-3
b=0.762e-3
n_wg=sqrt(3.55*(1-0.0027j))
WG=geom.Waveguide(a,b,n_wg,L)

# compute waveguide mode
f0=10e9     # frequency
P0=1        # input power
mu=1        # mode number in x direction
nu=0        # mode number in y direction
Mode=emag.WaveguideMode(WG,f0,mu,nu,'TE',P0)

# set up a vectorial coordinate grid
x_axis=np.linspace(0,WG.a,100)
y_axis=np.linspace(0,WG.b,100)
[x,y]=np.meshgrid(x_axis,y_axis)
z=np.zeros(x.shape)
r=xhat*x+yhat*y+zhat*z

# Plot waveguide mode
plt.close(0)
fig = plt.figure(0)
plt.imshow(np.real((Mode.E[yhat])(r)), origin='lower')
plt.colorbar()
plt.set_cmap("RdYlBu")

# Plot the waveguide itself
mlab.figure(figure=1,bgcolor=(1,1,1), fgcolor=(0.,0.,0.))
mlab.clf()
WG.MayaviPlot(1,plot_fields=False)
PlotTools.MayaviAxes(10,-L/2,L/2,-L/10,L,-L/10,L)


#%%  Polarizability Extraction ###

# Data files for polarizability extraction
S11MagFile='./HFSS/S11mag_MikeCell_DS_1p1.csv'
S11PhaseFile='./HFSS/S11phase_MikeCell_DS_1p1.csv'
S21MagFile='./HFSS/S21mag_MikeCell_DS_1p1.csv'
S21PhaseFile='./HFSS/S21phase_MikeCell_DS_1p1.csv'
HFSS_Files={'S11Mag':S11MagFile,'S11Phase':S11PhaseFile,
            'S21Mag':S21MagFile,'S21Phase':S21PhaseFile}

# Create a "waveguide" object that contains all the geometric properties of a waveguide. 
# Give it a, b, and refractive index
WG=geom.Waveguide(a,b,n_wg,L)

# Construct a dipole object.  Initially all it needs is a position, r0
r0=xhat*1.6e-3+yhat*0+zhat*L/2
dip=emag.Dipole(r0)
WG.add_dipole(dip)

# Plot the waveguide again.  This time you'll see the dipole.
mlab.figure(figure=1,bgcolor=(1,1,1), fgcolor=(0.,0.,0.))
mlab.clf()
WG.MayaviPlot(1,plot_fields=False)
PlotTools.MayaviAxes(10,-L/2,L/2,-L/10,L,-L/10,L)

# Extraction
dip.extract(WG,HFSS_Files)

# The extraction function returns polarizabilities as functions that are interpolants over the HFSS frequency data.
# Plot over whatever frequecy range you'd like to define.
f=np.linspace(9e9,11e9,100)
dip.analyze(f)

# To assign the polarizabilities we extracted to the dipole, we use the tuning function to tune it to
# a particular frequency.
f0=10e9
dip.tune(f0)
print(dip.alpha_e.components)
print(dip.alpha_m.components)


#%% Placement of geometries in space
# So far all of this was done assuming the waveguide's coordinates coincided with the global coordinate frame.
# A geometry might be rotated or translated using a rotation tensor and a displacement vector.  This will yield:
# (1) a new set of basis vectors for the local coordinate frame, and
# (2) a coordinate transformation from the global frome to the local frame, and vice versa.

# Right now the propagation axis of the waveguide is in the z-direction, since that is the default.  If our antenna is going
# to lie in the xy plane, then we need to rotate the waveguide by -pi/2 about the x-axis.  First let's make a fresh
# waveguide.
WG=geom.Waveguide(a,b,n_wg,L)

# And assign the dipole to the waveguide.  This way, when we rotate the waveguide it will also rotate the dipoles assigned
# to it.
dip.r0=xhat*1.6e-3+yhat*0+zhat*L/2
WG.add_dipole(dip)

# Now, rotate
R=math.RotTensor(-pi/2,'x',vec_space)
WG.Rotate(R)
dip.tune(f0)    # This updates the polarizabilities of the dipole after the rotation.
                     
mlab.figure(figure=1,bgcolor=(1,1,1), fgcolor=(0.,0.,0.))
mlab.clf()
WG.MayaviPlot(1,plot_fields=False)
PlotTools.MayaviAxes(10,-L/2,L/2,-L/10,L,-L/10,L)

# We can access the dipole's location in the local waveguide coordinates using the
# waveguide's built-in coordinate transformation.
print(WG.GlobalToLocal(dip.r0).components)

print(dip.alpha_e.components)
print(dip.alpha_m.components)

#%% Create a source for a geometry
# While a simulation may involve many geometries that may be coupled together, there needs to be at least one source
# in at least one geometry.  A source defines an incident field in the geometry that excites the dipoles (which may
# or may not be modeled as coupled together via DDA).
WG.Source(f0,amp=1)

# running the compute command on the geometry computes the total field in the waveguide and evaluates the dipole moments
# of any dipoles inside the waveguide.
WG.Compute(f0)

# Plot, including fields this time
mlab.figure(figure=1,bgcolor=(1,1,1), fgcolor=(0.,0.,0.))
mlab.clf()
WG.MayaviPlot(1,plot_fields=True)
PlotTools.MayaviAxes(10,-L/2,L/2,0,L,-L/10,L/2)

#%% Construct single dipole model
# In order to calculate far fields and radiated powers, we need to create a half space geometry
HS=geom.HalfSpace()

# And we need to add the dipole to the half space.  Dipoles that are complimentary metamaterial
# elements belong naturally to two geometries simultaneously.
HS.add_dipole(dip)

# We are ultimately going to plot Pz and Mx, so just allocate memory for these quantities.
Pz=np.zeros(np.shape(f),dtype=complex)
Mx=np.zeros(np.shape(f),dtype=complex)
Prad=np.zeros(np.shape(f))

# The study will be done as a function of frequency
for i, f0 in enumerate(f):
    
    # Update polarizability tensor
    dip.tune(f0)
    
    # Update the waveguide source for this frequency.  
    WG.Source(f0)
    
    # The compute function will excite dipole with the input emag vector field.
    # The default solver is unperturbed feed wave approximation.
    WG.Compute(f0)
    
    # Compute the far field in the half space geometry.
    HS.ComputeFarField(f0)
    
    # Collect the data we are monitoring.
    Pz[i]=dip.P[zhat]
    Mx[i]=dip.M[xhat]
    Prad[i]=HS.FarField.Prad

    

# Plot dipole moments
plt.close(0)
plt.figure(0)
plt.plot(f/1e9,np.abs(Z0*Mx),f/1e9,np.abs(Pz))
plt.grid()

# Plot radiation efficiency
plt.close(1)
plt.figure(1)
plt.plot(f/1e9,Prad/P0)
plt.grid()

#%%  Analyze a waveguide with a large array of dipoles
f0=9.85e9
lmbda=c/f0

# Creating the waveguide
pitch=lmbda
N=30
L=pitch*N
a=14e-3
b=0.762e-3
n_wg=sqrt(3.55*(1-0.0027j))
WG=geom.Waveguide(a,b,n_wg,L)
HS=geom.HalfSpace()

# Create an initial dipole
r0=xhat*1.6e-3+yhat*0+zhat*L/2
dip=emag.Dipole(r0)
dip.extract(WG,HFSS_Files)

# Create dipole array
z_pos=np.arange(pitch/2,(N+1/2)*pitch,pitch)
x_pos=np.ones(np.shape(z_pos))*1.6e-3
y_pos=np.zeros(np.shape(z_pos))
r_pos=math.GridToListOfVectors({xhat:x_pos,yhat:y_pos,zhat:z_pos})
dip_array=dip.Array(r_pos)

# Add all the dipoles to the waveguide.
for dipole in dip_array:
    WG.add_dipole(dipole)
    HS.add_dipole(dipole)

# Now, position it where you want it in space
R=math.RotTensor(pi/2,'x',vec_space)
WG.Rotate(R)

# Once everything is in position, tune the dipoles to ensure they all have the right
# polarizability tensors.
for dipole in dip_array: dipole.tune(f0)

# Add a source
WG.Source(f0)

# Compute
WG.Compute(f0)
HS.ComputeFarField(f0)

# Plot waveguide, and dipoles with near field
mlab.figure(figure=1,bgcolor=(1,1,1), fgcolor=(0.,0.,0.))
mlab.clf()
WG.MayaviPlot(1,plot_fields=True)
PlotTools.MayaviAxes(10,-L/2,L/2,0,L,-L/10,L/2)

# Far field plotting
# this function returns directivity as a function of the k-vector
directivity=HS.FarField.directivity

# This returns the k-vector as a function of theta and phi
k=HS.FarField.k

# create a grid for plotting
(theta_grid,phi_grid,th_axis,ph_axis,dth,dph)=math.HalfSphereGrid(360,90)

# Compute directivity in dB.  This step is where all the computation really happens, since
# the fields defined by the graph of nested function operations are finally executed on a set 
# of grid points.
Dir_dB=10*np.log10(np.real(directivity(k(theta_grid,phi_grid)))+1e-16) 
Dir_dB[Dir_dB<0]=0

# This is the standard routine for plotting 3D far fields.
rhat=math.SphericalBasis('r')
X=Dir_dB*(rhat.dot(xhat))(theta_grid,phi_grid)
Y=Dir_dB*(rhat.dot(yhat))(theta_grid,phi_grid)
Z=Dir_dB*(rhat.dot(zhat))(theta_grid,phi_grid)
mlab.figure(figure=2,bgcolor=(1,1,1), fgcolor=(0.,0.,0.))
mlab.clf()
mlab.mesh(X,Y,Z)
coordmax=np.max([X,Y,Z])
PlotTools.MayaviAxes(10,-coordmax,coordmax,-coordmax,coordmax,0,coordmax)

# Since this was a large periodic array of elements all with the same polarizability, 
# the far-field pattern you will see will be a set of very narrow grating lobes.

