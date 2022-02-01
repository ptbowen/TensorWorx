#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 14:10:34 2021

@author: ptbowen
"""

import ElectromagneticObjects as emag
import MathObjects as math
import numpy as np
import matplotlib.pyplot as plt
from Geometries import Waveguide
from Extraction import WaveguideExtraction
from numpy.lib.scimath import sqrt
from matplotlib import cm


# constants
pi=np.pi
Z0=376
c=2.998e8

# We are going to operate in a 3D euclidean global geometry.
(xhat,yhat,zhat)=math.CreateEuclideanBasis()
vec_space=[xhat,yhat,zhat]

# In spyder, run the command %matplotlib in the console before running the rest of this code to allow 
# interactive plots.

#%%  Waveguide modes
# Waveguide geometry
L=30e-3
a=14e-3
b=0.762e-3
n_wg=sqrt(3.55*(1-0.0027j))
WG=Waveguide(a,b,n_wg,L)

# compute waveguide mode
f0=10e9     # frequency
P0=1        # input power
mu=1        # mode number in x direction
nu=0        # mode number in y direction
Mode=emag.WaveguideMode(WG,f0,mu,nu,'TE',P0)

# set up a coordinate grid
x_axis=np.linspace(0,WG.a,100)
y_axis=np.linspace(0,WG.b,100)
[x,y]=np.meshgrid(x_axis,y_axis)
z=np.zeros(x.shape)
r=xhat*x+yhat*y+zhat*z

zorder_background = 0
zorder_foreground = 1


# Plot waveguide mode
plt.close()
fig0, ax0 = plt.subplots()
ctf = ax0.imshow(np.real((Mode.E[yhat])(r,0)),
                 origin='lower', 
                 zorder = zorder_foreground,
                  cmap = 'RdYlBu',
                 )
plt.colorbar(ctf, ax = ax0)
# ax0.set_cmap("RdYlBu")

# Plot the waveguide itself
scale=1e-3
plt.close(1)

fig1, ax1 = plt.subplots(nrows = 1, ncols = 1, subplot_kw=dict(projection='3d'))
WG.plot(ax1,scale)
plt.show()
ax1.set_xlim([-L/2/scale,L/2/scale])
ax1.set_ylim([-L/2/scale,L/2/scale])
ax1.set_zlim([0,L/scale])


#%%  Polarizability Extraction ###

# Data files for polarizability extraction
S11MagFile='./HFSS/S11mag_MikeCell_DS_1p1.csv'
S11PhaseFile='./HFSS/S11phase_MikeCell_DS_1p1.csv'
S21MagFile='./HFSS/S21mag_MikeCell_DS_1p1.csv'
S21PhaseFile='./HFSS/S21phase_MikeCell_DS_1p1.csv'
HFSS_SParamFiles=[S11MagFile,S11PhaseFile,S21MagFile,S21PhaseFile]

# Create a "waveguide" object that contains all the geometric properties of a waveguide. 
# Give it a, b, and refractive index
WG=Waveguide(a,b,n_wg,L)

# Construct a dipole object.  Initially all it needs is a position, r0
r0=xhat*1.6e-3+yhat*0
dip1=emag.Dipole(r0)

# Extraction
(alpha_ey,alpha_mx)=WaveguideExtraction(dip1.r0,WG,HFSS_SParamFiles,'EM')

# The extraction function returns polarizabilities as functions that are interpolants over the HFSS frequency data.
# Plot over whatever frequecy range you'd like to define.
plt.close(0)
plt.figure(0)
f=np.linspace(9e9,11e9,100)
plt.plot(f/1e9,np.real(alpha_mx(f))*1e9,f/1e9,np.imag(alpha_mx(f))*1e9)
plt.grid()

# To assign the polarizabilities we extracted to the dipole, we pick a frequency and use a tensor element.
f0=10e9
dip1.alpha_m=xhat.tensor_prod(xhat)*alpha_mx(f0)
dip1.alpha_e=yhat.tensor_prod(yhat)*alpha_ey(f0)


#%% Placement of geometries in space
# So far all of this was done assuming the waveguide's coordinates coincided with the global coordinate frame.
# A geometry might be rotated or translated using a rotation tensor and a displacement vector.  This will yield:
# (1) a new set of basis vectors for the local coordinate frame, and
# (2) a coordinate transformation from the global frome to the local frame, and vice versa.

# Right now the propagation axis of the waveguide is in the z-direction, since that is the default.  If our antenna is going
# to lie in the xy plane, then we need to rotate the waveguide by -pi/2 about the x-axis.  First let's make a fresh
# waveguide.
WG=Waveguide(a,b,n_wg,L)

# And assign the dipole to the waveguide.  This way, when we rotate the waveguide it will also rotate the dipoles assigned
# to it.
r0=xhat*1.6e-3+yhat*0+zhat*L/2
dip1=emag.Dipole(r0)
f0=10e9
dip1.alpha_m=xhat.tensor_prod(xhat)*alpha_mx(f0)
dip1.alpha_e=yhat.tensor_prod(yhat)*alpha_ey(f0)
WG.add_dipole(dip1)

# Now, rotate
R=math.RotTensor(-pi/2,'x',vec_space)
WG.Rotate(R)

# These are the definitions of the new local basis in terms of the global basis elements
print(WG.xhat_loc.components)
print(WG.yhat_loc.components)
print(WG.zhat_loc.components)
                     

# Plot the rotated waveguide
scale=1e-3
plt.close(1)
fig = plt.figure(1)
ax = fig.add_subplot(111, projection='3d')
WG.plot(ax,scale,top_alpha=0)
# plt.show()
ax.set_xlim([-L/scale/4,(3/4)*L/scale])
ax.set_ylim([0,L/scale])
ax.set_zlim([-L/scale/4,(3/4)*L/scale])

print(WG.GlobalToLocal(dip1.r0))

#%% Create a source for a geometry
# While a simulation may involve many geometries that may be coupled together, there needs to be at least one source
# in at least one geometry.  A source defines an incident field in the geometry that excites the dipoles (which may
# or may not be modeled as coupled together via DDA).
amp=1
WG.Source(f0,amp)

# running the compute command on the geometry computes the total field in the waveguide and evaluates the dipole moments
# of any dipoles inside the waveguide.
WG.Compute()

# Plot, including fields this time
scale=1e-3
plt.close(1)
fig = plt.figure(1)
ax = fig.add_subplot(111, projection='3d')
WG.plot(ax,scale,top_alpha=0,plot_fields=True)
plt.show()
ax.set_xlim([-L/scale/4,(3/4)*L/scale])
ax.set_ylim([0,L/scale])
ax.set_zlim([-L/scale/4,(3/4)*L/scale])

#%% Construct single dipole model


# We are ultimately going to plot Pz and Mx, so just allocate memory for these quantities.
Pz=np.zeros(np.shape(f),dtype=complex)
Mx=np.zeros(np.shape(f),dtype=complex)
Prad=np.zeros(np.shape(f))

# The study will be done as a function of frequency
ii=0
for f0 in f:
    
    # Compute polarizability tensor and assign to dipole
    dip1.alpha_m=WG.xhat_loc.tensor_prod(WG.xhat_loc)*alpha_mx(f0)
    dip1.alpha_e=WG.yhat_loc.tensor_prod(WG.yhat_loc)*alpha_ey(f0)
    
    # Compute the waveguide mode for this frequency.  The other arguments are set to their defaults.
    # The default P0 is 1W.
    WG.Source(f0,amp)
    
    # The compute function will excite dipole with the input emag vector field
    WG.Compute()
    
    # Collect the Pz and Mx components (in global coordinates) and save.
    Pz[ii]=dip1.P[zhat]
    Mx[ii]=dip1.M[xhat]
    
    # Compute far field
    (directivity,H,Prad[ii])=emag.FarField([dip1],f0)
    
    ii+=1
    

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
pitch=lmbda*1.2
N=5
L=pitch*N
a=14e-3
b=0.762e-3
n_wg=sqrt(3.55*(1-0.0027j))
WG=Waveguide(a,b,n_wg,L)

# Create an initial dipole
r0=xhat*1.6e-3+yhat*0+zhat*L/2
dip1=emag.Dipole(r0)
dip1.alpha_m=xhat.tensor_prod(xhat)*alpha_mx(f0)
dip1.alpha_e=yhat.tensor_prod(yhat)*alpha_ey(f0)
# WG.add_dipole(dip1)

# Create dipole array
z_pos=np.arange(pitch/2,(N+1/2)*pitch,pitch)
x_pos=np.ones(np.shape(z_pos))*1.6e-3
y_pos=np.zeros(np.shape(z_pos))
r_pos=math.GridToListOfVectors({xhat:x_pos,yhat:y_pos,zhat:z_pos})
dip_array=dip1.Array(r_pos)

# Add all the dipoles to the waveguide.
for dip in dip_array:
    WG.add_dipole(dip)

# Now, position it where you want it in space
R=math.RotTensor(-pi/2,'x',vec_space)
WG.Rotate(R)

# Add a source
WG.Source(f0)

# Compute
WG.Compute()

# Plot waveguide, and dipoles with near field
scale=1e-3
plt.close(0)
fig = plt.figure(0)
ax = fig.add_subplot(111, projection='3d')
WG.plot(ax,scale,top_alpha=0,plot_fields=True)
plt.show()
ax.set_xlim([-L/scale/2,(1/2)*L/scale])
ax.set_ylim([0,L/scale])
ax.set_zlim([-L/scale/4,(3/4)*L/scale])

# Create far field
(directivity,H,Prad)=emag.FarField(dip_array,f0)

# Define k-vector for far-field plot
(r_hat,th_hat,ph_hat)=math.VectorToSphericalBasis()
rhat=math.SphericalBasis('r')
k0=2*pi*f0/c
k=rhat*k0

# Create coordinate grid for far-field plot
(theta_grid,phi_grid,th_axis,ph_axis,dth,dph)=math.HalfSphereGrid(360,180)

# Plot far-field
R=(r_hat*directivity)(k(theta_grid,phi_grid)) # literally all of the actual computation happens just in this step.
X=np.real(R[xhat])
Y=np.real(R[yhat])
Z=np.real(R[zhat])
plt.close(1)
plt.figure(1)
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
R_norm=np.real(R.norm()/R.norm().max())
ax.plot_surface(X,Y,Z,facecolors=cm.coolwarm(R_norm),edgecolor='none')
plot_dim=R.norm().max()
ax.set_xlim([-plot_dim,plot_dim])
ax.set_ylim([-plot_dim,plot_dim])
ax.set_zlim([0,1.5*plot_dim])

