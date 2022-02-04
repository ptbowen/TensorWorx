#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 24 11:00:55 2021

@author: ptbowen
"""
import worx.MathObjects as math

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mayavi import mlab
from numpy.lib.scimath import sqrt
pi=np.pi


#### Scalar Fields ####
# Define scalar fields as functions
f=math.ScalarField((lambda x,y: x*y))
g=math.ScalarField((lambda x,y: x+y))

# But, the math.ScalarField() object returns functions that are operable!  This means we can
# multiply and add them and so on...
h=(f*g)+g**2

# They only evaluate once we provide the arguments with data
print(h(2,3))

# As an example, let's set up a coordinate grid and plot h(x,y)
x_axis=np.linspace(-1,1,100)
y_axis=x_axis
[x,y]=np.meshgrid(x_axis,y_axis)
z=np.zeros(np.shape(x))

plt.figure(0)
plt.imshow(h(x,y),origin='lower')
plt.colorbar()
plt.set_cmap("RdYlBu")



#%%  Euclidean Vector Space ######

#Now we can move from scalar fields to vector fields.  First we define a euclidean basis.
xhat=math.EuclideanBasis('x')
yhat=math.EuclideanBasis('y')
zhat=math.EuclideanBasis('z')

# A basis vector is an abstract object that cannot be reduced to the concept of numbers. It's a "thing".
# The only thing that's postulated about them is their dot products between each other.
# And they have labels:
print(xhat)

# These obey dot product rules:
print(xhat.dot(yhat))
print(xhat.dot(xhat))


#%%  Vectors ###########
#If you add or multiply basis elements, you get a vector...
print(type(xhat*2+yhat*3))

# A vector is an object that uses a dictionary connecting each basis element to a number,
# which is the coefficient of that basis element.
V1=math.Vector({xhat:1,yhat:2,zhat:3})
V2=math.Vector({xhat:3,zhat:4})

# You can index vectors using the basis
print(V1[xhat])

# and you can take dot products...
print(V1.dot(xhat))
print(V1.dot(V2))

# You can list the components of a vector using
print(V1.components)

#%%  Vector Fields ######
#Vector fields are formed using dictionaries connecting each basis element to a scalar field instead of
# to an number.  The same math.Vector object works for both vectors and vector fields.
F1=math.Vector({xhat:f,yhat:g})
F2=math.Vector({xhat:g,yhat:g})

# But unlike dictionaries, they are operable!
F3=F1+F2
plt.imshow(F3[xhat](x,y),origin='lower')


# The dot product of vector fields is a scalar field...
h2=F1.dot(F2)
plt.figure(0)
plt.imshow(h2(x,y),origin='lower')
plt.colorbar()
plt.set_cmap("RdYlBu")



#%%  Tensors
# we define tensor basis elements using two vector basis elements:
xTx=math.TensorElement(xhat,xhat)
xTy=math.TensorElement(xhat,yhat)
yTx=math.TensorElement(yhat,xhat)
yTy=math.TensorElement(yhat,yhat)


# We can define vectors and matrices like so.... 
v1=xhat*2+yhat*3
v2=xhat-yhat
T=xTx*4+xTy*5
print(xTx)
print('Now isnt that pretty...')

# However, beware that you can only right-multiply
# and right-add by numbers.  I'd have to edit the integer and float objects inside python in order
# to get left multiply to work right, and I'm just bystepping that nastiness for the moment.

# Tensors can be defined 
v1Tv2=v1.tensor_prod(v2)
print(xhat.dot((T+v1Tv2).dot(xhat)))

#%% Tensor Fields
# Now we can define tensor fields using the same approach as vectors.  Either expand in a basis:
T=xTx*f+xTy*g

# Or equivalently directly give it a dictionary mappint tensor elements to scalar fields:
T2=math.Tensor({xTx:f,xTy:g})

plt.figure(0)
plt.imshow(xhat.dot(T.dot(xhat)(x,y)),origin='lower')

plt.figure(1)
plt.imshow(f(x,y),origin='lower')
plt.colorbar()



#%%  Rotations on vectors ###
# Here is a subroutine that generates a rotation matrix:
R_Matrix=math.RotMatrix(pi/4,'z')

# Here's how a rotation would work in numpy
v1=np.array([1,1,0]/sqrt(2))
v1p=np.matmul(R_Matrix,v1)
print(v1p)

# Let's do the same in GoatWorx.  Given a vector space (a list of basis elements) we can promote a numpy matrix to a tensor, and then
# take the dot product with a vector to rotate it.  Note that the vector or the tensor can be a subset of the vector space and rotation still works just fine.
vec_space=[xhat,yhat,zhat]
R_Tensor=math.MatrixToTensor(R_Matrix,vec_space)
v2=(xhat*1+yhat*1)/sqrt(2)
v2p=R_Tensor.dot(v2)
print(v2p.components)

# There's also a built-in routine to rotate vectors.
v2p=v2.rotate(R_Tensor)
print(v2p.components)

#%% Rotations on tensors ###
# Tensors can be transposed...
R_Tensor=R_Tensor.T()
v2p=R_Tensor.dot(v2)
print(v2p.components)


# Which means now we can rotate other tensors:
alpha=xTx/2+xTy/2+yTx/2+yTy/2
R_Tensor=math.RotTensor(pi/4,'z',vec_space)
identity=R_Tensor.dot(R_Tensor.T())
alpha2=R_Tensor.dot(alpha.dot(R_Tensor.T()))
print(identity.components)
print(alpha2.components)


#%% Rotations  and translations on scalar fields
# Let's create a scalar field that takes single argument, which is a vector valued input. I'll choose the
# function f=cos(theta).  But to do that, we need to define a function theta that takes a vector r and
# returns theta, so that f(r)=cos(theta(r)), where r is a vector.  There is a built in function to return 
# functions that give the spherical coordinates of a vector:
(r,theta,phi)=math.VectorToSphericalCoordinates()
f=math.ScalarField((lambda r: np.cos(theta(r))))

# now let's define the space of coordinates in vector form,
r=xhat*x+yhat*y

# And plot the scalar field:
plt.figure(0)
plt.imshow(f(r),origin='lower')
plt.colorbar()

# Okay, so for a scalar valued function of a vector, we can rotate the function using a rotation tensor
# applied to the vector input to the function, f'(r)=f(R^Tr)
R=math.RotTensor(pi/10,'z',vec_space)
fp=math.ScalarField((lambda r: f(R.T().dot(r))))

# And plot the rotated scalar field:
plt.figure(1)
plt.imshow(fp(r),origin='lower')
plt.colorbar()

# There's a built in routine for the scalar field object that does just this if supplied with 
# a rotation tensor.  But mind that it only works if the scalar field is defined as a function of a vector.
# Let's test it by rotating backwards this time.
fp=f.rotate(R.T())
plt.figure(2)
plt.imshow(fp(r),origin='lower')
plt.colorbar()

# Translations shift a scalar field by a supplied vector r0, such that f'(r)=f(r-r0).  There is
# a built-in function for this as well.
r0=xhat*(-1/3)+yhat*(1/3)
fp=fp.translate(r0)
plt.figure(3)
plt.imshow(fp(r),origin='lower')
plt.colorbar()

#%% Rotations and translations on vector fields
# To rotate a vector field, you have to rotate both the basis and the coordinates.  The general
# formula is F'(r)=RF(R^Tr), for a rotation tensor R.  I've made a built-in function for this as well
# for vector objects.  Let's test it by first defining a vector field.  Let's choose cos(theta)*theta_hat.
# Broken apart, cos(theta) is a scalar field multiplied by the vector field theta_hat.  Both need to be 
# defined as functions that take a vector in the space as an argument.  For theta that's given by
(r,theta,phi)=math.VectorToSphericalCoordinates()

# And for theta_hat that's, 
(r_hat,th_hat,ph_hat)=math.VectorToSphericalBasis()

# Now let's form the vector field,
F=th_hat*math.ScalarField((lambda r: np.cos(theta(r))))

# And adjust the coordinate grid a bit (just to make it easier to see with quiver),
x_axis=np.linspace(-1,1,20)
y_axis=x_axis
[x,y]=np.meshgrid(x_axis,y_axis)
r=xhat*x+yhat*y

# And then plot using quiver.
plt.figure(0)
plt.quiver(x,y,F.dot(xhat)(r),F.dot(yhat)(r))

# Now, let's rotate this thing by pi/2.
R=math.RotTensor(pi/2,'z',vec_space)
F2=F.rotate(R)

# And then plot using quiver.
plt.figure(1)
plt.quiver(x,y,F2.dot(xhat)(r),F2.dot(yhat)(r))

# Alright, now we can translate vector fields in the same way that we translate scalar fields.
r0=xhat*(-1/3)+yhat*(1/3)
F3=F2.translate(r0)
plt.figure(2)
plt.quiver(x,y,F3.dot(xhat)(r),F3.dot(yhat)(r))


#%% General coordinate transformations
# General coordinate transformations are operations that take vectors in one coordinate frame and return 
# their representation in another coordinate frame.  Most common coordinate transformations can be represented
# using rotations and translations.  But in general, since a coordinate transformation is a 



#%% Spherical coordinates vector space

# I've made some special routines to return the vector fields that define spherical coordinates:
rhat=math.SphericalBasis('r')
th_hat=math.SphericalBasis('theta')
ph_hat=math.SphericalBasis('phi')

# But note that the coordinates for these vector fields are theta and phi, not x,y,z.  So we need
# to define new scalar fields theta and phi.  I've also defined a routine for this:
(r,theta,phi)=math.SphericalCoordinates()

# Since these are just scalar fields, the function doesn't need any arguments, since it just returns
# scalar fields that are assumed to operate on R^3
x_axis=np.linspace(-1,1,100)
y_axis=x_axis
[x,y]=np.meshgrid(x_axis,y_axis)
z=np.zeros(np.shape(x))


# Now we can just use these coordinates to evaluate functions on our coordinate grids:
plt.figure(0)
plt.imshow(yhat.dot(rhat)(theta(x,y,z),phi(x,y,z)))
plt.colorbar()

# But you don't have to operate on euclidean space... let's just look at theta space.
theta=np.linspace(0,2*pi)
phi=(pi/2)*np.ones(np.shape(theta))
plt.figure(1)
plt.plot(theta,xhat.dot(th_hat)(theta,phi))



#%%  Far-field Green's function for dipole at the origin

# First we need some emag constants
c=2.998e8
f=10e9
k=2*pi*f/c

# The green's function is a tensor on tangent space of a sphere:
G=(th_hat.tensor_prod(th_hat)+ph_hat.tensor_prod(ph_hat))*(-k**2/(4*pi))

# The electromagnetic Green's tensor (electric field from magnetic dipoles) is:
F=(th_hat.tensor_prod(ph_hat)-ph_hat.tensor_prod(th_hat))*(k**2/(4*pi))
    
# Let's compute the electric fields from a magnetic dipole.
M=xhat
E=G.dot(M)
Ephi=ph_hat.dot(E)
Etheta=th_hat.dot(E)
Enorm=(E.dot(E))**(1/2)

# Let's create a coordinate grid to plot the fields
th_axis=np.linspace(0,2*pi,360)
ph_axis=np.linspace(0,pi/2,90)
[theta_grid,phi_grid]=np.meshgrid(th_axis,ph_axis)

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
rhat=math.SphericalBasis('r')
R=(rhat*Enorm)(theta_grid,phi_grid) # literally all of the actual computation happens just in this step.
X=R[xhat]
Y=R[yhat]
Z=R[zhat]
R_norm=R.norm()/R.norm().max()
ax.plot_surface(X,Y,Z,facecolors=cm.coolwarm(R_norm),edgecolor='none')
plot_dim=R.norm().max()
ax.set_xlim([-plot_dim,plot_dim])
ax.set_ylim([-plot_dim,plot_dim])
ax.set_zlim([0,1.5*plot_dim])




#%% Far-Field Green's function for dipole at rp
# The Green's tensor is really a tensor valued function of r and rp, where r is
# the field observation point and rp is the location of the dipole.  In the far-field limit,
# we replace r with the k-vector, which is defined as the wavenumber times rhat.  
# This is purely a function of angular coordinates theta and phi.  So to do this, 
# let's get rhat as a vector field defined over theta and phi arguments.
rhat=math.SphericalBasis('r')

# Then the k-vector is then,
c=2.998e8
f=9.89e9
k0=2*pi*f/c
k=rhat*k0

# Let's then define a second vector which will represent the position of the dipole.
rp=yhat*0.5

# Alright, so the tensorial part of the Green's function just takes the k-vector as arguments.
# This function returns three vector fields, which are the spherical basis vectors.  The arguments 
# of these vector fields will be a vector, and they calculates th spherical basis vectors at the location
# of the vector provided in the arguments.
(r_hat,th_hat,ph_hat)=math.VectorToSphericalBasis()

# The tensorial bit of the Green's tensor is constructed in the same way using these new vector fields
T=(th_hat.tensor_prod(th_hat)+ph_hat.tensor_prod(ph_hat))

# However, T is a tensor that takes only k as an argument, not rp.  But in order to use it to construct a
# Green's function using our typical rules for multiplication of scalar fields, i.e. h(x)=f(x)*g(x), the 
# arguments for the two functions that are being multiplied need to be the same.  So in order to make 
# them the same, we will add a dummy variable to T(k), so that T(k) becomes T(k,rp), but rp doesn't 
# do anything.
T=T.add_dummy()

# Now this gets multiplied by a scalar field that depends two vectors: k and rp
g=math.ScalarField((lambda k,rp: k.norm()**2*np.exp(k.dot(rp)*1j)*(-1/(4*pi))))

# Putting these together, we get the free space far-field Green's function on k and rp.
G=T*g

# Again, we can choose dipole moment M and calculate the magnetic field H.
M=xhat

# G depends on k and rp, but we want to return H(k), not H(k,rp).  So we evaluate the rp variable
# to eleminate it in the Green's function before taking the dot product.
H=(G.evaluate(rp,1)).dot(M)
Hphi=ph_hat.dot(H)
Htheta=th_hat.dot(H)
Hnorm=H.norm()
Z0=376
Sr=H.dot(H.conj())*Z0

# Let's create a coordinate grid to plot the fields: 360 points in theta and 90 points in phi.
# (theta_grid,phi_grid,th_axis,ph_axis,dth,dph)=math.HalfSphereGrid(360,90)
dth=pi/180
dph=dth
(theta_grid,phi_grid)=np.mgrid[-pi:pi:dth,0:pi/2:dph]


# Directivity
Prad=np.trapz(np.trapz(Sr(k(theta_grid,phi_grid))*np.sin(phi_grid)))*dth*dph
directivity=Sr*(4*pi/Prad)
dir_max=directivity(k(theta_grid,phi_grid)).max()


# Now we plot in the usual way, but remember Hnorm is defined on k and rp.
R=(r_hat*directivity)(k(theta_grid,phi_grid)) # literally all of the actual computation happens just in this step.
X=np.real(R[xhat])
Y=np.real(R[yhat])
Z=np.real(R[zhat])
R_norm=np.real(R.norm()/R.norm().max())

mlab.figure(1)
mlab.clf()
mlab.axes()
mlab.outline()
mlab.view(-45,-125)
mlab.mesh(X,Y,Z)

# plt.figure(1)
# fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
# R_norm=np.real(R.norm()/R.norm().max())
# ax.plot_surface(X,Y,Z,facecolors=cm.coolwarm(R_norm),edgecolor='none')
# plot_dim=R.norm().max()
# ax.set_xlim([-plot_dim,plot_dim])
# ax.set_ylim([-plot_dim,plot_dim])
# ax.set_zlim([0,1.5*plot_dim])


#%%
#### Combined Products ####
# Define scalar fields as functions of single arguments
f=math.ScalarField((lambda x: np.sin(x)))
g=math.ScalarField((lambda y: np.sin(y)))

# The combined product creates a new function of two variables
# such that h(x,y)=f(x)*g(y), as opposed to the normal product which creates
# a new function of a single variable h(x)=f(x)*g(x):
h=f.comb_prod(g)

# They only evaluate once we provide the arguments with data
print(h(2,3))

# As an example, let's set up a coordinate grid and plot h(x,y)
x_axis=np.linspace(0,pi,100)
y_axis=x_axis
[x,y]=np.meshgrid(x_axis,y_axis)
z=np.zeros(np.shape(x))

plt.figure(0)
plt.imshow(h(x,y),origin='lower')
plt.colorbar()
plt.set_cmap("RdYlBu")

# Let's do the same with vector fields.
F1=xhat*f
F2=yhat*g
F3=F1.comb_prod(g)
T1=F1.comb_tensor_prod(F2)
print(T1(2,3).components)



        