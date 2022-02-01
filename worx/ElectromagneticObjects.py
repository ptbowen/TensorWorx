#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 16:12:21 2021

@author: ptbowen
"""

import numpy as np
import scipy.interpolate
import copy
import matplotlib.pyplot as plt
from numpy.lib.scimath import sqrt

import worx.HFSS_IO as HFSS_IO
import worx.PlotTools as PlotTools
import worx.MathObjects as math

pi=np.pi
c=2.998e8
Z0=376
(xhat,yhat,zhat)=math.CreateEuclideanBasis()
vec_space=[xhat,yhat,zhat]


class Dipole(object):
    def __init__(self,r0):
        
        self.alpha_e=math.ZeroTensor(vec_space)
        self.alpha_m=math.ZeroTensor(vec_space)
        self.P=math.ZeroVector(vec_space)
        self.M=math.ZeroVector(vec_space)
        self.r0=r0
        self.vec_space=vec_space
        self.PlotRadius=1.5e-3
        self.Geometries={}
        self.nu_m=xhat
        self.nu_e=yhat
        self.TuningFunction_m=math.ScalarField(lambda x: 0)
        self.TuningFunction_e=math.ScalarField(lambda x: 0)
        self.m=0
        self.Parameterized=0
        
        self.GeometryList=[]
        
        return None

    def Rotate(self,R):
        R=copy.copy(R)
        
        self.alpha_e=R.dot(self.alpha_e.dot(R.T()))
        self.alpha_m=R.dot(self.alpha_m.dot(R.T()))
        self.nu_e=R.dot(self.nu_e)
        self.nu_m=R.dot(self.nu_m)
        self.P=R.dot(self.P)
        self.M=R.dot(self.M)
        self.r0=R.dot(self.r0)

        return None
    
    def Translate(self,v):
        self.r0=self.r0+v

    def Excite(self,field):
       
        self.P=self.alpha_e.dot(field.E(self.r0))
        self.M=self.alpha_m.dot(field.H(self.r0))

    def Array(self,vec_list):
        dip_list=[0]*len(vec_list)
        for i in range(0,len(dip_list)):
            dip_list[i]=copy.deepcopy(self)
            dip_list[i].r0=vec_list[i]
            
        return dip_list
    
    def plot(self,ax,scale,arrow_scale):
        PlotTools.PlotSphere(ax,self.PlotRadius/scale,self.r0/scale)
        
        (lmbda,eigenvectors)=self.alpha_e.eig(self.vec_space)
        (u,v,w)=math.ListOfVectorsToGrid(eigenvectors)
        (x,y,z)=math.ListOfVectorsToGrid([self.r0,self.r0,self.r0])
        ax.quiver(x/scale,y/scale,z/scale,np.abs(u)/arrow_scale,np.abs(v)/arrow_scale,np.abs(w)/arrow_scale,cmap='Reds')
        
        (lmbda,eigenvectors)=self.alpha_m.eig(self.vec_space)
        (u,v,w)=math.ListOfVectorsToGrid(eigenvectors)
        (x,y,z)=math.ListOfVectorsToGrid([self.r0,self.r0,self.r0])
        ax.quiver(x/scale,y/scale,z/scale,np.abs(u)/arrow_scale,np.abs(v)/arrow_scale,np.abs(w)/arrow_scale,cmap='Blues')

    def tune(self,*args):
        self.alpha_e=self.nu_e.tensor_prod(self.nu_e)*self.TuningFunction_e(*args)
        self.alpha_m=self.nu_m.tensor_prod(self.nu_m)*self.TuningFunction_m(*args)
        
        if self.Parameterized==1:
            self.m=args[1]
        
    
    def extract(self,waveguide,HFSS_Files,freq_sweep=1,parameterized=0):
        
        
        if freq_sweep==1 and parameterized==1:
            (S11,S21,f_hfss,param_hfss)=HFSS_IO.Import_SParams_Parameterized(HFSS_Files)
            f_hfss=f_hfss[:,0]
            param_hfss=param_hfss[0,:]
            mod=(param_hfss-np.min(param_hfss))/(np.max(param_hfss)-np.min(param_hfss))
            
            alpha_ey_data=np.zeros((len(f_hfss),len(param_hfss)),dtype=complex)
            alpha_mx_data=np.zeros((len(f_hfss),len(param_hfss)),dtype=complex)
            
            r0=waveguide.GlobalToLocal(self.r0)
            r0=xhat*r0[xhat]+yhat*r0[yhat]
            
            for i, f in enumerate(f_hfss):
                k=2*pi*f/c
                Mode=WaveguideMode(waveguide,f,1,0,'TE',1)
                
                for j, m in enumerate(param_hfss):
                    alpha_ey_data[i][j]=-1j*Z0*Mode.N/(2*k*(Mode.E[yhat](r0)**2))*(S21[i][j]+S11[i][j]-1)
                    alpha_mx_data[i][j]=-1j*Mode.N/(2*Z0*k*(Mode.H[xhat](r0)**2))*(S21[i][j]-S11[i][j]-1)
                    
            alpha_ey_re=scipy.interpolate.RectBivariateSpline(f_hfss,mod,np.real(alpha_ey_data))
            alpha_ey_im=scipy.interpolate.RectBivariateSpline(f_hfss,mod,np.imag(alpha_ey_data))
            alpha_ey=math.ScalarField(alpha_ey_re)+math.ScalarField(alpha_ey_im)*1j
    
            alpha_mx_re=scipy.interpolate.RectBivariateSpline(f_hfss,mod,np.real(alpha_mx_data))
            alpha_mx_im=scipy.interpolate.RectBivariateSpline(f_hfss,mod,np.imag(alpha_mx_data))
            alpha_mx=math.ScalarField(alpha_mx_re)+math.ScalarField(alpha_mx_im)*1j

            self.nu_m=waveguide.xhat_loc
            self.nu_e=waveguide.yhat_loc
            self.TuningFunction_m=alpha_mx
            self.TuningFunction_e=alpha_ey
            self.Parameterized=1
            self.ParameterFunction=(lambda m: np.interp(m,mod,param_hfss))
        
        if freq_sweep==1 and parameterized==0:
            (S11,S21,f)=HFSS_IO.Import_SParams(HFSS_Files)
            
            r0=waveguide.GlobalToLocal(self.r0)
            r0=xhat*r0[xhat]+yhat*r0[yhat]
            
            alpha_ey_data=np.zeros(np.shape(f),dtype=complex)
            alpha_mx_data=np.zeros(np.shape(f),dtype=complex)
            for i in range(0,len(f)):
                k=2*pi*f[i]/c
                Mode=WaveguideMode(waveguide,f[i],1,0,'TE',1)
                
                alpha_ey_data[i]=-1j*Z0*Mode.N/(2*k*(Mode.E[yhat](r0)**2))*(S21[i]+S11[i]-1)
                alpha_mx_data[i]=-1j*Mode.N/(2*Z0*k*(Mode.H[xhat](r0)**2))*(S21[i]-S11[i]-1)

            def alpha_ey(f0):
                alpha=np.interp(f0,f,alpha_ey_data,left=0,right=0)
                return alpha
            
            def alpha_mx(f0):
                alpha=np.interp(f0,f,alpha_mx_data,left=0,right=0)
                return alpha
            
            self.nu_m=waveguide.xhat_loc
            self.nu_e=waveguide.yhat_loc
            self.TuningFunction_m=alpha_mx
            self.TuningFunction_e=alpha_ey
            
            
    def analyze(self,f=0,mod=0):
        if self.Parameterized==1:
            if isinstance(f,np.ndarray):

                fig,(AxRe,AxIm)=plt.subplots(nrows=2,sharex=True,sharey=True)
                
                for m in mod:
                    AxRe.plot(f/1e9,np.real(self.TuningFunction_m(f,m))*1e9,label=('m='+str(m)))
                AxRe.grid()
                AxRe.set_title('Magnetic polarizability, real part')
                AxRe.set_ylabel('Polarizability [mm^3]')
                AxRe.legend()
                
                for m in mod:
                    AxIm.plot(f/1e9,np.imag(self.TuningFunction_m(f,m))*1e9,label=('m='+str(m)))
                AxIm.grid()
                AxIm.set_title('Magnetic polarizability, imaginary part')
                AxIm.set_xlabel('Frequency [GHz]')
                AxIm.set_ylabel('Polarizability [mm^3]')
                AxIm.legend()
                
                fig,(AxRe,AxIm)=plt.subplots(nrows=2,sharex=True,sharey=True)
                
                for m in mod:
                    AxRe.plot(f/1e9,np.real(self.TuningFunction_e(f,m))*1e9,label=('m='+str(m)))
                AxRe.grid()
                AxRe.set_title('Electric polarizability, real part')
                AxRe.set_ylabel('Polarizability [mm^3]')
                AxRe.legend()
                
                for m in mod:
                    AxIm.plot(f/1e9,np.imag(self.TuningFunction_e(f,m))*1e9,label=('m='+str(m)))
                AxIm.grid()
                AxIm.set_title('Electric polarizability, imaginary part')
                AxIm.set_xlabel('Frequency [GHz]')
                AxIm.set_ylabel('Polarizability [mm^3]')
                AxIm.legend()
                
            else:
                fig,(AxM,AxP)=plt.subplots(nrows=1,ncols=2,sharex=True,sharey=True,subplot_kw=dict(polar=True))
                AxM.scatter(np.angle(self.TuningFunction_m(f,mod)),
                            np.abs(self.TuningFunction_m(f,mod)*1e9))
                AxM.set_title('Magnetic polarizability [mm^3]')
                
                AxP.scatter(np.angle(self.TuningFunction_e(f,mod)),
                            np.abs(self.TuningFunction_e(f,mod)*1e9))
                AxP.set_title('Electric polarizability [mm^3]')
        
        else:
            if isinstance(f,np.ndarray):

                fig,(AxRe,AxIm)=plt.subplots(nrows=2,sharex=True,sharey=True)

                AxRe.plot(f/1e9,np.real(self.TuningFunction_m(f))*1e9)
                AxRe.grid()
                AxRe.set_title('Magnetic polarizability, real part')
                AxRe.set_ylabel('Polarizability [mm^3]')

                AxIm.plot(f/1e9,np.imag(self.TuningFunction_m(f))*1e9)
                AxIm.grid()
                AxIm.set_title('Magnetic polarizability, imaginary part')
                AxIm.set_xlabel('Frequency [GHz]')
                AxIm.set_ylabel('Polarizability [mm^3]')
                
                fig,(AxRe,AxIm)=plt.subplots(nrows=2,sharex=True,sharey=True)

                AxRe.plot(f/1e9,np.real(self.TuningFunction_e(f))*1e9)
                AxRe.grid()
                AxRe.set_title('Electric polarizability, real part')
                AxRe.set_ylabel('Polarizability [mm^3]')

                AxIm.plot(f/1e9,np.imag(self.TuningFunction_e(f))*1e9)
                AxIm.grid()
                AxIm.set_title('Electric polarizability, imaginary part')
                AxIm.set_xlabel('Frequency [GHz]')
                AxIm.set_ylabel('Polarizability [mm^3]')
                
                

class EM_VectorField(object):
    def __init__(self):
        self.E=math.Vector({})
        self.H=math.Vector({})
        
    def __add__(self,other):
        return EM_VectorField(self.E+other.E,self.H+other.H)
    
    def __sub__(self,other):
        return EM_VectorField(self.E-other.E,self.H-other.H)
    
    @property
    def E(self):
        return self._E
    @E.setter
    def E(self,VF):
        self._E=VF
    @E.deleter
    def E(self):
        del self._E
            
    @property
    def H(self):
        return self._H
    @H.setter
    def H(self,VF):
        self._H=VF
    @H.deleter
    def H(self):
        del self._H

class FarField(EM_VectorField):
    def __init__(self):
        EM_VectorField.__init__(self)

        
        self.k=math.Vector({})

        self.Prad=0
        
        return None


class WaveguideMode(EM_VectorField):
    def __init__(self,WG,f,mu=1,nu=0,polarization='TE',P0=1,orientation='+'):
        
        lmbda=c/f
        k=2*pi/lmbda
        a=WG.a
        b=WG.b
        n=WG.n_wg
        
        (xhat,yhat,zhat)=math.CreateEuclideanBasis()
        
        gamma=sqrt((mu*pi/a)**2+(nu*pi/b)**2)
        beta=sqrt(n**2*k**2-gamma**2);
        E0=sqrt(P0*8*Z0*gamma**2/(k*np.real(beta)*a*b*(1+int(mu==0))*(1+int(nu==0))));
        
        if orientation=='-':
            beta=-beta
        
        if gamma!=0:
            N=-(E0**2/Z0)*k*beta*a*b*(1+int(mu==0))*(1+int(nu==0))/(2*gamma**2)
        else:
            N=1
        
        if polarization=='TE':
            def Hz(r):
                Hz=1j*(E0/Z0)*np.cos(mu*pi*r.dot(xhat)/a)*np.cos(nu*pi*r.dot(yhat)/b)
                return Hz
            Hz=math.ScalarField(Hz)
            
            def Hx(r):
                Hx=(E0/Z0)*(beta*(mu*pi/a)/gamma**2)*np.sin(mu*pi*r.dot(xhat)/a)*np.cos(nu*pi*r.dot(yhat)/b)
                return Hx
            Hx=math.ScalarField(Hx)
            
            def Hy(r):
                Hy=(E0/Z0)*(beta*(nu*pi/b)/gamma**2)*np.cos(mu*pi*r.dot(xhat)/a)*np.sin(nu*pi*r.dot(yhat)/b)
                return Hy
            Hy=math.ScalarField(Hy)
            
            Ex=Hy*(k*Z0/beta)
            Ey=Hx*(-k*Z0/beta)
            Ez=math.ScalarField(lambda r: 0)
            
        
        elif polarization=='TM':
            def Ez(r):
                Ez=E0*np.sin(mu*pi*r.dot(xhat)/a)*np.sin(nu*pi*r.dot(yhat)/b)
                return Ez
            Ez=math.ScalarField(Ez)
            
            def Ex(r):
                Ex=E0*(-1j*beta*(mu**pi/a)/gamma**2)*np.cos(mu*pi*r.dot(xhat)/a)*np.sin(nu*pi*r.dot(yhat)/b)
                return Ex
            Ex=math.ScalarField(Ex)
            
            def Ey(r):
                Ey=E0*(-1j*beta*(mu**pi/b)/gamma**2)*np.cos(mu*pi*r.dot(xhat)/a)*np.sin(nu*pi*r.dot(yhat)/b)
                return Ey
            Ey=math.ScalarField(Ey)
            
            Hz=math.ScalarField(lambda r: 0)
            Hx=Ey*(-k/(Z0*beta))
            Hy=Ex*(k/(Z0*beta))
            
        propagation=math.ScalarField(lambda r: np.exp(-1j*beta*(r.dot(zhat))))
        E=(xhat*Ex + yhat*Ey + zhat*Ez)*propagation
        H=(xhat*Hx + yhat*Hy + zhat*Hz)*propagation
            

        EM_VectorField.__init__(self)
        self.E=E
        self.H=H
        self.polarization=polarization
        self.mu=mu
        self.nu=nu
        self.beta=beta
        self.f=f
        self.N=N
        
        
        return None
    

