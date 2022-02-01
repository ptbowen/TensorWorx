#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 18:55:59 2021

@author: ptbowen
"""
import numpy as np
import worx.MathObjects as math
import worx.HFSS_IO 
import scipy
from worx.ElectromagneticObjects import WaveguideMode
    
Z0=376
c=2.998e8
pi=np.pi
(xhat,yhat,zhat)=math.CreateEuclideanBasis()

def WaveguideExtraction(r0,waveguide,HFSS_SParamFiles,ExtractionType):
    
    
    (S11,S21,f)=HFSS_IO.Import_SParams(HFSS_SParamFiles)
    
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
    
    return (alpha_ey,alpha_mx)




def WaveguideExtraction_Parameterized(r0,waveguide,HFSS_SParamFiles):
    
    (S11,S21,f_hfss,param_hfss)=HFSS_IO.Import_SParams_Parameterized(HFSS_SParamFiles)
    
    alpha_ey_data=np.zeros(np.shape(f_hfss),dtype=complex)
    alpha_mx_data=np.zeros(np.shape(f_hfss),dtype=complex)
    
    for i, f in enumerate(f_hfss[:,0]):
        k=2*pi*f/c
        Mode=WaveguideMode(waveguide,f,1,0,'TE',1)
        
        for j, m in enumerate(param_hfss[0,:]):
            alpha_ey_data[i][j]=-1j*Z0*Mode.N/(2*k*(Mode.E[yhat](r0)**2))*(S21[i][j]+S11[i][j]-1)
            alpha_mx_data[i][j]=-1j*Mode.N/(2*Z0*k*(Mode.H[xhat](r0)**2))*(S21[i][j]-S11[i][j]-1)
            
    alpha_ey_re=scipy.interpolate.RectBivariateSpline(f_hfss[:,0],param_hfss[0,:],np.real(alpha_ey_data))
    alpha_ey_im=scipy.interpolate.RectBivariateSpline(f_hfss[:,0],param_hfss[0,:],np.imag(alpha_ey_data))
    alpha_ey=math.ScalarField(alpha_ey_re)+math.ScalarField(alpha_ey_im)*1j

    alpha_mx_re=scipy.interpolate.RectBivariateSpline(f_hfss[:,0],param_hfss[0,:],np.real(alpha_mx_data))
    alpha_mx_im=scipy.interpolate.RectBivariateSpline(f_hfss[:,0],param_hfss[0,:],np.imag(alpha_mx_data))
    alpha_mx=math.ScalarField(alpha_mx_re)+math.ScalarField(alpha_mx_im)*1j
    
    return (alpha_ey,alpha_mx)




def WaveguideExtraction2(dip,waveguide,S11,S21,f):
    
    r0=waveguide.GlobalToLocal(dip.r0)
    r0=xhat*r0[xhat]+yhat*r0[yhat]
    print(r0.components)
        
    if isinstance(f,float):
        k=2*pi*f/c
        Mode=WaveguideMode(waveguide,f,1,0,'TE',1)
        alpha_ey=1j*Z0*Mode.N/(2*k*(Mode.E[yhat](r0)**2))*(S21+S11-1);
        alpha_mx=1j*Mode.N/(2*Z0*k*(Mode.H[xhat](r0)**2))*(S21-S11-1);
    
    else:
        alpha_ey_data=np.zeros(np.shape(f),dtype=complex)
        alpha_mx_data=np.zeros(np.shape(f),dtype=complex)
        

        for i in range(0,len(f)):
            k=2*pi*f[i]/c
            Mode=WaveguideMode(waveguide,f[i],1,0,'TE',1)
            
            alpha_ey_data[i]=1j*Z0*Mode.N/(2*k*(Mode.E[yhat](r0)**2))*(S21[i]+S11[i]-1);
            alpha_mx_data[i]=1j*Mode.N/(2*Z0*k*(Mode.H[xhat](r0)**2))*(S21[i]-S11[i]-1);
    
        def alpha_ey(f0):
            alpha=np.interp(f0,f,alpha_ey_data,left=0,right=0)
            return alpha
        
        def alpha_mx(f0):
            alpha=np.interp(f0,f,alpha_mx_data,left=0,right=0)
            return alpha
    
    return (alpha_ey,alpha_mx)