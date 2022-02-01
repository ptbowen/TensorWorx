#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 10:22:27 2021

@author: ptbowen
"""

import worx.MathObjects as math
import worx.ElectromagneticObjects as emag
import worx.PlotTools as PlotTools
import copy
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import pandas as pd
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from numpy.lib.scimath import sqrt
from matplotlib import cm
from mayavi import mlab
pi=np.pi
Z0=376
c=2.998e8
(xhat,yhat,zhat)=math.CreateEuclideanBasis()
vec_space=[xhat,yhat,zhat]
eps=np.finfo(float).eps


class HalfSpace(object):
    def __init__(self):
        self.DipoleList=[]
        self.Field=emag.EM_VectorField()
        self.SourceField=emag.EM_VectorField()
        
        self.FarField=emag.FarField()
        self.xhat_loc=xhat
        self.yhat_loc=yhat
        self.zhat_loc=zhat
        self.vec_space_loc=[self.xhat_loc,self.yhat_loc,self.zhat_loc]
        self.Rot_Basis=math.IdentityTensor(vec_space)
        self.Translation=xhat*0
        # self.GlobalToLocal=math.VectorIdentityMap(vec_space)
        # self.LocalToGlobal=math.VectorIdentityMap(vec_space)
        
        return None
    
    def GlobalToLocal(self,r):
        return r
    
    def add_dipole(self,dip):
        self.DipoleList.append(dip)
        dip.GeometryList.append(self)
    
    def GreensFunction(self,f0):
        k0=2*pi*f0/c
        
        I=math.IdentityTensor(vec_space)
        R_x=math.ScalarField(lambda r,rp: r[xhat]-rp[xhat])
        R_y=math.ScalarField(lambda r,rp: r[yhat]-rp[yhat])
        R_z=math.ScalarField(lambda r,rp: r[zhat]-rp[zhat])
        R=xhat*R_x+yhat*R_y+zhat*R_z
        
        # rneqrp=math.ScalarField(lambda r,rp: int(r!=rp))
        def r_neq_rp(r,rp):
            u=np.ones(np.shape(r[xhat]))
            u[R(r,rp).norm()==0]=0
            return u
        r_neq_rp=math.ScalarField(r_neq_rp)
        g=math.ScalarField((lambda r,rp: np.exp((r-rp).norm()*(-1j*k0))*(-1/((r-rp).norm()*4*pi+eps))))
        a=math.ScalarField((lambda r,rp: ((r-rp).norm()*(-1j*k0)-1) / ((r-rp).norm()**2+eps) + 1 ))
        b=math.ScalarField((lambda r,rp: ((r-rp).norm()*3*1j*k0-(r-rp).norm()**2*k0**2+3) / ((r-rp).norm()**4+eps) ))

        GroundPlaneBC_M=xhat.tensor_prod(xhat)*2+yhat.tensor_prod(yhat)*2
        GroundPlaneBC_P=zhat.tensor_prod(zhat)*2

        Gee=((I*a+R.tensor_prod(R)*b)*g*r_neq_rp).dot(GroundPlaneBC_P)
        Gmm=((I*a+R.tensor_prod(R)*b)*g*r_neq_rp).dot(GroundPlaneBC_M)
        # Gee=(I*a+R.tensor_prod(R)*b)*g*r_neq_rp
        # Gmm=(I*a+R.tensor_prod(R)*b)*g*r_neq_rp
        Gme=I*a*0
        Gem=I*a*0
        
        return (Gee,Gmm,Gem,Gme)
    
    
    def FarFieldGreensFunction(self):
        (r_hat,th_hat,ph_hat)=math.VectorToSphericalBasis()
        T_iso=(th_hat.tensor_prod(th_hat)+ph_hat.tensor_prod(ph_hat))
        T_iso=T_iso.add_dummy()
        
        T_aniso=(ph_hat.tensor_prod(th_hat)-th_hat.tensor_prod(ph_hat))
        T_aniso=T_aniso.add_dummy()
        
        g=math.ScalarField((lambda k,rp: (k(0,0)).norm()**2*np.exp(k.dot(rp)*1j)*(-1/(4*pi))))
        
        G=T_iso*g
        F=T_aniso*g
        
        return (G,F)


    def ComputeFarField(self,f):

        # Options
        N_pts_per_beam=8       

        # Define k-vector
        rhat=math.SphericalBasis('r')
        k0=(2*pi*f/c)
        k=rhat*k0
        
        # Far field calc
        (G,F)=self.FarFieldGreensFunction()
        H=math.Vector({})
        E=math.Vector({})
        for dip in self.DipoleList:
            H=H+(G.evaluate(dip.r0,1)).dot(xhat*dip.M.dot(xhat)*2+yhat*dip.M.dot(yhat)*2)            
            E=E+(F.evaluate(dip.r0,1)).dot(xhat*dip.M.dot(xhat)*2+yhat*dip.M.dot(yhat)*2)    
        Sr=H.dot(H.conj())*Z0/2
        
        #### Directivity ###
        
        # Find maximum dimension of aperture to find ideal angular resolution for integration
        L=0
        if len(self.DipoleList)>1:
            dist=np.zeros([len(self.DipoleList),len(self.DipoleList)])
            for i, dip1 in enumerate(self.DipoleList):
                for j, dip2 in enumerate(self.DipoleList):
                    dist[i,j]=(dip1.r0-dip2.r0).norm()
            L=np.max(dist)
            
            if L<10*2*pi/k0: L=10*2*pi/k0

        else: L=10*2*pi/k0
        
        dth=np.arcsin(2*pi/(k0*L))/N_pts_per_beam
        Nth=int(2*pi/dth)
        Nph=int(pi/2/dth)

        
        # Integrate radiated power
        (theta_grid,phi_grid,th_axis,ph_axis,dth,dph)=math.HalfSphereGrid(Nth,Nph)
        Sr_grid=Sr(k(theta_grid,phi_grid))
        Prad=np.real(np.trapz(np.trapz(Sr_grid*np.sin(phi_grid)))*dth*dph)       
        
        # Directivity
        directivity=Sr*(4*pi/Prad)
        
        # Beam Analysis (best done in k-space)
        dk=2*pi/L/N_pts_per_beam
        [kx,ky]=np.mgrid[-k0:k0+dk:dk,-k0:k0+dk:dk]
        k_grid=xhat*kx+yhat*ky+zhat*np.real(sqrt(k0**2-kx**2-ky**2))
        dir_grid=directivity(k_grid)
        dir_grid[(kx**2+ky**2)>k0**2]=0
        
        (r,theta,phi)=math.VectorToSphericalCoordinates()
        (peak_i,peak_j)=self.Find_2D_Peaks(np.real(dir_grid),N_pts_per_beam)
        Beams_Theta=np.squeeze(theta(k_grid)[peak_i,peak_j])
        Beams_Phi=np.squeeze(phi(k_grid)[peak_i,peak_j])
        Beams_Dir=np.squeeze((np.real(directivity(k(Beams_Theta,Beams_Phi)))))
        ind=np.flipud(np.argsort(Beams_Dir))      

        
        if len(ind)>1:
            SLL=10*np.log10(np.max(Beams_Dir))-10*np.log10(Beams_Dir)

            Beams=pd.DataFrame({'Directivtiy [dB]':10*np.log10(Beams_Dir[ind]),
                                'Theta [deg]': Beams_Theta[ind]*180/pi,
                                'Phi [deg]' : Beams_Phi[ind]*180/pi,
                                'Side Lobe Level [dB]': SLL[ind]})
            
        else: 
            Beams=pd.DataFrame({'Directivtiy [dB]':[10*np.log10(Beams_Dir)],
                                'Theta [deg]': [Beams_Theta*180/pi],
                                'Phi [deg]' : [Beams_Phi*180/pi]})
        
        self.FarField.E=E
        self.FarField.H=H
        self.FarField.directivity=directivity
        self.FarField.Prad=Prad
        self.FarField.Sr=Sr
        self.FarField.k=k
        self.FarField.Nth=Nth
        self.FarField.Nph=Nph
        self.FarField.Beams=Beams
            
    def Find_2D_Peaks(self,field,N_dist):
        peak_i=[]
        peak_j=[]
        for i in range(1,np.shape(field)[0]-1):
            (peaks_1D,properties)=scipy.signal.find_peaks((field[i,:]),prominence=None)
            for j in peaks_1D:
                (peaks_1D2,properties)=scipy.signal.find_peaks((field[:,j]),prominence=None)
                if i in peaks_1D2:
                    peak_i.append(i)
                    peak_j.append(j)
                

        return (peak_i,peak_j)






class Waveguide(object):
    def __init__(self,a,b,n,L=1):
        
        # Local coordinate system
        self.xhat_loc=xhat
        self.yhat_loc=yhat
        self.zhat_loc=zhat
        self.vec_space_loc=[self.xhat_loc,self.yhat_loc,self.zhat_loc]
        
        self.Rot_Basis=math.IdentityTensor(vec_space)
        self.Rot_Coord=math.IdentityTensor(vec_space)
        self.Translation=math.Vector({})
        self.GlobalToLocal=math.VectorIdentityMap()
        self.LocalToGlobal=math.VectorIdentityMap()
        
        # Dimensions
        self.a=a
        self.b=b
        self.L=L
        
        # emag stuff
        self.n_wg=n
        self.Field=emag.EM_VectorField()
        self.SourceField=emag.EM_VectorField()
        self.DipoleList=[]
        self.N_modes=1
        
        return None
    

    
    def Rotate(self,R):
        # Update coordinate transformatoins
        self.GlobalToLocal=self.GlobalToLocal.inv_rot(copy.copy(R))
        self.LocalToGlobal=copy.copy(R).dot(self.LocalToGlobal)
        
        # Redefine the basis and coordinate rotations
        self.Rot_Coord=copy.copy(R).dot(self.Rot_Coord)
        self.Rot_Basis=self.Rot_Basis.dot(copy.copy(R.T()))
        
        # Rotate the basis
        self.xhat_loc=R.dot(self.xhat_loc)
        self.yhat_loc=R.dot(self.yhat_loc)
        self.zhat_loc=R.dot(self.zhat_loc)
        self.vec_space_loc=[self.xhat_loc,self.yhat_loc,self.zhat_loc]
        
        # Rotate the dipoles
        for dip in self.DipoleList:
            dip.Rotate(R)
            
    def Translate(self,v):
        self.Translation=self.Translation+v
        self.GlobalToLocal=self.GlobalToLocal.translate(v)
        self.LocalToGlobal=self.LocalToGlobal+v
        
        for dip in self.DipoleList:
            dip.Translate(v)
        
        
    def mode(self,f,mu=1,nu=0,polarization='TE',P0=1):
        return emag.WaveguideMode(self,f,mu,nu,polarization,P0)
    
    def add_dipole(self,dip):
        self.DipoleList.append(dip)
        dip.GeometryList.append(self)
        
    def Source(self,f,source_type='mode',amp=1,dip=0):
        if source_type=='mode':
            mode=emag.WaveguideMode(self,f,mu=1,nu=0,polarization='TE',P0=1)
            self.SourceField.E=mode.E*amp
            self.SourceField.H=mode.H*amp
    
        if source_type=='dipole':
            (Gee,Gmm,Gem,Gme)=self.GreensFunction(f)
            self.SourceField.H=Gmm.evaluate(self.GlobalToLocal(dip.r0),1).dot(self.Rot_Basis.dot(dip.M))
    
    
    
    def Modulate(self,k_b,E_b,plot=0):
        k=k_b.norm()
        f=c*k/(2*pi)
        m_range=np.linspace(0,1,1000)
        alpha_achieved=np.zeros(len(self.DipoleList),dtype=complex)
        alpha_m_ideal=np.zeros(len(self.DipoleList),dtype=complex)
        alpha_max=np.zeros(len(self.DipoleList),dtype=complex)
        
        # Compute ideal pattern
        for i, dip in enumerate(self.DipoleList):
            # compute ideal tuning
            if self.SourceField.H(self.GlobalToLocal(dip.r0)).dot(self.Rot_Basis.dot(dip.nu_m))!=0:
                alpha_m_ideal[i]=E_b.dot(self.yhat_loc.cross(dip.nu_m))*np.exp(-1j*k_b.dot(dip.r0))/self.SourceField.H(self.GlobalToLocal(dip.r0)).dot(self.Rot_Basis.dot(dip.nu_m))
            else:
                alpha_m_ideal[i]=0
            
            alpha_max[i]=np.max(np.abs(dip.TuningFunction_m(f,m_range)))
        
        # normalization
        alpha_max=np.max(alpha_max)
        alpha_ideal_max=np.max(np.abs(alpha_m_ideal))
        alpha_m_ideal=(alpha_max/alpha_ideal_max)*alpha_m_ideal
            
            
        # Euclidean modulation
        for i, dip in enumerate(self.DipoleList):
            m=m_range[np.argmin(np.abs(alpha_m_ideal[i]-dip.TuningFunction_m(f,m_range)))]
            
            # tune dipole
            dip.tune(f,m)
            
            # Record polarizabiliites
            alpha_achieved[i]=dip.TuningFunction_m(f,dip.m)
        
        if plot:
            plt.close(1)
            fig = plt.figure(1)
            ax = fig.add_subplot(projection='polar')
            ax.scatter(np.angle(alpha_m_ideal),np.abs(alpha_m_ideal))
            ax.scatter(np.angle(alpha_achieved),np.abs(alpha_achieved))
            ax.quiver(np.angle(alpha_m_ideal),np.abs(alpha_m_ideal),
                      np.real(alpha_achieved-alpha_m_ideal),np.imag(alpha_achieved-alpha_m_ideal),
                      angles='uv', units='x',scale_units='xy', scale=1)
                    
                
        return None
        
    def Compute(self,f0,compute_type='unperturbed',DeEmbedPort1=0,DeEmbedPort2=0,N_iter=3):
        
        if len(self.DipoleList)==0:
            compute_type='unperturbed'
            
        
        if compute_type=='unperturbed':
            self.Field=self.SourceField
            self.UpdateDipoleMoments()
        
        
        if compute_type=='single_scattering':
            self.Field=self.SourceField
            self.UpdateDipoleMoments()
            
            (Gee,Gmm,Gem,Gme)=self.GreensFunction(f0)
            for dip in self.DipoleList:
                H_dip=(Gmm.evaluate(self.GlobalToLocal(dip.r0),1).dot(self.Rot_Basis.dot(dip.M))
                       +Gme.evaluate(self.GlobalToLocal(dip.r0),1).dot(self.Rot_Basis.dot(dip.P)))
                E_dip=(Gee.evaluate(self.GlobalToLocal(dip.r0),1).dot(self.Rot_Basis.dot(dip.P))
                       +Gem.evaluate(self.GlobalToLocal(dip.r0),1).dot(self.Rot_Basis.dot(dip.M)))
                self.Field.H=self.Field.H+H_dip
                self.Field.E=self.Field.E+E_dip

            self.ComputeS_Params(f0,DeEmbedPort1,DeEmbedPort2)
    
    
    
        if compute_type=='iterative':
            (Gee,Gmm,Gem,Gme)=self.GreensFunction(f0)
            
            self.Field=self.SourceField
            self.UpdateDipoleMoments()
            for i in range(1,N_iter):
                self.Field=self.SourceField
                for dip in self.DipoleList:
                    H_dip=Gmm.evaluate(self.GlobalToLocal(dip.r0),1).dot(self.Rot_Basis.dot(dip.M))
                    self.Field.H=self.Field.H+H_dip
                    self.UpdateDipoleMoments()
                
            for dip in self.DipoleList:
                H_dip=Gmm.evaluate(self.GlobalToLocal(dip.r0),1).dot(self.Rot_Basis.dot(dip.M))
                self.Field.H=self.Field.H+H_dip
            self.UpdateDipoleMoments()
            self.ComputeS_Params(f0,DeEmbedPort1,DeEmbedPort2)
            
            
            
        if compute_type=='coupled':
            

            # Build DDA matrix and excitation vector E0,H0
            N_dip=len(self.DipoleList)
            xTx=xhat.tensor_prod(xhat)
            Mat_HH=[[xTx*0 for i in range(0,N_dip)] for j in range(0,N_dip)]
            Mat_EH=[[xTx*0 for i in range(0,N_dip)] for j in range(0,N_dip)]
            Mat_HE=[[xTx*0 for i in range(0,N_dip)] for j in range(0,N_dip)]
            Mat_EE=[[xTx*0 for i in range(0,N_dip)] for j in range(0,N_dip)]
            
            # define source field and set diagonal elements of DDA matrix to identity.
            H0=[0 for i in range(0,N_dip)]
            E0=[0 for i in range(0,N_dip)]
            for i in range(0,N_dip):
                Mat_HH[i][i]=math.IdentityTensor(vec_space)
                Mat_EE[i][i]=math.IdentityTensor(vec_space)
                
                # Define source field in global coordinates
                ri=self.GlobalToLocal(self.DipoleList[i].r0)
                H0[i]=self.Rot_Basis.T().dot(self.SourceField.H(ri))
                E0[i]=self.Rot_Basis.T().dot(self.SourceField.E(ri))
                H0[i]=(math.VectorToColumn(H0[i],vec_space))
                E0[i]=(math.VectorToColumn(E0[i],vec_space))
            
            # Build the rest of the DDA matrix by including coupling between dipoles
            for i in range(0,N_dip):
                dip_i=self.DipoleList[i]
                
                for j in range(0,N_dip):
                    dip_j=self.DipoleList[j]
                    
                    for geometry in dip_i.GeometryList:
                        if geometry in dip_j.GeometryList:
                            # Green's functions take local coordinates as arguments
                            ri=geometry.GlobalToLocal(dip_i.r0)
                            rj=geometry.GlobalToLocal(dip_j.r0)
                            
                            # Polarizabilities are always expressed in global coordinates
                            alpha_ej=dip_j.alpha_e
                            alpha_mj=dip_j.alpha_m
                            
                            # Get Green's functions in global coordinates
                            R=geometry.Rot_Basis
                            (Gee,Gmm,Gem,Gme)=geometry.GreensFunction(f0)
                            Gee_ij=R.T().dot(Gee(ri,rj)).dot(R)
                            Gmm_ij=R.T().dot(Gmm(ri,rj)).dot(R)
                            Gem_ij=R.T().dot(Gem(ri,rj)).dot(R)
                            Gme_ij=R.T().dot(Gme(ri,rj)).dot(R)

                            Mat_EE[i][j]=Gee_ij.dot(alpha_ej)*(-1) + Mat_EE[i][j]
                            Mat_HH[i][j]=Gmm_ij.dot(alpha_mj)*(-1) + Mat_HH[i][j]
                            Mat_EH[i][j]=Gem_ij.dot(alpha_mj)*(-1) + Mat_EH[i][j]
                            Mat_HE[i][j]=Gme_ij.dot(alpha_ej)*(-1) + Mat_HE[i][j]
                    
                    
                    Mat_EE[i][j]=math.TensorToMatrix(Mat_EE[i][j],vec_space)
                    Mat_HH[i][j]=math.TensorToMatrix(Mat_HH[i][j],vec_space)
                    Mat_HE[i][j]=math.TensorToMatrix(Mat_HE[i][j],vec_space)
                    Mat_EH[i][j]=math.TensorToMatrix(Mat_EH[i][j],vec_space)
            
            MatE=np.hstack((np.block(Mat_EE),np.block(Mat_EH)))
            MatH=np.hstack((np.block(Mat_HE),np.block(Mat_HH)))
            Matrix=np.vstack((MatE,MatH))
            F0=np.hstack((np.block(E0),np.block(H0)))

            
            
            #  Solve for magnetic fields at dipole locations
            F=np.linalg.solve(Matrix,F0)
            E=F[0:int(len(F)/2)]
            H=F[int(len(F)/2):len(F)]
            E=np.reshape(H,(int(len(E)/3),3))
            H=np.reshape(H,(int(len(H)/3),3))
            
            # Compute dipole moments from fields
            for i in range(0,N_dip):
                Ei=math.ColumnToVector(E[i],vec_space)
                self.DipoleList[i].P=self.DipoleList[i].alpha_e.dot(Ei)
                
                Hi=math.ColumnToVector(H[i],vec_space)
                self.DipoleList[i].M=self.DipoleList[i].alpha_m.dot(Hi)
            
            # Sum up incident plus scattered fields to get final total field as a vector field
            self.Field=self.SourceField
            (Gee,Gmm,Gem,Gme)=self.GreensFunction(f0)
            for dip in self.DipoleList:
                E_dip=(Gee.evaluate(self.GlobalToLocal(dip.r0),1).dot(self.Rot_Basis.dot(dip.P))
                        +Gem.evaluate(self.GlobalToLocal(dip.r0),1).dot(self.Rot_Basis.dot(dip.M)))
                self.Field.E=self.Field.E+E_dip
                
                H_dip=(Gmm.evaluate(self.GlobalToLocal(dip.r0),1).dot(self.Rot_Basis.dot(dip.M))
                        +Gme.evaluate(self.GlobalToLocal(dip.r0),1).dot(self.Rot_Basis.dot(dip.P)))
                self.Field.H=self.Field.H+H_dip
                
                
            # Find S-parameters
            self.ComputeS_Params(f0,DeEmbedPort1,DeEmbedPort2)

            
    
    
    def UpdateDipoleMoments(self):
        for dip in self.DipoleList:
            r0=self.GlobalToLocal(dip.r0)
            dip.P=dip.alpha_e.dot(self.Rot_Basis.T().dot(self.Field.E(r0)))
            dip.M=dip.alpha_m.dot(self.Rot_Basis.T().dot(self.Field.H(r0)))
    
    
    def ComputeS_Params(self,f0,DeEmbedPort1=0,DeEmbedPort2=0):
        k0=2*pi*f0/c
        mode_p=emag.WaveguideMode(self,f0,orientation='+')
        mode_m=emag.WaveguideMode(self,f0,orientation='-')
        ap=np.exp(-1j*mode_p.beta*self.L)
        am=0
        for dip in self.DipoleList:
            r0p=self.GlobalToLocal(dip.r0)
            r0p_t=xhat*r0p[xhat]+yhat*r0p[yhat]
            ap=ap+(1j*k0/mode_p.N)*(mode_m.E(r0p_t).dot(self.Rot_Basis.dot(dip.P))/Z0
                                      -mode_m.H(r0p_t).dot(self.Rot_Basis.dot(dip.M))*Z0)*np.exp(-1j*mode_p.beta*(self.L-r0p[zhat]))
            am=am+(-1j*k0/mode_m.N)*(mode_m.E(r0p_t).dot(self.Rot_Basis.dot(dip.P))/Z0
                                       -mode_p.H(r0p_t).dot(self.Rot_Basis.dot(dip.M))*Z0)*np.exp(1j*mode_m.beta*r0p[zhat])
        
        self.S21=ap*np.exp(1j*mode_p.beta*(DeEmbedPort2+DeEmbedPort1))
        self.S11=am*np.exp(1j*mode_p.beta*2*DeEmbedPort1)
    
    
    def GreensFunction(self,f):
        
        N=self.N_modes
        k=2*pi*f/c
        
        # find first N modes with slowest decay
        [nu,mu]=np.mgrid[0:N+1:1,0:N+1:1]
        nu=nu.flatten()
        mu=mu.flatten()
        beta_TE=np.zeros(np.size(nu),dtype=complex)
        for i in range(0,len(nu)):
            beta_TE[i]=emag.WaveguideMode(self,f,mu=mu[i],nu=nu[i],polarization='TE').beta
        ind_map=np.argsort(-np.imag(beta_TE))
        nu_TE=nu[ind_map[1:N+1]]
        mu_TE=mu[ind_map[1:N+1]]
        
        beta_TM=np.zeros(np.size(nu),dtype=complex)
        for i in range(0,len(nu)):
            beta_TM[i]=emag.WaveguideMode(self,f,mu=mu[i],nu=nu[i],polarization='TM').beta
        ind_map=np.argsort(-np.imag(beta_TM))
        nu_TM=nu[ind_map[1:N+1]]
        mu_TM=mu[ind_map[1:N+1]]
        
        # You have to split the domain between the forwards and backwards directions
        def zpm(r,rp):
            u=np.zeros(np.shape(r[zhat]))
            u[r[zhat]>rp[zhat]]=1
            return u
        def zmp(r,rp):
            u=np.zeros(np.shape(r[zhat]))
            u[r[zhat]<rp[zhat]]=1
            return u
        zpm=math.ScalarField(zpm)
        zmp=math.ScalarField(zmp)
        
        # Compute the Green's tensor
        Gmm=math.Tensor({})
        Gee=math.Tensor({})
        Gem=math.Tensor({})
        Gme=math.Tensor({})
        for n in range(0,N):
            # TE modes
            Hp=emag.WaveguideMode(self,f,mu=mu_TE[n],nu=nu_TE[n],polarization='TE',orientation='+').H
            Hm=emag.WaveguideMode(self,f,mu=mu_TE[n],nu=nu_TE[n],polarization='TE',orientation='-').H
            Ep=emag.WaveguideMode(self,f,mu=mu_TE[n],nu=nu_TE[n],polarization='TE',orientation='+').E
            Em=emag.WaveguideMode(self,f,mu=mu_TE[n],nu=nu_TE[n],polarization='TE',orientation='-').E
            N_nu=emag.WaveguideMode(self,f,mu=mu_TE[n],nu=nu_TE[n]).N
            Gmm=Gmm+(Hp.comb_tensor_prod(Hm)*zpm+Hm.comb_tensor_prod(Hp)*zmp)*(-1j*Z0*k/N_nu)
            Gee=Gee+(Ep.comb_tensor_prod(Em)*zpm+Em.comb_tensor_prod(Ep)*zmp)*(1j*k/N_nu/Z0)
            Gem=Gem+(Ep.comb_tensor_prod(Hm)*zpm+Em.comb_tensor_prod(Hp)*zmp)*(-1j*Z0*k/N_nu)
            Gme=Gee+(Hp.comb_tensor_prod(Em)*zpm+Hm.comb_tensor_prod(Ep)*zmp)*(1j*k/N_nu/Z0)
            
            # TM modes
            Hp=emag.WaveguideMode(self,f,mu=mu_TM[n],nu=nu_TM[n],polarization='TM',orientation='+').H
            Hm=emag.WaveguideMode(self,f,mu=mu_TM[n],nu=nu_TM[n],polarization='TM',orientation='-').H
            Ep=emag.WaveguideMode(self,f,mu=mu_TM[n],nu=nu_TM[n],polarization='TM',orientation='+').E
            Em=emag.WaveguideMode(self,f,mu=mu_TM[n],nu=nu_TM[n],polarization='TM',orientation='-').E
            N_nu=emag.WaveguideMode(self,f,mu=mu_TM[n],nu=nu_TM[n]).N
            Gmm=Gmm+(Hp.comb_tensor_prod(Hm)*zpm+Hm.comb_tensor_prod(Hp)*zmp)*(-1j*Z0*k/N_nu)
            Gee=Gee+(Ep.comb_tensor_prod(Em)*zpm+Em.comb_tensor_prod(Ep)*zmp)*(1j*k/N_nu/Z0)
            Gem=Gem+(Ep.comb_tensor_prod(Hm)*zpm+Em.comb_tensor_prod(Hp)*zmp)*(-1j*Z0*k/N_nu)
            Gme=Gee+(Hp.comb_tensor_prod(Em)*zpm+Hm.comb_tensor_prod(Ep)*zmp)*(1j*k/N_nu/Z0)
            
        return (Gee,Gmm,Gem,Gme)
    
    
    def plot(self,ax,scale=1,plot_fields=False,
             facecolors='grey', linewidths=1, edgecolors='k', alpha=1,top_alpha=0):
        
        def WaveguidePolygons(self,scale,facecolors=facecolors,linewidths=linewidths, edgecolors=edgecolors, alpha=alpha,top_alpha=top_alpha):
            (xhat,yhat,zhat)=math.CreateEuclideanBasis()
            r=[0]*8
            r[0]=xhat*0
            r[1]=xhat*self.a/scale
            r[2]=(xhat*self.a+yhat*self.b)/scale
            r[3]=yhat*self.b/scale
            for i in range(4,8):
                r[i]=r[i-4]+zhat*self.L/scale
                
            for i in range(0,8):
                r[i]=self.Rot_Basis.dot(r[i])
                
            def VecToList(v):
                (xhat,yhat,zhat)=math.CreateEuclideanBasis()
                return [v.dot(xhat),v.dot(yhat),v.dot(zhat)]
                
            verts_left=[[VecToList(r[0]),VecToList(r[4]),VecToList(r[7]),VecToList(r[3])]]
            left=Poly3DCollection(verts_left,facecolors=facecolors,linewidths=linewidths, edgecolors=edgecolors, alpha=alpha)
        
            verts_bottom=[[VecToList(r[3]),VecToList(r[2]),VecToList(r[6]),VecToList(r[7])]]
            bottom=Poly3DCollection(verts_bottom,facecolors=facecolors,linewidths=linewidths, edgecolors=edgecolors, alpha=alpha)
            
            verts_right=[[VecToList(r[1]),VecToList(r[5]),VecToList(r[6]),VecToList(r[2])]]
            right=Poly3DCollection(verts_right,facecolors=facecolors,linewidths=linewidths, edgecolors=edgecolors, alpha=alpha)
            
            verts_top=[[VecToList(r[0]),VecToList(r[1]),VecToList(r[5]),VecToList(r[4])]]
            top=Poly3DCollection(verts_top,facecolors=facecolors,linewidths=linewidths, edgecolors=edgecolors, alpha=top_alpha)
        
            return (left,bottom,right,top)
        
        # Plot dipoles
        for dip in self.DipoleList:
            dip.plot(ax,scale,2e-8)
        
        # Plot fields 
        if plot_fields:
            (xhat,yhat,zhat)=math.CreateEuclideanBasis()
            [xp,zp]=np.meshgrid(np.linspace(0,self.a,15),np.arange(0,self.L,self.a/15))
            yp=np.ones(np.shape(xp))*self.b*(9/10)
            rp=xhat*xp+yhat*yp+zhat*zp
            r=self.LocalToGlobal(rp)
            (x,y,z)=((r.dot(xhat)/scale,r.dot(yhat)/scale,r.dot(zhat)/scale))
            field=np.real(self.Field.H.dot(xhat)(rp))
            field=(field-field.min())/(field.max()-field.min())
            ax.plot_surface(x,y,z,facecolors=cm.coolwarm(field),edgecolor='none',alpha=1)
        
        # Plot waveguide
        (left,bottom,right,top)=WaveguidePolygons(self,scale,facecolors,linewidths,edgecolors,alpha,top_alpha)
        ax.add_collection3d(left)
        ax.add_collection3d(bottom)
        ax.add_collection3d(right)
        ax.add_collection3d(top)
        
    def MayaviPlot(self,scale,plot_fields=False,phase=0,field_clip=0,):
        (xhat,yhat,zhat)=math.CreateEuclideanBasis()
        
        # Plot waveguide
        r=[0 for i in range(0,8)]
        r[0]=xhat*0
        r[1]=xhat*self.a/scale
        r[2]=(xhat*self.a+yhat*self.b)/scale
        r[3]=yhat*self.b/scale
        r[4]=zhat*self.L/scale
        r[5]=(xhat*self.a+zhat*self.L)/scale
        r[6]=(xhat*self.a+yhat*self.b+zhat*self.L)/scale
        r[7]=(yhat*self.b+zhat*self.L)/scale
            
        for i in range(0,len(r)):
            r[i]=self.LocalToGlobal(r[i])
        
        PlotTools.MayaviRect(r[0],r[1]-r[0],r[4]-r[0],PlotTools.material_color('copper'),0.3)
        PlotTools.MayaviRect(r[0],r[3]-r[0],r[4]-r[0],PlotTools.material_color('copper'),1)
        PlotTools.MayaviRect(r[1],r[2]-r[1],r[5]-r[1],PlotTools.material_color('copper'),1)
        PlotTools.MayaviRect(r[3],r[2]-r[3],r[7]-r[3],PlotTools.material_color('copper'),1)

        # Plot dipoles
        for dip in self.DipoleList:
            mlab.points3d(dip.r0.dot(xhat)/scale,
                          dip.r0.dot(yhat)/scale,
                          dip.r0.dot(zhat)/scale,scale_factor=0.001/scale)


        # Plot Fields
        if plot_fields:
            ds=np.max([self.a,self.b])/20
            [xp,zp]=np.mgrid[0:self.a+ds:ds,0:self.L+ds:ds]
            yp=np.ones(np.shape(xp))*self.b*(1/2)
            rp=xhat*xp+yhat*yp+zhat*zp
            r=self.LocalToGlobal(rp)
            field=np.real(xhat.dot(self.Field.H)(rp)*np.exp(1j*phase))
            if field_clip!=0:
                field[np.abs(field)>field_clip]=field_clip*np.sign(field[np.abs(field)>field_clip])

            field=(field/np.max(np.abs(field)))*(self.b/2.1/scale)
            (x,y,z)=((r.dot(xhat)/scale,r.dot(yhat)/scale,r.dot(zhat)/scale))
            mlab.mesh(x,y,z+field)

    
    
    