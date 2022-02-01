#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 14:24:03 2021

@author: ptbowen
"""
import numpy as np
import copy
from numpy.lib.scimath import sqrt
pi=np.pi



class EuclideanBasis(object):
    def __init__(self,element):
        self.element=element
        
    def __call__(self,*args):
        if isinstance(args[0],Vector):
            return self.dot(args[0])
        else:
            return self
        
    def __getitem__(self,element):
        if element==self:
            return 1
        else:
            return 0
    
    def __hash__(self):
        result=0
        for c in self.element: result+=ord(c)
        return result
    
    def __str__(self):
        return 'hat{'+str(self.element)+'}'
    
    __repr__=__str__
    
    def __eq__(self,other):
        if isinstance(other,EuclideanBasis):
            return int(self.element==other.element)
        else:
            return False
            
    def __add__(self,other):
        if isinstance(other,EuclideanBasis):
            v1=Vector({self:1})
            v2=Vector({other:1})
            return v1+v2
        elif isinstance(other,Vector):
            v1=Vector({self:1})
            return other+v1
        # elif isinstance(other,VectorField):
        #     vf1=VectorField({self:ScalarField(1)})
        #     return vf1+other
        
    def __sub__(self,other):
        if isinstance(other,EuclideanBasis):
            v1=Vector({self:1})
            v2=Vector({other:1})
            return v1-v2
        elif isinstance(other,Vector):
            vf1=Vector({self:1})
            return (other-vf1)*(-1)
    
    def __mul__(self,other):
        if isinstance(other,int) and other==1: return self
        else: return Vector({self:other})

    def dot(self,other):
        if isinstance(other,EuclideanBasis):
            return int(self.element==other.element)  
        
        elif isinstance(other,Vector):
            result=0
            for element in other.components:
                if isinstance(element,EuclideanBasis):
                    if element==self:
                        if callable(other[element]):
                            result=other[element]+result
                        else:
                            result=result+other[element]
                else:                    
                    if callable(other[element]):
                        result=other[element]*self.dot(element)+result
                    else:
                        result=self.dot(element)*other[element]+result
            return result
        
        elif isinstance(other,TensorElement):
            if self==other.v_out:
                return other.v_in
            else:
                return 0
            
        elif isinstance(other,Tensor):
            result=Vector({})
            for element in other.components:
                if self.dot(element)!=0:
                    result=result+self.dot(element)*other[element]
            return result

    def cross(self,other):
        (xhat,yhat,zhat)=CreateEuclideanBasis()
        vec_space=[xhat,yhat,zhat]
        if isinstance(other,EuclideanBasis):
            ind_delta=vec_space.index(other)-vec_space.index(self)
            if ind_delta==0:
                return 0
            elif ind_delta<0:
                return other.cross(self)*-1
            elif ind_delta==1:
                ind=vec_space.index(other)+1
                if ind==len(vec_space):
                    ind=0
                return vec_space[ind]
            else:
                ind=vec_space.index(other)+1
                if ind==len(vec_space):
                    ind=0
                return vec_space[ind]*-1
            
        if isinstance(other,Vector):
            result=Vector({});
            for element in other.components:
                cp=self.cross(element)
                if cp!=0:
                    if callable(other[element]):
                        result=other[element]*cp+result
                    else:
                        result=cp*other[element]+result
            return result
        
            
    def tensor_prod(self,other):
        if isinstance(other,EuclideanBasis):
            return TensorElement(self,other)
        
        if isinstance(other,Vector):
            result=Tensor({})
            for element in other.components:
                result=result+self.tensor_prod(element)*other[element]
            return result


class Vector(object):
    def __init__(self,dictionary):
        self.components=dictionary
        
    def __getitem__(self,element):
        if element in self.components:
            return self.components[element]
        else:
            return 0
    
    def __call__(self,*args):
        result=self.components.copy()
        for element in result:
            if callable(result[element]):
                result[element]=result[element](*args)
        return Vector(result)
    
    def __setitem__(self,element,other):
        self.components[element]=other
    
    
    def __hash__(self):
        return id(self)
    
    def keys(self):
        return self.components.keys()
    
    def __eq__(self,other):
        if isinstance(other,Vector):
            answer=True
            for element in self.components:
                if element in other.components:
                    answer=(answer and self[element]==other[element])
                else:
                    answer=False
            return answer
        else:
            return False
    
    def __add__(self,other):
        result=self.components.copy()
        if isinstance(other,EuclideanBasis):
            other=Vector({other:1})
        
        if isinstance(other,int):
            if other==0:
                return self
            else:
                raise ValueError('Cannot add vector and integer.')
        
        for element in result: 
            if element in other.components:
                if not isinstance(other[element],int):
                    if callable(result[element]):
                        result[element]=result[element]+other[element]
                    else:
                        result[element]=other[element]+result[element]
                elif other[element]!=0:
                    result[element]=result[element]+other[element]
                    
        for element in other.components:
            if element not in result:
                if not isinstance(other[element],int):
                    result[element]=other[element]
                else:
                    if other[element]!=0:
                        result[element]=other[element]
        return Vector(result)
    
    def __sub__(self,other):
        result=self.components.copy()
        for element in result: 
            if element in other.components:
                if callable(result[element]):
                    result[element]=result[element]-other[element]
                else:
                    result[element]=other[element]*(-1)+result[element]
        for element in other.components:
            if element not in result:
                result[element]=-other[element]
                
        return Vector(result)
    
    
    def __mul__(self,other):
        result=self.components.copy()
        for element in result:
            if callable(result[element]):
                result[element]=result[element]*other
            else:
                result[element]=other*result[element]
        return Vector(result)
    
    def __truediv__(self,other):
        result=self.components.copy()
        for element in result:
            if callable(result[element]):
                result[element]=result[element]/other
            else:
                result[element]=(1/other)*result[element]
        return Vector(result)
    
    
    def dot(self,other):
        if isinstance(other,EuclideanBasis): return other.dot(self)
        
        elif isinstance(other,Vector):
            result=0
            for element_self in self.components:
                if callable(self[element_self]):
                    result=self[element_self]*element_self.dot(other)+result
                else:
                    result=element_self.dot(other)*self[element_self]+result
            return result
            
        if isinstance(other,TensorElement):
            result=Vector({})
            for element in self.components:
                if not isinstance(element.dot(other),int):
                    result=result+element.dot(other)*self[element]
            return result
        
        if isinstance(other,Tensor):
            result=Vector({})
            for element in other.components:
                if not isinstance(self.dot(element),int):
                    result=result+self.dot(element)*other[element]
            return result
        
        
    def cross(self,other):
        if isinstance(other,EuclideanBasis): return other.cross(self)*-1
    
        elif isinstance(other,Vector):
            result=Vector({})
            for element in self.components:
                cp=element.cross(other)
                if cp!=0:
                    if callable(self[element]):
                        result=self[element]*cp+result
                    else:
                        result=cp*self[element]+result
            return result
        
    
    def conj(self):
        result={}
        for element in self.components:
            if callable(self[element]):
                result[element]=self[element].conj()
            else:
                result[element]=np.conj(self[element])
        return Vector(result)
    
    def norm(self):
        return (self.dot(self.conj()))**(1/2)
    
    def tensor_prod(self,other):
        if isinstance(other,EuclideanBasis):
            result=Tensor({})
            for element in self.components:
                result=result+element.tensor_prod(other)*self[element]
            return result
        
        if isinstance(other,Vector):
            result=Tensor({})
            for element in other.components:
                result=result+self.tensor_prod(element)*other[element]
            return result
    
    def rotate(self,R):
        result=Vector({})
        for element in self.components:
            if callable(self[element]):
                result=result+R.dot(element)*self[element].rotate(R)
            else:
                result=result+R.dot(element)*self[element]
        return result
    
    
    def translate(self,r0):
        result=Vector({})
        for element in self.components:
            if callable(self[element]):
                result=result+element*self[element].translate(r0)
            else:
                result=result+element*self[element]
        return result
    
    def inv_rot(self,R):
        result=Vector({})
        for element in self.components:
            if callable(self[element]):
                result[element]=self[element].rotate(R)
        return result
    
    def add_dummy(self):
        result={}
        for element in self.components:
            result[element]=self[element].add_dummy()
        return Vector(result)
    
    def comb_prod(self,other):
        result=Vector({})
        for element in self.components:
            result[element]=self[element].comb_prod(other)
        return result
    
    def comb_tensor_prod(self,other):
        result=Tensor({})
        for element in other.components:
            result=result+self.tensor_prod(element).comb_prod(other[element])
        return result
    
    
    
    
class TensorElement(object):
    def __init__(self,v_out,v_in):
        self.v_out=v_out
        self.v_in=v_in
        
    def __call__(self,*args):
        if isinstance(args[0],Vector):
            return self.dot(args[0])
        else:
            return self
    
    def __hash__(self):
        return (hash(self.v_out)+1)*hash(self.v_in)
    
    def __str__(self):
        return str(self.v_out)+u'\u2297'+str(self.v_in)
    
    __repr__=__str__
    
    def __eq__(self,other):
            return int(self.v_in==other.v_in and self.v_out==other.v_out)

    def dot(self,other):
            if isinstance(self.v_in,EuclideanBasis) and isinstance(other,EuclideanBasis):
                if self.v_in==other:
                    return self.v_out
                else:
                    return 0
            elif isinstance(other,Vector):
                return self.v_out*self.v_in.dot(other)
            
            elif isinstance(other,TensorElement):
                if self.v_in.dot(other.v_out)!=0:
                    return TensorElement(self.v_out,other.v_in)*self.v_in.dot(other.v_out)
                else:
                    return 0
            
            elif isinstance(other,Tensor):
                result=Tensor({})
                for element in other.components:
                    if self.dot(element)!=0:
                        result=result+self.dot(element)*other[element]
                return result
        
    def __add__(self,other):
        if isinstance(other,TensorElement):
            T1=Tensor({self:1})
            T2=Tensor({other:1})
            return T1+T2
        elif isinstance(other,Tensor):
            T=Tensor({self:1})
            return T+other
        
    def __sub__(self,other):
        if isinstance(other,TensorElement):
            T1=Tensor({self:1})
            T2=Tensor({other:1})
            return T1-T2
        elif isinstance(other,Tensor):
            T=Tensor({self:1})
            return T-other
        
    def __mul__(self,other):
        if other==1: return self
        else: return Tensor({self:other})
    
    def __truediv__(self,other):
        return Tensor({self:other**(-1)})

    def T(self):
        result=TensorElement(self.v_in,self.v_out)
        return result
        
    
    
    
class Tensor(object):
    def __init__(self,dictionary):
        self.components=dictionary
        
    def __getitem__(self,element):
        if element in self.components:
            return self.components[element]
        else:
            return 0
        
    def __setitem__(self,element,other):
        self.components[element]=other
    
    def __call__(self,*args):
        result=self.components.copy()
        for element in result:
            if callable(result[element]):
                result[element]=result[element](*args)
        return Tensor(result)
    
        
    def __hash__(self):
        return id(self)
    
    def keys(self):
        return self.components.keys()
    
    def __eq__(self,other):
        if isinstance(other,Tensor):
            answer=True
            for element in self.components:
                if element in other.components:
                    answer=(answer and self[element]==other[element])
                else:
                    answer=False
            return answer
        else:
            return False
    
    def __add__(self,other):
        result=self.components.copy()
        if isinstance(other,TensorElement):
            other=Tensor({other:1})
                
        for element in result: 
            if element in other.components:
                if other[element]!=0:
                    if callable(result[element]):
                        result[element]=result[element]+other[element]
                    else:
                        result[element]=other[element]+result[element]
        for element in other.components:
            if element not in result:
                if other[element]!=0:
                    result[element]=other[element]
        return Tensor(result)

    def __sub__(self,other):
        result=self.components.copy()
        if isinstance(other,TensorElement):
            other=Tensor({other:-1})
            
        for element in result: 
            if element in other.components and other[element]!=0:
                if callable(result[element]):
                    result[element]=result[element]-other[element]
                else:
                    result[element]=other[element]*(-1)+result[element]
        for element in other.components:
            if element not in result and other[element]!=0:
                result[element]=-other[element]

        return Tensor(result)
        
    def __mul__(self,other):
        result=self.components.copy()
        for element in result:
            if callable(result[element]):
                result[element]=result[element]*other
            else:
                result[element]=other*result[element]
        return Tensor(result)
    
    
    def dot(self,other):
        if isinstance(other,EuclideanBasis):
            result=Vector({})
            for element in self.components:
                if element.dot(other)!=0:
                    result=result+element.dot(other)*self[element]
            return result
        
        if isinstance(other,Vector):
            result=Vector({})
            for element in other.components:
                if self.dot(element)!=0:
                    result=result+self.dot(element)*other[element]
            return result
        
        if isinstance(other,TensorElement):
            result=Tensor({})
            for element in self.components:
                if not isinstance(element.dot(other),int):
                    result=result+element.dot(other)*self[element]
            return result
        
        if isinstance(other,Tensor):
            result=Tensor({})
            for element in other.components:
                if not isinstance(self.dot(element),int):
                    result=result+self.dot(element)*other[element]
            return result
        
    def T(self):
        result={}
        for element in self.components:
            result[element.T()]=self[element]
        return Tensor(result)
        
    def add_dummy(self):
        result={}
        for element in self.components:
            result[element]=self[element].add_dummy()
        return Tensor(result)
    
    def evaluate(self,x,ind):
        result={}
        for element in self.components:
            result[element]=self[element].evaluate(x,ind)
        return Tensor(result)
        
    def eig(self,vec_space):
        R_matrix=TensorToMatrix(self,vec_space)
        (lmbda,U)=np.linalg.eig(R_matrix)
        eigenvectors=[0]*len(vec_space)
        for i in range(0,len(vec_space)):
            for j in range(0,len(vec_space)):
                eigenvectors[i]=vec_space[j]*(U[j,i])*(lmbda[i])+eigenvectors[i]
        return (lmbda,eigenvectors)
    
    def comb_prod(self,other):
        result=Tensor({})
        for element in self.components:
            result[element]=self[element].comb_prod(other)
        return result
    
        
class ScalarField(object):
    def __init__(self,f):
        if callable(f):
            self.f=f
        else:
            self.f=(lambda *args: f)
    
    def __call__(self,*args):
        return self.f(*args)
    
    def __add__(self,other):
        if isinstance(other,ScalarField):
            return ScalarField((lambda *args: self(*args) + other(*args)))
        else:
            return ScalarField((lambda *args: self(*args) + other))
    
    def __sub__(self,other):
        if isinstance(other,ScalarField):
            return ScalarField((lambda *args: self(*args) - other(*args)))
        else:
            return ScalarField((lambda *args: self(*args) - other))
    
    def __mul__(self,other):
        if isinstance(other,ScalarField):
            return ScalarField((lambda *args: self(*args) * other(*args)))
        elif isinstance(other,EuclideanBasis):
            return other*self
        elif isinstance(other,Vector):
            return other*self
        else:
            return ScalarField((lambda *args: self(*args) * other))
        
    def __truediv__(self,other):
        if isinstance(other,ScalarField):
            return ScalarField((lambda *args: self(*args) / other(*args)))
        else:
            return ScalarField((lambda *args: self(*args) / other))
    
    def __pow__(self,power):
        return ScalarField((lambda *args: self(*args)**power))
    
    def __abs__(self):
        return ScalarField((lambda *args: np.abs(self(*args))))
    
    def conj(self):
        return ScalarField((lambda *args: np.conj(self(*args))))
    
    def add_dummy(self):
        return ScalarField(lambda *args: self(*args[0:-1]))
    
    def evaluate(self,x,ind):
        x=copy.copy(x)
        return ScalarField((lambda *args: self(*(args[0:ind]+(x,)+args[ind+1:]))))
    
    def rotate(self,R):
        return ScalarField((lambda *args: self(R.T().dot(args[0]))))
    
    def translate(self,r0):
        return ScalarField((lambda *args: self(args[0]-r0)))
    
    def comb_prod(self,other):
        return ScalarField(lambda *args: self(args[0])*other(args[1]))
    
        
class SphericalBasis(Vector):
    def __init__(self,element):
        (xhat,yhat,zhat)=CreateEuclideanBasis()
        
        if element=='r':
            r_x=ScalarField(lambda th,ph:np.cos(th)*np.sin(ph))
            r_y=ScalarField(lambda th,ph:np.sin(th)*np.sin(ph))
            r_z=ScalarField(lambda th,ph:np.cos(ph))
            Vector.__init__(self,{xhat:r_x,yhat:r_y,zhat:r_z})
        
        if element=='theta':
            th_x=ScalarField(lambda th,ph: -np.sin(th) )
            th_y=ScalarField(lambda th,ph: np.cos(th) )
            th_z=ScalarField(0)
            Vector.__init__(self,{xhat:th_x, yhat:th_y, zhat:th_z})
        
        if element=='phi':
            ph_x=ScalarField(lambda th,ph: np.cos(th)*np.cos(ph))
            ph_y=ScalarField(lambda th,ph: np.sin(th)*np.cos(ph))
            ph_z=ScalarField(lambda th,ph: -np.sin(ph))
            Vector.__init__(self,{xhat:ph_x, yhat:ph_y, zhat:ph_z})
                                 
        self.element=element
            
    def dot(self,other):
        if isinstance(other,SphericalBasis):
            return int(self.element==other.element)
        else:
            return other.dot(self)
        

def CreateEuclideanBasis():
    xhat=EuclideanBasis('x')
    yhat=EuclideanBasis('y')
    zhat=EuclideanBasis('z')
    return (xhat,yhat,zhat)
        
        
def SphericalCoordinates():
    r=ScalarField((lambda x,y,z: sqrt(x**2+y**2+z**2)))
    theta=ScalarField((lambda x,y,z: np.arctan2(y,x)))
    phi=ScalarField(((lambda x,y,z: np.arctan2(sqrt(x**2+y**2),z))))
    
    return (r,theta,phi)
        
    
    
def VectorToSphericalCoordinates():
    
    (xhat,yhat,zhat)=CreateEuclideanBasis()
    def r(vec):
        return r.norm()
    
    def theta(vec):
        return np.arctan2(vec.dot(yhat),vec.dot(xhat))
        
    def phi(vec):
        return np.arctan2(sqrt(vec.dot(xhat)**2+vec.dot(yhat)**2),vec.dot(zhat))
    
    return (ScalarField(r),ScalarField(theta),ScalarField(phi))
    



def VectorToSphericalBasis():
    
    (xhat,yhat,zhat)=CreateEuclideanBasis()
    eps=np.finfo(float).eps
    
    r_x=ScalarField(lambda vec: vec.dot(xhat))
    r_y=ScalarField(lambda vec: vec.dot(yhat))
    r_z=ScalarField(lambda vec: vec.dot(zhat))
    r_hat=Vector({xhat:r_x,yhat:r_y,zhat:r_z})    
    
    th_x=ScalarField(lambda vec: vec.dot(yhat)*(-1)/(sqrt(vec.dot(xhat)**2+vec.dot(yhat)**2)+eps) )
    th_y=ScalarField(lambda vec: (vec.dot(xhat)+eps)/(sqrt(vec.dot(xhat)**2+vec.dot(yhat)**2)+eps) )
    th_z=ScalarField(0)
    theta_hat=Vector({xhat:th_x,yhat:th_y,zhat:th_z})
    
    ph_x=ScalarField(lambda vec: (vec.dot(xhat)*vec.dot(zhat)+eps)
          /(sqrt((vec.dot(xhat)**2+vec.dot(yhat)**2)*(vec.dot(xhat)**2+vec.dot(yhat)**2+vec.dot(zhat)**2))+eps))
    
    ph_y=ScalarField(lambda vec: vec.dot(yhat)*vec.dot(zhat)
          /(sqrt((vec.dot(xhat)**2+vec.dot(yhat)**2)*(vec.dot(xhat)**2+vec.dot(yhat)**2+vec.dot(zhat)**2))+eps))
    
    ph_z=ScalarField(lambda vec: sqrt((vec.dot(xhat)**2+vec.dot(yhat)**2)
          /(vec.dot(xhat)**2+vec.dot(yhat)**2+vec.dot(zhat)**2+eps)))
    phi_hat=Vector({xhat:ph_x,yhat:ph_y,zhat:ph_z})
    

    
    return (r_hat,theta_hat,phi_hat)
    
        
        
    
def MatrixToTensor(M,vec_space):
    result={}
    ii=0
    for vec_in in vec_space:
        jj=0
        for vec_out in vec_space:
            tensor_element=TensorElement(vec_out,vec_in)
            result[tensor_element]=M[jj,ii]
            jj+=1
        ii+=1
    return Tensor(result)
        
def TensorToMatrix(T,vec_space):
    result=np.zeros([len(vec_space),len(vec_space)],dtype=complex)
    for i in range(0,len(vec_space)):
        for j in range(0,len(vec_space)):
            result[i,j]=vec_space[i].dot(T.dot(vec_space[j]))
    return result

def VectorToColumn(V,vec_space):
    result=np.zeros([len(vec_space)],dtype=complex)
    for i in range(0,len(vec_space)):
        result[i]=vec_space[i].dot(V)
    return result

def ColumnToVector(V,vec_space):
    result=Vector({})
    for i in range(0,len(V)):
        result[vec_space[i]]=V[i]
    return result
    

def ZeroTensor(vec_basis):
    result={}
    for vec_in in vec_basis:
        for vec_out in vec_basis:
            tensor_element=TensorElement(vec_out,vec_in)
            result[tensor_element]=0
    return Tensor(result)


def ZeroVector(vec_space):
    result={}
    for vec in vec_space: result[vec]=0
    return Vector(result)
    

def RotMatrix(th,axis):
    if axis=='x':
        T=np.array([[1, 0, 0],[0, np.cos(th), np.sin(th)],[0, -np.sin(th), np.cos(th)]])
        
    if axis=='y':
        T=np.array([[np.cos(th), 0, np.sin(th)],[0, 1, 0],[-np.sin(th), 0, np.cos(th)]])
    
    if axis=='z':
        T=np.array([[np.cos(th), -np.sin(th), 0],[np.sin(th), np.cos(th), 0],[0, 0, 1]])
        
    return T

def RotTensor(th,axis,vec_basis):
    return MatrixToTensor(RotMatrix(th,axis),vec_basis)


def HalfSphereGrid(Ntheta,Nphi):
    th_axis=np.linspace(0,2*pi,Ntheta)
    ph_axis=np.linspace(0,pi/2,Nphi)
    [theta_grid,phi_grid]=np.meshgrid(th_axis,ph_axis)
    dth=th_axis[1]-th_axis[0]
    dph=ph_axis[1]-ph_axis[0]
    
    return (theta_grid,phi_grid,th_axis,ph_axis,dth,dph)


def SphereGrid(Ntheta,Nphi):
    th_axis=np.linspace(0,pi,Ntheta)
    ph_axis=np.linspace(0,pi/2,Nphi)
    [theta_grid,phi_grid]=np.meshgrid(th_axis,ph_axis)
    dth=th_axis[1]-th_axis[0]
    dph=ph_axis[1]-ph_axis[0]
    
    return (theta_grid,phi_grid,th_axis,ph_axis,dth,dph)
    

def GridToListOfVectors(dictionary):
    i=0
    for element in dictionary:
        if i==0:
            vec_list=[0]*dictionary[element].size
            
        for j in range(0,len(vec_list)):
            vec_list[j]=element*dictionary[element][j]+vec_list[j]
        
        i+=1
    return vec_list
    
def ListOfVectorsToGrid(vecs):
    (xhat,yhat,zhat)=CreateEuclideanBasis()
    x=np.zeros([len(vecs)],dtype=complex)
    y=np.zeros([len(vecs)],dtype=complex)
    z=np.zeros([len(vecs)],dtype=complex)
    for i in range(0,len(vecs)):
        x[i]=vecs[i].dot(xhat)
        y[i]=vecs[i].dot(yhat)
        z[i]=vecs[i].dot(zhat)
    return (x,y,z)
    
    
def IdentityTensor(vec_space):
    result=Tensor({})
    for element in vec_space:
        result=result+element.tensor_prod(element)
            
    return result

def VectorIdentityMap():
    (xhat,yhat,zhat)=CreateEuclideanBasis()
    result=Vector({})
    result[xhat]=ScalarField(lambda r: r[xhat])
    result[yhat]=ScalarField(lambda r: r[yhat])
    result[zhat]=ScalarField(lambda r: r[zhat])
    # ScalarField(lambda r: r[vec_space[0]])

    return result
            
def LorentzianTrace(amp):
    f=(lambda m: amp*np.sin(m*pi)*np.exp(-1j*m*pi))
    return f

def Lorentzian(alpha0=1,Q=10):
    def L(f,f0):
        w0=2*pi*f0
        L= alpha0*w0**2/(w0**2*(1+1j/Q)-(2*pi*f)**2)
        return L
    
    return L
    
    
    
    