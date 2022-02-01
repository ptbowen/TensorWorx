#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 18:43:14 2021

@author: ptbowen
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 18:41:49 2021

@author: ptbowen
"""
import numpy as np
import pandas as pd
    
def Import_SParams(HFSS_SParamFiles):

    S11MagData = pd.read_csv(HFSS_SParamFiles['S11Mag']).to_numpy()
    S11PhaseData = pd.read_csv(HFSS_SParamFiles['S11Phase']).to_numpy()
    S21MagData = pd.read_csv(HFSS_SParamFiles['S21Mag']).to_numpy()
    S21PhaseData = pd.read_csv(HFSS_SParamFiles['S21Phase']).to_numpy()
    
    f_hfss=S11MagData[:,0]*1e9
    S11=S11MagData[:,1]*np.exp(1j*S11PhaseData[:,1]*np.pi/180)
    S21=S21MagData[:,1]*np.exp(1j*S21PhaseData[:,1]*np.pi/180)
    
    return (S11,S21,f_hfss)


def ImportHFSS_Data(filename):
    data=pd.read_csv(filename).to_numpy()
    f_hfss=data[:,0]*1e9
    data=data[:,1]
    
    return (data,f_hfss)


def Import_SParams_Parameterized(HFSS_SParamFiles):
    S11MagData = pd.read_csv(HFSS_SParamFiles['S11Mag']).to_numpy()
    S11PhaseData = pd.read_csv(HFSS_SParamFiles['S11Phase']).to_numpy()
    S21MagData = pd.read_csv(HFSS_SParamFiles['S21Mag']).to_numpy()
    S21PhaseData = pd.read_csv(HFSS_SParamFiles['S21Phase']).to_numpy()

    param_hfss=S11MagData[:,0]
    f_hfss=S11MagData[:,1]*1e9
    S11=S11MagData[:,2]*np.exp(1j*S11PhaseData[:,2]*np.pi/180)
    S21=S21MagData[:,2]*np.exp(1j*S21PhaseData[:,2]*np.pi/180)


    Nf=len(np.unique(f_hfss))
    Np=len(np.unique(param_hfss))
    f_hfss=np.reshape(f_hfss,(Np,Nf)).transpose()
    param_hfss=np.reshape(param_hfss,(Np,Nf)).transpose()
    S11=np.reshape(S11,(Np,Nf)).transpose()
    S21=np.reshape(S21,(Np,Nf)).transpose()

    return (S11,S21,f_hfss,param_hfss)