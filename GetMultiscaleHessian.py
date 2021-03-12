# -*- coding: utf-8 -*-
"""
Created on Sun Mar  7 23:55:47 2021

@author: hudew
"""

import sys
sys.path.insert(0,'E:\\tools\\')
import util
import numpy as np
import nrrd
import matplotlib.pyplot as plt
from skimage.feature import hessian_matrix, hessian_matrix_det


global α,β,c
α = 0.5
β = 0.5
c = 130

def ab_eigenVal(H):
    "eigen-decomposition"
    eigenValues, eigenVectors = np.linalg.eigh(H)
    abs_ev = np.abs(eigenValues)
    abs_ev = np.diag(abs_ev)
    opt = eigenVectors.dot(abs_ev).dot(eigenVectors.T)
    return opt

def symmetrize(elements):
    "complete the symmetric matrix given elements in upper triangle"
    if len(elements) == 6:
        # populate the upper triangle from vector
        opt = np.zeros((3,3))
        idx = np.triu_indices(len(opt))
        opt[idx] = elements
        # fill the lower trangle
        opt_diag = np.diag(np.diag(opt))
        opt = opt+opt.T-opt_diag
    elif len(elements) == 3:
        # populate the upper triangle from vector
        opt = np.zeros((2,2))
        idx = np.triu_indices(len(opt))
        opt[idx] = elements
        # fill the lower trangle
        opt_diag = np.diag(np.diag(opt))
        opt = opt+opt.T-opt_diag
    else:
        raise ValueError('Length of vector invalid.')
    
    return opt

def GetHessian(vol, sigma=0.6, order='rc'):
    h,d,w = vol.shape
    H_elements = np.array(hessian_matrix(vol,sigma,order='rc'))
    vec = H_elements.reshape((6,-1))
    _,num = vec.shape
    vec_Hessian = np.zeros((9,num),dtype=np.float32)
    
    for i in range(num):
        elements = vec[:,i]
        # fill the Hessian
        H = symmetrize(elements)
        # all positive eigenValue
        #H = ab_eigenVal(H)
        vec_Hessian[:,i] = H.reshape(-1)

    vol_Hessian = vec_Hessian.reshape((9,h,d,w))
    
    return vol_Hessian

def GetVesselness(vector):
    Hessian = vector.reshape((3,3))
    # e-value in accending order
    evals, _ = np.linalg.eigh(Hessian)
    # vesselness measurement
    if evals[0] or evals[1]>0:
        vesselness = 0
    else:    
        # λ1,λ2,λ3
        abs_evals = np.abs(evals)
        abs_evals = abs_evals.sort()
        
        Ra = abs_evals[1]/abs_evals[2]
        Rb = abs_evals[0]/(np.sqrt(abs_evals[1]*abs_evals[2]))
        S = np.sqrt(np.sum(np.multiply(evals,evals)))
        
        vesselness = (1-np.exp(-Ra**2/(2*α**2)))*np.exp(-Rb**2/(2*β**2))*(1-np.exp(-S**2/(2*c**2)))
    return vesselness
    
#%% volume
root = 'E:\\OCTA\\eval\\'
data = util.nii_loader(root+'seg_roi.nii.gz')

roi = np.array([[40,70],[8,13],[40,70]])
dim = roi[:,1]-roi[:,0]

data = data[roi[0,0]:roi[0,1],roi[1,0]:roi[1,1],roi[2,0]:roi[2,1]]

#%%
sig_min = 0.25
sig_max = 2
num_step = 10
sigma = np.linspace(sig_min,sig_max,num_step)

collection = []

for i in range(len(sigma)):
   collection.append(GetHessian(data,sigma[i]).reshape((9,-1))) 
   
#%% choose the tensor with maximum vesselness
_,num = collection[0].shape
vesselness = np.zeros([len(collection),num],dtype=np.float32)
   
for i in range(len(collection)):
    vol = collection[i]
    
    for j in range(num):
        vesselness[i,j] = GetVesselness(vol[:,j])

indices = np.argmax(vesselness,axis=0)
opt = np.zeros([9,num],dtype=np.float32)

for i in range(len(indices)):
    idx = int(indices[i])
    vol = collection[idx]
    
    # take absolute value of eigenvalues and save in opt
    h_mat = vol[:,i].reshape((3,3))
    abs_h_mat = ab_eigenVal(h_mat)
    opt[:,i] = abs_h_mat.reshape((9,)) 

opt = opt.reshape((9,dim[0],dim[1],dim[2]))

#%% save
dataroot = 'E:\\dti\\'
file_exp = 'latent.nrrd' 
data_dti, header_dti = nrrd.read(dataroot+file_exp)

name = 'octa_multiscale_hessian.nrrd'

header = header_dti
header['sizes'] = [9,dim[0],dim[1],dim[2]]
header['space origin'] = np.float64(np.array([dim[0]/2,dim[1]/2,dim[2]/2]))

nrrd.write(dataroot+name,opt,header=header)
util.nii_saver(data,dataroot,'field_roi.nii.gz')

#%%
#a = opt.reshape((9,-1))
#l = []
#
#for i in range(4500):
#    mat = a[:,i].reshape((3,3))
#    ev,_ = np.linalg.eigh(mat) 
#    l.append(np.sqrt(np.sum(np.multiply(ev,ev))))


