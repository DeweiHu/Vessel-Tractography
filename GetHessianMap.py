# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 00:17:55 2021

@author: hudew
"""

import sys
sys.path.insert(0,'E:\\tools\\')
import util
import numpy as np
import nrrd
import matplotlib.pyplot as plt
from skimage.feature import hessian_matrix, hessian_matrix_det

def ab_eigenVal(H):
    "eigen-decomposition"
    eigenValues, eigenVectors = np.linalg.eig(H)
    eigenValues = np.abs(np.diag(eigenValues))
    opt = eigenVectors.dot(eigenValues).dot(eigenVectors.T)
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

dataroot = 'E:\\dti\\'
file_exp = 'latent.nrrd'
data_dti, header_dti = nrrd.read(dataroot+file_exp)

#%%
root = 'E:\\OCTA\\paper_img\\'
name = 'latent.nrrd'
vol = util.nii_loader(root+'vol_seg5.nii.gz')
h,d,w = vol.shape

header = header_dti
header['sizes'] = [9,h,d,w]
header['space origin'] = np.float64(np.array([h/2,d/2,w/2]))

H_elements = np.array(hessian_matrix(vol, sigma=0.9, order='rc'))
vec = H_elements.reshape((6,-1))
_,num = vec.shape
vec_Hessian = np.zeros((9,num),dtype=np.float32)

for i in range(num):
    elements = vec[:,i]
    # fill the Hessian
    H = symmetrize(elements)
    # all positive eigenValue
    H = ab_eigenVal(H)
    vec_Hessian[:,i] = H.reshape(-1)

vol_Hessian = vec_Hessian.reshape((9,h,d,w))

nrrd.write(dataroot+name,vol_Hessian,header=header)



