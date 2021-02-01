# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 18:57:40 2021

@author: hudew
"""

import sys
sys.path.insert(0,'E:\\tools\\')
import util
import random, math
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from random_geometry_points.plane import Plane

# define the radius of sigma_1 and height of sigma_2
global r,h
r = 2
h = 5

def sample_vec(num):
    global r
    theta_phi = np.float32(np.random.uniform(0.0, np.pi*2, size = (2,num)))
    return np.concatenate((r*np.ones((1,num),dtype=np.float32),theta_phi),axis=0)

def sample_from_plane(n,ref,radius,num):
    '''
    n[tuple]: normal vector in cartesian coordinate
    ref[tuple]: reference point 
    num[int]: number of sample points required
    '''
    plane = Plane(n,0.0,ref,radius)
    samples = plane.create_random_points(num)
    return np.float32(np.asarray(samples))

def sphere2cart(vector):
    _,num = vector.shape
    opt = np.zeros((3,num),dtype=np.float32)
    opt[0,:] = np.multiply(np.sin(vector[1,:]),
                           np.cos(vector[2,:]))
    opt[1,:] = np.multiply(np.sin(vector[1,:]),
                           np.sin(vector[2,:]))
    opt[2,:] = np.cos(vector[1,:])
    return opt

def interpolate(volume,vec_cart):
    _,num = vec_cart.shape
    opt = np.zeros([num,],dtype=np.float32)
    for i in range(num):
        p = vec_cart[:,i]
        # suppose there is no out-of-boundary problem
        neighbors = np.array([[np.floor(p[0]),np.floor(p[1]),np.floor(p[2])],
                              [np.floor(p[0]),np.ceil(p[1]),np.floor(p[2])],
                              [np.floor(p[0]),np.floor(p[1]),np.ceil(p[2])],
                              [np.floor(p[0]),np.ceil(p[1]),np.ceil(p[2])],
                              [np.ceil(p[0]),np.floor(p[1]),np.floor(p[2])],
                              [np.ceil(p[0]),np.ceil(p[1]),np.floor(p[2])],
                              [np.ceil(p[0]),np.floor(p[1]),np.ceil(p[2])],
                              [np.ceil(p[0]),np.ceil(p[1]),np.ceil(p[2])]]).astype(int)
        values = np.zeros([8,],dtype=np.float32)    
        for j in range(8):
            values[j] = volume[neighbors[j,0],neighbors[j,1],neighbors[j,2]]
        opt[i] = griddata(neighbors,values,p,method='linear')
    return opt

def classify(points,center):
    '''
    points: cartesian coordinates of samples arranged in shape: [3,num].
    center: the center point
    compare the distance to center point with the radius r:
         d > r: sigma_2 (marked 1)
        d <= r: sigma_1 (marked 0)
    '''
    global r
    d = np.sum(np.square(points-center),axis=0)
    opt = np.uint8(d > r**2)
    return opt
    
def cylinder_sample(vec_sphere,center,num_sample):
    '''
    center: the voxel coordinate 
    vec_sphere: a random vector in spherical coordinate
    num_sample: number of sample points in each integer step along the norm
    
    The aim of this function is sample in the cylinder indicated by the orientation 
    vector. The output will be the coordinate of samples and their corresponding 
    class(in/out of the sphere)
    '''
    global r, h
    
    sample_points = []
    sample_class = []
    
    # h samples along each dir which defines the center of the plane
    for j in range(-r,h):
        norm = sphere2cart(vec_sphere)
        n = (np.float(norm[0,0]),np.float(norm[1,0]),np.float(norm[2,0]))
        # sample on the plane that include (0,0,0) and shift by p
        samples = (sample_from_plane(n,(0,0,0),r,num_sample)).T+center+norm*j
        # compare the distance from sample points to center
        pout = classify(samples,center)
        
        sample_points.append(samples)
        sample_class.append(pout)
    
    return sample_points,sample_class

def Get_M(sample_points,sample_class,volume):
    '''
    compute the M value of the specific direction
    '''
    # vectorize
    for i in range(len(sample_points)):
        if i == 0:
            idx = sample_points[i]
            cla = sample_class[i]
        else:
            idx = np.concatenate((idx,sample_points[i]),axis=1)
            cla = np.concatenate((cla,sample_class[i]))
    # interpolate
    dim = len(cla)
    cla = cla.reshape([dim,1])
    intensity = interpolate(volume,idx).reshape([dim,1])
    mu_out = np.sum(np.multiply(intensity,cla))/np.sum(cla)
    mu_in = np.sum(np.multiply(intensity,1-cla))/np.sum(1-cla)
    
    return np.square(mu_in-mu_out)

def H_matrix(orients):
    '''
    takes the orients(cartesian) with shape [3,vec_num]
    '''
    _,num = orients.shape
    H = np.zeros([num,6],dtype=np.float32)
    for i in range(num):
        H[i,:] = np.array([orients[0,i]**2,orients[1,i]**2,orients[2,i]**2,
                          2*orients[0,i]*orients[1,i],2*orients[0,i]*orients[2,i],
                          2*orients[1,i]*orients[2,i]],dtype=np.float32)
    return H

def GetTensor(volume,vec_num,layer_sample_num,center):
    '''
    for voxel [center], collect data from [vec_num] of directions
    orients: orientation in cartesian 
    m_values: correspoinding m value
    '''
    orients = np.zeros([3,vec_num],dtype=np.float32)
    m_values = np.zeros([1,vec_num],dtype=np.float32)
    ori_sphere = sample_vec(vec_num) #[3,vec_num]
    
    for i in range(vec_num):
        vec_sphere = (ori_sphere[:,i]).reshape([3,1])
        orients[:,i] = sphere2cart(vec_sphere).reshape([3,])
        sp, sc = cylinder_sample(vec_sphere,center,layer_sample_num)
        m_values[:,i] = Get_M(sp,sc,volume)
        
        H = H_matrix(orients) 
        d_vec = np.matmul(np.linalg.pinv(H),m_values.T)
        d_tensor = np.array([[d_vec[0,0],d_vec[3,0],d_vec[4,0]],
                         [d_vec[3,0],d_vec[1,0],d_vec[5,0]],
                         [d_vec[4,0],d_vec[5,0],d_vec[2,0]]],dtype=np.float32)
    return d_tensor
    
    
#%% Function Test
#vec_sphere = sample_vec(1)
#center = np.array([[0,0,0]]).T
#sp,sc = cylinder_sample(vec_sphere,center,200)
#orient = sphere2cart(vec_sphere)
#
#from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
#
#xs = []
#ys = []
#zs = []
#ms = []
#
#for i in range(len(sp)):
#    mat = sp[i]
#    xs.append(mat[0,:])
#    ys.append(mat[1,:])
#    zs.append(mat[2,:])
#    
#    cl = sc[i]
#    for j in range(len(cl)):
#        if cl[j] == 0:
#            ms.append('x')    
#        else:
#            ms.append('o')
#    
#fig = plt.figure(figsize=(10,10))
#ax = fig.add_subplot(111, projection='3d')
#
## Data for a three-dimensional line
#zline = np.linspace(-1,4,10)
#xline = orient[0,0]/orient[2,0]*(zline)
#yline = orient[1,0]/orient[2,0]*(zline)
#ax.plot3D(xline, yline, zline, 'black')
#
#for i in range(len(xs)):
#    x = xs[i]
#    y = ys[i]
#    z = zs[i]
#    m = ms[i]
#    ax.scatter(x, y, z, marker=m)
#    ax.set_xlabel('X Label')
#    ax.set_ylabel('Y Label')
#    ax.set_zlabel('Z Label')
#plt.show()


#%%
root = 'E:\\OCTA\\eval\\'
volume = util.nii_loader(root+'seg_roi.nii.gz')

vec_num = 30
layer_sample_num = 10
center = np.array([[64,10,69]],dtype=np.float32).T 

d_tensor = GetTensor(volume,vec_num,layer_sample_num,center)
    

