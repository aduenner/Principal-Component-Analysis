# -*- coding: utf-8 -*-
"""
Created on Sat May 12 14:40:48 2018

@author: aduen
"""

import matplotlib.pyplot as plt
from PCA import *
import numpy as np
import timeit
import functools
from memory_profiler import memory_usage

save_dir = "New Results"


# num_components = 70 # Number of principal components
num_images = 1000  # ber of images from set to test with
image_set_full = np.load('set1/noised_data_test_50.npy')
image_set_int = image_set_full
image_set = np.asarray(image_set_int,dtype=np.float32)/255.0
image_set = image_set[:num_images]

def time_funcs(ncomponents):
    
    simuliter = functools.partial(pca_transform,image_set,ncomponents,'Simultaneous_Iteration')
    fullsvd     = functools.partial(pca_transform,image_set,ncomponents,'Full_SVD')
    incremental = functools.partial(pca_transform,image_set,ncomponents,'Incremental_PCA')
    nipals      = functools.partial(pca_transform,image_set,ncomponents,'NIPALS')
    svdnumpy    = functools.partial(pca_transform,image_set,ncomponents,'SVD_Numpy')

    print('Testing Simultaneous Iteration')
    t_simuliter   = timeit.timeit(simuliter,number=1)
    #m_simuliter = max(memory_usage((pca_transform,(image_set,num_components,'Simultaneous_Iteration'))))
    print('Simultaneous Iteration took ' + repr(t_simuliter)+' seconds to compute ' + repr(ncomponents)+' PCs')
    #print('memory usage= '+repr(m_simuliter))

    print('Testing scikit fullsvd')
    t_fullsvd     = timeit.timeit(fullsvd,number=1)
    #m_fullsvd     = max(memory_usage(fullsvd))
    print('scikit fullsvd took ' + repr(t_fullsvd) + ' seconds to compute ' + repr(ncomponents) + ' PCs')
    #print('memory usage= ' + repr(m_fullsvd))

    print('Testing scikit incremental SVD')
    t_incremental = timeit.timeit(incremental,number=1)
    #m_incremental = max(memory_usage(incremental))
    print('scikit incremental svd took ' + repr(t_incremental) + ' seconds to compute ' + repr(ncomponents) + ' PCs')
    #print('memory usage= ' + repr(m_incremental))

    print('Testing NIPALS')
    t_nipals      = timeit.timeit(nipals,number=1)
    #m_nipals = max(memory_usage(nipals))
    print('nipals took ' + repr(t_nipals) + ' seconds to compute ' + repr(ncomponents) + ' PCs')
    #print('memory usage= ' + repr(m_nipals))

    print('Testing numpysvd')
    t_svdnumpy    = timeit.timeit(svdnumpy,number=1)
    #m_svdnumpy = max(memory_usage(svdnumpy))
    print('numpy svd took ' + repr(t_svdnumpy) + ' seconds to compute ' + repr(ncomponents) + ' PCs')
    #print('memory usage= ' + repr(m_svdnumpy))
    print('')
    #return t_simuliter, t_fullsvd, t_incremental, t_nipals, t_svdnumpy, m_simuliter, m_fullsvd, m_incremental, m_nipals, m_svdnumpy
    return t_simuliter, t_fullsvd, t_incremental, t_nipals, t_svdnumpy


componentlist = [10, 20, 30, 40, 50, 60, 80, 100, 120, 140, 160, 200, 240, 280, 320, 360, 440, 520, 600, 680,768]

results = np.zeros((20, 5))

for i in range(len(componentlist)):
    components = componentlist[i]
    results[i,:] = time_funcs(components)
legend=('Simultaneous Power Iteration', 'Full SVD', 'Incremental SVD','NIPALS','SVD Numpy')
np.save(os.path.join(save_dir, 'testresults.npy'), results)
