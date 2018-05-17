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


num_components = 70 # Number of principal components
num_images = 1000  # ber of images from set to test with
image_set_full = np.load('NNHelper/noised_data_test_50.npy')
image_set_int = image_set_full
image_set = np.asarray(image_set_int,dtype=np.float32)/255.0

def time_funcs(ncomponents):
    simuliter = functools.partial(pca_transform,image_set,ncomponents,'Simultaneous_Iteration')
    fullsvd     = functools.partial(pca_transform,image_set,ncomponents,'Full_SVD')
    incremental = functools.partial(pca_transform,image_set,ncomponents,'Incremental_PCA')
    nipals      = functools.partial(pca_transform,image_set,ncomponents,'NIPALS')
    svdnumpy    = functools.partial(pca_transform,image_set,ncomponents,'SVD_Numpy')

    print('Testing Simultaneous Iteration')
    t_simuliter   = timeit.timeit(simuliter,number=1)
    #m_simuliter = max(memory_usage((pca_transform,(image_set,num_components,'Simultaneous_Iteration'))))
    print('Simultaneous Iteration took'+repr(t_simuliter)+' seconds to compute '+repr(ncomponents)+' PCs')
    #print('memory usage= '+repr(m_simuliter))

    print('Testing scikit fullsvd')
    t_fullsvd     = timeit.timeit(fullsvd,number=1)
    #m_fullsvd     = max(memory_usage(fullsvd))
    print('scikit fullsvd took' + repr(t_fullsvd) + ' seconds to compute ' + repr(ncomponents) + ' PCs')
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

results=np.zeros((20,5))
for i in range(len(componentlist)):
    components = componentlist[i]
    results[i,:] = time_funcs(components)
legend=('Simultaneous Power Iteration', 'Full SVD', 'Incremental SVD','NIPALS','SVD Numpy')
np.save('testresults',results)



#tfullsvd   = timeit.Timer(functools.partial(pca_transform,image_set,num_components,'Full_SVD'),5)
    #transformed_set, components = pca_transform(imageset, ncomponents, 'Simultaneous_Iteration')
    #return transformed_set, components
#
#
# def fullsvd(imageset, ncomponents):
#     transformed_set, components = pca_transform(imageset, ncomponents, 'Full_SVD')
#     return transformed_set, components
#
#
# def incremental_pca(imageset, ncomponents):
#     transformed_set, components = pca_transform(imageset, ncomponents, 'Incremental_PCA')
#     return transformed_set, components
#
# def nipals(imageset, ncomponents):
#     transformed_set, components = pca_transform(imageset, ncomponents, 'NIPALS')
#     return transformed_set, components
#
# def nipalsgs(imageset, ncomponents):
#     transformed_set, components = pca_transform(imageset, ncomponents, 'NIPALS_GS')
#     return transformed_set, components
#
# def nipalsgpu(imageset, ncomponents):
#     transformed_set, components = pca_transform(imageset, ncomponents, 'NIPALS_GPU')
#     return transformed_set, components
#
# def svdnumpy(imageset, ncomponents):
#     transformed_set, components = pca_transform(imageset, ncomponents, 'SVD_Numpy')
#     return transformed_set,components
#
# t=timeit.Timer(functools.partial())
#
# #PCorig,explainedorig = PrincipleComponents(zeromean_orig,15)
# # pca_orig = PCA(n_components=784)
# # ReducedNoisy=pca_orig.fit_transform(zeromean_noisy)
# # X_inv_proj = pca_orig.inverse_transform(ReducedNoisy)
# # #reshaping as 400 images of 64x64 dimension
# # X_proj_img = np.reshape(X_inv_proj,(1000,28,28))
# #
# # show_image(X_proj_img[0])
# #
# # OurReduced = PrincipalComponents(zeromean_noisy,10)
# # OurReducedTrans = TransformSpace(zeromean_noisy,OurReduced,mean_noisy)
# # OurReducedFirstImg = extract_image(OurReducedTrans, 28,0)
# # plt.subplot(224)
# # show_image(OurReducedFirstImg)
# ##
# #pc1 = extract_image(PCorig,28,0)
# #show_image(pc1)
# #
# #plt.subplot(224)
# #pc2 = extract_image(PCnoise,28,0)
# #show_image(pc2)
# # varrat = pca_orig.explained_variance_ratio_
# # cumvar = np.zeros(np.shape(varrat))
# # cumvar[1]=varrat[1]
# #
# # for i in range(1,np.shape(varrat)[0]):
# #     cumvar[i] = varrat[i]+cumvar[i-1]
