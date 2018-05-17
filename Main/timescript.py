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
import torch


num_components = 784 # Number of principal components
num_images = 1000  # ber of images from set to test with
image_set_full = np.load('NNHelper/noised_data_test_50.npy')
image_set_int = image_set_full
image_set = np.asarray(image_set_int,dtype=np.float32)/255.0
reduced_set = image_set[:num_images]

# image_set = np.random.uniform(size = (num_images, num_components))
def time_funcs(ncomponents, image_set):

    simuliter = functools.partial(pca_transform,image_set,ncomponents,'Simultaneous_Iteration')
    # fullsvd     = functools.partial(pca_transform,image_set,ncomponents,'Full_SVD')
    # incremental = functools.partial(pca_transform,image_set,ncomponents,'Incremental_PCA')
    nipals      = functools.partial(pca_transform,image_set,ncomponents,'NIPALS')
    nipals_gpu = functools.partial(pca_transform,image_set,ncomponents,'NIPALS_GPU')
    svdnumpy    = functools.partial(pca_transform,image_set,ncomponents,'SVD_Numpy')

    print('Testing Simultaneous Iteration')
    t_simuliter   = timeit.timeit(simuliter,number=3)
    #m_simuliter = max(memory_usage((pca_transform,(image_set,num_components,'Simultaneous_Iteration'))))
    print('Simultaneous Iteration took'+repr(t_simuliter)+' seconds to compute '+repr(ncomponents)+' PCs')
    #print('memory usage= '+repr(m_simuliter))

    # print('Testing scikit fullsvd')
    # t_fullsvd     = timeit.timeit(fullsvd,number=1)
    # #m_fullsvd     = max(memory_usage(fullsvd))
    # print('scikit fullsvd took' + repr(t_fullsvd) + ' seconds to compute ' + repr(ncomponents) + ' PCs')
    # #print('memory usage= ' + repr(m_fullsvd))

    # print('Testing scikit incremental SVD')
    # t_incremental = timeit.timeit(incremental,number=1)
    # #m_incremental = max(memory_usage(incremental))
    # print('scikit incremental svd took ' + repr(t_incremental) + ' seconds to compute ' + repr(ncomponents) + ' PCs')
    # #print('memory usage= ' + repr(m_incremental))

    print('Testing NIPALS')
    t_nipals      = timeit.timeit(nipals,number=3)
    #m_nipals = max(memory_usage(nipals))
    print('nipals took ' + repr(t_nipals) + ' seconds to compute ' + repr(ncomponents) + ' PCs')
    #print('memory usage= ' + repr(m_nipals))
    # print('Testing NIPALS_GPU')
    # t_nipals_gpu     = timeit.timeit(nipals_gpu,number=3)
    # #m_nipals = max(memory_usage(nipals))
    # print('nipals took ' + repr(t_nipals_gpu) + ' seconds to compute ' + repr(ncomponents) + ' PCs')
    # #print('memory usage= ' + repr(m_nipals))

    print('Testing numpysvd')
    t_svdnumpy    = timeit.timeit(svdnumpy,number=3)
    #m_svdnumpy = max(memory_usage(svdnumpy))
    print('numpy svd took ' + repr(t_svdnumpy) + ' seconds to compute ' + repr(ncomponents) + ' PCs')
    #print('memory usage= ' + repr(m_svdnumpy))
    print('')
    #return t_simuliter, t_fullsvd, t_incremental, t_nipals, t_svdnumpy, m_simuliter, m_fullsvd, m_incremental, m_nipals, m_svdnumpy
    return t_simuliter, t_svdnumpy, t_nipals

def time_funcs3(ncomponents, image_set):

    nipals      = functools.partial(pca_transform,image_set,ncomponents,'NIPALS')
    nipals_gpu = functools.partial(pca_transform,image_set,ncomponents,'NIPALS_GPU')
  
  
    print('Testing NIPALS')
    t_nipals      = timeit.timeit(nipals,number=3)
    #m_nipals = max(memory_usage(nipals))
    print('nipals took ' + repr(t_nipals) + ' seconds to compute ' + repr(ncomponents) + ' PCs')
    #print('memory usage= ' + repr(m_nipals))
    print('Testing NIPALS_GPU')
    t_nipals_gpu     = timeit.timeit(nipals_gpu,number=3)
    #m_nipals = max(memory_usage(nipals))
    print('nipals took ' + repr(t_nipals_gpu) + ' seconds to compute ' + repr(ncomponents) + ' PCs')
    #print('memory usage= ' + repr(m_nipals))
    return t_nipals, t_nipals_gpu

def time_funcs2(n_images, image_set):

    rset = image_set[:n_images]

    simuliter = functools.partial(pca_transform,rset, 50,'Simultaneous_Iteration')
    # fullsvd     = functools.partial(pca_transform,image_set,ncomponents,'Full_SVD')
    # incremental = functools.partial(pca_transform,image_set,ncomponents,'Incremental_PCA')
    nipals      = functools.partial(pca_transform,rset,50,'NIPALS')
    nipals_gpu = functools.partial(pca_transform,rset,50,'NIPALS_GPU')
    svdnumpy    = functools.partial(pca_transform,rset,50,'SVD_Numpy')

    print('Testing Simultaneous Iteration')
    t_simuliter   = timeit.timeit(simuliter,number=1)
    #m_simuliter = max(memory_usage((pca_transform,(image_set,num_components,'Simultaneous_Iteration'))))
    print('Simultaneous Iteration took'+repr(t_simuliter)+' seconds to compute')
    #print('memory usage= '+repr(m_simuliter))

    # print('Testing scikit fullsvd')
    # t_fullsvd     = timeit.timeit(fullsvd,number=1)
    # #m_fullsvd     = max(memory_usage(fullsvd))
    # print('scikit fullsvd took' + repr(t_fullsvd) + ' seconds to compute ' + repr(ncomponents) + ' PCs')
    # #print('memory usage= ' + repr(m_fullsvd))

    # print('Testing scikit incremental SVD')
    # t_incremental = timeit.timeit(incremental,number=1)
    # #m_incremental = max(memory_usage(incremental))
    # print('scikit incremental svd took ' + repr(t_incremental) + ' seconds to compute ' + repr(ncomponents) + ' PCs')
    # #print('memory usage= ' + repr(m_incremental))

    # print('Testing NIPALS')
    # t_nipals      = timeit.timeit(nipals,number=1)
    # #m_nipals = max(memory_usage(nipals))
    # t_nipals *= 5
    # print('nipals took ' + repr(t_nipals) + ' seconds to compute')
    # #print('memory usage= ' + repr(m_nipals))
    # # print('Testing NIPALS_GPU')
    # # t_nipals_gpu     = timeit.timeit(nipals_gpu,number=3)
    # # #m_nipals = max(memory_usage(nipals))
    # # print('nipals took ' + repr(t_nipals_gpu) + ' seconds to compute ' + repr(ncomponents) + ' PCs')
    # # #print('memory usage= ' + repr(m_nipals))

    # # print('Testing numpysvd')

    # aa = []

    # for i in range(5):
    #     aa.append(timeit.timeit(svdnumpy,number=1))

    # t_svdnumpy = np.median(np.array(aa)) * 5
    # #m_svdnumpy = max(memory_usage(svdnumpy))
    # print('numpy svd took ' + repr(t_svdnumpy) + ' seconds to compute')
    # #print('memory usage= ' + repr(m_svdnumpy))
    # print('')
    # #return t_simuliter, t_fullsvd, t_incremental, t_nipals, t_svdnumpy, m_simuliter, m_fullsvd, m_incremental, m_nipals, m_svdnumpy
    return t_simuliter, 0,0


with torch.cuda.device(0):

    componentlist = [10, 20, 30, 40, 50, 60, 80, 100, 120, 140, 160, 200, 240, 280, 320, 360, 440, 520, 600, 680,768]

    results=np.zeros((20,3))



    # for i in range(len(componentlist)):
    #     components = componentlist[i]
    #     results[i,:] = time_funcs(components, image_set)
    #     np.save('new_results',results)

    image_set = np.random.uniform(size = (200000, num_components))

    componentlist = [10000, 20000, 30000, 40000, 50000, 75000, 100000, 150000, 200000]

    results=np.zeros((10,3))

    for i in range(len(componentlist)):
        components = componentlist[i]
        results[i,:] = time_funcs2(components, image_set)
        np.save('new_results',results)



    # #     print(results)
    # # legend=('Simultaneous Power Iteration', 'Full SVD', 'Incremental SVD','NIPALS','SVD Numpy')
    
    # image_set = np.random.uniform(size = (50000, num_components))

    # componentlist = [10, 20, 30, 40, 50, 60, 80, 100, 120, 140, 160]

    # results=np.zeros((20,2))

    # for i in range(len(componentlist)):
    #     components = componentlist[i]
    #     results[i,:] = time_funcs3(components, image_set)
    #     np.save('gpu_results',results)



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
