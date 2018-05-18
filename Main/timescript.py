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

CONST_COMPONENTS = 784
CONST_SET = np.load('set1/noised_data_test_50.npy')
CONST_SET = np.asarray(CONST_SET, dtype="float32") / 255.0

def numba_compare(ncomponents, data_set):

    simuliter   = functools.partial(pca_transform,data_set,ncomponents,'Simultaneous_Iteration')
    nipals      = functools.partial(pca_transform,data_set,ncomponents,'NIPALS')
    svdnumpy    = functools.partial(pca_transform,data_set,ncomponents,'SVD_Numpy')

    t_simuliter = timeit.timeit(simuliter,number=5)
    t_nipals    = timeit.timeit(nipals,number=5)
    t_svdnumpy  = timeit.timeit(svdnumpy,number=5)

    return t_simuliter, t_svdnumpy, t_nipals


def nipals():

def gpu_compare_core(ncomponents, image_set):

    nipals       = functools.partial(pca_transform,image_set,ncomponents,'NIPALS')
    nipals_gpu   = functools.partial(pca_transform,image_set,ncomponents,'NIPALS_GPU')
  
    t_nipals     = timeit.timeit(nipals,number=5)
    t_nipals_gpu = timeit.timeit(nipals_gpu,number=5)
  
    return t_nipals, t_nipals_gpu

def gpu_compare(n_images = 25000, checklist=[10, 20, 30, 40, 50, 60]):

    with torch.cuda.device(0):

        max_components = max(checklist)

        ret = np.zeros((len(checklist), 2), dtype="float32")

        image_set = np.random.uniform(size = (n_images, max_components))

        for i in range(len(checklist))

            n_components = checklist[i]
        
            results[i,:] = gpu_compare(components, image_set)
    #     np.save('gpu_results',results)




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
