# -*- coding: utf-8 -*-
"""
Created on Sat May 12 14:40:48 2018

@author: aduen
"""

import matplotlib.pyplot as plt
from PCA import *
import numpy as np

num_components = 100  # Number of principal components
num_images = 1000  # number of images from set to test with
image_set_full = np.load('NNHelper\original_images.npy')
image_set = image_set_full[0:num_images, :]


def simuliter(input_set):
    transformed_set = pca_transform(input_set, num_components, 'Simul_Iter')
    plot_images(transformed_set)
    return transformed_set


def fullsvd(imageset):
    transformed_set = pca_transform(imageset, num_components, 'Full_SVD')
    plot_images(transformed_set)
    return transformed_set


def Incremental_PCA(imageset):
    transformed_set = pca_transform(imageset, num_components, 'Incremental_PCA')
    plot_images(transformed_set)
    return transformed_set

def NIPALS(imageset):
    return

def extract_image(imgset, shape, index):
    """Extracts a single square image from the image db
    Parameters:
        imgset - Image database
        shape - n pixels per row/col in square image: n^2 = #cols in imgset
        index - Index in image database that is to be extracted
    Outputs:
        image_out - nxn uint8 image that can be displayed or manipulated
    """
    image_out = np.reshape(imgset[index, :], (shape, shape))
    return image_out


def plot_images(imageset, num_images=16):
    shape = int(np.ceil(np.sqrt(np.shape(imageset)[1])))
    numrows = int(np.ceil(np.sqrt(num_images)))
    for i in range(num_images):
        plt.subplot[numrows,numrows,i+1] = extract_image(imageset,shape,i)
        plt.imshow()




#PCorig,explainedorig = PrincipleComponents(zeromean_orig,15)
# pca_orig = PCA(n_components=784)
# ReducedNoisy=pca_orig.fit_transform(zeromean_noisy)
# X_inv_proj = pca_orig.inverse_transform(ReducedNoisy)
# #reshaping as 400 images of 64x64 dimension
# X_proj_img = np.reshape(X_inv_proj,(1000,28,28))
#
# show_image(X_proj_img[0])
#
# OurReduced = PrincipalComponents(zeromean_noisy,10)
# OurReducedTrans = TransformSpace(zeromean_noisy,OurReduced,mean_noisy)
# OurReducedFirstImg = extract_image(OurReducedTrans, 28,0)
# plt.subplot(224)
# show_image(OurReducedFirstImg)
##
#pc1 = extract_image(PCorig,28,0)
#show_image(pc1)
#
#plt.subplot(224)
#pc2 = extract_image(PCnoise,28,0)
#show_image(pc2)
# varrat = pca_orig.explained_variance_ratio_
# cumvar = np.zeros(np.shape(varrat))
# cumvar[1]=varrat[1]
#
# for i in range(1,np.shape(varrat)[0]):
#     cumvar[i] = varrat[i]+cumvar[i-1]
