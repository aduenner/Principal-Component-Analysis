# -*- coding: utf-8 -*-
"""
Created on Sat May 12 14:40:48 2018

@author: aduen
"""

import matplotlib.pyplot as plt
from PCA import *
import numpy as np

def datatest(imageset):
    plot_images(imageset,10)
    return imageset

def meantest(imageset):
    means = np.mean(imageset,axis=1,keepdims=True)
    imagesetout=imageset-means
    plot_images(imagesetout)
    return imagesetout


def simuliter(imageset, ncomponents):
    transformed_set, components, _ = pca_transform(imageset, ncomponents, 'Simultaneous_Iteration')
    plot_images(transformed_set)
    plot_images(components, min(np.shape(components)[0],16))
    return transformed_set, components


def fullsvd(imageset, ncomponents):
    transformed_set, components, _ = pca_transform(imageset, ncomponents, 'Full_SVD')
    plot_images(transformed_set,1)
    plot_images(components, min(np.shape(components)[0],4))
    return transformed_set, components


def incremental_pca(imageset, ncomponents):
    transformed_set, components, _ = pca_transform(imageset, ncomponents, 'Incremental_PCA')
    plot_images(transformed_set)
    plot_images(components, min(np.shape(components)[0],16))
    return transformed_set, components

def nipals(imageset, ncomponents):
    transformed_set, components, _ = pca_transform(imageset, ncomponents, 'NIPALS')
    plot_images(transformed_set)
    plot_images(components,min(np.shape(components)[0],16))
    return transformed_set, components

def nipalsgs(imageset, ncomponents):
    transformed_set, components, _ = pca_transform(imageset, ncomponents, 'NIPALS_GS')
    plot_images(transformed_set)
    plot_images(components,min(np.shape(components)[0],16))
    return transformed_set, components

def nipalsgpu(imageset, ncomponents):
    transformed_set, components, _ = pca_transform(imageset, ncomponents, 'NIPALS_GPU')
    plot_images(transformed_set)
    plot_images(components,min(np.shape(components)[0],16))
    return transformed_set, components

def svdnumpy(imageset, ncomponents):
    transformed_set, components, _ = pca_transform(imageset, ncomponents, 'SVD_Numpy')
    plot_images(transformed_set)
    plot_images(components,min(np.shape(components)[0],16))
    return transformed_set,components

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


def plot_images(imageset, num_images=4):
    shape = int(np.ceil(np.sqrt(np.shape(imageset)[1])))
    numrows = int(np.ceil(np.sqrt(num_images)))
    for i in range(num_images):
        plt.subplot(numrows, numrows, i+1)
        plt.imshow(extract_image(imageset,shape,i),cmap='gray')
        plt.axis('off')

    plt.show()
