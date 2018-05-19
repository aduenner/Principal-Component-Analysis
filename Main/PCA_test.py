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
    plot_images(transformed_set,'simuliter images')
    plot_images(components, 'simuliter components',min(np.shape(components)[0],16))
    return transformed_set, components


def fullsvd(imageset, ncomponents):
    transformed_set, components, _ = pca_transform(imageset, ncomponents, 'Full_SVD')
    plot_images(transformed_set,'fullsvd images')
    plot_images(components, 'fullsvd components',min(np.shape(components)[0],16))
    return transformed_set, components


def incremental_pca(imageset, ncomponents):
    transformed_set, components, _ = pca_transform(imageset, ncomponents, 'Incremental_PCA')
    plot_images(transformed_set,'incremental pca images')
    plot_images(components, 'incremental pca components',min(np.shape(components)[0],16))
    return transformed_set, components

def nipals(imageset, ncomponents):
    transformed_set, components, _ = pca_transform(imageset, ncomponents, 'NIPALS')
    plot_images(transformed_set,'nipals images'),
    plot_images(components,'nipals components',min(np.shape(components)[0],16))
    return transformed_set, components

def nipalsgs(imageset, ncomponents):
    transformed_set, components, _ = pca_transform(imageset, ncomponents, 'NIPALS_GS')
    plot_images(transformed_set,'nipalsgs images')
    plot_images(components,'nipalsgs components',min(np.shape(components)[0],16))
    return transformed_set, components

def nipalsgpu(imageset, ncomponents):
    transformed_set, components, _ = pca_transform(imageset, ncomponents, 'NIPALS_GPU')
    plot_images(transformed_set,'nipalsgpu images')
    plot_images(components,'nipalsgpu components',min(np.shape(components)[0],16))
    return transformed_set, components

def svdnumpy(imageset, ncomponents):
    transformed_set, components, _ = pca_transform(imageset, ncomponents, 'SVD')
    plot_images(transformed_set,'svdnumpy images')
    plot_images(components,'svdnumpy components',min(np.shape(components)[0],16))
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


def plot_images(imageset, Title, num_images=4):
    shape = int(np.ceil(np.sqrt(np.shape(imageset)[1])))
    numrows = int(np.ceil(np.sqrt(num_images)))
    for i in range(num_images):
        plt.subplot(numrows, numrows, i+1)
        plt.imshow(extract_image(imageset,shape,i),cmap='gray')
        plt.axis('off')
        plt.title(Title)

    plt.show()
