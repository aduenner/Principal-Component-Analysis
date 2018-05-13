import numpy as np
import SimulPowerIter as spi
import NIPALS as nip
from sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA



def pca_transform(image_set, num_components, analysis_type, stop_condition=1e-6):
    """ Calculate principal components of an image set and and return the image as represented by its principal components.
    Parameters:
        image_set - Int16 Array - 1 row for each image with the number of cols representing pixels per image
        num_components --  Number of principal components to use
        analysis_type -- Analysis type (Simultaneous_Iteration, NIPALS, Full_SVD, Incremental_PCA)
        stop_condition -- % Change in explained covariance deemed adequate for stopping calculation of principal components

    Returns:
        transformed_image_set - Image set transformed by principal components
        principal_components - Principal Components
        Scores
        Loadings
    Note:  Stopping condition is not supported by all analysis types. Choose either a stop_condition value or number of
    components but not both"""

    # Transform image set to double
    image_set = image_set/255.0
    images = np.shape(image_set)[0]
    pixels = np.shape(image_set)[1]
    # Initialize transformed image set
    transformed_image_set = np.zeros((images,pixels))
    principal_components = np.zeros((images,num_components))

    if analysis_type == "Simultaneous_Iteration":
        zero_mean_set,image_means = zero_mean(image_set)
        # image_set_covariance = np.cov(image_set.T)
        image_set_covariance = np.dot(zero_mean_set.T,zero_mean_set)
        principal_components,evalues_,_ = spi.SimulIter(image_set_covariance,neigs=num_components,maxiters=100,tol=1e-6)
        principal_components = principal_components.T
        transformed_image_set = pca_transform_set(zero_mean_set,principal_components,image_means)

    elif analysis_type == "Full_SVD":
        builtin_pca = PCA(num_components)
        builtin_pca.fit(image_set)
        principal_components = builtin_pca.components_
        imageset_reduced = np.dot(image_set - builtin_pca.mean_,builtin_pca.components_.T)

        transformed_image_set = np.dot(imageset_reduced, principal_components) + builtin_pca.mean_

    elif analysis_type == "Incremental_PCA":
        incremental_pca = IncrementalPCA(num_components)
        incremental_pca.fit(image_set)
        principal_components = incremental_pca.components_
        imageset_reduced = np.dot(image_set - incremental_pca.mean_, incremental_pca.components_.T)

        transformed_image_set = np.dot(imageset_reduced, principal_components) +incremental_pca.mean_

    elif analysis_type == "NIPALS":
        zero_mean_set, image_means = zero_mean(image_set)
        scores, loadings, eigenvals = nip.NIPALS(zero_mean_set,num_components,stop_condition)
        transformed_image_set = np.dot(scores,loadings.T)
        principal_components = loadings.T
        for i in range(np.shape(transformed_image_set)[0]):
            transformed_image_set[i, :] += image_means[i]

    return transformed_image_set, principal_components


def zero_mean(image_set):
    """Use to zero the mean of images in rows of input matrix"""
    num_img = np.shape(image_set)[0]
    mean_val = np.zeros(num_img)

    for i in range(num_img):
        # Mean value for each image is the mean of its pixels which are represented in the columns of the image's row
        mean_val[i] = np.mean(image_set[i, :])
        # Subtract the mean from each image
        image_set[i, :] = image_set[i, :] - mean_val[i]
    return image_set, mean_val


def pca_transform_set(OriginalSet, EigenVectors, Mean):
    """Use projection matrix to transform images to reduced rank subspace"""
    reduced_set = np.dot(OriginalSet,EigenVectors.T)
    new_set = np.dot(reduced_set,EigenVectors)
    for i in range(np.shape(new_set)[0]):
        new_set[i, :] += Mean[i]

    return new_set
