import numpy as np
import SimulPowerIter as spi
import NIPALS as nip
import NIPALS_MGS as ngs
import NIPALS_torch as nip_gpu
from sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA



def pca_transform(image_set, num_components, analysis_type, stop_condition=1e-6):
    """ Calculate principal components of an image set and and return the image as represented by its principal components.
    Parameters:
        image_set - Float64 Array - 1 row for each image with the number of cols representing pixels per image
        num_components --  Number of principal components to use
        analysis_type -- Analysis type (Simultaneous_Iteration, NIPALS, NIPALS_GS Full_SVD, Incremental_PCA)
        stop_condition -- % Change in explained covariance deemed adequate for stopping calculation of principal components

    Returns:
        transformed_image_set - Image set transformed by principal components
        principal_components - Principal Components
        Scores
        Loadings
    Note:  Stopping condition is not supported by all analysis types. Choose either a stop_condition value or number of
    components but not both"""

    images = np.shape(image_set)[0]
    pixels = np.shape(image_set)[1]
    # Initialize transformed image set
    transformed_image_set = np.zeros((images,pixels))
    principal_components = np.zeros((images,num_components))

    if analysis_type == "Simultaneous_Iteration":
        image_means = np.mean(image_set,axis=1,keepdims=True)
        image_set -= image_means
        # image_set_covariance = np.cov(image_set.T)
        image_set_covariance = np.dot(image_set.T,image_set)/(np.shape(image_set)[1]-1)
        eigenvectors,evalues_,_ = spi.SimulIter(image_set_covariance,neigs=num_components,maxiters=100,tol=1e-6)
        reduced_set = np.dot(image_set, eigenvectors)
        principal_components = eigenvectors.T
        transformed_image_set = np.dot(reduced_set, principal_components)+image_means


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
        image_means=np.mean(image_set, axis=1,keepdims=True)
        image_set -= image_means
        scores, loadings, eigenvals = nip.NIPALS(image_set,num_components,stop_condition)
        transformed_image_set = np.dot(scores,loadings.T)
        #transformed_image_set = add_mean(transformed_image_set,image_means)
        principal_components = loadings.T

    elif analysis_type == "NIPALS_GS":
        image_means=np.mean(image_set, axis=1,keepdims=True)
        image_set -= image_means
        scores, loadings, eigenvals = ngs.NIPAL_GS(image_set,num_components,stop_condition)
        transformed_image_set = np.dot(scores,loadings.T)
        transformed_image_set+=image_means
        principal_components = loadings.T

    elif analysis_type == "NIPALS_GPU":
        image_means = np.mean(image_set, axis=1, keepdims=True)
        image_set -= image_means
        scores, loadings, eigenvals = nip_gpu.NIPALS(image_set, num_components, stop_condition)
        transformed_image_set = np.dot(scores, loadings.T)
        transformed_image_set += image_means
        principal_components = loadings.T


    return transformed_image_set, principal_components


def pca_transform_set(OriginalSet, EigenVectors, Mean):
    """Use projection matrix to transform images to reduced rank subspace"""
    reduced_set = np.dot(OriginalSet,EigenVectors.T)
    new_set = np.dot(reduced_set,EigenVectors)

    return new_set
