import numpy as np


def SVD_PCA(X, num_components):
    u, s, vh = np.linalg.svd(X, full_matrices=False)
    samples=np.shape(X)[0]
    variables=np.shape(X)[1]


    Components = V[:,0:num_components]

    Eigenvals = s^2/
    return Scores, Loadings, Eigenvals
