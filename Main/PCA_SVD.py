import numpy as np

def PCA_SVD(X, num_components):

    u, s, vh = np.linalg.svd(X, full_matrices=False)
    samples = np.shape(X)[0]
    variables = np.shape(X)[1]

    Components = (vh.T)[:, 0:num_components]
    Scores = np.dot(u,np.diag(s))
    Scores = Scores[:,0:num_components]
    Eigenvals = np.dot(s,s) / (samples-1)

    return Scores, Components, Eigenvals
