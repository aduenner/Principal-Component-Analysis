import numpy as np
import torch

import numpy as np
import torch

def NIPALS(X, num_components, threshold=1e-6, max_iter=200):

    Scores = np.zeros((np.shape(X)[0],num_components))
    Loadings = np.zeros([np.shape(X)[1],num_components])

    Eigenvals = np.zeros(num_components)

    for i in range(num_components):

        old_lambda = 0
        t = X[:,i]

        for j in range(max_iter):

            # Compute loadings
            p = np.dot(X.T, t)
            p /= np.dot(t.T,t)
            Loadings[:, i] = p

            p = p / np.linalg.norm(p)
     
            # Project X onto p to find score vector t
            t = np.dot(X, p)
            t /= np.dot(p.T,p)
            Scores[:, i] = t
      
            new_lambda = np.dot(t.T,t)

            # Add score vector to matrix of score vectors
            Eigenvals[i] = new_lambda

            # Check for convergence
            diff = np.abs(old_lambda - new_lambda)

            if diff < threshold:
                break

            old_lambda = new_lambda
            
        # Update Xh
        X -= np.dot(t[:,np.newaxis], p[np.newaxis,:])

    return Scores, Loadings, Eigenvals






