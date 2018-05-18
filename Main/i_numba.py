from numba import jit
import numpy as np
import operator

# @jit(nopython=True)
def _NIPALS(X, scores, loadings, lambdas, n_components, threshold=1e-6, max_iter=1000):

    for i in range(n_components):

        t = X[:, i]
        _lambda = t
        _lambda_n = np.dot(_lambda, _lambda)

        for j in range(max_iter):

            # Compute loadings
            p = np.dot(X.T, t)
            p /= _lambda_n
            loadings[:, i] = p

            p /= np.linalg.norm(p)
    
            # Project X onto p to find score vector t
            t = np.dot(X, p)
            t /= np.dot(p, p)
            scores[:, i] = t

            diff = np.linalg.norm(_lambda - t)
            _lambda[:] = t[:]

      
            # Add score vector to matrix of score vectors
            _lambda_n = np.dot(_lambda, _lambda)
            lambdas[i] = _lambda_n

            # Check for convergence
            if diff < threshold:
                print(j)
                break
           
        # Update X
        X -= np.outer(t, p) * 1


