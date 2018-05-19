import numpy as np
import operator

def _NIPALS(X, scores, loadings, lambdas, n_components, threshold=1e-6, max_iter=200):

    for i in range(n_components):

        t = X[:, i]
        _lambda = np.dot(t, t) 

        for j in range(max_iter):

            # Compute loadings
            p = np.dot(X.T, t)
            p /= _lambda
            loadings[:, i] = p

            p /= np.linalg.norm(p)
    
            # Project X onto p to find score vector t
            t = np.dot(X, p)
            t /= np.dot(p, p)
            scores[:, i] = t
      
            _lambda_n = np.dot(t, t)

            # Add score vector to matrix of score vectors
            lambdas[i] = _lambda_n

            # Check for convergence
            diff = np.abs(_lambda - _lambda_n)

            if diff < threshold:
                break

            _lambda = _lambda_n
           
        # Update X
        X -= np.outer(t, p) 

def _SVD(X, scores, loadings, lambdas, n_components):

    u, s, vt = np.linalg.svd(X, full_matrices=False)
    m = np.shape(X)[0]

    loadings[:] = (vt.T)[:, 0:n_components]
    scores[:] = np.dot(u, np.diag(s))[:,0:n_components]
    lambdas[:] = np.dot(s, s) / (m - 1)

def _SI(X, Q, Q_o, R, I, threshold=1e-6, max_iter=200): 

    z = 0

    for k in range(max_iter):

        A = operator.matmul(X, Q)

        Q_o[:] = Q

        _MGS(A, Q, R)

        delta = operator.matmul(I - np.dot(Q, Q.T), Q_o)
        
        if np.linalg.norm(delta) <= threshold:

            break

        z = k

def _MGS(A, Q, R):

    (m, n) = A.shape

    for k in range(n):

        R[k, k] = np.sqrt(np.dot(A[:, k], A[:, k]))
        Q[:, k] = A[:, k] / R[k,k]

        for j in range (k + 1,n):

            R[k, j] = np.dot(Q[:, k].T, A[:, j])
            A[:, j] -= R[k, j] * Q[:, k]
