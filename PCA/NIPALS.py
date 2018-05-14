import numpy as np

def NIPALS(X, num_components, threshold=1e-3):


    Scores = np.zeros((np.shape(X)[0],num_components))
    Loadings = np.zeros((np.shape(X)[1],num_components))

    Eigenvals = np.zeros(num_components)

    num_iters=10
    for i in range(num_components):
        old_eigen = 0
        t=X[:,i]
        for j in range(num_iters):
            # compute loadings
            p = np.dot(X.T, t)/np.dot(t.T,t)
            Loadings[:, i] = p
            #p = np.dot(p,np.sqrt(np.dot(p.T,p)))
            p=p/np.linalg.norm(p)
            # project X onto p to find score vector t
            t = np.dot(X, p)/ np.dot(p.T,p)
            Scores[:, i] = t
            # Compute eigenvalue from t
            new_eigen = np.dot(t.T,t)
            Eigenvals[i] = new_eigen
            # add score vector to matrix of score vectors

            # check for convergence
            diff = np.abs(old_eigen-new_eigen)
            if diff < threshold:
                break
            old_eigen = new_eigen


            # Update Xh
        X = X - np.dot(t[:,np.newaxis], p[np.newaxis,:])

    return Scores, Loadings, Eigenvals




