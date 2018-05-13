import numpy as np

def NIPALS(X, num_components, threshold=1e-6):

    Residuals = X
    Scores = np.zeros((np.shape(X)[0],num_components))
    Loadings = np.zeros((np.shape(X)[1],num_components))
    t = X[:,1]
    Eigenvals = np.zeros(num_components)

    for i in range(num_components):

        mu = 0
        v=R[:,i]
        for j in range(100)
            U=np.dot(R.T,V)




        # compute loadings
        p = np.dot(X.T, t)/np.dot(t.T, t)
        # add loadings to matrix of loadings
        Loadings[:, i] = p
        # normalize loadings
        p = np.dot(p, np.sqrt(np.dot(p.T, p)))
        # project X onto p to find score vector t
        t = np.dot(X, p)/np.dot(p.T, p)

        # Compute eigenvalue from t
        Eigenvals[i] = np.dot(t.T, t)
        # add score vector to matrix of score vectors
        Scores[:, i] = t
        # check for convergence
        if i > 0:
            diff = np.abs(Eigenvals[i] - Eigenvals[i-1])
            print(diff)
            if diff < threshold:
                break

        # Update Xh
        X -= np.dot(t[:,np.newaxis], p[np.newaxis,:])

    return Scores, Loadings, Eigenvals




