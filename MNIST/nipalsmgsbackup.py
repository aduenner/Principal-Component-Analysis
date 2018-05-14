import numpy as np

def NIPAL_GS(X, num_components, threshold=1e-6):
    num_iters=10
    R = np.copy(X)
    rows=np.shape(X)[0]
    cols=np.shape(X)[1]
    U = np.zeros((cols,cols))
    V = np.zeros((rows,rows))

    Scores = np.zeros((np.shape(X)[0],num_components))
    Loadings = np.zeros((np.shape(X)[1],num_components))
    t = X[:,1]
    Eigenvals = np.zeros(num_components)
    for i in range(num_components):
        mu = 0
        V[:,i]=R[:,i]
        for j in range(num_iters):
            U[:,i]=np.dot(R.T,V[:,i])
            if i>0:
                A=np.dot(U[i,:].T,U[:,i])
                U[:,i] -= np.dot(U[i,:],A)

            U[:,i]=U[:,i]/np.linalg.norm(U[:,i])
            V[:,i] = np.dot(R,U[:,i])
            if i>=0:
                B=np.dot(V[i,:].T,V[:,i])
                V[:,i] -= np.dot(V[i,:],B)
            eigenval = np.linalg.norm(V[:,i])
            V[:,i] /= eigenval

            if abs(eigenval-mu)<threshold:
                break
            mu=eigenval

        Vtemp = V[:,i]
        Utemp = U[:,i]
        R -=eigenval*np.dot(Vtemp[:,np.newaxis],Utemp[np.newaxis,:])
        Eigenvals[i]=eigenval

    Scores = V[:,0:num_components]*Eigenvals
    Loadings = U[:,0:num_components]

    return Scores, Loadings, Eigenvals




