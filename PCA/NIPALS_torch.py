import numpy as np
import torch

def NIPALS(X, num_components, threshold=1e-6):
    X=torch.from_numpy(X).cuda()
    #Scores = np.zeros((np.shape(X)[0], num_components))
    Scores = torch.zeros([np.shape(X)[0], num_components]).cuda()
    #Loadings = np.zeros([np.shape(X)[1], num_components])
    Loadings = torch.zeros([np.shape(X)[1],num_components]).cuda()
    #Eigenvals = np.zeros(num_components)
    Eigenvals = torch.zeros(num_components).cuda()
    num_iters = 100
    for i in range(num_components):
        old_eigen = 0
        t = X[:, i]
        for j in range(num_iters):
            # compute loadings
            #p = np.dot(X.T, t) / np.dot(t.T, t)
            p = torch.mv(torch.t(X),t)
            p = torch.div(p,torch.norm(p))
            Loadings[:, i] = p
            # p = np.dot(p,np.sqrt(np.dot(p.T,p)))

            # project X onto p to find score vector t
            #t = np.dot(X, p) / np.dot(p.T, p)
            t = torch.mv(X,p)
            Scores[:, i] = t
            # Compute eigenvalue from t
            new_eigen = torch.norm(t)
            Eigenvals[i] = new_eigen
            # add score vector to matrix of score vectors

            # check for convergence
            diff = np.abs(old_eigen - new_eigen)
            if diff < threshold:
                print(new_eigen)
                break
            old_eigen = new_eigen

        # Update Xh
        r = torch.ger(t,p)
        X -= r

    return Scores.cpu().numpy(), Loadings.cpu().numpy(), Eigenvals.cpu().numpy()




