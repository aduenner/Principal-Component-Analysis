import numpy as np
import torch

def NIPALS(X, num_components, threshold=1e-6, max_iter=50):

    X = torch.from_numpy(X).cuda()

    Scores = torch.zeros([np.shape(X)[0], num_components]).cuda()

    Loadings = torch.zeros([np.shape(X)[1],num_components]).cuda()

    Eigenvals = torch.zeros(num_components).cuda()

    for i in range(num_components):

        old_eigen = 0
        t = X[:, i]
     
        for j in range(max_iter):

            p = torch.mv(X.t(), t)      
            p.div_(t.dot(t))
  
            Loadings[:, i] = p          

            p.div_(p.norm())
          
            t = torch.mv(X,p)
            t.div_(p.dot(p))
            Scores[:, i] = t

            new_eigen = t.dot(t)

            Eigenvals[i] = new_eigen             
            diff = np.abs(old_eigen - new_eigen)

            if diff < threshold:                
                break
            old_eigen = new_eigen

        X.sub_(torch.ger(t,p))

    return Scores.cpu().numpy(), Loadings.cpu().numpy(), Eigenvals.cpu().numpy()






