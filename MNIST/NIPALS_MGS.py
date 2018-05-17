import numpy as np
import torch


def NIPALS_GS(X, num_components, threshold=1e-6, max_iter=50):
    
    X = torch.from_numpy(X).cuda()

    Scores = torch.zeros([np.shape(X)[0], num_components]).cuda()

    Loadings = torch.zeros([np.shape(X)[1], num_components]).cuda()

    Eigenvals = torch.zeros(num_components).cuda()

    # Gram Schmit terms
    PPt = torch.zeros([np.shape(X)[1], np.shape(X)[1]]).cuda()
    TTt = torch.zeros([np.shape(X)[0], np.shape(X)[0]]).cuda()

    for i in range(num_components):

        old_eigen = 0
        t = X[:, i]

        for j in range(max_iter):

            p = torch.mv(X.t(), t)
            p.div_(t.dot(t))

            # GS for p
            if i > 0:
                p.sub_(torch.mv(PPt, p))

            Loadings[:, i] = p

            p.div_(p.norm())

            t = torch.mv(X, p)
            t.div_(p.dot(p))

            # GS for t
            if i > 0:
                t.sub_(torch.mv(TTt, t))

            Scores[:, i] = t

            new_eigen = t.dot(t)

            Eigenvals[i] = new_eigen
            diff = np.abs(old_eigen - new_eigen)

            if diff < threshold:
                break
            old_eigen = new_eigen

        X.sub_(torch.ger(t, p))
        # Add outer product of P*P' to PPp
        PPt = torch.ger(p, p)
        # Add outer product of T*T' to TTt
        TTt = torch.ger(t, t).div_(new_eigen)

    return Scores.cpu().numpy(), Loadings.cpu().numpy(), Eigenvals.cpu().numpy()
