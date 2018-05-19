import torch

def _NIPALS_GPU(X, scores, loadings, lambdas, n_components, threshold=1e-6, max_iter=200):

    for i in range(n_components):

        t = X[:, i]
        _lambda = t.dot(t) 

        for j in range(max_iter):

            # Compute loadings
            p = torch.mv(X.t(), t)   
            p.div_(_lambda)
            loadings[:, i] = p

            p.div_(p.norm())
    
            # Project X onto p to find score vector t
            t = torch.mv(X,p)
            t.div_(p.dot(p))
            scores[:, i] = t
      
            _lambda_n = t.dot(t)

            # Add score vector to matrix of score vectors
            lambdas[i] = _lambda_n

            # Check for convergence
            diff = torch.abs(_lambda - _lambda_n)

            if diff <= threshold:
                break

            _lambda = _lambda_n
           
        # Update X
        X.sub_(torch.ger(t,p))