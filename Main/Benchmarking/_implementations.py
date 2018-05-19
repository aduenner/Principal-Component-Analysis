import numpy as np
import time

import i_numpy as inp
import i_numba as inb
import i_torch as itc

from sklearn.decomposition import IncrementalPCA as _bI
from sklearn.decomposition import PCA as _bSVD

import torch

def benchmark_GPU(data, n_components, threshold=1e-6, max_iter=200):

    with torch.cuda.device(0):

        dtype = data.dtype

        scores = torch.zeros((np.shape(data)[0], n_components), dtype=torch.float).cuda()
        loadings = torch.zeros((np.shape(data)[1], n_components), dtype=torch.float).cuda()
        lambdas = torch.zeros(n_components, dtype=torch.float)

        data_means = np.mean(data, axis=0, keepdims=True)
        data -= data_means

        data = torch.from_numpy(data).cuda()

        start_time = time.time()

        itc._NIPALS_GPU(data, scores, loadings, lambdas, n_components, threshold, max_iter)

        _elapsed = time.time() - start_time

        scores = scores.cpu().numpy()
        loadings = loadings.cpu().numpy()

        pc = loadings.T

        ret = np.dot(scores, pc)    
        ret += data_means

        return ret, pc, _elapsed

def benchmark_NIPALS(data, n_components, use_np=True, threshold=1e-6, max_iter=200):

    dtype = data.dtype
    eps = np.finfo(dtype).eps

    scores = np.zeros((np.shape(data)[0], n_components), dtype=dtype)
    loadings = np.zeros((np.shape(data)[1], n_components), dtype=dtype)
    lambdas = np.zeros(n_components, dtype=dtype)

    data_means = np.mean(data, axis=0, keepdims=True)
    data -= data_means

    start_time = time.time()

    if use_np:

        inp._NIPALS(data, scores, loadings, lambdas, n_components, threshold, max_iter)

    else: 

        inb._NIPALS(data, scores, loadings, lambdas, n_components, threshold, max_iter)

    _elapsed = time.time() - start_time

    pc = loadings.T

    ret = np.dot(scores, pc)    
    ret += data_means

    return ret, pc, _elapsed


def benchmark_SVD(data, n_components, use_np=True):

    dtype = data.dtype
    eps = np.finfo(dtype).eps

    scores = np.zeros((np.shape(data)[0], n_components), dtype=dtype)
    loadings = np.zeros((np.shape(data)[1], n_components), dtype=dtype)
    lambdas = np.zeros(n_components, dtype=dtype)

    data_means = np.mean(data, axis=0, keepdims=True)
    data -= data_means

    start_time = time.time()

    if use_np:

        inp._SVD(data, scores, loadings, lambdas, n_components)

    else: 

        inb._SVD(data, scores, loadings, lambdas, n_components)

    _elapsed = time.time() - start_time

    pc = loadings.T

    ret = np.dot(scores, pc)    
    ret += data_means

    return ret, pc, _elapsed

def benchmark_SI(data, n_components, use_np=True, threshold=1e-4, max_iter=200):

    dtype = data.dtype

    data_means = np.mean(data, axis=0, keepdims=True)
    data -= data_means

    X = data.T.dot(data) / (data.shape[1] - 1)

    (m, n) = X.shape

    Q = np.random.randn(m, n_components)
    Q = np.array(Q, dtype=dtype)

    Q_o = Q.copy()

    R = np.zeros((n,n), dtype=dtype)
    I = np.eye(m, dtype=dtype)

    start_time = time.time()

    if use_np:

        inp._SI(X, Q, Q_o, R, I, threshold, max_iter)

    else: 

        inb._SI(X, Q, Q_o, R, I, threshold, max_iter)

    _elapsed = time.time() - start_time

    scores = np.dot(data, Q)
    pc = Q.T

    ret = np.dot(scores, pc)    
    ret += data_means

    return ret, pc, _elapsed


# # Benchmark the builtin Incremental PCA iteration
# def benchrmark_builtin_I(data, n_components, threshold=1e-4, max_iter=200):

#     dtype = data.dtype

#     data_means = np.mean(data, axis=0, keepdims=True)
#     data -= data_means

#     _alg = _bI(n_components=n_components, whiten=False)
#     _iter = 0


#     while _alg.var_ > threshold and max_iter > _iter:

#         _alg.partial_fit(dat)


#     (m, n) = X.shape

#     Q = np.random.randn(m, n_components)
#     Q = np.array(Q, dtype=dtype)

#     Q_o = Q.copy()

#     R = np.zeros((n,n), dtype=dtype)
#     I = np.eye(m, dtype=dtype)

#     start_time = time.time()

#     if use_np:

#         inp._SI(X, Q, Q_o, R, I, threshold, max_iter)

#     else: 

#         inb._SI(X, Q, Q_o, R, I, threshold, max_iter)

#     _elapsed = time.time() - start_time

#     scores = np.dot(data, Q)
#     pc = Q.T

#     ret = np.dot(scores, pc)    
#     ret += data_means

#     return ret, pc, _elapsed

if __name__ == "__main__":

    dtype = "float32"

    train_images = np.load('noised_data_training_30.npy')
    test_images = np.load('noised_data_test_30.npy')

    dataset = np.concatenate((train_images, test_images), axis=0)
    dataset = np.array(dataset, dtype=dtype)

    n_images = 70000
    n_pixels = 784
    n_components = 10

    data = dataset[:n_images, :n_pixels]
    minimal = dataset[:10, :10]

    # Quickly Compile Numba Functions
    benchmark_NIPALS(minimal.copy(), 1, False)
    benchmark_SVD(minimal.copy(), 1, False)
    benchmark_SI(minimal.copy(), 1, False)

    print("Testing NIPALS on Numba...")
    _, _, _elapsed = benchmark_NIPALS(data.copy(), n_components, False)
    print("Time Elapsed: ", _elapsed, " seconds")

    print("Testing NIPALS on GPU...")
    _, _, _elapsed = benchmark_GPU(data.copy(), n_components)
    print("Time Elapsed: ", _elapsed, " seconds")

    print("Testing NIPALS on Numpy...")
    ret_n, pcnp, _elapsed = benchmark_NIPALS(data.copy(), n_components, True)
    print("Time Elapsed: ", _elapsed, " seconds")

    print("Testing SVD on Numba...")
    _, _, _elapsed = benchmark_SVD(data.copy(), n_components, False)
    print("Time Elapsed: ", _elapsed, " seconds")

    print("Testing SVD on Numpy...")
    ret_svd, pcsvd, _elapsed = benchmark_SVD(data.copy(), n_components, True)
    print("Time Elapsed: ", _elapsed, " seconds")

    print("Testing SI on Numba...")
    _, _, _elapsed = benchmark_SI(data.copy(), n_components, False)
    print("Time Elapsed: ", _elapsed, " seconds")

    print("Testing SI on Numpy...")
    ret_si, pcsi, _elapsed = benchmark_SI(data.copy(), n_components, True)
    print("Time Elapsed: ", _elapsed, " seconds")
 



