import numpy as np
import time

import PCA

import i_numpy as inp
import i_numba as inb

from line_profiler import LineProfiler

def nipals_nb(data, n_components):

    dtype = data.dtype
    eps = np.finfo(dtype).eps

    scores = np.zeros((np.shape(data)[0], n_components), dtype)
    loadings = np.zeros((np.shape(data)[1], n_components), dtype)
    lambdas = np.zeros(n_components)

    data_means = np.mean(data, axis=0, keepdims=True)
    data -= data_means
    data += eps

    lp = LineProfiler()

    start_time = time.time()

    inb._NIPALS(data, scores, loadings, lambdas, n_components)

    _elapsed = time.time() - start_time

    pc = loadings.T

    ret = np.dot(scores, pc)    
    ret += data_means
    ret -= eps
    return ret, pc, _elapsed


if __name__ == "__main__":

	n_images = 500
	n_pixels = 784
	n_components = 20

	test_array = np.random.randint(0, 255, (n_images, n_pixels), dtype="uint8")
	test_array = np.array(test_array, dtype="float64") / 255.0

	print("Testing NIPALS on Optimized...")
	# _, _, _ = nipals_nb(test_array.copy(), 1)
	_, pcnb, _elapsed_nb = nipals_nb(test_array.copy(), n_components)
	print(_elapsed_nb)
	print("Testing NIPALS on Original...")
	_, pcnp, _elapsed_np = PCA.pca_transform(test_array.copy(), n_components, "NIPALS", stop_condition=1e-6)
	print(_elapsed_np)
	print("Testing NIPALS on SVD...")
	_, pcnx, _elapsed_nx = PCA.pca_transform(test_array.copy(), n_components, "Full_SVD", stop_condition=1e-6)
	print(_elapsed_nx)

	print(np.all(pcnb == pcnp))




