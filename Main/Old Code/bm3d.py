import pybm3d

import numpy as np

import skimage.data
from skimage.measure import compare_psnr

def bm3d(data, std):

	ret = np.zeros(data.shape, dtype="float32")

	for i in range(len(data)):

		if i % 100 == 99:

			print(i + 1)

		img = data[i].swapaxes(0, 2)
		ret[i] = pybm3d.bm3d.bm3d(img, std).swapaxes(2, 0)

	return ret


