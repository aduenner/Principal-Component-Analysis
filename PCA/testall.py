import numpy as np
import PCA_test as pcatest
num_components = 20 # Number of principal components
num_images = 10000  # number of images from set to test with
image_set_full = np.load('NNHelper\original_images.npy')
image_set_int = image_set_full[0:num_images, :]
image_set = np.asarray(image_set_int)/255.0

t_data_img = pcatest.datatest(image_set)
t_mean_img = pcatest.meantest(image_set)
t_simuliter_img, t_simuliter_pc = pcatest.simuliter(image_set, num_components)
t_fullsvd_img, t_fullsvd_pc = pcatest.fullsvd(image_set, num_components)
t_incremental_pca_img, t_incremental_pca_pc = pcatest.incremental_pca(image_set, num_components)
t_nipals_img, t_nipals_pc = pcatest.nipals(image_set, num_components)
t_nipalsgs_img, t_nipalsgs_pc = pcatest.nipalsgs(image_set, num_components)