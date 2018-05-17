import numpy as np
import PCA_test as pcatest
num_components = 100 # Number of principal components
num_images = 1000  # number of images from set to test with
image_set_full = np.load('NNHelper/test_data.npy')
image_set_int = np.load('NNHelper/noised_data_test_50_2.npy')

image_set = np.asarray(image_set_int,dtype=np.float32)/1.0
image_set_2=np.asarray(image_set_full,dtype=np.float32)/1.0
t_data_img = pcatest.datatest(image_set_2)
t_data_img = pcatest.datatest(image_set)
#t_mean_img = pcatest.meantest(image_set)
#t_simuliter_img, t_simuliter_pc = pcatest.simuliter(image_set, num_components)
t_fullsvd_img, t_fullsvd_pc = pcatest.fullsvd(image_set, num_components)
#t_incremental_pca_img, t_incremental_pca_pc = pcatest.incremental_pca(image_set, num_components)
#t_nipals_img, t_nipals_pc = pcatest.nipals(image_set, num_components)
#t_nipalsgs_img, t_nipalsgs_pc = pcatest.nipalsgs(image_set, num_components)
#t_nipalsgpu_img, t_nipalsgpu_pc = pcatest.nipalsgpu(image_set,num_components)
#t_svdnumpy_img, t_svdnumpy_pc = pcatest.svdnumpy(image_set,num_components)

