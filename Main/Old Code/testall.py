import numpy as np
import PCA_test as pcatest
num_components = 2 # Number of principal components
num_images = 1000  # number of images from set to test with
image_set_full = np.load('NNHelper/Set1/speckled_data_test_0.01.npy')
image_set_int = np.load('NNHelper/Set1/noised_data_test_5.npy')

image_set = np.asarray(image_set_int,dtype=np.float32)/1.0
image_set_2=np.asarray(image_set_full,dtype=np.float32)/1.0
t_data_img = pcatest.datatest(image_set_2)
t_data_img = pcatest.datatest(image_set)
#t_mean_img = pcatest.meantest(image_set)
#t_simuliter_img, t_simuliter_pc = pcatest.simuliter(image_set, num_components)
t_fullsvd_img, t_fullsvd_pc = pcatest.fullsvd(image_set_2, num_components)
image3 = pcatest.datatest(image_set_2-image_set)
image4 = pcatest.datatest(image_set_2 - image_set)
#t_incremental_pca_img, t_incremental_pca_pc = pcatest.incremental_pca(image_set, num_components)
#t_nipals_img, t_nipals_pc = pcatest.nipals(image_set, num_components)
#t_nipalsgs_img, t_nipalsgs_pc = pcatest.nipalsgs(image_set, num_components)
#t_nipalsgpu_img, t_nipalsgpu_pc = pcatest.nipalsgpu(image_set,num_components)
#t_svdnumpy_img, t_svdnumpy_pc = pcatest.svdnumpy(image_set,num_components)



