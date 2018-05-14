import os

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import matplotlib.pyplot as plt

import time
import csv

import gen_dataset
import gen_images
import PCA

import mnist_model as mm

if __name__ == "__main__":

    n_classes = 10
    class_names = list(str(i) for i in range(10))

    shape = (28, 28)
    n_dims = 1

    data_dir = "set1"

    training_data, test_data, training_labels, test_labels = gen_dataset.load_dataset(data_dir=data_dir)

    training_data = np.array(training_data, dtype="float32")
    test_data = np.array(test_data, dtype="float32")
    
    denoise_types = ["NIPALS", "NIPALS_GPU"]
    component_values = [25]


    with torch.cuda.device(1):

        # train(training_data, test_data, training_labels, test_labels, n_classes, n_dims, class_names, save_params="model_params_" + data_dir + ".pt")

        for val in [50.0]:

            print("\n===========================================================\n")
            print("Evaluating model on noised test images...\n")
            print("Noise Level: " + str(float(val)))

            # noise_images_training = np.load(os.path.join(data_dir, "noised_data_training_" + str(int(val)) + ".npy")).reshape(-1, 1, 28, 28)
            noise_images_test = np.load(os.path.join(data_dir, "noised_data_test_" + str(int(val)) + ".npy")).reshape(-1, 1, 28, 28)

            # noise_images_training = np.array(noise_images_training, dtype="float32")
            noise_images_test = np.array(noise_images_test, dtype="float32")

            # acc, _ = mm.eval_(noise_images_test, test_labels, n_classes, n_dims, class_names, "model_params_" + data_dir + ".pt")

            print("\n-----------------------------------------------\n")
            print("Evaluating model on denoised test images...\n")

            n_methods = len(denoise_types)
            n_tests = len(component_values)

            ret = np.zeros((n_methods, n_tests), dtype="float64")

            val = []

            for j in range(n_methods):

                for k in range(n_tests):

                    denoised_images_test, _ = PCA.pca_transform(noise_images_test, component_values[k], denoise_types[j])

                    val.append(denoised_images_test)


                    
                    denoised_images_test = np.array(denoised_images_test, dtype="float32")

                    overall_accuracy, _ = mm.eval_(denoised_images_test, test_labels, n_classes, n_dims, class_names, "model_params_" + data_dir + ".pt")

                    ret[j, k] = overall_accuracy

            np.savetxt(str(float(val)) + ".csv", ret, delimiter=",")