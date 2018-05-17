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

import skimage.measure as measure

import bm3d

import mnist_model as mm

N_CLASSES = 10
CLASS_NAMES = list(str(i) for i in range(10))

SHAPE = (28, 28)
N_DIMS = 1

# NOISE_LEVELS = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]
NOISE_LEVELS = [50.0, 50.0, 50.0]
# DENOISE_TYPES = ["None", "Full_SVD", "Simultaneous_Iteration", "NIPALS_GPU"]
DENOISE_TYPES = ["SVD", "Simultaneous_Iteration", "NIPALS", "NIPALS_GPU"]
# COMPONENT_NUMBERS = [10, 20, 30, 40, 50, 60, 80, 100, 120, 140, 160, 200, 240, 280, 320, 360, 440, 520, 600, 680]
COMPONENT_NUMBERS = [10, 20, 30, 40, 50, 60, 80, 100, 120, 140, 160, 200, 240, 280, 320, 360, 400]
DATASETS = ["set1"]
# CLASSIFIERS = ["cs1", "cs2", "cs3", "cs4", "cs5"]
CLASSIFIERS = ["cs1"]

N_NOISE_LEVELS = len(NOISE_LEVELS)
N_DENOISE_TYPES = len(DENOISE_TYPES)
N_COMPONENT_NUMBERS = len(COMPONENT_NUMBERS)
N_DATASETS = len(DATASETS)
N_CLASSIFIERS = len(CLASSIFIERS)

RESULTS_DIR = "New Results"


def test_accuracies():

    with torch.cuda.device(0):

        accuracies = np.zeros((N_NOISE_LEVELS + 1, N_CLASSIFIERS + 1), dtype="float64")
        accuracies[1:, 0] = NOISE_LEVELS[:]

        data_dir = DATASETS[0]

        _, test_data, _, test_labels = gen_dataset.load_dataset(data_dir=data_dir)

        test_data = np.array(test_data, dtype="float32")

        accuracies[0, 0] = 0.0

        losses = np.array(accuracies, dtype="float64")

        for i in range(N_CLASSIFIERS):

            classifier = CLASSIFIERS[i]

            o_acc, _, o_loss = mm.eval_(test_data, test_labels, N_CLASSES, N_DIMS, CLASS_NAMES, "model_params_" + classifier + ".pt")

            accuracies[0, i + 1] = o_acc
            losses[0, i + 1] = o_loss

            for j in range(N_NOISE_LEVELS):

                val = NOISE_LEVELS[j]

                noise_images_test = np.load(os.path.join(data_dir, "noised_data_test_" + str(int(val)) + ".npy")).reshape(-1, 1, 28, 28)
                noise_images_test = np.array(noise_images_test, dtype="float32")           

                acc, _, loss = mm.eval_(noise_images_test, test_labels, N_CLASSES, N_DIMS, CLASS_NAMES, "model_params_" + classifier + ".pt")

                accuracies[j + 1, i + 1] = acc
                losses[j + 1, i + 1] = loss

    np.savetxt(os.path.join(RESULTS_DIR, "accuracies.csv"), accuracies, delimiter=",")
    np.savetxt(os.path.join(RESULTS_DIR, "losses.csv"), losses, delimiter=",")


def train_models():  

    with torch.cuda.device(1):
    
        data_dir = DATASETS[0]

        training_data, test_data, training_labels, test_labels = gen_dataset.load_dataset(data_dir=data_dir)

        training_data = np.array(training_data, dtype="float32")
        test_data = np.array(test_data, dtype="float32")
        
        for i in range(N_CLASSIFIERS):

            classifier = CLASSIFIERS[i]    

            mm.train(training_data, test_data, training_labels, test_labels, N_CLASSES, N_DIMS, CLASS_NAMES, save_params="model_params_" + classifier + ".pt")


def run_full_set():
   
    data_dir = DATASETS[0]

    ret = np.zeros((N_CLASSIFIERS, N_NOISE_LEVELS, N_COMPONENT_NUMBERS, N_DENOISE_TYPES), dtype="float64")
    times = np.zeros(ret.shape, dtype="float64")
    losses = np.zeros(ret.shape, dtype="float64")

    n_to_complete = ret.size
    n_completed = 1    

    print("\n===========================================================\n")
    print("Evaluating model on dataset " + data_dir + "\n")

    training_data, test_data, training_labels, test_labels = gen_dataset.load_dataset(data_dir=data_dir)

    training_data = np.array(training_data, dtype="float32")
    test_data = np.array(test_data, dtype="float32")

    n_test = len(test_data)

    mse_values = np.zeros((N_NOISE_LEVELS, N_COMPONENT_NUMBERS, N_DENOISE_TYPES, n_test), dtype="float64")
    noised_mse = np.zeros((N_NOISE_LEVELS, n_test), dtype="float64")

    for j in range(N_NOISE_LEVELS):

        val = NOISE_LEVELS[j]

        print("\n++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        print("Noise Level: " + str(float(val)))

        noise_images_test = np.load(os.path.join(data_dir, "noised_data_test_" + str(int(val)) + ".npy")).reshape(-1, 1, 28, 28)
        noise_images_test = np.array(noise_images_test, dtype="float32")
                            
        for k in range(N_COMPONENT_NUMBERS):

            print("\n------------------------------------------------------")
            print("Number of Components: " + str(float(COMPONENT_NUMBERS[k])))

            for l in range(N_DENOISE_TYPES):

                print("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
                print("Denoising Method: " + str(DENOISE_TYPES[l]))

                with torch.cuda.device(1):
                    denoised_images_test, _, time_elapsed = PCA.pca_transform(noise_images_test, 
                                                               COMPONENT_NUMBERS[k], 
                                                               DENOISE_TYPES[l])

                denoised_images_test = np.array(denoised_images_test, dtype="float32")

                print("Method Time: " + str(float(time_elapsed)))

                new_times[j, k, l] = time_elapsed

                for i in range(N_CLASSIFIERS):

                    classifier = CLASSIFIERS[i]

                    print("\n===========================================================\n")
                    print("Evaluating Classifier " + classifier + "\n")

                    with torch.cuda.device(1):

                        overall_accuracy, _, overall_loss = mm.eval_(denoised_images_test, test_labels, N_CLASSES, N_DIMS, CLASS_NAMES, "model_params_" + classifier + ".pt")

                    ret[i, j, k, l] = overall_accuracy
                    times[i, j, k, l] = time_elapsed
                    losses[i, j, k, l] = overall_loss

                    print("Completed Trial: " + str(n_completed) + " of " + str(n_to_complete))

                    n_completed += 1

                mse_values[j, k, l, :] = get_mean_mse(test_data, denoised_images_test)
              
        noised_mse[j, :] = get_mean_mse(test_data, noise_images_test)

        print("\nSaving..................\n\n\n\n\n\n")

        np.save(os.path.join(RESULTS_DIR, "denoised_accuracies.npy"), ret)
        np.save(os.path.join(RESULTS_DIR, "runtimes.npy"), times)
        np.save(os.path.join(RESULTS_DIR, "denoised_losses.npy"), losses)

        np.save(os.path.join(RESULTS_DIR, "noised_mse_values.npy"), noised_mse)
        np.save(os.path.join(RESULTS_DIR, "denoised_mse_values.npy"), mse_values)


def get_mean_mse(truth, denoised, dtype="uint8"):

    n_images = len(truth)
    assert(len(truth) == len(denoised))

    dmin = np.iinfo(dtype).min
    dmax = np.iinfo(dtype).max

    u_denoised = np.clip(denoised, dmin, dmax)
    u_denoised = np.array(u_denoised, dtype=dtype)

    u_truth = np.clip(truth, dmin, dmax)
    u_truth = np.array(u_truth, dtype=dtype)

    mse_values = np.zeros(n_images, dtype="float64")

    for i in range(n_images):

        mse_values[i] = measure.compare_mse(u_truth[i], u_denoised[i])

    return mse_values


def run_bm3d():

    with torch.cuda.device(1):
    
        data_dir = DATASETS[0]

        training_data, test_data, training_labels, test_labels = gen_dataset.load_dataset(data_dir=data_dir)

        training_data = np.array(training_data, dtype="float32")
        test_data = np.array(test_data, dtype="float32")

        for val in NOISE_LEVELS:

            print("\n===========================================================\n")
            print("Evaluating model on noised test images...\n")
            print("Noise Level: " + str(float(val)))

            noise_images_test = np.load(os.path.join(data_dir, "noised_data_test_" + str(int(val)) + ".npy")).reshape(-1, 1, 28, 28)
            noise_images_test = np.array(noise_images_test, dtype="float32")

            acc, _ = mm.eval_(noise_images_test, test_labels, N_CLASSES, N_DIMS, CLASS_NAMES, "model_params_" + data_dir + ".pt")

            output_images = bm3d.bm3d(noise_images_test, val)
            
            bm3d, _ = mm.eval_(output_images, test_labels, N_CLASSES, N_DIMS, CLASS_NAMES, "model_params_" + data_dir + ".pt")

            np.save(os.path.join(data_dir, "bm3d_" + str(int(val)) + ".npy"), output_images)
        

if __name__ == "__main__":   

   
    # train_models()
    run_full_set()
    # test_accuracies()
