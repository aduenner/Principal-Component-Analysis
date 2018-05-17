import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

N_CLASSES = 10
CLASS_NAMES = list(str(i) for i in range(10))

SHAPE = (28, 28)
N_DIMS = 1

NOISE_LEVELS = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]
DENOISE_TYPES = ["None", "Full_SVD", "Simultaneous_Iteration", "Incremental_PCA", "NIPALS_GPU"]
COMPONENT_NUMBERS = [10, 20, 30, 40, 60, 80, 100, 120, 150, 180, 210, 240, 300, 360, 480, 600]
DATASETS = ["set1", "set2", "set3"]

N_NOISE_LEVELS = len(NOISE_LEVELS)
N_DENOISE_TYPES = len(DENOISE_TYPES)
N_COMPONENT_NUMBERS = len(COMPONENT_NUMBERS)
N_DATASETS = len(DATASETS)

RESULTS_DIR = "results"
SAVE_DIR = "graphs"

def graph_accuracies_test():

    sns.set(context='notebook', style='whitegrid', font="serif")
    # rc={"lines.linewidth": 5.5}

    plt.figure(figsize=(6, 4), dpi=240)

    raw_data = np.loadtxt(os.path.join(RESULTS_DIR, "accuracies.csv"), dtype="float64", delimiter=",")

    noise_densities = raw_data[:, 0]
    n_datasets = raw_data.shape[1] - 1

    labels = list("Classifier " + str(i + 1) for i in range(n_datasets))

    colours = sns.color_palette("GnBu_d", n_colors=n_datasets)

    for i in range(n_datasets):

        accuracies = raw_data[:, i + 1]
        accuracies = 1.0 - accuracies

        ax = sns.pointplot(x=noise_densities, 
                           y=accuracies, 
                           linestyles='-', 
                           color=colours[i], 
                           ci=None, 
                           scale=0.5)

    ax.set_ylabel("Error Rate", fontsize=14)
    ax.set_xlabel(r"Mean Magnitude of Noise Mask", fontsize=14)
    ax.set_yscale("log")
    ax.set_yticks(np.logspace(-2.5, -0.5, 5))

    ax.set_title("Classifier Error Rate on Noised Images", fontsize=16)

    plt.legend(labels)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, 'accuracies.png'))
    plt.show()    

def graph_losses_test():

    sns.set(context='notebook', style='whitegrid', font="serif")
    # rc={"lines.linewidth": 5.5}

    plt.figure(figsize=(6, 4), dpi=240)

    raw_data = np.loadtxt(os.path.join(RESULTS_DIR, "losses.csv"), dtype="float64", delimiter=",")

    print(raw_data)

    noise_densities = raw_data[:, 0]
    n_datasets = raw_data.shape[1] - 1

    labels = list("Classifier " + str(i + 1) for i in range(n_datasets))

    colours = sns.color_palette("GnBu_d", n_colors=n_datasets)

    for i in range(n_datasets):

        losses = raw_data[:, i + 1]

        ax = sns.pointplot(x=noise_densities, 
                           y=losses, 
                           linestyles='-', 
                           color=colours[i], 
                           ci=None, 
                           scale=0.5)

    ax.set_ylabel("Mean loss", fontsize=14)
    ax.set_xlabel(r"Mean Magnitude of Noise Mask", fontsize=14)
    ax.set_yscale("log")
    ax.set_yticks(np.logspace(-3.5, -1, 6))

    ax.set_title("Classifier Loss on Noised Images", fontsize=16)

    plt.legend(labels)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, 'losses.png'))
    plt.show()   


def graph_principal_components():

    ret = np.load(os.path.join(RESULTS_DIR, "denoised_accuracies.npy"))
    times = np.load(os.path.join(RESULTS_DIR, "runtimes.npy"))
    losses = np.load(os.path.join(RESULTS_DIR, "denoised_losses.npy"))

    print(times[1, 5, 12, :])
    print(losses[1, 7, 5:8, :])










if __name__ == "__main__": 

    graph_accuracies_test()
    graph_losses_test()
    # graph_principal_components()