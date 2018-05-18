import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import matplotlib.ticker as ticker

N_CLASSES = 10
CLASS_NAMES = list(str(i) for i in range(10))

SHAPE = (28, 28)
N_DIMS = 1

NOISE_LEVELS = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]
DENOISE_TYPES = ["None", "Full_SVD", "Simultaneous_Iteration", "NIPALS_GPU"]
COMPONENT_NUMBERS = [10, 20, 30, 40, 50, 60, 80, 100, 120, 140, 160, 200, 240, 280, 320, 360, 440, 520, 600, 680]
DATASETS = ["set1"]
CLASSIFIERS = ["cs1", "cs2", "cs3", "cs4", "cs5"]

N_NOISE_LEVELS = len(NOISE_LEVELS)
N_DENOISE_TYPES = len(DENOISE_TYPES)
N_COMPONENT_NUMBERS = len(COMPONENT_NUMBERS)
N_DATASETS = len(DATASETS)
N_CLASSIFIERS = len(CLASSIFIERS)

RESULTS_DIR = "Results"
SAVE_DIR = "Graphs"

def graph_error_rate():

    sns.set(context='notebook', style='darkgrid', font="serif")

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
    ax.set_xlabel("Magnitude of Noise Mask", fontsize=14)

    ax.set_title("Classifier Error Rate on Noised Images", fontsize=16)

    plt.legend(labels, title="Classifier")
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, 'Error Rate.png'))
    # plt.show()    


def graph_losses():

    sns.set(context='notebook', style='darkgrid', font="serif")

    plt.figure(figsize=(6, 4), dpi=240)

    raw_data = np.loadtxt(os.path.join(RESULTS_DIR, "losses.csv"), dtype="float64", delimiter=",")
    noise_densities = raw_data[:, 0]
    n_datasets = raw_data.shape[1] - 1

    labels = list(str(i + 1) for i in range(n_datasets))

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
    ax.set_xlabel(r"Magnitude of Noise Mask", fontsize=14)

    ax.set_title("Classifier Loss on Noised Images", fontsize=16)

    plt.legend(labels, title="Classifier")
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, 'Loss.png'))
    # plt.show()   


def graph_principal_components(noise_levels, metric="Accuracy"):

    if metric == "Accuracy" or metric == "Error Rate":

        data = np.load(os.path.join(RESULTS_DIR, "denoised_accuracies.npy"))
        metric = "Error Rate"
    
    elif metric == "MSE":

        data = np.load(os.path.join(RESULTS_DIR, "denoised_mse_values.npy"))
    
    elif metric == "Loss":

        data = np.load(os.path.join(RESULTS_DIR, "denoised_losses.npy"))
    
    elif metric == "Runtime":

        data = np.load(os.path.join(RESULTS_DIR, "runtimes.npy"))
   
    else:
        raise

    n_datasets = data.shape[0] 
    
    for j in noise_levels:

        noise_level = NOISE_LEVELS[j]            

        if metric != "MSE":

            plot_data = data[:, j, :, :]
            plot_data = np.mean(plot_data, axis=0)

        else:

            plot_data = data[j, :, :, :]
            plot_data = np.mean(plot_data, axis=2)

        sns.set(context='notebook', style='darkgrid', font="serif")
    
        colours = sns.color_palette(n_colors=n_datasets)
        labels = DENOISE_TYPES[1:]
                   
        plt.figure(figsize=(6, 4), dpi=240)

        if metric == "Error Rate":

            plot_data = 1.0 - plot_data

        for l in range(N_DENOISE_TYPES):

            if l == 0:

                continue

            plt.plot(COMPONENT_NUMBERS[1:], 
                     plot_data[1:, l].flatten(), 
                     color=colours[l],
                     linestyle='-',
                     linewidth=1,
                     marker='o',
                     markersize=4)

        ax = plt.gca()

        if metric != "Error Rate":
            ax.set_ylabel(r"$\mu_{" + metric + r"}$" , fontsize=14)
        else:
            ax.set_ylabel(metric , fontsize=14)

        # ax.set_xlabel("Number of Principal Components", fontsize=14)
        ax.set_xlabel(r"$n_{components}$", fontsize=14)

        max_yticks = 5
        yloc = plt.MaxNLocator(max_yticks, prune="both")
        ax.yaxis.set_major_locator(yloc)

        # ax.set_title(metric + " vs. Principal Component Count", fontsize=16)
        ax.set_title(metric + r" vs. $n_{components}$ || $\mu_{loss}=$" + str(noise_level), fontsize=16)

        plt.legend(labels, title="PCA Type")
        plt.tight_layout()
        plt.savefig(os.path.join(SAVE_DIR, "Noise" + str(noise_level) + "_" + metric + ".png"))
        # plt.savefig(os.path.join(SAVE_DIR, "Noise_" + str(noise_level) + "_" + metric + " vs. Principal Component Count"+ ".png"))

        plt.close('all')

    
def graph_principal_components_noises(noise_levels, metric):

    if metric == "Accuracy" or metric == "Error Rate":
        data = np.load(os.path.join(RESULTS_DIR, "denoised_accuracies.npy"))
        metric = "Error Rate"

    elif metric == "MSE":
        data = np.load(os.path.join(RESULTS_DIR, "denoised_mse_values.npy"))
    
    elif metric == "Loss":
        data = np.load(os.path.join(RESULTS_DIR, "denoised_losses.npy"))
    
    elif metric == "Runtime":
        data = np.load(os.path.join(RESULTS_DIR, "runtimes.npy"))
  
    n_datasets = len(noise_levels)

    sns.set(context='notebook', style='darkgrid', font="serif")
    
    colours = sns.light_palette("navy", reverse=False, n_colors=n_datasets)
    labels = list(NOISE_LEVELS[i] for i in noise_levels)
                
    plt.figure(figsize=(7, 4), dpi=240)

    for i in range(len(noise_levels)):

        j = noise_levels[i]

        if metric != "MSE":

            plot_data = data[:, j, :, :]
            plot_data = np.mean(plot_data, axis=0)
            plot_data = np.mean(plot_data, axis=1)

        else:

            plot_data = data[j, :, :, :]
            plot_data = np.mean(plot_data, axis=1)
            plot_data = np.mean(plot_data, axis=1)

        if metric == "Error Rate":

            plot_data = 1.0 - plot_data

        plt.plot(COMPONENT_NUMBERS[1:], 
                 plot_data[1:], 
                 color=colours[i],
                 linestyle='-',
                 linewidth=1,
                 marker='o',
                 markersize=4)

    ax = plt.gca()

    if metric != "Error Rate":
        ax.set_ylabel(r"$\mu_{" + metric + r"}$" , fontsize=14)
    
    else:
        ax.set_ylabel(metric , fontsize=14)

    # ax.set_xlabel("Number of Principal Components", fontsize=14)
    ax.set_xlabel(r"$n_{components}$", fontsize=14)

    max_yticks = 5
    yloc = plt.MaxNLocator(max_yticks, prune="both")
    ax.yaxis.set_major_locator(yloc)
    
    ax.set_title(metric + r" vs. $n_{components}$", fontsize=16)

    plt.legend(labels, title=r'$\mu{noise}$', loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, metric + " vs. Principal Component Count"+ ".png"))
    # plt.show()   
    plt.close('all')

def graph_NIPALS_error():

    old_data = np.loadtxt(os.path.join(RESULTS_DIR, "accuracies.csv"), dtype="float64", delimiter=",")

    noise_densities = old_data[1:, 0]
    n_datasets = old_data.shape[1] - 1

    labels = DENOISE_TYPES


    old_data = np.mean(old_data[1:, 1:], axis=1)

    error_rates = 1.0 - old_data

    sns.set(context='notebook', style='darkgrid', font="serif")
    colours = sns.color_palette(n_colors=6)          

    plt.figure(figsize=(7, 4), dpi=240)

    ax = sns.pointplot(x=noise_densities, 
                       y=error_rates, 
                       linestyles='-', 
                       color=colours[0], 
                       ci=None, 
                       scale=0.5)

    data = np.load(os.path.join(RESULTS_DIR, "denoised_accuracies.npy"))

    for j in range(4):

        if j == 0:

            continue

        new_accuracies = list()

        for i in range(8):

            new_accuracies.append(np.max(data[:, i, :, j], axis=0))

        new_accuracies = 1 - np.array(new_accuracies, dtype="float64")
        new_accuracies = np.mean(new_accuracies, axis=1)

        ax = sns.pointplot(x=noise_densities, 
                       y=new_accuracies, 
                       linestyles='-', 
                       color=colours[j], 
                       ci=None, 
                       scale=0.5)
    ax.set_ylabel("Error Rate", fontsize=14)
    ax.set_xlabel(r"Magnitude of Noise Mask", fontsize=14)

    ax.set_title("Effect of PCA on Error Rate", fontsize=16)

    plt.legend(labels, title="PCA Type")
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, 'NIPALS Error Rate.png'))
    # plt.show()    
    plt.close()

def graph_NIPALS_loss():

    old_data = np.loadtxt(os.path.join(RESULTS_DIR, "losses.csv"), dtype="float64", delimiter=",")

    noise_densities = old_data[1:, 0]
    n_datasets = old_data.shape[1] - 1

    labels = DENOISE_TYPES

    old_data = np.mean(old_data[1:, 1:], axis=1)

    accuracies = old_data

    sns.set(context='notebook', style='darkgrid', font="serif")
    colours = sns.color_palette(n_colors=6)         
    plt.figure(figsize=(7, 4), dpi=240)

    ax = sns.pointplot(x=noise_densities, 
                       y=accuracies, 
                       linestyles='-', 
                       color=colours[0], 
                       ci=None, 
                       scale=0.5)

    data = np.load(os.path.join(RESULTS_DIR, "denoised_losses.npy"))

    for j in range(4):

        if j == 0:

            continue

        new_accuracies = list()

        for i in range(8):

            new_accuracies.append(np.min(data[:, i, :, j], axis=0))

        new_accuracies = np.array(new_accuracies, dtype="float64")
        new_accuracies = np.mean(new_accuracies, axis=1)

        ax = sns.pointplot(x=noise_densities, 
                       y=new_accuracies, 
                       linestyles='-', 
                       color=colours[j], 
                       ci=None, 
                       scale=0.5)

    ax.set_ylabel("Loss", fontsize=14)
    ax.set_xlabel(r"Magnitude of Noise Mask", fontsize=14)

    ax.set_title("Selected PCA vs. Loss", fontsize=16)

    plt.legend(labels, title="PCA Type")
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, 'NIPALS Loss.png'))
   
def graph_runtimes_components():

    data = np.load(os.path.join(RESULTS_DIR, "time_results.npy"))

    sns.set(context='notebook', style='darkgrid', font="serif")
    colours = sns.color_palette(n_colors=6)
    labels = ["Full_SVD", "Simultaneous_Iteration", "NIPALS"]
    colour_indices = [1, 2, 4]
    plt.figure(figsize=(7, 4), dpi=240)

    for l in range(N_DENOISE_TYPES):

        if l == 0:

            continue      

        plt.plot(COMPONENT_NUMBERS[:7], 
                 data[:7, l - 1] / 5, 
                 color=colours[colour_indices[l - 1]],
                 linestyle='-',
                 linewidth=1,
                 marker='o',
                 markersize=4)
    ax = plt.gca()

    ax.set_ylabel("Runtime (seconds)" , fontsize=14)

    ax.set_xlabel("Number of Principal Components", fontsize=14)
  
    max_yticks = 5
    yloc = plt.MaxNLocator(max_yticks, prune="both")
    ax.yaxis.set_major_locator(yloc)
    
    ax.set_title("Runtime vs. Principal Component Count", fontsize=16)
    plt.legend(labels, title="PCA Type", loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "Runtime vs. Principal Component Count"+ ".png"))
    # plt.show()   
    plt.close('all')

def graph_runtimes_images():

    data = np.load(os.path.join(RESULTS_DIR, "new_results.npy"))

    vals = [10000, 20000, 30000, 40000, 50000, 75000, 100000, 150000, 200000]
    sns.set(context='notebook', style='darkgrid', font="serif")
 
    colours = sns.color_palette(n_colors=6)
    labels = ["Full_SVD", "Simultaneous_Iteration", "NIPALS"]
    colour_indices = [1, 2, 4]

    plt.figure(figsize=(7, 4), dpi=240)

    for l in range(N_DENOISE_TYPES):

        if l == 0:

            continue

        plt.plot(vals[:7], 
                 data[:7, l - 1] / 5, 
                 color=colours[colour_indices[l - 1]],
                 linestyle='-',
                 linewidth=1,
                 marker='o',
                 markersize=4)
    ax = plt.gca()

    ax.set_ylabel("Runtime (seconds)" , fontsize=14)

    ax.set_xlabel("Number of Images", fontsize=14)

    max_yticks = 5
    yloc = plt.MaxNLocator(max_yticks, prune="both")
    ax.yaxis.set_major_locator(yloc)
 
    ax.set_title("Runtime vs. Image Count", fontsize=16)
    plt.legend(labels, title="PCA Type", loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "Runtime vs. Image Count"+ ".png"))
    # plt.show()   
    plt.close('all')

def graph_GPU_speedup():

    data = np.load(os.path.join(RESULTS_DIR, "gpu_results.npy"))
    sns.set(context='notebook', style='darkgrid', font="serif")
 
    colours = sns.color_palette(n_colors=6)
    labels = ["NIPALS", "NIPALS_GPU"]
    color_indices = [4, 3]

                     
    plt.figure(figsize=(7, 4), dpi=240)

    for l in range(2):

        plt.plot(COMPONENT_NUMBERS[:5], 
                 data[:5, l] / 5, 
                 color=colours[color_indices[l]],
                 linestyle='-',
                 linewidth=1,
                 marker='o',
                 markersize=4)
    ax = plt.gca()

    ax.set_ylabel("Runtime (seconds)" , fontsize=14)

    ax.set_xlabel("Number of Principal Components", fontsize=14)
 
    max_yticks = 5
    yloc = plt.MaxNLocator(max_yticks, prune="both")
    ax.yaxis.set_major_locator(yloc)
    ax.set_xlim(0, 60)

    ax.set_title("Runtime vs. Principal Component Count", fontsize=16)
    plt.legend(labels, title="PCA Type", loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "GPU Runtime vs. Principal Component Count"+ ".png"))
    # plt.show()   
    plt.close('all')



if __name__ == "__main__": 

    noise_levels = range(8)

    # graph_error_rate()
    # graph_losses()

    # graph_principal_components(noise_levels, metric="Accuracy")
    # graph_principal_components(noise_levels, metric="MSE")
    # graph_principal_components(noise_levels, metric="Loss")

    # graph_principal_components_noises(noise_levels, metric="Accuracy")
    # graph_principal_components_noises(noise_levels, metric="MSE")
    # graph_principal_components_noises(noise_levels, metric="Loss")
  
   
    graph_NIPALS_error()
    graph_NIPALS_loss()

    # graph_runtimes_components()
    # graph_runtimes_images()

    # graph_GPU_speedup()


