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
# NOISE_LEVELS = [50.0]
DENOISE_TYPES = ["None", "Full_SVD", "Simultaneous_Iteration", "NIPALS_GPU"]
# DENOISE_TYPES = ["NIPALS_GPU"]
COMPONENT_NUMBERS = [10, 20, 30, 40, 50, 60, 80, 100, 120, 140, 160, 200, 240, 280, 320, 360, 440, 520, 600, 680]
# COMPONENT_NUMBERS = [10]
DATASETS = ["set1"]
CLASSIFIERS = ["cs1", "cs2", "cs3", "cs4", "cs5"]

N_NOISE_LEVELS = len(NOISE_LEVELS)
N_DENOISE_TYPES = len(DENOISE_TYPES)
N_COMPONENT_NUMBERS = len(COMPONENT_NUMBERS)
N_DATASETS = len(DATASETS)
N_CLASSIFIERS = len(CLASSIFIERS)

RESULTS_DIR = "results"
SAVE_DIR = "graphs"

def graph_accuracies_test():

    sns.set(context='notebook', style='darkgrid', font="serif")
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
    ax.set_xlabel(r"Magnitude of Noise Mask", fontsize=14)
    # ax.set_yscale("log")
    # ax.set_yticks(np.logspace(-2.5, -0.5, 5))

    ax.set_title("Classifier Error Rate on Noised Images", fontsize=16)

    plt.legend(labels, title="Classifier")
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, 'accuracies.png'))
    plt.show()    

def graph_losses_test():

    sns.set(context='notebook', style='darkgrid', font="serif")
    # rc={"lines.linewidth": 5.5}

    plt.figure(figsize=(6, 4), dpi=240)

    raw_data = np.loadtxt(os.path.join(RESULTS_DIR, "losses.csv"), dtype="float64", delimiter=",")

    print(raw_data)

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
    # ax.set_yscale("log")
    # ax.set_yticks(np.logspace(-3.5, -1, 6))

    ax.set_title("Classifier Loss on Noised Images", fontsize=16)

    plt.legend(labels, title="Classifier")
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, 'losses.png'))
    plt.show()   


def graph_principal_components(noise_levels, metric="Accuracy"):

    if metric == "Accuracy":
        data = np.load(os.path.join(RESULTS_DIR, "denoised_accuracies.npy"))
        metric = "Error Rate"
    elif metric == "MSE":
        data = np.load(os.path.join(RESULTS_DIR, "denoised_mse_values.npy"))
    elif metric == "Loss":
        data = np.load(os.path.join(RESULTS_DIR, "denoised_losses.npy"))
    elif metric == "Runtime":
        data = np.load(os.path.join(RESULTS_DIR, "runtimes.npy"))
    # losses = np.load(os.path.join(RESULTS_DIR, "denoised_losses.npy"))

    # noised_mse = np.load(os.path.join(RESULTS_DIR, "noised_mse_values.npy"))
    # mse_values = np.load(os.path.join(RESULTS_DIR, "denoised_mse_values.npy"), mse_values)

    n_datasets = data.shape[0] 
    
    for j in noise_levels:

        noise_level = NOISE_LEVELS[j]

        for l in range(N_DENOISE_TYPES):

            if l == 0:

                continue

            else:

                # colours = sns.color_palette("GnBu_d", n_colors=n_datasets)

                # labels = list("Classifier " + str(i + 1) for i in range(n_datasets))

                # denoise_type = DENOISE_TYPES[l]

                # plot_data = data[:, j, :, l]

                # if metric == "Error Rate":

                #     plot_data = 1.0 - plot_data

                # sns.set(context='notebook', style='whitegrid', font="serif")
                # # rc={"lines.linewidth": 5.5}

                # plt.figure(figsize=(12, 8), dpi=240)

                # for i in range(n_datasets):

                #     ax = sns.pointplot(x=COMPONENT_NUMBERS, 
                #                        y=plot_data[i, :].flatten(), 
                #                        linestyles='-', 
                #                        color=colours[i], 
                #                        ci=None, 
                #                        scale=0.5)

                # if metric != "Error Rate":
                #     ax.set_ylabel("Mean " + metric , fontsize=14)
                # else:
                #     ax.set_ylabel(metric , fontsize=14)

                # ax.set_xlabel("Number of Principal Components", fontsize=14)
                # ax.set_yscale("log")
                # # ax.set_yticks(np.logspace(-3.5, -1, 6))

                # ax.set_title("Classifier " + metric + " on Images Processed by " + denoise_type, fontsize=16)

                # plt.legend(labels)
                # plt.tight_layout()
                # plt.savefig(os.path.join(SAVE_DIR, "Noise_" + str(noise_level) + "_" + denoise_type + "_principal_components_vs_" + metric + ".png"))
                # plt.show()   

                pass

        if metric != "MSE":

            plot_data = data[:, j, :, :]
            plot_data = np.mean(plot_data, axis=0)

        else:

            plot_data = data[j, :, :, :]
            plot_data = np.mean(plot_data, axis=2)

        colours = sns.color_palette(n_colors=n_datasets)
        labels = DENOISE_TYPES[1:]

        sns.set(context='notebook', style='darkgrid', font="serif")
                       
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


            # ax = sns.pointplot(x=COMPONENT_NUMBERS, 
            #                    y=plot_data[:, l].flatten(), 
            #                    linestyles='-', 
            #                    color=colours[l], 
            #                    ci=None, 
            #                    scale=0.5)

        ax = plt.gca()

        if metric != "Error Rate":
            ax.set_ylabel("Mean " + metric , fontsize=14)
        else:
            ax.set_ylabel(metric , fontsize=14)

        ax.set_xlabel("Number of Principal Components", fontsize=14)
        # ax.set_yscale("log")
        max_yticks = 5
        yloc = plt.MaxNLocator(max_yticks, prune="both")
        ax.yaxis.set_major_locator(yloc)

        ax.set_title(metric + " vs. Principal Component Count", fontsize=16)

        plt.legend(labels, title="PCA Type")
        plt.tight_layout()
        plt.savefig(os.path.join(SAVE_DIR, "Loss_" + str(noise_level) + "_" + metric + " vs. Principal Component Count"+ ".png"))
        # plt.show()   
        plt.close('all')

    
def graph_principal_components_noises(noise_levels, metric):

    if metric == "Accuracy":
        data = np.load(os.path.join(RESULTS_DIR, "denoised_accuracies.npy"))
        metric = "Error Rate"
    elif metric == "MSE":
        data = np.load(os.path.join(RESULTS_DIR, "denoised_mse_values.npy"))
    elif metric == "Loss":
        data = np.load(os.path.join(RESULTS_DIR, "denoised_losses.npy"))
    elif metric == "Runtime":
        data = np.load(os.path.join(RESULTS_DIR, "runtimes.npy"))
  
    n_datasets = len(noise_levels)

    colours = sns.light_palette("navy", reverse=False, n_colors=n_datasets)
    labels = list(NOISE_LEVELS[i] for i in noise_levels)

    sns.set(context='notebook', style='darkgrid', font="serif")
                   
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
        # ax.set_yticklabels([])

    if metric != "Error Rate":
        ax.set_ylabel("Mean " + metric , fontsize=14)
    else:
        ax.set_ylabel(metric , fontsize=14)

    ax.set_xlabel("Number of Principal Components", fontsize=14)
    # ax.set_yscale("log")

    # # ax.yaxis.set_major_locator(ticker.MultipleLocator(base=1e-1))
    max_yticks = 5
    yloc = plt.MaxNLocator(max_yticks, prune="both")
    ax.yaxis.set_major_locator(yloc)
    
    # ax.set_yticklabels(ax.get_yticks(minor=True))

    ax.set_title(metric + " vs. Principal Component Count", fontsize=16)
    plt.legend(labels, title="Noise Magnitude", loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, metric + " vs. Principal Component Count"+ ".png"))
    plt.show()   
    plt.close('all')

def graph_best_accuracy():

    # data = np.load(os.path.join(RESULTS_DIR, "denoised_accuracies.npy"))

    old_data = np.loadtxt(os.path.join(RESULTS_DIR, "accuracies.csv"), dtype="float64", delimiter=",")

    noise_densities = old_data[1:, 0]
    n_datasets = old_data.shape[1] - 1

    labels = ["None", "NIPALS_GPU"]

    colours = sns.color_palette("muted")
    print(colours)

    old_data = np.max(old_data[1:, 1:], axis=1)

    accuracies = 1.0 - old_data

    sns.set(context='notebook', style='darkgrid', font="serif")
                   
    plt.figure(figsize=(7, 4), dpi=240)

    ax = sns.pointplot(x=noise_densities, 
                       y=accuracies, 
                       linestyles='-', 
                       color=colours[0], 
                       ci=None, 
                       scale=0.5)

    data = np.load(os.path.join(RESULTS_DIR, "denoised_accuracies.npy"))

    new_accuracies = list()

    for i in range(8):

        new_accuracies.append(np.max(data[:, i, :, 3], axis=0))

    new_accuracies = np.array(new_accuracies, dtype="float64")
    new_accuracies = np.max(new_accuracies, axis=1)
    new_accuracies = 1 - new_accuracies

    ax = sns.pointplot(x=noise_densities, 
                   y=new_accuracies, 
                   linestyles='-', 
                   color=colours[3], 
                   ci=None, 
                   scale=0.5)

    ax.set_ylabel("Error Rate", fontsize=14)
    ax.set_xlabel(r"Magnitude of Noise Mask", fontsize=14)
    # ax.set_yscale("log")
    # ax.set_yticks(np.logspace(-2.5, -0.5, 5))

    ax.set_title("Effect of PCA on Error Rate", fontsize=16)

    plt.legend(labels, title="PCA Type")
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, 'new_accuracies.png'))
    plt.show()    
    plt.close()

def graph_best_loss():

    # data = np.load(os.path.join(RESULTS_DIR, "denoised_accuracies.npy"))

    old_data = np.loadtxt(os.path.join(RESULTS_DIR, "losses.csv"), dtype="float64", delimiter=",")

    noise_densities = old_data[1:, 0]
    n_datasets = old_data.shape[1] - 1

    labels = ["None", "NIPALS_GPU"]

    colours = sns.color_palette()
    print(colours)

    old_data = np.min(old_data[1:, 1:], axis=1)

    accuracies = old_data

    sns.set(context='notebook', style='darkgrid', font="serif")
                   
    plt.figure(figsize=(7, 4), dpi=240)

    ax = sns.pointplot(x=noise_densities, 
                       y=accuracies, 
                       linestyles='-', 
                       color=colours[0], 
                       ci=None, 
                       scale=0.5)

    data = np.load(os.path.join(RESULTS_DIR, "denoised_losses.npy"))

    new_accuracies = list()

    for i in range(8):

        new_accuracies.append(np.min(data[:, i, :, 3], axis=0))

    new_accuracies = np.array(new_accuracies, dtype="float64")
    new_accuracies = np.min(new_accuracies, axis=1)

    ax = sns.pointplot(x=noise_densities, 
                   y=new_accuracies, 
                   linestyles='-', 
                   color=colours[3], 
                   ci=None, 
                   scale=0.5)

    ax.set_ylabel("Loss", fontsize=14)
    ax.set_xlabel(r"Magnitude of Noise Mask", fontsize=14)
    # ax.set_yscale("log")
    # ax.set_yticks(np.logspace(-2.5, -0.5, 5))

    ax.set_title("Effect of PCA on Loss", fontsize=16)

    plt.legend(labels, title="PCA Type")
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, 'new_losses.png'))
    plt.show()    

def graph_times():

    data = np.load(os.path.join(RESULTS_DIR, "time_results.npy"))


    n_datasets = 6
    colours = sns.color_palette(n_colors=n_datasets)
    labels = ["Full_SVD", "Simultaneous_Iteration", "NIPALS"]

    sns.set(context='notebook', style='darkgrid', font="serif")
                   
    plt.figure(figsize=(7, 4), dpi=240)

    for l in range(N_DENOISE_TYPES):

        if l == 0:

            continue

        color=colours[l + 1]

        index = l

        if l == 2:

            index = 1

        if l == 1:

            index = 2


        if l == 3:

            color = colours[5]


        plt.plot(COMPONENT_NUMBERS[:7], 
                 data[:7, index - 1] / 3, 
                 color=color,
                 linestyle='-',
                 linewidth=1,
                 marker='o',
                 markersize=4)
    ax = plt.gca()

    ax.set_ylabel("Runtime (seconds)" , fontsize=14)

    ax.set_xlabel("Number of Principal Components", fontsize=14)
    # ax.set_yscale("log")

    # # ax.yaxis.set_major_locator(ticker.MultipleLocator(base=1e-1))
    max_yticks = 5
    yloc = plt.MaxNLocator(max_yticks, prune="both")
    ax.yaxis.set_major_locator(yloc)
    
    # ax.set_yticklabels(ax.get_yticks(minor=True))

    ax.set_title("Runtime vs. Principal Component Count", fontsize=16)
    plt.legend(labels, title="PCA Type", loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "Runtime vs. Principal Component Count"+ ".png"))
    plt.show()   
    plt.close('all')

def graph_times_2():

    data = np.load(os.path.join(RESULTS_DIR, "new_results.npy"))

    vals = [10000, 20000, 30000, 40000, 50000, 75000, 100000, 150000, 200000]

    n_datasets = 6
    colours = sns.color_palette(n_colors=n_datasets)
    labels = ["Full_SVD", "Simultaneous_Iteration", "NIPALS"]

    sns.set(context='notebook', style='darkgrid', font="serif")
                   
    plt.figure(figsize=(7, 4), dpi=240)

    for l in range(N_DENOISE_TYPES):

        if l == 0:

            continue

        color=colours[l + 1]

        index = l

        if l == 2:

            index = 1

        if l == 1:

            index = 2


        if l == 3:

            color = colours[5]

        print(len(vals))

        print(data.shape)


        plt.plot(vals[:7], 
                 data[:7, index - 1] / 5, 
                 color=color,
                 linestyle='-',
                 linewidth=1,
                 marker='o',
                 markersize=4)
    ax = plt.gca()

    ax.set_ylabel("Runtime (seconds)" , fontsize=14)

    ax.set_xlabel("Number of Images", fontsize=14)
    # ax.set_yscale("log")

    # # ax.yaxis.set_major_locator(ticker.MultipleLocator(base=1e-1))
    max_yticks = 5
    yloc = plt.MaxNLocator(max_yticks, prune="both")
    ax.yaxis.set_major_locator(yloc)
    
    # ax.set_yticklabels(ax.get_yticks(minor=True))

    ax.set_title("Runtime vs. Image Count", fontsize=16)
    plt.legend(labels, title="PCA Type", loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "Runtime vs. Image Count"+ ".png"))
    plt.show()   
    plt.close('all')


def graph_times_3():

    data = np.load(os.path.join(RESULTS_DIR, "gpu_results.npy"))


    n_datasets = 6
    colours = sns.color_palette(n_colors=n_datasets)
    labels = ["NIPALS", "NIPALS_GPU"]

    sns.set(context='notebook', style='darkgrid', font="serif")
                   
    plt.figure(figsize=(7, 4), dpi=240)

    for l in range(2):

        if l == 0:

            index = 5

        if l == 1:

            index = 4


        if l == 3:

            color = colours[5]


        plt.plot(COMPONENT_NUMBERS[:5], 
                 data[:5, l] / 5, 
                 color=colours[index],
                 linestyle='-',
                 linewidth=1,
                 marker='o',
                 markersize=4)
    ax = plt.gca()

    ax.set_ylabel("Runtime (seconds)" , fontsize=14)

    ax.set_xlabel("Number of Principal Components", fontsize=14)
    # ax.set_yscale("log")

    # # ax.yaxis.set_major_locator(ticker.MultipleLocator(base=1e-1))
    max_yticks = 5
    yloc = plt.MaxNLocator(max_yticks, prune="both")
    ax.yaxis.set_major_locator(yloc)
    ax.set_xlim(0, 60)
    
    # ax.set_yticklabels(ax.get_yticks(minor=True))

    ax.set_title("Runtime vs. Principal Component Count", fontsize=16)
    plt.legend(labels, title="PCA Type", loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "GPU Runtime vs. Principal Component Count"+ ".png"))
    plt.show()   
    plt.close('all')



if __name__ == "__main__": 

    noise_levels = range(8)

    # graph_accuracies_test()
    # graph_losses_test()

    # graph_principal_components(noise_levels, metric="Accuracy")
    # graph_principal_components(noise_levels, metric="MSE")
    # graph_principal_components(noise_levels, metric="Loss")

    # graph_principal_components_noises(range(8), "Accuracy")
    # graph_principal_components_noises(range(8), metric="MSE")
    # graph_principal_components_noises(range(8), metric="Loss")
    # colours = sns.color_palette()
    # graph_best_accuracy()
    # graph_best_loss()
    # graph_times()
    graph_times_2()


