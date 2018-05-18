from _implementations import *
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import matplotlib.ticker as ticker

COMPONENT_VALUES = [5, 10, 15, 20, 30, 40, 50, 60, 80, 100]
IMAGE_NUMBERS = [100, 200, 400, 600, 800, 1000, 1500, 2000, 2500, 3000, 4000, 5000]
PIXEL_COUNTS = [20, 40, 60, 80, 100, 150, 200, 250, 300, 400, 500, 600, 800, 1000, 1500, 2000]

ALGORITHMS = ['Full_SVD', 'Simultaneous_Iteration', 'NIPALS']

N_CV_TRIALS = len(COMPONENT_VALUES)
N_IN_TRIALS = len(IMAGE_NUMBERS)
N_ALGORITHMS = len(ALGORITHMS)
N_PC_TRIALS = len(PIXEL_COUNTS)

NDIR = "Normalized Graphs"
DIR = "Graphs"

def benchmark_pixels(n_images=5000, n_components=5, EACH=8):

    dtype = "float32"

    ret = np.zeros((N_ALGORITHMS, N_PC_TRIALS, EACH), dtype="float64")
    n_trials = ret.size

    minimal = dataset[:10, :10]

    # Quickly Compile Numba Functions
    benchmark_NIPALS(minimal.copy(), 1, False)
    benchmark_SVD(minimal.copy(), 1, False)
    benchmark_SI(minimal.copy(), 1, False)

    completed = 0

    for k in range(EACH):

        images = np.random.normal(0, 1, (n_images, max(PIXEL_COUNTS)))
        images = np.array(images, dtype)

        for j in range(N_PC_TRIALS):

            n_pixels = PIXEL_COUNTS[j]
            subset = images[:, :n_pixels]

            _, _, _elapsedN = benchmark_NIPALS(subset.copy(), n_components, False)
            _, _, _elapsedSVD = benchmark_SVD(subset.copy(), n_components, False)
            _, _, _elapsedSI = benchmark_SI(subset.copy(), n_components, False)

            completed += N_ALGORITHMS

            ret[0, j, k] = _elapsedSVD
            ret[1, j, k] = _elapsedSI
            ret[2, j, k] = _elapsedN

            print(_elapsedSVD)
            print(_elapsedSI)
            print(_elapsedN)

            print("Completed: ", completed, " of ", n_trials, " trials!")
        np.save('full_pixels.npy', ret)

def benchmark_all(dataset, EACH=8):

    ret = np.zeros((N_CV_TRIALS, N_IN_TRIALS, N_ALGORITHMS, EACH), dtype="float64")
    n_trials = ret.size

    minimal = dataset[:10, :10]

    # Quickly Compile Numba Functions
    benchmark_NIPALS(minimal.copy(), 1, False)
    benchmark_SVD(minimal.copy(), 1, False)
    benchmark_SI(minimal.copy(), 1, False)

    n_total = len(dataset)

    completed = 0

    for i in range(N_CV_TRIALS):

        n_components = COMPONENT_VALUES[i]

        for j in range(N_IN_TRIALS):

            n_images = IMAGE_NUMBERS[j]

            for l in range(EACH):

                perm = np.random.permutation(n_total)
                subset = dataset[perm][:n_images]

                _, _, _elapsedN = benchmark_NIPALS(subset.copy(), n_components, False)
                _, _, _elapsedSVD = benchmark_SVD(subset.copy(), n_components, False)
                _, _, _elapsedSI = benchmark_SI(subset.copy(), n_components, False)

                completed += N_ALGORITHMS

                ret[i, j, 0, l] = _elapsedSVD
                ret[i, j, 1, l] = _elapsedSI
                ret[i, j, 2, l] = _elapsedN

            print("Completed: ", completed, " of ", n_trials, " trials!")
            np.save('full_run_new.npy', ret)

def graph_vs_components(normalized=False):

    runtimes = np.load('full_run_new.npy')
    runtimes = np.mean(runtimes, axis=3)

    for j in range(N_IN_TRIALS):

        sns.set(context='notebook', style='darkgrid', font="serif")
        colours = sns.color_palette(n_colors=6)
        colour_indices = [1, 2, 4]
        plt.figure(figsize=(6, 4), dpi=240)
       
        n_images = IMAGE_NUMBERS[j]

        for k in range(N_ALGORITHMS):

            values = runtimes[:, j, k].flatten()

            if normalized:

                values /= values[0]

            plt.plot(COMPONENT_VALUES, 
                     values, 
                     color=colours[colour_indices[k]],
                     linestyle='-',
                     linewidth=1,
                     marker='o',
                     markersize=4)

        ax = plt.gca()

        if normalized:

            ax.set_title(r"Normalized Runtime vs. $n_{components}$", fontsize=16)
     
        else:

            ax.set_ylabel(r"$Runtime (seconds)$" , fontsize=14)
            ax.set_title(r"$Runtime$ vs. $n_{components}$", fontsize=16)
     

        ax.set_xlabel(r"$n_{components}$", fontsize=14)

        max_yticks = 5
        yloc = plt.MaxNLocator(max_yticks, prune="both")
        ax.yaxis.set_major_locator(yloc)

        plt.legend(ALGORITHMS, title="PCA Type")
        plt.tight_layout()
        if normalized:

            plt.savefig(os.path.join(NDIR, "N_Components_Image" + str(n_images) + ".png"))
        
        else:

            plt.savefig(os.path.join(DIR, "N_Components_Image" + str(n_images) + ".png"))
                  
        plt.close('all')

def graph_vs_images(normalized=False):

    runtimes = np.load('full_run_new.npy')
    runtimes = np.mean(runtimes, axis=3)

    for i in range(N_CV_TRIALS):

        sns.set(context='notebook', style='darkgrid', font="serif")
        colours = sns.color_palette(n_colors=6)
        colour_indices = [1, 2, 4]
        plt.figure(figsize=(6, 4), dpi=240)
       
        n_components = COMPONENT_VALUES[i]

        for k in range(N_ALGORITHMS):

            values = runtimes[i, :, k].flatten()

            if normalized:

                values /= values[0]

            plt.plot(IMAGE_NUMBERS, 
                     values, 
                     color=colours[colour_indices[k]],
                     linestyle='-',
                     linewidth=1,
                     marker='o',
                     markersize=4)

        ax = plt.gca()

        if normalized:

            ax.set_title(r"Normalized Runtime vs. $n_{images}$", fontsize=16)
     
        else:

            ax.set_ylabel(r"$Runtime (seconds)$" , fontsize=14)
            ax.set_title(r"$Runtime$ vs. $n_{images}$", fontsize=16)
     

        ax.set_xlabel(r"$n_{images}$", fontsize=14)

        max_yticks = 5
        yloc = plt.MaxNLocator(max_yticks, prune="both")
        ax.yaxis.set_major_locator(yloc)

        plt.legend(ALGORITHMS, title="PCA Type")
        plt.tight_layout()
        if normalized:

            plt.savefig(os.path.join(NDIR, "N_Image_Components" + str(n_components) + ".png"))
        
        else:

            plt.savefig(os.path.join(DIR, "N_Image_Components" + str(n_components) + ".png"))
                  
        plt.close('all')

def graph_vs_pixels(normalized=False):

    runtimes = np.load('full_pixels.npy')
    runtimes = np.mean(runtimes, axis=2)

    print(runtimes)

    sns.set(context='notebook', style='darkgrid', font="serif")
    colours = sns.color_palette(n_colors=6)
    colour_indices = [1, 2, 4]
    plt.figure(figsize=(6, 4), dpi=240)

    for k in range(N_ALGORITHMS):

        values = runtimes[k, :].flatten()

        if normalized:

            values /= values[0]

        plt.plot(PIXEL_COUNTS, 
                 values, 
                 color=colours[colour_indices[k]],
                 linestyle='-',
                 linewidth=1,
                 marker='o',
                 markersize=4)

    ax = plt.gca()

    if normalized:

        ax.set_title(r"Normalized Runtime vs. $n_{pixels}$", fontsize=16)
 
    else:

        ax.set_ylabel(r"$Runtime (seconds)$" , fontsize=14)
        ax.set_title(r"$Runtime$ vs. $n_{pixels}$", fontsize=16)
 

    ax.set_xlabel(r"$n_{pixels}$", fontsize=14)

    max_yticks = 5
    yloc = plt.MaxNLocator(max_yticks, prune="both")
    ax.yaxis.set_major_locator(yloc)

    plt.legend(ALGORITHMS, title="PCA Type")
    plt.tight_layout()

    if normalized:

        plt.savefig(os.path.join(NDIR, "N_Pixels.png"))
    
    else:

        plt.savefig(os.path.join(DIR, "N_Pixels.png"))

              
    plt.close('all')

def graph_vs_pixels_large(normalized=False):

    runtimes = np.load('full_pixels.npy')
    runtimes = np.mean(runtimes, axis=2)

    print(runtimes)

    sns.set(context='notebook', style='darkgrid', font="serif")
    colours = sns.color_palette(n_colors=6)
    colour_indices = [1, 4]
    plt.figure(figsize=(6, 4), dpi=240)

    for k in range(N_ALGORITHMS - 1):

        values = runtimes[k, :].flatten()

        if normalized:

            values /= values[0]

        plt.plot(PIXEL_COUNTS, 
                 values, 
                 color=colours[colour_indices[k]],
                 linestyle='-',
                 linewidth=1,
                 marker='o',
                 markersize=4)

    ax = plt.gca()

    if normalized:

        ax.set_title(r"Normalized Runtime vs. $n_{pixels}$", fontsize=16)
 
    else:

        ax.set_ylabel(r"$Runtime (seconds)$" , fontsize=14)
        ax.set_title(r"$Runtime$ vs. $n_{pixels}$", fontsize=16)
 

    ax.set_xlabel(r"$n_{pixels}$", fontsize=14)

    max_yticks = 5
    yloc = plt.MaxNLocator(max_yticks, prune="both")
    ax.yaxis.set_major_locator(yloc)

    plt.legend(list(ALGORITHMS[0], ALGORITHMS[2]), title="PCA Type")
    plt.tight_layout()

    if normalized:

        plt.savefig(os.path.join(NDIR, "N_Pixels_Large.png"))
    
    else:

        plt.savefig(os.path.join(DIR, "N_Pixels_Large.png"))

              
    plt.close('all')


if __name__ == '__main__':

    dtype = "float32"

    train_images = np.load('noised_data_training_30.npy')
    test_images = np.load('noised_data_test_30.npy')

    dataset = np.concatenate((train_images, test_images), axis=0)
    dataset = np.array(dataset, dtype=dtype)

    # benchmark_all(dataset)
    benchmark_pixels()
    # benchmark_pixels_large()

    # graph_vs_components(True)
    # graph_vs_images(True)
    # graph_vs_components(False)
    # graph_vs_images(False)

    graph_vs_pixels(True)
    graph_vs_pixels(False)