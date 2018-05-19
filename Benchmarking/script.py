from _implementations import *
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import matplotlib.ticker as ticker

COMPONENT_VALUES = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
IMAGE_NUMBERS = [100, 200, 300, 400, 500, 600, 700, 800, 1000]
PIXEL_COUNTS = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300, 320]
GPU_IMAGES = [2000, 4000, 6000, 8000, 10000, 12000, 14000, 16000, 18000, 20000]


ALGORITHMS = ['Full_SVD', 'Simultaneous_Iteration', 'NIPALS']

N_CV_TRIALS = len(COMPONENT_VALUES)
N_IN_TRIALS = len(IMAGE_NUMBERS)
N_ALGORITHMS = len(ALGORITHMS)
N_PC_TRIALS = len(PIXEL_COUNTS)
N_GPU = len(GPU_IMAGES)

NDIR = "Normalized Graphs"
DIR = "Unnormalized Graphs"
IMGDIR = "Images"

def benchmark_gpu(dataset, savename='', EACH=15):

    ret = np.zeros((N_GPU, EACH, 2), dtype="float64")
    n_trials = ret.size
    minimal = dataset[:10, :10]

    benchmark_GPU(minimal.copy(), 1)
    benchmark_NIPALS(minimal.copy(), 1, False)

    completed = 0
    n_total = dataset.shape[0]


    for j in range(EACH):
        for k in range(N_GPU):

            n_images = GPU_IMAGES[k]

            perm = np.random.permutation(n_total)[:n_images]
            subset = dataset[perm]

            _, _, _elapsed = benchmark_NIPALS(subset.copy(), 5, False)
            _, _, _elapsed_GPU = benchmark_GPU(subset.copy(), 5)

            completed += 2

            ret[k, j, 0] = _elapsed
            ret[k, j, 1] = _elapsed_GPU


            print("Completed: ", completed, " of ", n_trials, " trials!")
        np.save('GPU_' + savename + '.npy', ret)



def benchmark_pixels(dataset, n_images=5000, n_components=5, EACH=25, savename=''):

    dtype = "float32"

    ret = np.zeros((N_ALGORITHMS, N_PC_TRIALS, EACH), dtype="float64")
    n_trials = ret.size

    minimal = dataset[:10, :10]

    # Quickly Compile Numba Functions
    benchmark_NIPALS(minimal.copy(), 1, False)
    benchmark_SVD(minimal.copy(), 1, False)
    benchmark_SI(minimal.copy(), 1, False)

    completed = 0
    n_total = dataset.shape[0]
    max_pixels = dataset.shape[1]

    for k in range(EACH):

        for j in range(N_PC_TRIALS):

            n_pixels = PIXEL_COUNTS[j]
            perm = np.random.permutation(n_total)

            perm_pixels = np.random.permutation(max_pixels)[:n_pixels]

            subset = dataset[perm][:n_images, perm_pixels]

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
        np.save('full_pixels' + savename + '.npy', ret)

def benchmark_all(dataset, EACH=25, savename=''):

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
            np.save('full' + savename + '.npy', ret)

def graph_vs_components(normalized=False, savename=''):

    runtimes = np.load('full' + savename + '.npy')
    runtimes = np.median(runtimes, axis=3)

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
     
        ax.set_title(r"$n_{images}$ = " + str(n_images), fontsize=14)

        ax.set_xlabel(r"$n_{components}$", fontsize=14)

        max_yticks = 5
        yloc = plt.MaxNLocator(max_yticks, prune="both")
        ax.yaxis.set_major_locator(yloc)

        plt.legend(ALGORITHMS, title="PCA Type")
        plt.tight_layout()
        if normalized:

            plt.savefig(os.path.join(NDIR, "N_Components_Image" + str(n_images) + ".png"))
        
        else:

            plt.savefig(os.path.join(DIR, "U_Components_Image" + str(n_images) + ".png"))
                  
        plt.close('all')

def graph_vs_images(normalized=False, savename=''):

    runtimes = np.load('full' + savename + '.npy')
    runtimes = np.median(runtimes, axis=3)

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
     
        ax.set_title(r"$n_{components}$ = " + str(n_components), fontsize=14)

        ax.set_xlabel(r"$n_{images}$", fontsize=14)

        max_yticks = 5
        yloc = plt.MaxNLocator(max_yticks, prune="both")
        ax.yaxis.set_major_locator(yloc)

        plt.legend(ALGORITHMS, title="PCA Type")
        plt.tight_layout()
        if normalized:

            plt.savefig(os.path.join(NDIR, "N_Image_Components" + str(n_components) + ".png"))
        
        else:

            plt.savefig(os.path.join(DIR, "U_Image_Components" + str(n_components) + ".png"))
                  
        plt.close('all')

def graph_vs_pixels(normalized=False, savename=""):

    runtimes = np.load('full_pixels' + savename + '.npy')
    runtimes = np.mean(runtimes, axis=2)

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

        plt.savefig(os.path.join(DIR, "U_Pixels.png"))

    plt.show()              
    plt.close('all')



def graph_vs_GPU(savename=""):

    runtimes = np.load('GPU_' + savename + '.npy')
    runtimes = np.mean(runtimes, axis=1)

    sns.set(context='notebook', style='darkgrid', font="serif")
    colours = sns.color_palette(n_colors=6)
    colour_indices = [3, 4]
    plt.figure(figsize=(6, 4), dpi=240)

    labels = ["NIPALS", "NIPALS_GPU"]
    for k in range(2):

        values = runtimes[:, k].flatten()

        plt.plot(GPU_IMAGES, 
                 values, 
                 color=colours[colour_indices[k]],
                 linestyle='-',
                 linewidth=1,
                 marker='o',
                 markersize=4)

    ax = plt.gca()

    ax.set_ylabel(r"$Runtime (seconds)$" , fontsize=14)
    ax.set_title(r"$Runtime$ vs. $n_{images}$", fontsize=16)
 

    ax.set_xlabel(r"$n_{images}$", fontsize=14)

    max_yticks = 5
    yloc = plt.MaxNLocator(max_yticks, prune="both")
    ax.yaxis.set_major_locator(yloc)

    plt.legend(labels, title="PCA Type")
    plt.tight_layout()

    plt.savefig(os.path.join(DIR, "U_GPU.png"))

    plt.show()              
    plt.close('all')


def extract_image(data, shape, index):

    ret = np.reshape(data[index, :], (shape, shape))
    return ret

def plot_images(data, titles, savelocation):

    n_images = len(data)

    shape = int(np.ceil(np.sqrt(np.shape(data)[1])))
    numrows = int(np.ceil(np.sqrt(n_images)))

    plt.figure(figsize=(6, 6), dpi=240)
    sns.set(context='notebook', style='darkgrid', font="serif")

    for i in range(n_images):
        ax = plt.subplot(numrows, numrows, i+1)
        ax.imshow(extract_image(data,shape,i),cmap='gray')
        ax.set_title(r'$n=' + str(titles[i]) + r'$', fontsize=14)
        plt.axis('off')


    fig = plt.gcf()
    plt.tight_layout()
    fig.subplots_adjust(top=0.85)
    fig.suptitle(r'Effect of Varying $n_{components}$ on PCA Output', fontsize=16)
    plt.savefig(os.path.join(IMGDIR, savelocation))
    plt.close()



if __name__ == '__main__':

    dtype = "float32"

    train_images = np.load('noised_data_training_30.npy')
    test_images = np.load('noised_data_test_30.npy')

    dataset = np.concatenate((train_images, test_images), axis=0)
    dataset = np.array(dataset, dtype=dtype)

    # benchmark_all(dataset, savename='2')

    # graph_vs_components(True, savename='2')
    # graph_vs_images(True, savename='2')
    # graph_vs_components(False, savename='2')
    # graph_vs_images(False, savename='2')


    # benchmark_pixels(dataset, savename='2')

    # graph_vs_pixels(True, '2')
    # graph_vs_pixels(False, '2')


    # benchmark_gpu(dataset)

    graph_vs_GPU()





    # component_numbers = [1, 2, 5, 10, 20, 50, 100, 200, 784]
    # dataset = np.load('noised_data_training_60.npy')

    # dataset = np.array(dataset, dtype=dtype)

    # perm = np.random.permutation(50)

    # images = list()

    # for n_components in component_numbers:

    #     ret, _, _ = benchmark_SVD(dataset, n_components, False)

    #     images.append(ret[perm])

    # images = np.array(images)
    # np.save('save.npy', images)
    # images = np.load('save.npy')

    # for j in range(images.shape[1]):

    #     display = images[:, j, :]
    #     savelocation = 'set_' + str(j) + ".png"
    #     plot_images(display, component_numbers, savelocation)









