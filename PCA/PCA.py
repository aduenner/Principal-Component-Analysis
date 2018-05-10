import numpy as np
import matplotlib.pyplot as plt
import IterativeEigenvals as IE
from matplotlib.mlab import PCA
from functools import partial

def load_images_from_file(imagedbname):
    '''Loads image data with extension .npy and returns number of images
    Inputs:
        imagedbname: Name of .npy file w/ cols representing pixels and rows individual images
    Outputs: 
        images: Image db
        n_images: number of images in db
    '''
    images = np.load(imagedbname+".npy")
    n_images=images.shape[0]
    return images,n_images

def extract_image(imgset, shape, index):
    '''Extracts a single square image from the image db
    Inputs:
        imgset - Image database
        shape - n pixels per row/col in square image: n^2 = #cols in imgset
        index - Index in image database that is to be extracted
    Outputs:
        image_out - nxn uint8 image that can be displayed or manipulated
    '''
    image_out = np.reshape(imgset[index,:],(shape,shape))
    return image_out

def show_image(image):
    '''Show an image extracted with extract_image'''
    plt.imshow(image)
    plt.show()

def zero_mean(imageset,num_img):
    '''Use to zero the mean of images in rows of input matrix'''
    num_img = np.int64(np.sqrt(np.shape(imageset)[1]))
    meanval=np.zeros(num_img)
    
    for i in range(num_img):
        meanval[i]=np.uint8(np.mean(images[i,:]))
        imageset[i,:] = imageset[i,:] - meanval[i]
    return imageset

noisy_image,n_noisy=load_images_from_file("noised_images")
orig_image,n_orig=load_images_from_file("original_images")

zeromean_noisy = zero_mean(orig_image,n_orig)
zeromean_orig = zero_mean(noisy_image,n_noisy)

'Calculate the covariance matrix'
covar_orig  = np.cov(orig_image)
covar_noisy = np.cov(noisy_image)

def PrincipleComponents(imageset,num_components,method):
    imageset,nimages = load_images_from_file(imageset)
    imagecovariance = np.cov(imageset)
    
    Components = {
            0: PCA(imageset,num_components),
            2: IE.SimultaneousIteration(imagecovariance,num_components)
            }
    
    PC = Components[method]
    return PC

plt.subplot(221)
zeromean_orig_0 = extract_image(zeromean_orig,28,0)
show_image(zeromean_orig_0)


plt.subplot(222)
noise_orig_0 = extract_image(zeromean_noisy,28,0)
show_image(noise_orig_0)

plt.subplot(223)


Q,V,Err=IE.SimulIter(noise_orig_0,neigs=2)
EV_image=np.uint8(Q*V*np.transpose(Q))

orig_image_0 = extract_image(orig_image,28,0)
show_image(EV_image)

plt.subplot(224)
noisy_image_0 = extract_image(noisy_image,28,0)
show_image(noisy_image_0)

