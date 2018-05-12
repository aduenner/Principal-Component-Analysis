import numpy as np
import matplotlib.pyplot as plt
import IterativeEigenvals as IE
from sklearn.decomposition import PCA

def load_images_from_file(imagedbname,num_images=1000):
    '''Loads image data with extension .npy and returns number of images
    Inputs:
        imagedbname: Name of .npy file w/ cols representing pixels and rows individual images
        numimmages: Number of images to import
    Outputs: 
        images: Image db
        n_images: number of images in db
    '''
    images = np.load(imagedbname+".npy")
    images = images[0:num_images,:]/255.
    
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

def zero_mean(imageset):
    '''Use to zero the mean of images in rows of input matrix'''
    num_img = np.shape(imageset)[0]
    meanval=np.zeros(num_img)
    
    for i in range(num_img):
        meanval[i]=np.mean(imageset[i,:])
        imageset[i,:] = imageset[i,:] - meanval[i]
    return imageset,meanval

noisy_image,n_noisy=load_images_from_file("noised_images")
orig_image,n_orig=load_images_from_file("original_images")

zeromean_noisy,mean_noisy = zero_mean(noisy_image)
zeromean_orig,mean_orig = zero_mean(orig_image)


def PrincipalComponents(imageset,num_components):
    imagecovariance = np.cov(np.transpose(imageset))
    PC,EV,Err = IE.SimulIter(imagecovariance,num_components)

    variance=EV
    return PC, variance

def TransformSpace(OriginalSet,EigenVectors,Mean):
    '''use projection matrix to transform images to reduced rank subspace'''
    
    NewSet = (OriginalSet @ EigenVectors) @ np.transpose(EigenVectors)
    for i in range(np.shape(NewSet)[0]):
        NewSet[i,:]+=Mean[i]
    
    return NewSet

plt.subplot(221)
noisy_0 = extract_image(zeromean_noisy,28,0)
show_image(noisy_0)

plt.subplot(222)
#noise_0 = extract_image(zeromean_noisy,28,0)
show_image(orig_0)
#
#plt.subplot(223)
#
plt.subplot(223)
#PCorig,explainedorig = PrincipleComponents(zeromean_orig,15)
pca_orig = PCA(n_components=784)
ReducedNoisy=pca_orig.fit_transform(zeromean_noisy)
X_inv_proj = pca_orig.inverse_transform(ReducedNoisy)
#reshaping as 400 images of 64x64 dimension 
X_proj_img = np.reshape(X_inv_proj,(1000,28,28))

show_image(X_proj_img[0])

OurReduced = PrincipalComponents(zeromean_noisy,10)
OurReducedTrans = TransformSpace(zeromean_noisy,OurReduced,mean_noisy)
OurReducedFirstImg = extract_image(OurReducedTrans, 28,0)
plt.subplot(224)
show_image(OurReducedFirstImg)
##
#pc1 = extract_image(PCorig,28,0)
#show_image(pc1)
#
#plt.subplot(224)
#pc2 = extract_image(PCnoise,28,0)
#show_image(pc2)
varrat = pca_orig.explained_variance_ratio_
cumvar = np.zeros(np.shape(varrat))
cumvar[1]=varrat[1]

for i in range(1,np.shape(varrat)[0]):
    cumvar[i] = varrat[i]+cumvar[i-1]

    