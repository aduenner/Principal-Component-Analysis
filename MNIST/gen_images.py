import numpy as np

import pickle
import cv2


def load_original_dataset():

	images = np.load("original_images.npy")
	labels = np.load("original_labels.npy")

	return images, labels


'''
Applies gaussian noise filter over input images

	:param: scale - magnitude of noise filter
	:return: new array with same shape as input containing noised images
'''
def apply_gaussian_noise(images, scale=30.0):


	noise_mask = np.random.normal(scale=scale, size=images.shape)
	noise_mask = np.array(noise_mask, dtype="int8")

	images = np.array(np.clip((noise_mask + images), 0, 255), dtype="uint8")

	return images


'''
Applies constrast filter over input images

	:param: contrast - contrast correction factor value (between -128 and 128); 0 is no change in contrast
	:return: new array with same shape as input containing altered images
'''
def apply_contrast_filter(images, contrast=-50.0):

	f = (259.0 * (contrast + 255.0)) / (255.0 * (259.0 - contrast))

	images = np.array(np.clip((f * (images - 128.0) + 128.0), 0, 255), dtype="uint8")

	return images

'''
Applies brightness filter over input images

	:param: brightness - integer brightness to be added to image (between -255 and 255);
	:return: new array with same shape as input containing brightned images
'''
def apply_contrast_filter(images, brightness=30):
	
	images = np.array(np.clip((images + brightness), 0, 255), dtype="uint8")

	return images


'''
Display unflattened image until keypress

	:param: win_name - name of the displayed window
	:param: shape - shape of the unflattened image
'''
def show_image(flattened, win_name="Display", shape=(28, 28)):

	image = flattened.reshape(shape)

	cv2.imshow(win_name, image)
	cv2.waitKey(0)


def generate_noised_data():

	images, labels = load_original_dataset()

	noise_images = apply_gaussian_noise(images)
	noise_images = apply_contrast_filter(noise_images)

	show_image(images[0], str(labels[0]))
	show_image(noise_images[0], str(labels[0]))

	np.save("noised_images.npy", noise_images)




if __name__ == '__main__':
	
	generate_noised_data()