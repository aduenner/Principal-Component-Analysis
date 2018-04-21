import numpy as np
import os
import cv2

CONST_TOTAL_DATA_SETS = 6
CONST_DATA_DIRECTORY = "Images"


def generate_training_sets(training_percentage = 0.75, total_images=None, permutation=True,
						   datatype="float32"):
	"""
	Function to generate both a training and testing set from the CIFAR-10 dataset

	:param training_percentage: Defines the percentage of used samples devoted to training
	:param total_images: Determines the number of images to be used in training+testing
	:param permutation: Determines whether or not the images from the dataset will be randomly shuffled
	:param datatype: Determines the datatype of the numpy arrays that will store the data
	
	:return: Returns a set of four numpy arrays, consisting of the training and testing data set image arrays
				as well as the corresponding label arrays
	"""

	data = []
	labels = []

	for i in range(CONST_TOTAL_DATA_SETS):

		data_filename = os.path.join(CONST_DATA_DIRECTORY, "data_" + str(i) + ".npy")
		data_file = np.load(data_filename)

		if data_file.dtype != datatype:
			data_file = np.array(data_file, dtype=datatype)

		data.append(data_file)

		label_filename = os.path.join(CONST_DATA_DIRECTORY, "labels_" + str(i) + ".npy")
		label_file = np.load(label_filename)

		labels.append(label_file)

	data = np.concatenate(data)
	labels = np.concatenate(labels)

	# Split image into three channels and set channels to the last axis
	data = data.reshape(-1, 3, 32, 32)
	# data = np.rollaxis(data, 1, 4)

	assert(len(data) == len(labels))

	if total_images is not None and total_images < len(data):

		data = data[:total_images]
		labels = labels[:total_images]

	if permutation:
		
		perm = np.random.permutation(len(data))

		data = data[perm]
		labels = labels[perm]

	training_count = int(training_percentage * len(data))

	training_data = data[:training_count]
	training_labels = labels[:training_count]

	testing_data = data[training_count:]
	testing_labels = labels[training_count:]

	return training_data, testing_data, training_labels, testing_labels


if __name__ == "__main__":

	generate_training_sets()