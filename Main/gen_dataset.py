import numpy as np
import os

CONST_TRAIN_DATA = "train_data.npy"
CONST_TRAIN_LABELS = "train_labels.npy"

CONST_TEST_DATA = "test_data.npy"
CONST_TEST_LABELS = "test_labels.npy"

def generate_dataset(data, labels, shape, n_dims, 
					 train_data_save=CONST_TRAIN_DATA, train_labels_save=CONST_TRAIN_LABELS,
					 test_data_save=CONST_TEST_DATA, test_labels_save=CONST_TEST_LABELS,
					 training_percentage=0.75, total_images=None, permutation=True, dtype="uint8"):

	data = np.load(data)
	labels = np.load(labels)

	data = np.array(data, dtype=dtype)
	labels = np.array(labels, dtype="int64")

	data = data.reshape(-1, n_dims, shape[0], shape[1])

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

	np.save(train_data_save, training_data)
	np.save(train_labels_save, training_labels)

	if (training_count != len(data)):

		np.save(test_data_save, testing_data)
		np.save(test_labels_save, testing_labels)

def load_dataset(train_data_save=CONST_TRAIN_DATA, train_labels_save=CONST_TRAIN_LABELS,
				 test_data_save=CONST_TEST_DATA, test_labels_save=CONST_TEST_LABELS, data_dir=""):

	if data_dir != '':

		training_data = np.load(os.path.join(data_dir, train_data_save))
		testing_data = np.load(os.path.join(data_dir, test_data_save))

		training_labels = np.load(os.path.join(data_dir, train_labels_save))
		testing_labels = np.load(os.path.join(data_dir, test_labels_save))

	else:

		training_data = np.load(train_data_save)
		testing_data = np.load(test_data_save)

		training_labels = np.load(train_labels_save)
		testing_labels = np.load(test_labels_save)

	return training_data, testing_data, training_labels, testing_labels

	

if __name__ == '__main__':
	
	generate_dataset()
