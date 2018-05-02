import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import matplotlib.pyplot as plt

import time

import gen_dataset
import gen_images

class Net(nn.Module):   

    def __init__(self, n_classes, n_dims=1):

        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(n_dims, 12, 3, padding=1)
        self.conv2 = nn.Conv2d(12, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, padding=1)

        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(512, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, n_classes)

    def forward(self, x):

        x = F.max_pool2d(F.relu(self.conv1(x)), (3, 3), stride=2, padding=1)
        x = F.max_pool2d(F.relu(self.conv2(x)), (3, 3), stride=2, padding=1)
        x = F.max_pool2d(F.relu(self.conv3(x)), (3, 3), stride=2, padding=1)

        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

    def num_flat_features(self, x):

        size = x.size()[1:]  # all dimensions except the batch dimension

        num_features = 1

        for s in size:
            num_features *= s

        return num_features


def train(training_data, validation_data, training_labels, validation_labels, n_classes, n_dims, class_names, save_params=None,
          train_epochs=25, learning_rate=0.001, batch_size=100):

    print("Preparing Dataset...")

    n_training = len(training_data)
    n_validation = len(validation_data)

    training_data = torch.from_numpy(training_data).cuda()
    validation_data = torch.from_numpy(validation_data).cuda()

    training_labels = torch.from_numpy(training_labels).cuda()
    validation_labels = torch.from_numpy(validation_labels).cuda()

    print("Initializing Network...")

    net = Net(n_classes, n_dims)
    net.cuda()
    net.train()

    print("Network Details...")
    print("\t" + str(net))

    # Define the loss function and the method of optimization (SGD w/ momentum)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)

    print("\nBeginning Training...")

    training_start = time.time()

    for epoch in range(train_epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        epoch_start = time.time()

        for batch in iterate_batches(training_data, training_labels,
                                     batch_size, shuffle=True):

            batch_inputs, batch_labels = batch

            batch_inputs, batch_labels = Variable(batch_inputs), Variable(batch_labels)

            optimizer.zero_grad()

            batch_outputs = net(batch_inputs)
            batch_loss = criterion(batch_outputs, batch_labels)

            batch_loss.backward()
            optimizer.step()

            running_loss += batch_loss.data[0]
       
        print('\nEpoch: [%d]\n\tLoss: %.6f Elapsed Time: %.6fs' %
              (epoch + 1, running_loss / n_training, time.time() - epoch_start))
        running_loss = 0.0

    print("Finished Training in %.6fs" % (time.time() - training_start))
    print("\nBeginning Testing...")

    validation_start = time.time()
    test_loss = 0.0

    class_correct = [0] * n_classes
    class_total = [0] * n_classes

    net.eval()

    for batch in iterate_batches(validation_data, validation_labels,
                                batch_size, shuffle=False):

        batch_inputs, batch_labels = batch

        batch_inputs, batch_labels = Variable(batch_inputs), Variable(batch_labels)
        
        batch_outputs = net(batch_inputs)

        batch_loss = criterion(batch_outputs, batch_labels)
        test_loss += batch_loss.data[0]

        _, batch_predictions = torch.max(batch_outputs.data, 1)
        batch_correct = (batch_predictions == batch_labels.data).squeeze()

        for i in range(batch_size):

            current_label = batch_labels.data[i]

            class_total[current_label] += 1
            class_correct[current_label] += batch_correct[i]

    print("Finished Testing in %.6fs" % (time.time() - validation_start))

    print("\nAverage Loss per Sample: %.6f" % (test_loss / n_validation))
    print("Overall Accuracy: %.6f %%" % (sum(class_correct) / float(n_validation)))

    print("\nClass by Class Accuracy: ")
    for i in range(n_classes):
        print('\tAccuracy of class %5s: %2f %%' % (
              class_names[i], 100 * class_correct[i] / float(class_total[i])))

    if save_params:

        assert(isinstance(save_params, str))

        torch.save(net.state_dict(), save_params)


def iterate_batches(inputs, targets, batch_size, shuffle=True):
        """
        Iterate over batches of a given size when the input is simply the raw image itself

        :param inputs: The input images to be classified
        :param targets: The target values (classifications) corresponding to those same images

        :param batch_size: The batch size to be iterated over
        :param shuffle: Variable determines whether or not to shuffle the order of the batches' indices each time

        :return: An iterator over the inputs/targets with each iteration being of size batch
        """
        # Make sure that the input and target arrays are also of the same size
        assert len(inputs) == len(targets)

        indices = np.arange(len(inputs))

        if shuffle:
            np.random.shuffle(indices)

        for start_index in range(0, len(inputs), batch_size):

            batch_indices = indices[start_index:start_index + batch_size]
            batch_indices = torch.from_numpy(batch_indices).type(torch.LongTensor).cuda()

            yield inputs[batch_indices], targets[batch_indices]

def eval_(inputs, targets, n_classes, n_dims, class_names, load_params, batch_size=100):

    assert(len(inputs) == len(targets))

    n_testing = len(targets)

    inputs = torch.from_numpy(inputs).cuda()
    targets = torch.from_numpy(targets).cuda()

    criterion = nn.CrossEntropyLoss()

    net = Net(n_classes, n_dims)
    net.cuda()

    net.load_state_dict(torch.load(load_params))
    net.eval()

    testing_start = time.time()
    test_loss = 0.0

    class_correct = [0] * n_classes
    class_total = [0] * n_classes

    for batch in iterate_batches(inputs, targets, batch_size, shuffle=False):

        batch_inputs, batch_labels = batch

        batch_inputs, batch_labels = Variable(batch_inputs), Variable(batch_labels)
        
        batch_outputs = net(batch_inputs)

        batch_loss = criterion(batch_outputs, batch_labels)
        test_loss += batch_loss.data[0]

        _, batch_predictions = torch.max(batch_outputs.data, 1)
        batch_correct = (batch_predictions == batch_labels.data).squeeze()

        for i in range(batch_size):

            current_label = batch_labels.data[i]

            class_total[current_label] += 1
            class_correct[current_label] += batch_correct[i]

    print("Finished Testing in %.6fs" % (time.time() - testing_start))

    print("\nAverage Loss per Sample: %.6f" % (test_loss / n_testing))
    print("Overall Accuracy: %.6f %%" % (sum(class_correct) / float(n_testing)))

    print("\nClass by Class Accuracy: ")
    for i in range(n_classes):
        print('\tAccuracy of class %5s: %2f %%' % (
              class_names[i], 100 * class_correct[i] / float(class_total[i])))




def imshow(img):

    img = img / 2 + 0.5   

    npimg = img.numpy()

    plt.imshow(np.transpose(npimg, (1, 2, 0)))

if __name__ == "__main__":

    n_classes = 10
    class_names = list(str(i + 1) for i in range(10))

    data_file = "original_images.npy"
    label_file = "original_labels.npy"

    shape = (28, 28)
    n_dims = 1

    gen_dataset.generate_dataset(data_file, label_file, shape, n_dims)
    training_data, test_data, training_labels, test_labels = gen_dataset.load_dataset()
    
    noise_images = gen_images.apply_gaussian_noise(test_data)
    noise_images = gen_images.apply_contrast_filter(noise_images)

    np.save("noised_data.npy", noise_images)
    np.save("noised_images.npy", noise_images.reshape(-1, shape[0] * shape[1]))

    noise_data = np.load("noised_data.npy")
    noise_data = np.array(noise_data, dtype="float32")

    with torch.cuda.device(1):

        train(training_data, test_data, training_labels, test_labels, n_classes, n_dims, class_names, save_params="model_params.pt")

        print("\n===========================================================\n")
        print("Evaluating model on noised test images...\n")

        eval_(noise_data, test_labels, n_classes, n_dims, class_names, "model_params.pt")


