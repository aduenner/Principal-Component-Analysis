import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision

import numpy as np
import matplotlib.pyplot as plt

import read
import time

import cv2

CONST_NUM_CLASSES = 10
CONST_CLASSES = ["plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

class Net(nn.Module):   

    def __init__(self):

        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel

        self.conv1 = nn.Conv2d(3, 12, 5, padding=2)
        self.conv2 = nn.Conv2d(12, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, padding=1)

        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(512, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):

        # Max pooling over a (3, 3) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (3, 3), stride=2, padding=1)
              # If the size is a square you can only specify a single number
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


def main(train_epochs=30, learning_rate=0.001, batch_size=100):

    print("Preparing Dataset...")

    training_data, testing_data, training_labels, testing_labels = \
    read.generate_training_sets(0.8, None, True)

    n_training = len(training_data)
    n_testing = len(testing_data)

    c_training_data = torch.from_numpy(training_data).cuda()
    c_testing_data = torch.from_numpy(testing_data).cuda()

    c_training_labels = torch.from_numpy(training_labels).cuda()
    c_testing_labels = torch.from_numpy(testing_labels).cuda()

    print("Initializing Network...")

    net = Net()
    net.cuda()

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

        for batch in iterate_batches(c_training_data, c_training_labels,
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

    testing_start = time.time()
    test_loss = 0.0

    class_correct = [0] * 10
    class_total = [0] * 10

    for batch in iterate_batches(c_testing_data, c_testing_labels,
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

    print("Finished Testing in %.6fs" % (time.time() - testing_start))

    print("\nAverage Loss per Sample: %.6f" % (test_loss / n_testing))
    print("Overall Accuracy: %.6f %%" % (sum(class_correct) / float(n_testing)))

    print("\nClass by Class Accuracy: ")
    for i in range(10):
        print('\tAccuracy of class %5s: %2f %%' % (
              CONST_CLASSES[i], 100 * class_correct[i] / float(class_total[i])))


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
            batch_indices = torch.from_numpy(batch_indices).cuda()

            yield inputs[batch_indices], targets[batch_indices]

def imshow(img):

    img = img / 2 + 0.5   

    npimg = img.numpy()

    plt.imshow(np.transpose(npimg, (1, 2, 0)))

if __name__ == "__main__":

    main();

    # with torch.cuda.device(1):
    #     main()