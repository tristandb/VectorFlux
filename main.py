"""
Implements a Neural Network

"""

import numpy as np
from neuralnetwork import NeuralNetwork
from mnist import read, show, normalize

# Import training and test set
train = list(read('train'))
test = list(read('test'))

# Split up MNIST dataset into training, test and validation set
np.random.shuffle(test)
test_split = int(0.99 * len(test))
test, validation = test[:test_split], test[test_split:]

print("Train size: {}".format(len(train)))
print("Test size: {}".format(len(test)))
print("Validation size: {}".format(len(validation)))

# Normalization for values
validation_x, validation_y = normalize(validation)
test_x, test_y = normalize(test)
train_x, train_y = normalize(train)

neural_network = NeuralNetwork(784)
neural_network.train(training_set_inputs=train_x, training_set_outputs=train_y, iterations=100000, test_interval=1000, test_set_inputs=validation_x, test_set_outputs=validation_y)

neural_network.test(test_set_inputs=validation_x, test_set_outputs=validation_y)

