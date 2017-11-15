"""
Implements a Neural Network

"""

import numpy as np
from mnist import read, show

# Import training and test set
train = list(read('train'))
test = list(read('test'))

# Split up MNIST dataset into training, test and validation set
np.random.shuffle(test)
test_split = int(0.9 * len(test))
test, validation = test[:test_split], test[test_split:]

print("Train size: {}".format(len(train)))
print("Test size: {}".format(len(test)))
print("Validation size: {}".format(len(validation)))