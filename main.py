"""
Implements a Neural Network

"""
from vectorflux import VectorFlux
from mnist import read, show, normalize

from vectorflux.layers import Dense
from vectorflux.layers.Dropout import Dropout

train = list(read('train'))
test = list(read('test'))

print("Train size: {}".format(len(train)))
print("Test size: {}".format(len(test)))

# Normalization for values
test_x, test_y = normalize(test)
train_x, train_y = normalize(train)

vf = VectorFlux()
vf.add(Dense(800, activation='sigmoid', input_shape=784, optimizer='Momentum'))
vf.add(Dropout(0.5, input_shape=800))
vf.add(Dense(800, activation='sigmoid', input_shape=800, optimizer='ADAM'))
vf.add(Dense(10, activation='sigmoid', input_shape=800))

vf.train(x_train = train_x, y_train = train_y, x_test=test_x, y_test = test_y, epochs=100000, alpha=0.001, mini_batch_size=100)

