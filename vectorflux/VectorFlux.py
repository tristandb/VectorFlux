import numpy as np
from tqdm import trange


def get_mini_batches(x, y, mini_batch_size):
    n = len(x)

    s = np.arange(x.shape[0])
    np.random.shuffle(s)

    x_train_shuffle = x[s]
    y_train_shuffle = y[s]

    mini_batches_x = [
        x_train_shuffle[k:k + mini_batch_size]
        for k in range(0, n, mini_batch_size)
    ]

    mini_batches_y = [
        y_train_shuffle[k:k + mini_batch_size]
        for k in range(0, n, mini_batch_size)
    ]
    return mini_batches_x, mini_batches_y


class VectorFlux():
    """
    VectorFlux (Name is derived from TensorFlow) implements a neural network that resembles Keras.
    """

    def __init__(self):
        """
        Initialize Neural Network by setting initial values for the layers.
        :param sizes:
        """
        self.num_layers = 0
        self.layers = []

    def add(self, layer):
        """
        Add a new layer.
        :param layer:
        :return:
        """
        self.num_layers += 1
        self.layers.append(layer)

    def train(self, x_train, y_train, epochs, alpha=0.1, mini_batch_size=100, x_test=None, y_test=None):
        """
        Train the neural network.
        :param x_train:
        :param y_train:
        :param epochs:
        :param alpha:
        :param mini_batch_size:
        :param x_test:
        :param y_test:
        :return:
        """
        for epoch in range(epochs):
            # Get mini batches
            mini_batches_x, mini_batches_y = get_mini_batches(x_train, y_train, mini_batch_size)

            # Run iteration for mini batches
            t = trange(len(mini_batches_x))
            t.unit = " minibatches"
            t.ncols = 100
            t.desc = "Epoch {}".format(epoch + 1)
            for i in t:
                self.update_batch(mini_batches_x[i], mini_batches_y[i], alpha=alpha)
                pass

            # Evaluate
            if epoch % 1 == 0:
                self.evaluate(x_test, y_test)

    def evaluate(self, x_test, y_test):
        layer_output = x_test
        layer_output_list = [layer_output]
        for layer in self.layers:
            layer_output = layer.call(layer_output, evaluate=True)
            layer_output_list.append(layer_output)
        print("Perf: {}".format(np.sum(np.argmax(y_test, axis=1) == np.argmax(layer_output, axis=1)) / len(x_test)))

    def update_batch(self, x_train, y_train, alpha):
        # Feed forward
        layer_output = x_train
        layer_output_list = [layer_output]
        for layer in self.layers:
            layer_output = layer.call(layer_output, )
            layer_output_list.append(layer_output)

        # Determine gradient
        delta = (layer_output - y_train)

        # Back propagate
        for i in range(len(layer_output_list) - 1, 0, -1):
            delta = self.layers[i - 1].back_propagate(delta, layer_output_list[i],
                                                      layer_output_list[i - 1], alpha=alpha)
