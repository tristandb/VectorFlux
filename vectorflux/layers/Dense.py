import numpy as np
from vectorflux.layers.Layer import Layer
from vectorflux.engine.activations import get_activation


class Dense(Layer):
    """
    Regular densely-connected NN layer.

    output = activation(dot(input, kernel) + bias)

    """

    def __init__(self, units, input_shape, activation=None, use_bias=False):
        super().__init__()
        self.units = units
        self.kernel = 2 * np.random.random((input_shape, units)) - 1
        self.use_bias = use_bias
        if self.use_bias is not False:
            self.bias = b = np.zeros(units)
        self.activation = get_activation(activation)

    def call(self, input, evaluate = False):
        layer_output = np.dot(input, self.kernel)
        if self.use_bias is not False:
            pass
            layer_output += self.bias
        if self.activation is not None:
            layer_output = self.activation.activate(layer_output)
        return layer_output

    def back_propagate(self, delta, value, previousvalue, alpha=0.1, output=False):
        if self.activation is not None:
            delta *= self.activation.prime_activation(value)

        # Calculate delta for next iteration
        return_delta = delta.dot(self.kernel.T).copy()

        self.kernel -= alpha * previousvalue.T.dot(delta)

        return return_delta
