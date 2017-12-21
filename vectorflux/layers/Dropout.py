from vectorflux.layers.Layer import Layer
import numpy as np


class Dropout(Layer):
    def __init__(self, rate, input_shape):
        super().__init__()
        self.input_shape = input_shape
        self.rate = min(1., max(0., rate))

    def call(self, input, evaluate=False):
        if evaluate:
            return input

        return input * np.random.binomial([np.ones((len(input), self.input_shape))], 1 - self.rate)[0] \
               * (1.0 / (1 - self.rate))

    def back_propagate(self, delta, value, previousvalue, alpha=0.1, output=False):
        return delta