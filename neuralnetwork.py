from numpy import exp, array, random, dot
import numpy as np
import math

class NeuralNetwork():
    def __init__(self, input_size):
        random.seed(1)
        self.synaptic_weights = 2 * random.random((input_size, 1)) - 1

    def __sigmoid(self, x):
        return 1 / (1 + exp(-x))

    def __sigmoid_derivative(self, x):
        return x * (1 - x)

    def train(self, training_set_inputs, training_set_outputs, iterations, test_set_inputs, test_set_outputs, test_interval):
        for iteration in range(iterations):
            if (iteration % test_interval == 0):
                print("Iteration {} of {}".format(iteration, iterations))
                self.test(test_set_inputs, test_set_outputs)
            output = self.think(training_set_inputs)

            error = (training_set_outputs - output) ** 2

            adjustment = dot(training_set_inputs.T, error * self.__sigmoid_derivative(output))

            self.synaptic_weights += adjustment

    def test(self, test_set_inputs, test_set_outputs):
        output = self.think(test_set_inputs)
        print(np.mean((output - test_set_outputs) ** 2))

    def think(self, inputs):
        return self.__sigmoid(dot(inputs, self.synaptic_weights))
