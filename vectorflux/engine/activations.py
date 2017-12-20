import numpy as np


class Activation(object):
    def activate(self, z):
        raise NotImplementedError

    def prime_activation(self, z):
        raise NotImplementedError


class Sigmoid(Activation):
    def activate(self, z):
        """
        Sigmoid
        :param z:
        :return:
        """
        return 1 / (1 + np.exp(np.negative(z)))

    def prime_activation(self, z):
        """
        Sigmoid prime function
        :param z:
        :return:
        """
        sigm = self.activate(z)
        return sigm * (1. - sigm)


class Softmax(Activation):
    def activate(self, z):
        """
        Softmax
        :param z:
        :return:
        """
        e = np.exp(z / 1.0)
        dist = e / np.sum(e)
        return dist

    def prime_activation(self, z):
        """
        Softmax prime activation
        :param z:
        :return:
        """
        return (z * (1 - z))


class NoneActivation(Activation):
    def activate(self, z):
        return z

    def prime_activation(self, z):
        return z


def get_activation(name):
    switcher = {
        "sigmoid": Sigmoid(),
        "softmax": Softmax(),
        None: NoneActivation()
    }
    # Get the functions from the switcher dictionary
    func = switcher.get(name, lambda: NotImplementedError)
    # Return the activation
    return func
