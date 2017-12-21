import numpy as np

class Optimizer(object):
    def get_updates(self, grads, alpha):
        raise NotImplementedError