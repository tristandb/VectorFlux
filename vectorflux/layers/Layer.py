class Layer(object):
    def __init__(self):
        self.kernel = None
        pass

    def call(self, input, evaluate = False):
        raise NotImplementedError

    def back_propagate(self, delta, value, previousvalue, alpha=0.1, output=False):
        raise NotImplementedError