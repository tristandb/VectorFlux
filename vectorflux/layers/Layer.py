class Layer(object):
    def __init__(self):
        self.kernel = None
        pass

    def call(self, input):
        raise NotImplementedError

    def back_propagate(self, delta, value, previousvalue, alpha, output):
        raise NotImplementedError