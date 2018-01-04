from vectorflux.optimizers.Optimizer import Optimizer


class Momentum(Optimizer):
    def __init__(self):
        self.previous_gradients = 0

    def get_updates(self, grads, alpha, momentum=0.81):
        self.previous_gradients = momentum * self.previous_gradients + grads * alpha
        return self.previous_gradients
