from vectorflux.optimizers.Optimizer import Optimizer


class SGD(Optimizer):
    def get_updates(self, grads, alpha):
        return alpha * grads