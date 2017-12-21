from vectorflux.optimizers.Optimizer import Optimizer
import numpy as np

class ADAM(Optimizer):
    def __init__(self):
        self.m = 0
        self.v = 0

    def get_updates(self, grads, alpha, beta_1 = 0.9, beta_2 = 0.999, epsilon=1e-8):
        self.m = beta_1 * self.m + (1 - beta_1) * grads
        self.v = beta_2 * self.v + (1 - beta_2) * (grads ** 2)

        m_hat = self.m / (1 - beta_1)
        v_hat = self.v / (1 - beta_2)

        return (alpha / (np.sqrt(v_hat) + epsilon)) * m_hat