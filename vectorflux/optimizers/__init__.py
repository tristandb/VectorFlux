from .Momentum import *
from .ADAM import *
from .SGD import *


def get_optimizer(name):
    switcher = {
        "SGD": SGD(),
        "ADAM": ADAM(),
        "Momentum": Momentum(),
        None: SGD()
    }
    # Get the functions from the switcher dictionary
    func = switcher.get(name, lambda: NotImplementedError)
    # Return the activation
    return func
