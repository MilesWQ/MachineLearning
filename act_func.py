import numpy as np


def sigmod(z):
    return 1 / (1 + np.exp(-z))


def relu(z):
    return 0 if z < 0 else z
