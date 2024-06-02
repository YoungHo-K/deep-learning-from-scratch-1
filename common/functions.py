import numpy as np


def relu(x):
    return np.maximum(x, 0)


def softmax(x):
    x_exp = np.exp(x - np.max(x))
    exp_sum = np.sum(x_exp)

    return x_exp / exp_sum


def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]
    delta = 1e-7

    return -np.sum(t * np.log(y + delta)) / batch_size

