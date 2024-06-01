import numpy as np


def step_function(x):
    return np.array(x > 0, dtype=np.int8)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def relu(x):
    return np.maximum(x, 0)


def identity_function(x):
    return x


def softmax(x):
    exp_x = np.exp(x - np.max(x))
    exp_sum = np.sum(exp_x)
    y = exp_x / exp_sum

    return y

