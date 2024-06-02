import numpy as np


def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)

    it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        index = it.multi_index
        tmp_val = x[index]

        x[index] = tmp_val + h
        fxh1 = f(x)

        x[index] = tmp_val - h
        fxh2 = f(x)

        grad[index] = (fxh1 - fxh2) / (2 * h)
        x[index] = tmp_val
        it.iternext()

    return grad
