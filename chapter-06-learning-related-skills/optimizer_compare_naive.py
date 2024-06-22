from collections import OrderedDict

import matplotlib.pyplot as plt

from common.optimizer import *


def f(x, y):
    return x ** 2 / 20.0 + y ** 2


def df(x, y):
    return x / 10.0, 2.0 * y


init_position = (-7.0, 2.0)
params = {
    "x": init_position[0],
    "y": init_position[1],
}

grads = {
    "x": 0,
    "y": 0
}

optimizers = OrderedDict()
optimizers["SGD"] = SGD(lr=0.95)
optimizers["Momentum"] = Momentum(lr=0.1)
optimizers["AdaGrad"] = AdaGrad(lr=1.5)
optimizers["Adam"] = Adam(lr=0.3)


for index, key in enumerate(optimizers):
    optimizer = optimizers[key]

    x_history = []
    y_history = []

    params["x"], params["y"] = init_position[0], init_position[1]
    for iteration in range(0, 30):
        x_history.append(params["x"])
        y_history.append(params["y"])

        grads["x"], grads["y"] = df(params["x"], params["y"])
        optimizer.update(params, grads)

    x = np.arange(-10, 10, 0.01)
    y = np.arange(-5, 5, 0.01)

    X, Y = np.meshgrid(x, y)
    Z = f(X, Y)

    mask = Z > 7
    Z[mask] = 0

    plt.subplot(2, 2, index + 1)
    plt.plot(x_history, y_history, 'o-', color="red", ms=3)
    plt.contour(X, Y, Z)
    plt.ylim(-10, 10)
    plt.xlim(-10, 10)
    plt.plot(0, 0, '+')

    plt.spring()
    plt.title(key)
    plt.xlabel("x")
    plt.ylabel("y")

plt.tight_layout(pad=1.0)
plt.show()
