import numpy as np


def numerical_diff(f, x):
    h = 1e-4
    return (f(x + h) - f(x - h)) / (2 * h)


def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)

    for index in range(x.size):
        tmp_val = x[index]

        x[index] = tmp_val + h
        fxh1 = f(x)

        x[index] = tmp_val - h
        fxh2 = f(x)

        grad[index] = (fxh1 - fxh2) / (2 * h)
        x[index] = tmp_val

    return grad


def gradient_descent(f, init_x, lr=0.1, step_num=100):
    x = init_x

    for index in range(0, step_num):
        grad = numerical_gradient(f, x)
        x -= lr * grad

    return x


def function_2(x):
    return np.sum(x ** 2)


print(gradient_descent(function_2, np.array([-3.0, 4.0])))