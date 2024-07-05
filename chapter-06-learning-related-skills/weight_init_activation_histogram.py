import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


x = np.random.randn(1000, 100)
node_num = 100
hidden_layer_size = 5
activations = {}

for index in range(hidden_layer_size):
    if index != 0:
        x = activations[index - 1]

    w = np.random.randn(node_num, node_num) / np.sqrt(node_num)
    a = np.dot(x, w)
    z = sigmoid(a)

    activations[index] = z

for index, a in activations.items():
    plt.subplot(1, len(activations), index + 1)
    plt.title(str(index + 1) + "-layer")
    plt.hist(a.flatten(), 30, range=(0, 1))

plt.tight_layout(pad=1.0)
plt.show()
