import numpy as np
import matplotlib.pyplot as plt

from simple_conv_net import SimpleConvNet


def filter_show(filters, nx=8):
    """
    c.f. https://gist.github.com/aidiary/07d530d5e08011832b12#file-draw_weight-py
    """
    FN, C, FH, FW = filters.shape
    ny = int(np.ceil(FN / nx))

    fig = plt.figure()
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

    for index in range(FN):
        ax = fig.add_subplot(ny, nx, index + 1, xticks=[], yticks=[])
        ax.imshow(filters[index, 0], cmap=plt.cm.gray_r, interpolation='nearest')

    plt.show()


network = SimpleConvNet()

# Random weights
filter_show(network.params['W1'])

# Learned weights
network.load_params("params.pkl")
filter_show(network.params['W1'])
