
import matplotlib.pyplot as plt
from matplotlib.image import imread

from common.layers import Convolution
from simple_conv_net import SimpleConvNet


def filter_show(filters, show_num=16):
    """
    c.f. https://gist.github.com/aidiary/07d530d5e08011832b12#file-draw_weight-py
    """
    fig = plt.figure()
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

    for index in range(show_num):
        ax = fig.add_subplot(4, 4, index + 1, xticks=[], yticks=[])
        ax.imshow(filters[index, 0], cmap=plt.cm.gray_r, interpolation='nearest')


network = SimpleConvNet(
    input_dim=(1, 28, 28),
    conv_params={'filter_num': 30, 'filter_size': 5, 'pad': 0, 'stride': 1},
    hidden_size=100,
    output_size=10,
    weight_init_std=0.01
)

# 학습된 가중치
network.load_params("params.pkl")

filter_show(network.params['W1'], 16)
print(network.params["W1"].shape)

img = imread('../dataset/cactus_gray.png')
img = img.reshape(1, 1, *img.shape)

fig = plt.figure()

for index in range(16):
    w = network.params['W1'][index]
    b = 0  # network.params['b1'][i]

    w = w.reshape(1, *w.shape)
    # b = b.reshape(1, *b.shape)
    conv_layer = Convolution(w, b)
    out = conv_layer.forward(img)
    out = out.reshape(out.shape[2], out.shape[3])

    ax = fig.add_subplot(4, 4, index + 1, xticks=[], yticks=[])
    ax.imshow(out, cmap=plt.cm.gray_r, interpolation='nearest')

plt.show()