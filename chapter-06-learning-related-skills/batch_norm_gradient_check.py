import numpy as np

from dataset.mnist import load_mnist
from common.multi_layer_net_extend import MultiLayerNetExtend

(x_train, y_train), (x_test, y_test) = load_mnist(normalize=True, one_hot_label=True)

network = MultiLayerNetExtend(
    input_size=784,
    hidden_size_list=[100, 100],
    output_size=10,
    use_batchnorm=True
)

x_batch = x_train[:1]
y_batch = y_train[:1]

grad_backprop = network.gradient(x_batch, y_batch)
grad_numerical = network.numerical_gradient(x_batch, y_batch)

for key in grad_numerical.keys():
    diff = np.average(np.abs(grad_backprop[key] - grad_numerical[key]))
    print(f"{key}: {diff:.6f}")
