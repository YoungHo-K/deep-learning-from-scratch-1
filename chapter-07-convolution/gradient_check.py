import numpy as np

from simple_conv_net import SimpleConvNet

network = SimpleConvNet(
    input_dim=(1,10, 10),
    conv_params={"filter_num": 10, "filter_size": 3, "pad": 0, "stride": 1},
    hidden_size=10,
    output_size=10,
    weight_init_std=0.01
)

X = np.random.rand(100).reshape((1, 1, 10, 10))
T = np.array([1]).reshape((1, 1))

grad_num = network.numerical_gradient(X, T)
grad = network.gradient(X, T)

for key, val in grad_num.items():
    print(key, np.abs(grad_num[key] - grad[key]).mean())
