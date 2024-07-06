import numpy as np
import matplotlib.pyplot as plt

from dataset.mnist import load_mnist
from simple_conv_net import SimpleConvNet
from common.trainer import Trainer


(x_train, y_train), (x_test, y_test) = load_mnist(flatten=False)

x_train, y_train = x_train[:5000], y_train[:5000]
x_test, y_test = x_test[:1000], y_test[:1000]

max_epochs = 20

network = SimpleConvNet(
    input_dim=(1, 28, 28),
    conv_params={"filter_num": 30, "filter_size": 5, "pad": 0, "stride": 1},
    hidden_size=100,
    output_size=10,
    weight_init_std=0.01,
)

trainer = Trainer(
    network,
    x_train, y_train, x_test, y_test,
    max_epochs, 100, "Adam", {"lr": 0.001}, 1000, True
)

trainer.train()

network.save_params("params.pkl")
print("Saved Network Parameters!")

markers = {'train': 'o', 'test': 's'}
x = np.arange(max_epochs)

plt.plot(x, trainer.train_acc_list, marker='o', label='train', markevery=2)
plt.plot(x, trainer.test_acc_list, marker='s', label='test', markevery=2)
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()
