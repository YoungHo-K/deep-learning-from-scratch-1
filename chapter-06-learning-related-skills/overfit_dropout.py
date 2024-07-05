import numpy as np
import matplotlib.pyplot as plt

from dataset.mnist import load_mnist
from common.trainer import Trainer
from common.multi_layer_net_extend import MultiLayerNetExtend

(x_train, y_train), (x_test, y_test) = load_mnist(normalize=True)

x_train = x_train[:300]
y_train = y_train[:300]

use_dropout = True
dropout_ratio = 0.2

network = MultiLayerNetExtend(
    input_size=784,
    hidden_size_list=[100, 100, 100, 100, 100, 100],
    output_size=10,
    use_dropout=use_dropout,
    dropout_ratio=dropout_ratio,
)

trainer = Trainer(
    network=network,
    x_train=x_train,
    y_train=y_train,
    x_test=x_test,
    y_test=y_test,
    epochs=301,
    mini_batch_size=100,
    optimizer='sgd',
    optimizer_params={'lr': 0.01},
    verbose=True,
)

trainer.train()

train_acc_list, test_acc_list = trainer.train_acc_list, trainer.test_acc_list

markers = {'train': 'o', 'test': 's'}
x = np.arange(len(train_acc_list))

plt.plot(x, train_acc_list, marker='o', label='train', markevery=10)
plt.plot(x, test_acc_list, marker='s', label='test', markevery=10)
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()
