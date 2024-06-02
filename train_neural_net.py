import tqdm
import numpy as np
import matplotlib.pyplot as plt

from dataset.mnist import load_mnist
from two_layer_net import TwoLayerNet


(x_train, y_train), (x_test, y_test) = load_mnist(normalize=True, one_hot_label=True)

train_loss_list = []
train_acc_list = []
test_acc_list = []

iters_num = 20
train_size = x_train.shape[0]
batch_size = 16
learning_rate = 0.1
iter_per_epoch = max(train_size / batch_size, 1)


network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

for index in tqdm.tqdm(range(iters_num)):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    y_batch = y_train[batch_mask]

    grad = network.numerical_gradient(x_batch, y_batch)

    for key in ("W1", "b1", "W2", "b2"):
        network.params[key] -= learning_rate * grad[key]

    loss = network.loss(x_batch, y_batch)
    train_loss_list.append(loss)

    if index % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, y_train)
        test_acc = network.accuracy(x_test, y_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print(f"Train acc, Test acc | {train_acc}, {test_acc}")


fig = plt.figure(figsize=(20, 10))
ax = fig.add_subplot(1, 1, 1)

ax.plot(range(len(train_loss_list)), train_loss_list, color="royalblue")

plt.grid(True)
plt.tight_layout(pad=1.0)
plt.show()
