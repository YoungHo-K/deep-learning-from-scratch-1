import numpy as np
import matplotlib.pyplot as plt

from dataset.mnist import load_mnist
from common.multi_layer_net_extend import MultiLayerNetExtend
from common.optimizer import SGD, Adam

(x_train, y_train), (x_test, y_test) = load_mnist(normalize=True)

x_train = x_train[:1000]
y_train = y_train[:1000]

max_epochs = 20
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.01


def __train(weight_init_std):
    bn_network = MultiLayerNetExtend(
        input_size=784,
        hidden_size_list=[100, 100, 100, 100, 100],
        output_size=10,
        weight_init_std=weight_init_std,
        use_batchnorm=True
    )

    network = MultiLayerNetExtend(
        input_size=784,
        hidden_size_list=[100, 100, 100, 100, 100],
        output_size=10,
        weight_init_std=weight_init_std,
        use_batchnorm=False
    )

    optimizer = SGD(lr=learning_rate)

    train_acc_list = []
    bn_train_acc_list = []

    iter_per_epoch = max(train_size / batch_size, 1)
    epoch = 0
    for index in range(1000000000):
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        y_batch = y_train[batch_mask]

        for _network in (bn_network, network):
            grads = _network.gradient(x_batch, y_batch)
            optimizer.update(_network.params, grads)

        if index % iter_per_epoch == 0:
            train_acc = network.accuracy(x_train, y_train)
            bn_train_acc = bn_network.accuracy(x_train, y_train)
            train_acc_list.append(train_acc)
            bn_train_acc_list.append(bn_train_acc)

            print(f"Epoch: {epoch:,} | Accuracy: {train_acc:.4f} - BatchNorm Accuracy: {bn_train_acc:.4f}")

            epoch += 1
            if epoch >= max_epochs:
                break

    return train_acc_list, bn_train_acc_list


weight_scale_list = np.logspace(0, -4, num=16)
x = np.arange(max_epochs)

for index, w in enumerate(weight_scale_list):
    print("============== " + str(index + 1) + "/16" + " ==============")
    train_acc_list, bn_train_acc_list = __train(w)

    plt.subplot(4, 4, index + 1)
    plt.title("W:" + str(w))

    if index == 15:
        plt.plot(x, bn_train_acc_list, label='Batch Normalization', markevery=2)
        plt.plot(x, train_acc_list, linestyle="--", label='Normal(without BatchNorm)', markevery=2)

    else:
        plt.plot(x, bn_train_acc_list, markevery=2)
        plt.plot(x, train_acc_list, linestyle="--", markevery=2)

    plt.ylim(0, 1.0)
    if index % 4:
        plt.yticks([])
    else:
        plt.ylabel("accuracy")

    if index < 12:
        plt.xticks([])
    else:
        plt.xlabel("epochs")

    plt.legend(loc='lower right')

plt.show()