import numpy as np
import matplotlib.pyplot as plt

from dataset.mnist import load_mnist
from common.multi_layer_net import MultiLayerNet
from common.util import shuffle_dataset
from common.trainer import Trainer


(x_train, y_train), (x_test, y_test) = load_mnist(normalize=True)

x_train = x_train[:500]
y_train = y_train[:500]

validation_rate = 0.2
validation_num = int(x_train.shape[0] * validation_rate)

x_train, y_train = shuffle_dataset(x_train, y_train)

x_val = x_train[:validation_num]
y_val = y_train[:validation_num]
x_train = x_train[validation_num:]
y_train = y_train[validation_num:]


def __train(lr, weight_decay, epochs=50):
    network = MultiLayerNet(
        input_size=784,
        hidden_size_list=[100, 100, 100, 100, 100, 100],
        output_size=10,
        weight_decay_lambda=weight_decay,
    )

    trainer = Trainer(network, x_train, y_train, x_val, y_val, epochs, 100, 'sgd', {'lr': lr}, verbose=False)
    trainer.train()

    return trainer.train_acc_list, trainer.test_acc_list


optimization_trial = 100
results_val = {}
results_train = {}
for _ in range(optimization_trial):
    weight_decay = 10 ** np.random.uniform(-8, -4)
    lr = 10 ** np.random.uniform(-6, -2)

    train_acc_list, val_acc_list = __train(lr, weight_decay)
    print(f"Validation Accuracy: {val_acc_list[-1]:.4f} | lr: {lr}, weight decay: {weight_decay}")

    key = f"lr: {lr}, weight decay: {weight_decay}"
    results_val[key] = val_acc_list
    results_train[key] = train_acc_list


print("=========== Hyper-Parameter Optimization Result ===========")
graph_draw_num = 20
col_num = 5
row_num = int(np.ceil(graph_draw_num / col_num))
index = 0

for key, val_acc_list in sorted(results_val.items(), key=lambda x: x[1][-1], reverse=True):
    print(f"Best-{index + 1:,} (val acc: {val_acc_list[-1]:.4f}) | {key}")

    plt.subplot(row_num, col_num, index + 1)
    plt.title("Best-" + str(index + 1))
    plt.ylim(0.0, 1.0)

    if index % 5:
        plt.yticks([])
    plt.xticks([])

    x = np.arange(len(val_acc_list))
    plt.plot(x, val_acc_list)
    plt.plot(x, results_train[key], "--")
    index += 1

    if index >= graph_draw_num:
        break

plt.show()
