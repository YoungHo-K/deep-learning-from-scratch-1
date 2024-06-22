# coding: utf-8

import numpy as np
from collections import OrderedDict
from common.layers import *
from common.gradient import numerical_gradient


class MultiLayerNet:
    """완전연결 다층 신경망

    Parameters
    ----------
    input_size : 입력 크기（MNIST의 경우엔 784）
    hidden_size_list : 각 은닉층의 뉴런 수를 담은 리스트（e.g. [100, 100, 100]）
    output_size : 출력 크기（MNIST의 경우엔 10）
    activation : 활성화 함수 - 'relu' 혹은 'sigmoid'
    weight_init_std : 가중치의 표준편차 지정（e.g. 0.01）
        'relu'나 'he'로 지정하면 'He 초깃값'으로 설정
        'sigmoid'나 'xavier'로 지정하면 'Xavier 초깃값'으로 설정
    weight_decay_lambda : 가중치 감소(L2 법칙)의 세기
    """

    def __init__(
            self,
            input_size,
            hidden_size_list,
            output_size,
            activation='relu',
            weight_init_std='relu',
            weight_decay_lambda=0
    ):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size_list = hidden_size_list
        self.hidden_layer_num = len(hidden_size_list)
        self.weight_decay_lambda = weight_decay_lambda
        self.params = {}

        self._init_weights(weight_init_std)

        activation_layer = {'sigmoid': Sigmoid, 'relu': ReLU}
        self.layers = OrderedDict()
        for index in range(1, self.hidden_layer_num + 1):
            self.layers['Affine' + str(index)] = Affine(self.params['W' + str(index)], self.params['b' + str(index)])
            self.layers['Activation_function' + str(index)] = activation_layer[activation]()

        index = self.hidden_layer_num + 1
        self.layers['Affine' + str(index)] = Affine(self.params['W' + str(index)], self.params['b' + str(index)])
        self.last_layer = SoftmaxWithLoss()

    def _init_weights(self, weight_init_std):
        all_size_list = [self.input_size] + self.hidden_size_list + [self.output_size]
        for index in range(1, len(all_size_list)):
            scale = weight_init_std
            if str(weight_init_std).lower() in ('relu', 'he'):
                scale = np.sqrt(2.0 / all_size_list[index - 1])  # ReLU를 사용할 때의 권장 초깃값

            elif str(weight_init_std).lower() in ('sigmoid', 'xavier'):
                scale = np.sqrt(1.0 / all_size_list[index - 1])  # sigmoid를 사용할 때의 권장 초깃값

            self.params['W' + str(index)] = scale * np.random.randn(all_size_list[index - 1], all_size_list[index])
            self.params['b' + str(index)] = np.zeros(all_size_list[index])

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)

        return x

    def loss(self, x, t):
        y = self.predict(x)

        weight_decay = 0
        for index in range(1, self.hidden_layer_num + 2):
            W = self.params['W' + str(index)]
            weight_decay += 0.5 * self.weight_decay_lambda * np.sum(W ** 2)

        return self.last_layer.forward(y, t) + weight_decay

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)

        if t.ndim != 1:
            t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)

        grads = {}
        for index in range(1, self.hidden_layer_num + 2):
            grads['W' + str(index)] = numerical_gradient(loss_W, self.params['W' + str(index)])
            grads['b' + str(index)] = numerical_gradient(loss_W, self.params['b' + str(index)])

        return grads

    def gradient(self, x, t):
        self.loss(x, t)

        dout = 1
        dout = self.last_layer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        grads = {}
        for index in range(1, self.hidden_layer_num + 2):
            grads['W' + str(index)] = (
                    self.layers['Affine' + str(index)].dW
                    + self.weight_decay_lambda * self.layers['Affine' + str(index)].W
            )
            grads['b' + str(index)] = self.layers['Affine' + str(index)].db

        return grads
