
from collections import OrderedDict

from common.layers import *
from common.gradient import numerical_gradient


class MultiLayerNetExtend:
    def __init__(
            self,
            input_size,
            hidden_size_list,
            output_size,
            activation='relu',
            weight_init_std='relu',
            weight_decay_lambda=0,
            use_dropout=False,
            dropout_ratio=0.5,
            use_batchnorm=False,
    ):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size_list = hidden_size_list
        self.hidden_layer_num = len(hidden_size_list)
        self.use_dropout = use_dropout
        self.weight_decay_lambda = weight_decay_lambda
        self.use_batchnorm = use_batchnorm
        self.params = {}

        self._init_weights(weight_init_std)

        activation_layer = {'sigmoid': Sigmoid, 'relu': ReLU}
        self.layers = OrderedDict()
        for index in range(1, self.hidden_layer_num + 1):
            self.layers['Affine' + str(index)] = Affine(self.params['W' + str(index)], self.params['b' + str(index)])

            if self.use_batchnorm:
                self.params['gamma' + str(index)] = np.ones(hidden_size_list[index - 1])
                self.params['beta' + str(index)] = np.zeros(hidden_size_list[index - 1])
                self.layers['BatchNorm' + str(index)] = BatchNormalization(self.params['gamma' + str(index)], self.params['beta' + str(index)])

            self.layers['Activation_function' + str(index)] = activation_layer[activation]()

            if self.use_dropout:
                self.layers['Dropout' + str(index)] = Dropout(dropout_ratio)

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

    def predict(self, x, train_flag=False):
        for key, layer in self.layers.items():
            if ("Dropout" in key) or ("BatchNorm" in key):
                x = layer.forward(x, train_flag)

            else:
                x = layer.forward(x)

        return x

    def loss(self, x, t, train_flag=False):
        y = self.predict(x, train_flag)

        weight_decay = 0
        for index in range(1, self.hidden_layer_num + 2):
            W = self.params['W' + str(index)]
            weight_decay += 0.5 * self.weight_decay_lambda * np.sum(W ** 2)

        return self.last_layer.forward(y, t) + weight_decay

    def accuracy(self, x, t):
        y = self.predict(x, train_flag=False)
        y = np.argmax(y, axis=1)

        if t.ndim != 1:
            t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t, train_flag=True)

        grads = {}
        for index in range(1, self.hidden_layer_num + 2):
            grads['W' + str(index)] = numerical_gradient(loss_W, self.params['W' + str(index)])
            grads['b' + str(index)] = numerical_gradient(loss_W, self.params['b' + str(index)])

            if self.use_batchnorm and (index != self.hidden_layer_num + 1):
                grads['gamma' + str(index)] = numerical_gradient(loss_W, self.params['gamma' + str(index)])
                grads['beta' + str(index)] = numerical_gradient(loss_W, self.params['beta' + str(index)])

        return grads

    def gradient(self, x, t):
        self.loss(x, t, train_flag=True)

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

            if self.use_batchnorm and (index != self.hidden_layer_num + 1):
                grads['gamma' + str(index)] = self.layers['BatchNorm' + str(index)].dgamma
                grads['beta' + str(index)] = self.layers['BatchNorm' + str(index)].dbeta

        return grads
