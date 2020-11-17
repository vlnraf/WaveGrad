"""
Optimizer Module
"""
import numpy as np


class Optimizer:
    def __init__(self, params):
        """
        Abstract base class for the optimizer.

        :param params: the layers of the network.
        """
        self.params = params

    def step(self):
        """
        Step pass for the optimizer.
        """
        raise NotImplementedError

    def zeroGrad(self):
        """
        Inizialize pass for the optimizer.
        """
        for p in self.params:
            p.weights_grad = 0.
            p.bias_grad = 0.


class DiscreteOptimizer(Optimizer):
    def __init__(self, params, lr=0.001):
        """
        Abstract class for optimizer that are Discrete.

        :param lr: the learning rate to use
        """
        super().__init__(params)
        self.lr = lr


class GD(DiscreteOptimizer):
    def __init__(self, params, lr=0.001, momentum=0.0):
        """
        The Gradient Descent optimizer (GD)

        :param lr: the learning rate to use
        :param momentum: Value variable between 0 and 1. Determines the velocity of the gradient descent
        """
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.v_weights = [np.zeros_like(t.weights) for t in self.params]
        self.v_bias = [np.zeros_like(t.bias) for t in self.params]

    def step(self):
        """
        Step pass for the optimizer
        """
        for i, object in enumerate(self.params):
            self.v_weights[i] = self.momentum * self.v_weights[i]
            self.v_bias[i] = self.momentum * self.v_bias[i]
            object.weights_grad = -self.lr * object.weights_grad + self.v_weights[i]
            object.bias_grad = -self.lr * object.bias_grad + self.v_bias[i]
            object.weights = object.weights + object.weights_grad
            object.bias = object.bias + object.bias_grad
            self.v_weights[i] = object.weights_grad
            self.v_bias[i] = object.bias_grad


class StochasticOptimizer(Optimizer):
    def __init__(self, params, lr=0.001):
        """
        Abstract class for optimizer that are Stochastic.

        :param lr: the learning rate to use
        """
        super().__init__(params)
        self.lr = lr


class SGD(StochasticOptimizer):
    def __init__(self, params, lr=0.001, momentum=0.0):
        """
        The Stochastic Gradient Descent optimzer (SGD)

        :param lr: the learning rate to use
        :param momentum: Value variable between 0 and 1. Determines the velocity of the gradient descent
        """
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.v_weights = [np.zeros_like(t.weights) for t in self.params]
        self.v_bias = [np.zeros_like(t.bias) for t in self.params]

    def step(self):
        """
        Step pass for the optimizer
        """
        for i, object in enumerate(self.params):
            self.v_weights[i] = self.momentum * self.v_weights[i]
            self.v_bias[i] = self.momentum * self.v_bias[i]
            object.weights_grad = -self.lr * object.weights_grad + self.v_weights[i]
            object.bias_grad = -self.lr * object.bias_grad + self.v_bias[i]
            object.weights = object.weights + object.weights_grad
            object.bias = object.bias + object.bias_grad
            self.v_weights[i] = object.weights_grad
            self.v_bias[i] = object.bias_grad
