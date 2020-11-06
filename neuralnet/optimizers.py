import numpy as np


class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError

    def zeroGrad(self):
        for p in self.params:
            p.weights_grad = 0.
            p.bias_grad = 0.


class GD(Optimizer):
    def __init__(self, params, lr=0.001):
        super().__init__(params)
        self.lr = lr

    def step(self):
        for object in self.params:
            object.weights -= self.lr * object.weights_grad
            object.bias -= self.lr * object.bias_grad

            # class SGD(Optimizer):
            #     def __init__(self, params, lr=0.001):
            #         super().__init__(params)
            #         self.lr = lr
            #
            #     def step():
