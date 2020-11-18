"""
Regularizer Moduel
"""
import numpy as np

class Regularizer:
    """
    Abstract base class for the Regularizer.
    """

    def call(self):
        """
        Write it!!!!!!!!
        """
        pass

    def gradient(self):
        """
        Write it!!!!!!!!
        """
        pass

class L2(Regularizer):
    def __init__(self, l2=0):
        """
        Initizalize the lamda component that will penalize weights

        :params l2: the regularization parameter
        """
        self.l2 = l2

    def call(self, params):
        """
        Calculate the weights decay.
        :params layers: the layers of the neural network
        """
        loss_penalty = 0
        for layer in params:
            concat = np.concatenate((layer.weights, layer.bias))
            loss_penalty += np.square(concat).mean()

        loss_penalty = 0.5 * self.l2 * loss_penalty
            
        return loss_penalty

    def gradient(self, params):
        """
        Calculate the weights decay.

        :params layers: the layers of the neural network
        """
        loss_penalty_grad = 0
        for layer in params:
            concat = np.concatenate((layer.weights, layer.bias))
            loss_penalty_grad += np.square(concat).mean()

        loss_penalty_grad = 0.5 * self.l2 * loss_penalty_grad

        return loss_penalty_grad
