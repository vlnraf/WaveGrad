"""
Activation Module
"""
import numpy as np


def relu(x, derivative=False):
    """
    The ReLu activation function is to performs a threshold
    operation to each input element where values less 
    than zero are set to zero.

    :param x: an array.
    :param derivative: compute derivative if =True, don't compute if =False.
    """
    if derivative == False:
        return np.maximum(0, x)
    else:
        x[x <= 0] = 0
        x[x > 0] = 1
        return x


# def relu_prime(x):
    #x[x<=0] = 0
    #x[x>0] = 1
    # return x


def tanh(x, derivative=False):
    """
    The Tanh activation function is computed on each element of the input array.

    :param x: an array.
    :param derivative: compute derivative if =True, don't compute if =False.
    """
    if derivative == False:
        return np.tanh(x)
    else:
        return 1-np.tanh(x)**2


# def tanh_prime(x):
    # return 1-np.tanh(x)**2


def sigmoid(x, derivative=False):
    """
    The sigmoid function takes in real numbers in any range and 
    squashes it to a real-valued output between 0 and 1.

    :param x: an array.
    :param derivative: compute derivative if =True, don't compute if =False.
    """
    if derivative == False:
        return 1.0/(1.0+np.exp(-x))
    else:
        return sigmoid(x) * (1 - sigmoid(x))
