"""
Losses Module
"""
import numpy as np


def MSE(target, y, derivative=False):
    """
    Compute the Mean Squared Error (MSE) of the predicted example.

    :param y: the predicted outuput
    :param target: the target output
    :param derivative: True for compute derivative, False for compute MSE
    :return: the MSE or it's derivative.
    """
    if derivative == False:
        return 0.5 * np.sum(np.square(target - y))
    else:
        return 2*(y - target)/y.size


def MAE(target, y, derivative=False):
    """
    Compute the Mean Absolute Error (MAE) of the predicted example.

    :param y: the predicted outuput
    :param target: the target output
    :param derivative: True for compute derivative, False for compute MAE
    :return: the MAE or it's derivative.
    """
    if derivative == False:
        return np.mean(np.abs(target - y))
    else:
        return np.sign(y - target)
