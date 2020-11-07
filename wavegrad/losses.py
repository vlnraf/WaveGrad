"""
Losses Module
"""
import numpy as np


def MSE(y, target, derivative=False):
    """
    Compute the Mean Squared Error (MSE) of the predicted example.

    :param y: the predicted outuput
    :param target: the target output
    :param derivative: True for compute derivative, False for compute MSE
    :return: the MSE or it's derivative.
    """
    if derivative == False:
        return np.mean(np.power(y - target, 2))
    else:
        return 2*(target-y)/y.size
