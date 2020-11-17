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

class L2Regularizer(Regularizer):
    def __init__(self, l2=0):
        """
        Initizalize the lamda component that will penalize weights

        :params l2: the regularization parameter
        """
        self.l2 = l2

    def call(self):


