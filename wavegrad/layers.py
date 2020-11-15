"""
Layer Module
"""
import numpy as np


class Layer():
    """
    Main Layer Object.
    """

    def __init__(self):
        self.input = None
        self.output = None

    def forward_propagation(self, input):
        raise NotImplementedError

    def backward_propagation(self, output_error):
        raise NotImplementedError


class LayerDense(Layer):
    """
    Layer with dense neurons Object.
    """

    def __init__(self, input_size, output_size, activation):
        """
        Create a new layer with a given number of input neurons and output neurons, specify the activation function to use.

        :param input_size: the number of input neurons.
        :param output_size: the number of output neurons.
        :param activation: the activation funciton.
        """
        self.data = input_size
        self.weights = np.random.rand(input_size, output_size) - 0.5
        self.bias = np.random.rand(1, output_size) - 0.5
#         self.weights = np.random.uniform(-0.05, 0.05, size=(input_size, output_size))
#         self.bias = np.random.uniform(-0.05, 0.05, size = (1, output_size))
        self.weights_grad = np.zeros((input_size, output_size))
        self.bias_grad = np.zeros((1, output_size))
        self.activation = activation

    # returns output for a given input
    def forward_propagation(self, input_data):
        """
        Function to calculate the forward propagation

        :param input_data: the input weights to perform the forward propagation.
        """

        self.input = input_data
        self.net = np.dot(self.input, self.weights) + self.bias
        self.output = self.activation(self.net)
        return self.output

    # Returns input_error=dE/dX for a given output_error=dE/dY.
    # computes dE/dW, dE/dB for a given output_error=dE/dY. Returns input_error=dE/dX.
    def backward_propagation(self, output_error):
        """
        Function to calculate the backward propagation

        :param output_error: the output_error of the next layer.
        """
        self.bias_grad = self.activation(self.net, derivative=True) * output_error
        input_error = np.dot(self.bias_grad, self.weights.T)
        self.weights_grad = np.dot(self.input.T, self.bias_grad)
        # dBias = output_error

        # update parameters
#         self.weights -= learning_rate * weights_grad
#         self.bias -= learning_rate * self.bias_grad
        return input_error
