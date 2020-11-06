import numpy as np


class Layer():

    def __init__(self):
        self.input = None
        self.output = None

    def forward_propagation(self, input):
        raise NotImplementedError

    def backward_propagation(self, output_error):
        raise NotImplementedError


class LayerDense(Layer):
    # input_size = number of input neurons
    # output_size = number of output neurons
    def __init__(self, input_size, output_size, activation):
        self.data = input_size
        self.weights = np.random.rand(input_size, output_size) - 0.5
        self.bias = np.random.rand(1, output_size) - 0.5
        self.weights_grad = np.zeros((input_size, output_size))
        self.bias_grad = np.zeros((1, output_size))
        self.activation = activation

    # returns output for a given input
    def forward_propagation(self, input_data):
        '''
        function to calculate the forward propagation
        '''

        self.input = input_data
        self.net = np.dot(self.input, self.weights) + self.bias
        self.output = self.activation(self.net)
        return self.output

    # Returns input_error=dE/dX for a given output_error=dE/dY.
    # computes dE/dW, dE/dB for a given output_error=dE/dY. Returns input_error=dE/dX.
    def backward_propagation(self, output_error):
        '''
        function to calculate the backward propagation
        '''
        self.bias_grad = self.activation(
            self.net, derivative=True) * output_error
        input_error = np.dot(self.bias_grad, self.weights.T)
        self.weights_grad = np.dot(self.input.T, self.bias_grad)
        # dBias = output_error

        # update parameters
#         self.weights -= learning_rate * weights_grad
#         self.bias -= learning_rate * self.bias_grad
        return input_error


# class ActivationLayer(Layer):
#    def __init__(self, activation):
#        self.activation = activation
#        #self.activation_prime = activation_prime
#
#    # returns the activated input
#    def forward_propagation(self, input_data):
#        self.input = input_data
#        self.output = self.activation(self.input)
#        return self.output
#
#    # Returns input_error=dE/dX for a given output_error=dE/dY.
#    # learning_rate is not used because there is no "learnable" parameters.
#    def backward_propagation(self, output_error, learning_rate):
#        return self.activation(self.input, derivative=True) * output_error
#
#    # def __init__(self, n_inputs, n_neurons, activation=sigmoid):
#        #self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
#        #self.biases = np.zeros((1, n_neurons))
#        #self.activation = activation
#
#    # def forward(self, inputs):
#        #net = np.dot(inputs, self.weights) + self.biases
#        # return self.activation.function(net)
#
#    # def backward(self, X, y, yhat):
#        #error = y - yhat
#        #adjustment = error * self.activation.jacobian(yhat)
#        #self.weights += np.dot(X.T, adjustment)
