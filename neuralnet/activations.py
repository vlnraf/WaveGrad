import numpy as np

def sign(Z):
    return (Z >= 0.5).astype(int)


def relu(x):
    '''
    The ReLu activation function is to performs a threshold
    operation to each input element where values less 
    than zero are set to zero.
    '''
    return np.maximum(0,x)

def relu_prime(x):
    x[x<=0] = 0
    x[x>0] = 1
    return x


def tanh(x):
    return np.tanh(x)

def tanh_prime( x):
    return 1-np.tanh(x)**2


def sigmoid(x):
    '''
    The sigmoid function takes in real numbers in any range and 
    squashes it to a real-valued output between 0 and 1.
    '''
    return 1.0/(1.0+np.exp(-x))

def sigmoid_prime(x):
    x = sigmoid(x) 
    return x * (1 - x)

