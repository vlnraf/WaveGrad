import numpy as np

# loss function and its derivative


def mse(y_true, y_pred, derivative=False):
    if derivative == False:
        return np.mean(np.power(y_true - y_pred, 2))
    else:
        return 2*(y_pred-y_true)/y_true.size

# def mse_prime(y_true, y_pred):
#     return 2*(y_pred-y_true)/y_true.size
