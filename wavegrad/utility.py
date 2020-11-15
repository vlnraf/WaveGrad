import numpy as np


def train_test_split(X_train, y_train, validation_split):
    size = int(len(X_train) * (validation_split * 100) / 100)
    X = X_train[size:]
    y = y_train[size:]
    X_val = X_train[:size]
    y_val = y_train[:size]

    return X, y, X_val, y_val
