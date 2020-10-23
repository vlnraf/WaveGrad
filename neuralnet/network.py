import numpy as np
from tqdm import trange
# from sklearn.metrics import accuracy_score


class Sequential:
    def __init__(self):
        self.layers = []
        self.loss = None
        self.loss_prime = None
        self.train_loss_history = []
        self.val_accuracy_history = []

    # add layer to network
    def add(self, layer):
        self.layers.append(layer)

    # set loss to use
    def use(self, loss, loss_prime):
        self.loss = loss
        self.loss_prime = loss_prime

    # predict output for given input
    def predict(self, input_data):
        # sample dimension first
        samples = len(input_data)
        result = []

        # run network over all samples
        for i in range(samples):
            # forward propagation
            output = input_data[i]
            for layer in self.layers:
                output = layer.forward_propagation(output)

            if (output.shape == (1, 1)):
                # output = output[0][0]
                output = np.max(output)

            result.append(output)

        return result

    # train the network
    def fit(self, x_train, y_train, epochs, learning_rate):
        # sample dimension first
        samples = len(x_train)
        acc = 0

        # training loop
        for i in (t := trange(epochs)):
            err = 0
            for j in range(samples):
                # forward propagation
                output = x_train[j]
                output = output.reshape(1, output.shape[0])
                for layer in self.layers:
                    output = layer.forward_propagation(output)

                # compute loss (for display purpose only)
                err += self.loss(y_train[j], output)

                # backward propagation
                error = self.loss_prime(y_train[j], output)
                for layer in reversed(self.layers):
                    error = layer.backward_propagation(error, learning_rate)

            # calculate average error on all samples
            err /= samples
            # append error to plot it later
            self.train_loss_history.append(err)
            # do prediction, cal
            pred = self.predict(x_train)
            acc = self.accuracy(y_train, pred)
            self.val_accuracy_history.append(acc)
            t.set_description('epoch %d/%d   error=%.2f    accuracy=%.2f' %
                              (i+1, epochs, err, acc))

    def accuracy(self, y_train, pred):
        # acc = np.sum(y_pred == y_train) / len(y_train)
        acc = np.sum((np.round(pred) == y_train)) / len(y_train)
        # acc = accuracy_score(y_train, np.round(pred))
        return acc
