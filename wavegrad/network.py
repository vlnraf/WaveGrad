"""
Network Module
"""
import numpy as np
from tqdm import trange
from .optimizers import DiscreteOptimizer, StochasticOptimizer


class Sequential:
    """
    Sequential Object
    """

    def __init__(self):
        """
        Create a Sequential Object that contain the :py:class:`.Layer`
        """
        self.layers = []
        self.loss = None
        self.loss_prime = None
        self.train_loss_history = []
        self.val_accuracy_history = []

    # add layer to network
    def add(self, layer):
        """
        Add a layer to the neural network.

        :param layer: the :py:class:`.Layer` to add.
        """
        self.layers.append(layer)

    # set loss to use
    def use(self, loss):
        """
        Add the loss function to the neural netwok.

        :param loss: the :py:func:`.MSE` to add.
        """
        self.loss = loss

    # predict output for given input
    def predict(self, input_data):
        """
        Insert the input data to the neural network to make a prediction

        :param input_data: the input. Its size must match the ``input_size`` of the neural network.
        :return: the output of the neural network.
        """
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
    def fit(self, x_train, y_train, epochs, optimizer, batch_size=None):
        """
        Insert the X_train y_train the optimizer to use and the epochs to fit the neural network with your datas, you can add the batch size if you need it.

        :param x_train: the input data
        :param y_train: the target data
        :param epoch: the number of epochs to permorm the train
        :param optimizer: the tipe of :py:class:`.Optimizer` to use.
        """
        if issubclass(type(optimizer), DiscreteOptimizer):
            # sample dimension first
            samples = len(x_train)
            acc = 0

            # training loop
            for i in (t := trange(epochs)):
                err = 0
                for j in range(samples):
                    # forward propagation
                    optimizer.zeroGrad()
                    output = x_train[j]
                    output = output.reshape(1, output.shape[0])
                    for layer in self.layers:
                        output = layer.forward_propagation(output)

                    # compute loss (for display purpose only)
                    err += self.loss(y_train[j], output)

                    # backward propagation
                    error = self.loss(y_train[j], output, derivative=True)
                    for layer in reversed(self.layers):
                        error = layer.backward_propagation(error)

                    optimizer.step()

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

        if issubclass(type(optimizer), StochasticOptimizer):
            acc = 0

            # training loop
            for i in (t := trange(epochs)):
                samp = np.random.randint(
                    0, x_train.shape[0], size=(batch_size))
                samples = len(samp)
                err = 0
                for j in range(samples):
                    optimizer.zeroGrad()
                    output = x_train[samp[j]]
                    output = output.reshape(1, output.shape[0])
                    for layer in self.layers:
                        output = layer.forward_propagation(output)

                    err += self.loss(y_train[samp[j]], output)
                    error = self.loss(
                        y_train[samp[j]], output, derivative=True)
                    for layer in reversed(self.layers):
                        error = layer.backward_propagation(error)

                    optimizer.step()

                err /= samples
                self.train_loss_history.append(err)
                pred = self.predict(x_train)
                acc = self.accuracy(y_train, pred)
                self.val_accuracy_history.append(acc)
                t.set_description('epoch %d/%d   error=%.2f    accuracy=%.2f' %
                                  (i+1, epochs, err, acc))

    def accuracy(self, y, pred):
        """
        Compute the accuracy of a given prediction

        :param y: the predicted output.
        :param pred: the target (binary) output.
        :return: the accuracy of the predicted output.
        """
        # acc = np.sum(y_pred == y) / len(y)
        acc = np.sum((np.round(pred) == y)) / len(y)
        # acc = accuracy_score(y, np.round(pred))
        return acc