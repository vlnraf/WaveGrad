"""
Utility Module
"""
import numpy as np
import matplotlib.pyplot as plt

def plot_loss(net):
    """
    Plot the loss of the network during the train and it's validation

    :param net: A network
    """
    fig, loss = plt.subplots()
    plt.style.use('ggplot')
    loss.plot(net.train_loss_history, color='navy', lw=2)
    loss.plot(net.val_loss_history, color='darkorange', lw=2)
    loss.set_title('model loss')
    loss.set_xlabel('epoch')
    loss.set_ylabel('loss')
    loss.legend(['training', 'validation']).get_frame().set_facecolor('white')
    plt.show()

def plot_accuracy(net):
    """
    Plot the accuracy of the network during the train and it's validation

    :param net: A network
    """
    fig, loss = plt.subplots()
    plt.style.use('ggplot')
    loss.plot(net.accuracy_history, color='navy', lw=2)
    loss.plot(net.val_accuracy_history, color='darkorange', lw=2)
    loss.set_title('model accuracy')
    loss.set_xlabel('epoch')
    loss.set_ylabel('accuracy')
    loss.legend(['training', 'validation']).get_frame().set_facecolor('white')
    plt.show()

def accuracy(y, pred):
    """
    Compute the accuracy of a given prediction

    :param y: the predicted output.
    :param pred: the target (binary) output.
    :return: the accuracy of the predicted output.
    """
    # acc = np.sum(y_pred == y) / len(y)
    #print(np.round(pred))
    acc = 0
    for x in range(len(pred)):
        acc += np.sum(np.array_equal(np.round(pred[x]), y[x]))
    # acc = np.sum(np.round(pred) == y) / len(y)
    # acc = accuracy_score(y, np.round(pred))
    return acc / len(y)
