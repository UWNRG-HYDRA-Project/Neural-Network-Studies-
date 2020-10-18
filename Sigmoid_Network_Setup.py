"""
A module defining the network class containing the 
stochastic gradient descent algorithm to optimize output correctness.

"""


import random
import numpy as np


class Network:
    def __init__(self, neuronNums):
        """The list ``neuronNums`` defines the number of neurons in each layer of the network. The
        weights and biases are initiallized using a Gaussian distribution with a mean of 0 and a variance
        of 1. ``biases`` and ``weights`` are lists of arrays storing the biases and weights of the network
        respectively.
        """
        self.neuronLayers = len(neuronNums)
        self.neuronNums = neuronNums
        self.biases = []
        self.weights = []
        for y in neuronNums[1:]:
            self.biases.append(np.random.randn(y, 1))

        for x, y in zip(neuronNums[:2], neuronNums[1:]):
            self.weights.append(np.random.randn(y,x))

    def feedforwardoutput(self, a):
        """Returns the output of the network given the input ``a`` where ``a`` is an array of 28 * 28 = 784
        pixels with greyscale values
        """
        for w, b in zip(self.weights, self.biases):
            a = sigmoid(np.dot(w,a) + b)




def sigmoid(z):
    """ 
    Normalizes input to range between 0 and 1
    """
    return 1.0/(1.0 + np.exp(-z))