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
            self.weights.append(np.random.randn(y, x))

    def feedforwardoutput(self, a):
        """Returns the output of the network given the input ``a`` where ``a`` is an array of 28 * 28 = 784
        pixels with greyscale values
        """
        for w, b in zip(self.weights, self.biases):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def stochastic_gradient_descent(self, training_data, mini_batch_size, eta, epoches):
        """Trains the neural network using the stochastic gradient descent aglorithm. The ``training_data`` is 
        randomly shuffled and seperated into ``mini_batches`` of size defined by ``mini_batch_size``. Each ``mini_batch`` is evaluated to adjust the 
        weights and biases of the network using the ``update_weights_and_biases`` function, whcih also requires the 
        learning rate ``eta``.``epoches`` is the number of times the process described above will loop. In a single epoch,
        all of the mini batches are iterated through to train the network. 
        """
        n = len(training_data)
        mini_batches = []
        for j in range(epoches):
            random.shuffle(training_data)
            for k in range(0, n, mini_batch_size):
                mini_batches.append(training_data[k:k + mini_batch_size])
            for mini_batch in mini_batches:
                self.update_weights_and_biases(mini_batch, eta)
            print("Epoch {0} completed".format(j))

    def update_weights_and_biases(self, mini_batch, eta):
        """Updates weights and biases of the network using the training examples from a single mini-batch. 
        ``delta_del_b`` and ``delta_del_w`` are each a list of arrays. The elements of each arrays are the partial 
        derivatives of the cost function with respect to every bias and weight in the network. These partial
        derivatives are summed over every training example in the ``mini_batch`` to get
        ``del_b`` and ``del_w``. the ``weights`` and ``biases`` are then updated accordingly.
        """
        del_b = []
        del_w = []
        for b in self.biases:
            del_b.append(np.zeros(b.shape))
        for w in self.weights:
            del_w.append(np.zeros(w.shape))
        for x, y in mini_batch:
            delta_del_b, delta_del_w = self.backpropogation(x, y)
            del_b = [single_layer_del_b + single_layer_delta_del_b
                     for single_layer_del_b, single_layer_delta_del_b in zip(del_b, delta_del_b)]
            del_w = [single_layer_del_w + single_layer_delta_del_w
                     for single_layer_del_w, single_layer_delta_del_w in zip(del_w, delta_del_w)]
        self.weights = [w - (eta/len(mini_batch))*nw for w,
                        nw in zip(self.weights, del_w)]
        self.biases = [b - (eta/len(mini_batch))*db for b,
                       db in zip(self.biases, del_b)]

    def backpropogation(self, x, y):
        """Returns a tuple representing the gradient of the cost function for a single training example. 
        I.e. the partial derivatives of the cost function with respect to every weight and bias in the 
        network. 
        """
        del_b = [np.zeros(b.shape) for b in self.biases]
        del_w = [np.zeros(w.shape) for w in self.weights]

        activation = x
        # Stores activation vectors layer by layer. Initialized with input activation x
        activation_list = [x]
        z_vectors = []

        # Feedforward. Calculates the activation vector for each layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            activation = sigmoid(z)
            z_vectors.append(z)
            activation_list.append(activation)

        # Calculates the error in the output layer
        output_error = (self.cost_prime(
            activation_list[-1], y))*sigmoid_derivative(z_vectors[-1])

        # Backpropogates the output error to calculate the error in every other layer of the network.
        del_b[-1] = output_error
        del_w[-1] = np.dot(output_error[-1], activation_list[-2].transpose())
        for i in range(2, self.neuronLayers):
            z = z_vectors[-i]
            sd = sigmoid_derivative(z)
            error = np.dot(self.weights[-i + 1].transpose(), error) * sd
            del_b[-i] = error
            del_w[-i] = np.dot(error, activation_list[i - 1].transpose())
        return(del_b, del_w)


    def cost_prime(self, output_activations, y):
        """The derivative of the cost function used in the backpropogation algorithm. Returns a vector of
        derivatives for C (cost function) with respect to each output activation.
        """
        return (output_activations - y)


def sigmoid(z):
    """ 
    Normalizes input to range between 0 and 1.
    """
    return 1.0/(1.0 + np.exp(-z))


def sigmoid_derivative(z):
    """
    Derivative of sigmoid function
    """
    return np.exp(-z)/((1.0 + np.exp(-z)) ^ 2)
