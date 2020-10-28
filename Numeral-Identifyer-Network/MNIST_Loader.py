"""
A module to load the MNIST image data for the number identifyer neural network. 
"""


import pickle
import gzip
import numpy as np


def load_data():
    """Opens the nmist data file.
    """
    f = gzip.open(
        'Numeral-Identifyer-Network/MNIST-Training-Data/mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = pickle.load(
        f, encoding='latin1')
    f.close()
    return training_data, validation_data, test_data


def load_data_wrapper():
    """Reshaps the data in the mnist.pkl.gz file to agree with the assumed array dimensions used in the 
    stochastic gradient descent algorithm.
    """
    tr_d, va_d, te_d = load_data()

    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = list(zip(training_inputs, training_results))

    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = list(zip(validation_inputs, va_d[1]))

    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = list(zip(test_inputs, te_d[1]))
    return training_data, validation_data, test_data


def vectorized_result(y):
    """Creats a 10 dimensional column vector containing a 1.0 in the nth row where n is the correct 
    numeral. Takes the parameter ``y`` where ``y`` is the correct number corresponding to a particular 
    sample. ``y`` could be 4, 5, 3, 6 etc.
    """
    vector = np.zeros((10, 1))
    vector[y] = 1.0
    return vector
