"""This code executes the neural network. The user can change the number of hidden layers, mini batch size,
learning rate, and number of epochs to train the network over. The user can optionally include test data 
to evaluate the network performace after each training epoch.
"""

import MNIST_Loader
import Sigmoid_Network_Setup

training_batch_size = 10
learning_rate = 3.0
number_of_epochs = 30

training_data, validation_data, test_data = MNIST_Loader.load_data_wrapper()
network = Sigmoid_Network_Setup.Network([784, 30, 10])
network.stochastic_gradient_descent(
    training_data, training_batch_size, learning_rate, number_of_epochs, test_data=test_data)
