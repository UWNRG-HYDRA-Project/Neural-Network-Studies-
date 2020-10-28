import MNIST_Loader
import Sigmoid_Network_Setup

training_data, validation_data, test_data = MNIST_Loader.load_data_wrapper()
network = Sigmoid_Network_Setup.Network([784,30,10])
network.stochastic_gradient_descent(training_data,10,3.0,30,test_data=test_data)