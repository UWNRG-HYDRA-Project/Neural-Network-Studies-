"""A module to load the MNIST image data for the number identifyer neural network. 

"""


import pickle
import gzip
import numpy as np

def load_data():
    f = gzip.open()
    