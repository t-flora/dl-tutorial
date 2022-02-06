import numpy as np

class FFNN(object):

    def __init__(self, dimensions, epochs = 10, l_rate = 0.001):
        self.dimensions = dimensions
        self.epochs = epochs
        self.l_rate = l_rate

        self.params = self.initialize()

    def initialize(self):
        input_l = self.dimensions["input_l"]
        h1 = self.dimensions["h1"]
        
        pass

    def init_layer(self):
        pass

    def add_layer(self, size):
        new_layer = self.init_layer(size)
        self.layers.push(new_layer)

    def relu(self):
        pass

    def fforward(self):
        pass

    def train(self):
        pass
