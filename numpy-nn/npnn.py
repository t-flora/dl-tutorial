import numpy as np

class FFNN(object):

    def __init__(self, x, y, nn_architecture) -> None:
        super().__init__()
        self.x = x
        self.w1 = np.random.rand(self.x.shape[1], 5)
        self.w2 = np.random.rand(self.w1.shape[1], 1)
        self.y = y
        self.b1 = np.random.rand(self.x.shape[1], 5)
        self.b2 = np.random.rand(self.w1.shape[1], 1)
        self.pred = np.zeros(self.y.shape)

    def add_layer(self, neurons = 10) -> None:
        pass

    def leaky_relu(self, x):
        """
        Computes the ReLU value for a pre-activation input
        """
        return np.where(x>0, x, x*0.01)

    def sigmoid(self, x):
        """
        Computes the sigmoid function for pre-activation input
        """
        return 1/(1+np.exp(-x))

    def dsig(self, x):
        sig = self.sigmoid(x)
        der = sig*(1-sig)
        return der

    def feedforward(self) -> None:
        """
        Calculates the activation values for each neuron in a layer, up to the output layer
        """
        self.a1 = sigmoid(np.dot(self.x, self.w1) + self.b1)
        self.pred = sigmoid(np.dot(self.a1, self.w2) + self.b2)

    def backprop(self) -> None:
        """
        Applies backpropagation to each layer to compute the derivative of the loss function as a function
        of each weight
        """
        dC_dpred = 2*(self.y - self.pred)
        dpred_dz = der_sigmoid(self.pred)
        dz_dw2 = self.a1

        # The code below is pretty wrong for now
        dC_da1 = 2*(self.y-self.pred)*der_sigmoid(self.pred)
        da1_dz = der_sigmoid(self.a1)
        dz_dw1 = self.x.T

        dC_dw2 = dC_dpred * dpred_dz * dz_dw2
        dC_dw1 = dC_da1 * da1_dz * dz_dw1

        self.w1 += dC_dw1
        self.w2 += dC_dw2
    
