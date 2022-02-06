import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
import time

STD_DIM = {"input_l":784, "h1":128, "h2":64, "output_l":10}

class FFNN(object):

    def __init__(self, dimensions = STD_DIM, epochs = 10, l_rate = 0.001, method = "he"):
        self.dimensions = dimensions
        self.n_layers = len(self.dimensions)
        # Initialize indices for calls in weight initialization
        self.idx = {i:ns for i, ns in zip(range(len(self.dimensions), self.dimensions.values()))}
        self.epochs = epochs
        self.l_rate = l_rate
        # Initialize weights using He method by default
        self.init_ws(method = method)

    def init_ws(self, method) -> None:
        """Initialize weights for method 

        Args:
            method (str): method to use in weight initialization. Can be either He or Xavier
        """
        n = len(self.dimensions)
        method = method.lower()
        self.params = {}

        if method == "he":
            for i in range(1, n):
                ns_in = self.idx[i-1]
                ns_out = self.idx[i]

                self.params[f"W{i}"] = np.random.randn(ns_in, ns_out) / np.sqrt(ns_in/2)

        elif method == "xavier":
           for i in range(1, n):
                ns_in = self.idx[i-1]
                ns_out = self.idx[i]

                self.params[f"W{i}"] = np.random.randn(ns_in, ns_out) / np.sqrt(ns_in)
        else:
            raise("ValueError")

    # def initialize(self):
    #     input_l = self.dimensions["input_l"]
    #     h1 = self.dimensions["h1"]
    #     h2 = self.dimensions["h2"]
    #     output_l = self.dimensions["output_l"]

    #     pass

    # def add_layer(self, size):
    #     new_layer = self.init_layer(size)
    #     self.layers.push(new_layer)

    def relu(self, x):
        relu_actv = np.maximum(0, x)
        return relu_actv

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

    def derivative(self, x, func = "sigmoid"):
        if func == "relu":
            return (x>0)*1
        elif func == "sigmoid":
            return self.sigmoid(x)*(1-self.sigmoid)
        elif func == "softmax":
            exponentials = np.exp(x - x.max())
            return self.softmax(x)*(1-exponentials/np.sum(exponentials, axis=0))
        else:
            raise("ValueError")

    # def dsig(self, x):
    #     sig = self.sigmoid(x)
    #     der = sig*(1-sig)
    #     return der

    def softmax(self, x):
        """Returns the softmax function for a given vector input

        Args:
            x (ndarray): [description]

        Returns:
            [ndarray]: [description]
        """
        exponentials = np.exp(x - x.max())
        return exponentials/np.sum(exponentials, axis=0)

    def feedforward(self, x_train, func = sigmoid):
        """
        Calculates the activation values for each neuron in a layer, up to the output layer
        """

        self.params['A0'] = x_train

        for i in range(1, len(self.params)):
            self.params[f"Z{i}"] = np.dot(self.params[f"W{i}"], self.params[f"A{i-1}"])
            self.params[f"A{i}"] = func(self.params[f'Z{i}'])

        return self.params[f"A{self.n_layers-1}"]
        # self.a1 = func(np.dot(self.x, self.w1) + self.b1)
        # self.pred = func(np.dot(self.a1, self.w2) + self.b2)

    def backprop(self, output, y_train) -> dict:
        """
        Applies backpropagation to each layer to compute the derivative of the loss function as a function
        of each weight
        """
        
        delta_params = {}

        error = (output - y_train)/output.shape[0] * self.derivative(self.params[f'Z{self.n_layers-1}'], func = "softmax")
        delta_params[f"W{self.n_layers-1}"] = np.outer(error, self.params[f"A{self.n_layers-2}"])

        for idx in range(self.n_layers-1, 1, -1):
            error = np.dot(self.params[f"W{idx}"].T, error) * self.derivative(self.params[f"Z{idx-1}"], func = "sigmoid")
            delta_params[f"W{idx-1}"] = np.outer(error, self.params[f"A{idx-2}"])

        return delta_params

        # dC_dpred = 2*(self.y - self.pred)
        # dpred_dz = der_sigmoid(self.pred)
        # dz_dw2 = self.a1

        # # The code below is pretty wrong for now
        # dC_da1 = 2*(self.y-self.pred)*der_sigmoid(self.pred)
        # da1_dz = der_sigmoid(self.a1)
        # dz_dw1 = self.x.T

        # dC_dw2 = dC_dpred * dpred_dz * dz_dw2
        # dC_dw1 = dC_da1 * da1_dz * dz_dw1

        # self.w1 += dC_dw1
        # self.w2 += dC_dw2

    def update_params(self, delta_params):
        """Update parameter values stored in the network as obtained in the backprop step

        Args:
            delta_params (dict): Dictionary containing the computed gradient values for parameter matrices
        """        
        for delta in delta_params:
            self.params[delta] -= self.l_rate*delta_params[delta]


    # def stochastic_gd(self):
    #     """
    #     Custom stochastic gradient descent optimizer
    #     """
    #     pass

    # def loss(self, output, X_test, y_test):
    #     """MSE loss

    #     Args:
    #         output ([type]): [description]
    #     """
    #     mse = np.square(y_test - output).mean()
    #     for xi, yi in zip(X_test, y_test):
    #         output = self.feedforward(X_test)
    #         mse = (np.square(yi-output))

    #     pass

    def crossEntropy(self, output, y_train):
        '''Cross entropy cost function '''
        eps = 1e-10
        yhat = np.clip(yhat, eps, 1-eps)
        return - np.nansum(y_train*np.log(yhat))

    def train(self, X_train, y_train):
        for it in range(self.epochs):
            # Iterate over each training example
            for xi, yi in zip(X_train, y_train):
                output = self.feedforward(xi)
                delta_params = self.backprop(yi, output)
                self.update_params(delta_params)
            
            print(f"Epoch: {it}, loss: {l}")
            # l = self.loss()
