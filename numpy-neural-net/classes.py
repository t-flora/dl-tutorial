import numpy as np

def softmax(self, x, df = False):
    """Returns the softmax function for a given vector input

    Args:
        x (ndarray): [description]

    Returns:
        [ndarray]: [description]
    """
    exponentials = np.exp(x - x.max())
    sm = exponentials/np.sum(exponentials, axis = 0)
    if df:
        # This is wrong :(
        dsm = sm * np.identity(sm.size) - sm.T @ sm
        return dsm
    return sm

def sigmoid(self, x, df = False):
    """[summary]

    Args:
        x ([type]): [description]
        df (bool, optional): [description]. Defaults to False.

    Returns:
        [type]: [description]
    """    
    s = 1/(1+np.exp(-x))
    if df:
        return s*(1-s)
    return s

def loss(y_true, y_pred, df = False):
    """MSE

    Args:
        y_true ([type]): [description]
        y_pred ([type]): [description]
        df (bool, optional): [description]. Defaults to False.

    Returns:
        [type]: [description]
    """    
    if df:
        return 2*(y_pred-y_true)/y_true.shape[0]
    return np.square(y_true - y_pred).mean()

class Layer(object):

    def __init__(self):
        self.input = None
        self.output = None
        self.l_rate = None

    def feedforward(self, input):
        raise NotImplementedError

    def backprop(self, output_error):
        """
        Backpropagation methods return dE/dX given the output error
        We update our parameter matrices based on the gradients computed at each step of backprop

        Args:
            output_error ([type]): [description]

        Raises:
            NotImplementedError: [description]
        """        
        raise NotImplementedError

class FullyConnectedLayer(Layer):
    def __init__(self, input_size, output_size, init_method = "he", l_rate = 0.001):
        # Xavier or He initialization
        self.method = init_method
        self.W = self.init_ws()[0]
        self.bias = self.init_ws(b = True)
        self.l_rate = l_rate

    def init_ws(self, bias = False) -> None:
        """Initialize weights for method 

        Args:
            method (str): method to use in weight initialization. Can be either He or Xavier
        """
        n = len(self.dimensions)
        method = self.method.lower()

        if method == "he":
            W = np.random.randn(input_size, output_size) / np.sqrt(input_size/2)
            b = np.random.randn(1, output_size) / np.sqrt(input_size/2)

        elif method == "xavier":
            W = np.random.randn(input_size, output_size) / np.sqrt(input_size)
            b = np.random.randn(1, output_size) / np.sqrt(input_size)

        else:
            raise("ValueError")
        if bias:
            return b
        return W

    def feedforward(self, input_data):
        self.input = input_data
        # Y = WX + B
        self.output = np.dot(self.input, self.W) + self.bias
        return self.output

    def backprop(self, output_error):
        # output_error = dE/dY
        # dE/dX = dE/dY * W^T
        delX = np.dot(output_error, self.W.T)
        # dE/dW = X^T * dE/dY
        delW = np.dot(self.input.T, output_error)
        # dE/db = dE/dY
        delB = output_error

        # Update parameters
        self.W -= self.l_rate*delW
        self.bias -= self.l_rate*delB
        return delX

class ActivationLayer(Layer):
    """
        The activation layer class serves to obtain dE/dX adding nonlinearity to forward and 
        backward propagation. It contains the activation function and its derivative, and is added
        "on top" of each fully connected layer to compute the appropriate activation values of the input
        passed into it
    """    
    def __init__(self, activation_func):
        self.activation_func = activation_func

    def feedforward(self, input_data):
        self.input = input_data
        self.output = self.activation_func(self.input)
        return self.output

    def backprop(self, output_error):
        # Multiply derivative of activation function by output error elementwise
        # Returns dE/dX = dE/dY \Hadamard f'(X)
        return self.activation_func(self.input, df=True)*output_error

class Network(object):
    def __init__(self, epochs = 10, l_rate = 0.001):
        self.layers = []
        self.l_rate = l_rate
        self.loss = None
        self.dloss = None

    def add_layer(self, layer):
        self.layers.append(layer)

    def set_loss(self, loss, dloss):
        """Set loss function to use

        Args:
            loss (function): loss function to be used
            dloss (function): derivative of loss function to be used
        """        
        self.loss = loss
        self.dloss = dloss

    def output(self, input_data):
        # Obtain the output for each data point in the dataset
        outs = []
        if type(input_data) == numpy.ndarray:
            n = input_data.shape[0]
        else:
            n = len(input_data)

        for i in range(n):
            out = input_data[i]
            for layer in self.layers:
                # Feed output of each layer as input into the next layer
                out = layer.feedforward(out)
            # Store output of the final layer as prediction
            outs.append(out)
        
        return outs

    def train(self, X_train, y_train):
        if type(X_train) == numpy.ndarray:
            n = X_train.shape[0]
        else:
            n = len(X_train)

        # For each epoch, run forward and backpropagation
        for e in range(self.epochs):
            #TODO: Write loss function
            total_loss = 0
            for xi in range(n):
                # Initiate output to feed into first hidden layer
                out = X_train[xi]
                for layer in self.layers:
                    # Obtain output of network for each training example
                    out = layer.feedforward(X_train[xi])
                # Compute loss for training example
                total_loss += self.loss(y_train[xi], out)

                # Get dE/dY
                dE = self.dloss(y_train[xi], out)
                
                # Iterate over layers from output to input
                for l in range(len(self.layers), -1, -1):
                    dE = self.layers[l].backprop(dE, self.l_rate)
            # Average loss over size of training data
            total_loss /= n
            print(f"Epoch: {e}/{self.epochs}, loss: {total_loss}")