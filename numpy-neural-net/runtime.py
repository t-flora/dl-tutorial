# from nn import FFNN
from classes import Network, FullyConnectedLayer, ActivationLayer, loss, sigmoid, softmax
from sklearn import datasets, metrics
from sklearn.model_selection import train_test_split
import time
digits = datasets.load_digits()

# Data loaded as shown in:
# https://scikit-learn.org/stable/auto_examples/classification/plot_digits_classification.html
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(data, digits.target,\
 test_size = 0.3)
X_train = X_train.astype("float32")
X_train /= 255

network = Network()
network.add_layer(FullyConnectedLayer(28*28, 128))

#network.add_layer()

# dnn = FFNN()
# start = time.time()
# dnn.train(X_train, y_train)
# print("Training length: ", time.time() - start)

# loss = dnn.loss()
# print("Custom loss measurement: ", loss)