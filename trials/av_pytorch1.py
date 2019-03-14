import numpy as np

#Input array
X = np.array([
    [1, 0, 1, 0],
    [1, 0, 1, 1],
    [0, 1, 0, 1]])

y = np.array([[1], [1], [0]])


# Sigmoid Function
def sigmoid (x):
    return 1/(1 + np.exp(-x))

# Derivative of Sigmoid Function
def derivatives_sigmoid(x):
    return x * (1 - x)


epoch = 5000
lr = 0.1
inputlayer_neurons = X.shape[1]
hiddenlayer_neurons = 3
output_neurons = 1

wh = np.random.uniform(inputlayer_neurons, hiddenlayer_neurons)
bh = np.random.uniform(1, hiddenlayer_neurons)

wout = np.random.uniform(hiddenlayer_neurons, output_neurons)
bout = np.random.uniform(hiddenlayer_neurons, output_neurons)

