import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
from public_tests import *
from init_utils import sigmoid, relu, compute_loss, forward_propagation, backward_propagation
from init_utils import update_parameters, predict, load_dataset, plot_decision_boundary, predict_dec

#Data Vizualization in 2D plot
plt.rcParams['figure.figsize'] = (7.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

#Loading the Dataset
train_X, train_Y, test_X, test_Y = load_dataset()


# Model Def
def model(X, Y, learning_rate=0.01, num_iterations=15000, print_cost=True, initialization="he"):

    grads = {}
    costs = []  # to keep track of the loss
    m = X.shape[1]  #fetching m (teh numbe of examles)
    layers_dims = [X.shape[0], 10, 5, 1] # The input layer size is the numbers X-features (rows of X) , the output layer is one unit
                                         # the two hidden layers has 10 and 5 units respectevly

    # Initialize parameters dictionary.
    if initialization == "zeros":
        parameters = initialize_parameters_zeros(layers_dims)
    elif initialization == "random":
        parameters = initialize_parameters_random(layers_dims)
    elif initialization == "he":
        parameters = initialize_parameters_he(layers_dims)

    # Loop (gradient descent)


     for i in range(num_iterations):
    # Forward propagation: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID.
         a3, cache = forward_propagation(X, parameters)

    # Loss
         cost = compute_loss(a3, Y)

    # Backward propagation.
      grads = backward_propagation(X, Y, cache)

    # Update parameters.
    parameters = update_parameters(parameters, grads, learning_rate)

    # Print the loss every 1000 iterations
    if print_cost and i % 1000 == 0:
        print("Cost after iteration {}: {}".format(i, cost))
        costs.append(cost)


# plot the loss
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

return parameters

# Initialize with Zeros Params Function
def initialize_parameters_zeros(layers_dims):
    parameters = {}
    L = len(layers_dims)  # number of layers in the network

    for l in range(1, L):
        parameters['W' + str(l)] = np.zeros((layers_dims[l], layers_dims[l - 1])) #WL -- weight matrix of shape (layers_dims[L], layers_dims[L-1])
         parameters['b' + str(l)] = np.zeros((layers_dims[l], 1)) #b1 -- bias vector of shape (layers_dims[1], 1)
return parameters

