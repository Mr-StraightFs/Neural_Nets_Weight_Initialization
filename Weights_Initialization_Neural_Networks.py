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
def model(X, Y, learning_rate=0.01, num_iterations=1000, print_cost=True, initialization="he"):

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

# Testing the Function and training the Model
parameters = initialize_parameters_zeros([3, 2, 1])
print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))
initialize_parameters_zeros_test(initialize_parameters_zeros)
# Training on 1000 iterations
parameters = model(train_X, train_Y, initialization = "zeros")
print ("On the train set:")
predictions_train = predict(train_X, train_Y, parameters)
print ("On the test set:")
predictions_test = predict(test_X, test_Y, parameters)
print ("predictions_train = " + str(predictions_train))
print ("predictions_test = " + str(predictions_test))
plt.title("Model with Zeros initialization")
axes = plt.gca()
axes.set_xlim([-1.5,1.5])
axes.set_ylim([-1.5,1.5])
plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)
# The performance is bad , the cost doesn't decrease, and the algorithm performs no better than random guessing
# Conclusion : initializing all the weights to zero results in the network failing to break symmetry.
# The weights  𝑊[𝑙]  should be initialized randomly to break symmetry.
# However, it's okay to initialize the biases  𝑏[𝑙]  to zeros. Symmetry is still broken so long as  𝑊[𝑙]  is initialized randomly.


# The code for the Random Initialize Func.
def initialize_parameters_random(layers_dims):
    parameters = {}
    L = len(layers_dims)  # integer representing the number of layers

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layers_dims[l], layers_dims[l - 1]) * 10 #WL -- weight matrix of shape (layers_dims[L], layers_dims[L-1])
        parameters['b' + str(l)] = np.zeros((layers_dims[l], 1)) * 10 #bL -- bias vector of shape (layers_dims[L], 1)
return parameters

# Testing and training with random initialization
parameters = initialize_parameters_random([3, 2, 1])
print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))
initialize_parameters_random_test(initialize_parameters_random)
parameters = model(train_X, train_Y, initialization = "random")
print ("On the train set:")
predictions_train = predict(train_X, train_Y, parameters)
print ("On the test set:")
predictions_test = predict(test_X, test_Y, parameters)
print (predictions_train)
print (predictions_test)
plt.title("Model with large random initialization")
axes = plt.gca()
axes.set_xlim([-1.5,1.5])
axes.set_ylim([-1.5,1.5])
plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)
# the results are much better than Zero initialization :
# the cost learning curve seems to be systemteclly decresing wth the num of iterations . That's wot we desire .
# the cost starts very high. This is because with large random-valued weights . The next initialization methods will solve this .
# Poor initialization can lead to vanishing/exploding gradients, which also slows down the optimization algorithm.
# Initializing weights to very large random values doesn't work well.
# Initializing with small random values should do better. The important question is, how small should be these random values be?

#
# the Func. Code for the He Initialization Method
def initialize_parameters_he(layers_dims):
    np.random.seed(3)
    parameters = {}
    L = len(layers_dims) - 1  # integer representing the number of layers

    for l in range(1, L + 1):
        parameters['W' + str(l)] = np.random.randn(layers_dims[l], layers_dims[l - 1]) * (
                    (2 / layers_dims[l - 1]) ** (1 / 2))
        parameters['b' + str(l)] = np.zeros((layers_dims[l], 1)) * ((2 / layers_dims[l - 1]) ** (1 / 2))

return parameters
# Testing and Training with the He Method
parameters = initialize_parameters_he([2, 4, 1])
print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))
initialize_parameters_he_test(initialize_parameters_he)
# parameters
parameters = model(train_X, train_Y, initialization = "he")
print ("On the train set:")
predictions_train = predict(train_X, train_Y, parameters)
print ("On the test set:")
predictions_test = predict(test_X, test_Y, parameters)
# plotting and Vizualization
plt.title("Model with He initialization")
axes = plt.gca()
axes.set_xlim([-1.5,1.5])
axes.set_ylim([-1.5,1.5])
plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)

# The model with He initialization separates the blue and the red dots very well in a small number of iterations.
#Different initializations lead to very different results
#Random initialization is used to break symmetry and make sure different hidden units can learn different things
#Resist initializing to values that are too large!
#He initialization works well for networks with ReLU activations