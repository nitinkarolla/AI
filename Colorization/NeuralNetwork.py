import numpy as np
import math
from copy import deepcopy


class NeuralNetwork():

    def __init__(self, num_hidden_layers = 2, learning_rate = 0.01, num_neurons_each_layer = None, weights = None):
        self.weights = weights
        self.num_hidden_layers = num_hidden_layers
        self.num_neurons_each_layer = num_neurons_each_layer
        self.activationHidden = self.sigmoid
        self.activationOut = self.sigmoid
        self.learning_rate = learning_rate

    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))

    def sigmoid_derivative(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def _square_loss(self, pred, y):
        return (pred - y) ** 2

    def _initialise_weights(self, input_shape):

        self.num_neurons_each_layer.append(1)
        self.total_layers = self.num_hidden_layers + 1

        # Initialising a numpy array of
        # shape = (number of hidden layers, number of neurons, number of weights per neuron) to store weights
        self.weights = np.empty(shape = (self.total_layers, self.num_neurons_each_layer, None))
        print(self.weights)

        # Iterate through the layers
        for layer in range(self.total_layers):
            self.weights.append([])

            # Iterate through neurons in this layer
            number_of_neurons_in_this_layer = self.num_neurons_each_layer[layer]
            for _ in range(number_of_neurons_in_this_layer):

                # first hidden layer
                if layer == 0:
                    self.weights[layer].append(np.random.normal(0, 0.5, size = input_shape))

                # rest hidden layers
                else:
                    # Each neuron will have number of weights equal to the number of neuron outputs from the
                    # previous layer. The first weight for each neuron is the bias weight.
                    self.weights[layer].append(np.random.normal(0, 0.5, size = 1 + self.num_neurons_each_layer[layer - 1]))

    def _update_weights(self, out, output_derivatives):
        for curr_layer in range(self.total_layers):
            next_layer = curr_layer + 1
            next_layer_outputs = out[next_layer][1:]
            curr_layer_outputs = out[curr_layer]
            for curr_layer_neuron in range(self.num_neurons_each_layer[curr_layer]):
                sigmoid_der = self.sigmoid_derivative(next_layer_outputs[curr_layer_neuron])
                self.weights[curr_layer][curr_layer_neuron] -= self.learning_rate * \
                    output_derivatives[next_layer][curr_layer_neuron] * sigmoid_der * curr_layer_outputs[curr_layer_neuron]

    def _backward(self, y, out):

        # The derivatives array will have the same shape as weights array. - one derivative for each
        # weight
        self.weight_derivatives = deepcopy(self.weights)
        output_derivatives = deepcopy(out)

        # Calculate the output derivatives w.r.t loss
        layers_reversed = range(self.total_layers - 1, -1, -1)
        for curr_layer in layers_reversed:

            # We start from 1st index to neglect the bias term at the start of the output
            outputs_of_current_layer = out[curr_layer][1:]
            curr_layer_neurons = self.num_neurons_each_layer[curr_layer]

            # Cycle through the neurons in this layer and calculate the
            # derivative of output from each neuron.
            for curr_layer_neuron in range(curr_layer_neurons):

                # For last layer use the gradient formula directly
                if curr_layer == self.total_layers - 1:
                    output_derivatives[curr_layer][curr_layer_neuron + 1] = 2 * (
                            outputs_of_current_layer[curr_layer_neuron] - y)

                # For the rest of the layers
                else:

                    # Sum over all the neurons in the next layer.
                    next_layer = curr_layer + 1
                    next_layer_neurons = self.num_neurons_each_layer[next_layer]
                    next_layer_outputs = out[next_layer][1:]

                    derivative = 0
                    for next_layer_neuron in range(next_layer_neurons):
                        sigmoid_der = self.sigmoid_derivative(next_layer_outputs[next_layer_neuron])
                        second_term = self.weights[next_layer][next_layer_neuron][curr_layer_neuron]
                        derivative += output_derivatives[next_layer][next_layer_neuron] * sigmoid_der * second_term

                    output_derivatives[curr_layer][curr_layer_neuron + 1] = derivative

        # Make bias term derivatives 0
        for index in range(len(output_derivatives)):
            output_derivatives[index][0] = 0

        # Update the weights based on the above computed derivatives
        self._update_weights(out, output_derivatives)

    def _forward(self, x):
        out = []
        for curr_layer in range(self.total_layers):

            # We append a 1 to the output of each layer for the bias weight
            out.append([1])

            # Multiply the input with each neuron's weights in this layer
            # and store the output of each neuron
            for weights in self.weights[curr_layer]:
                if curr_layer == 0:
                    out[curr_layer].append(self.sigmoid(weights.T @ x))
                else:
                    previous_layer_output = out[curr_layer - 1]
                    out[curr_layer].append(self.sigmoid(weights.T @ previous_layer_output))

        return out

    def fit(self, X, y):

        # Add a bias column to X
        X = np.column_stack((np.ones(len(X)), X))
        data = zip(*(X, y))

        # Initialise the weights of the network
        self._initialise_weights(X.shape[1])

        # Update weights using gradient descent. For this we do a forward pass and backward pass
        # for each data point and update weights after each pass.
        for x, y in data:
            out = self._forward(x)
            pred = out[-1][-1]
            loss = self._square_loss(pred, y)
            self._backward(y, out)


x = np.random.randint(-50, 50, (5, 3))
nn = NeuralNetwork(num_hidden_layers = 2, num_neurons_each_layer = [2, 3])
nn.fit(X = x, y = [1, 0, 1, 0])
