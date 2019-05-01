import numpy as np
import math
from copy import deepcopy


class NeuralNetwork():

    SigmoidActivation = "sigmoid"
    ReLUActivation = "relu"
    LinearActivation = "linear"

    def __init__(self,
                 num_hidden_layers = 2,
                 learning_rate = 0.1,
                 num_neurons_each_layer = None,
                 epochs = 10,
                 weights = None):
        self.weights = weights
        self.num_hidden_layers = num_hidden_layers
        self.num_neurons_each_layer = num_neurons_each_layer
        self.learning_rate = learning_rate
        self.epochs = epochs

        # Sigmoid activation for other layers. Linear activation for last layer
        self.activations = [self.SigmoidActivation] * self.num_hidden_layers + [self.SigmoidActivation]
        self.activations_functions = {
            self.SigmoidActivation: self._sigmoid,
            self.ReLUActivation: self._relu,
            self.LinearActivation: self._linear
        }
        self.activations_derivatives = {
            self.SigmoidActivation: self._sigmoid_derivative,
            self.ReLUActivation: self._relu_derivative,
            self.LinearActivation: self._linear_derivative
        }

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def _relu(self, x):
        return np.maximum(np.zeros_like(x), x)

    def _linear(self, x):
        return x

    def _sigmoid_derivative(self, x):
        return self._sigmoid(x) * (1 - self._sigmoid(x))

    def _relu_derivative(self, x):
        return (1 if x > 0 else 0)

    def _linear_derivative(self, x):
        return np.ones_like(x)

    def _mse_loss(self, pred, y):
        return np.mean((pred - y) ** 2)

    def _initialise_weights(self, input_shape):

        self.num_neurons_each_layer.append(1)
        self.total_layers = self.num_hidden_layers + 1
        self.layers = range(self.total_layers)

        # Initialising a numpy array of
        # shape = (number of hidden layers, number of neurons, number of weights per neuron) to store weights
        self.weights = []

        # Iterate through the layers
        for layer in self.layers:
            self.weights.append([])

            number_of_neurons_in_this_layer = self.num_neurons_each_layer[layer]
            if layer == 0:
                self.weights[layer] = np.random.normal(0, 0.5, size = (number_of_neurons_in_this_layer, input_shape))
            else:
                # Adding 1 for the bias neuron
                self.weights[layer] = np.random.normal(0, 0.5, size = (number_of_neurons_in_this_layer,
                                                     1 + self.num_neurons_each_layer[layer - 1]))

        self.weights = np.array(self.weights)

    def _backward(self, x, y, out):

        # The derivatives array will have the same shape as weights array. - one derivative for each
        # weight
        output_derivatives = deepcopy(out)
        old_weights = deepcopy(self.weights)

        # Compute the output derivatives
        layers_reversed = self.layers[::-1]
        for curr_layer in layers_reversed:

            # For the last layer simply use the formula
            if curr_layer == self.total_layers - 1:
                output_derivatives[curr_layer] = 2*(out[curr_layer] - y)
                continue

            next_layer = curr_layer + 1
            activation_for_next_layer = self.activations[next_layer]
            activation_derivative = self.activations_derivatives[activation_for_next_layer]

            # The first term
            first_term = output_derivatives[next_layer]

            # Calculate the activation derivative
            current_layer_output = out[curr_layer].copy()
            current_layer_output = np.insert(current_layer_output, obj = 0, values = 1)
            activation_derivatives = activation_derivative(old_weights[next_layer] @ current_layer_output)

            for curr_layer_neuron in range(self.num_neurons_each_layer[curr_layer]):

                # Calculate the second term of the output derivative. We multiply only those weights which
                # are at the index of current layer neuron. Also, we remove the bias from the weights.
                next_layer_weights_without_bias = old_weights[next_layer][:, 1:]
                second_term = activation_derivatives * next_layer_weights_without_bias[:, curr_layer_neuron]
                output_derivatives[curr_layer][curr_layer_neuron] = first_term @ second_term

        # Update the weights using the output derivative calculated above
        for curr_layer in layers_reversed:

            # Get the activation for this layer and its derivative function
            activation_for_this_layer = self.activations[curr_layer]
            activation_derivative = self.activations_derivatives[activation_for_this_layer]

            # If first layer then use the data as the previous layer.
            if curr_layer == 0:
                previous_layer_output = x
            else:
                prev_layer = curr_layer - 1
                previous_layer_output = out[prev_layer].copy()
                previous_layer_output = np.insert(previous_layer_output, obj = 0, values = 1)

            first_term = output_derivatives[curr_layer]
            activation_derivatives = activation_derivative(old_weights[curr_layer] @ previous_layer_output)

            # For all neurons in the layer, update the weight indices simultaneously
            num_weights_in_this_layer_neurons = old_weights[curr_layer].shape[1]
            for curr_layer_weight_index in range(num_weights_in_this_layer_neurons):
                second_term = activation_derivatives * previous_layer_output[curr_layer_weight_index]
                weight_derivatives = first_term * second_term
                self.weights[curr_layer][:, curr_layer_weight_index] = \
                    old_weights[curr_layer][:, curr_layer_weight_index] - self.learning_rate * weight_derivatives

        # print(old_weights)
        # print("######################################################################################################")
        # print(self.weights)

    def _forward(self, x):
        out = []
        for curr_layer in self.layers:
            out.append([])

            # Get the activation for this layer and its function
            activation_for_this_layer = self.activations[curr_layer]
            activation_function = self.activations_functions[activation_for_this_layer]

            if curr_layer == 0:
                previous_layer_output = x
            else:
                previous_layer_output = out[curr_layer - 1].copy()
                previous_layer_output = np.insert(previous_layer_output, obj = 0, values = 1)

            out[curr_layer] = activation_function(self.weights[curr_layer] @ previous_layer_output)

        out = np.array(out)
        return out

    def fit(self, X, y):

        # Add a bias column to X
        X_new = np.column_stack((np.ones(len(X)), X))
        data = zip(*(X_new, y))

        # Initialise the weights of the network
        self._initialise_weights(X_new.shape[1])

        for epoch in range(self.epochs):

            # Update weights using gradient descent. For this we do a forward pass and backward pass
            # for each data point and update weights after each pass.
            for x_, y_ in data:
                out = self._forward(x_)
                self._backward(x_, y_, out)

            predictions = self.predict(X)
            loss = self._mse_loss(predictions, y)
            print("Epoch = ", str(epoch + 1), " - ", "Loss = ", str(loss))

    def predict(self, X):

        # Add a bias column to X
        X_new = np.column_stack((np.ones(len(X)), X))

        preds = []
        for x in X_new:
            pred = self._forward(x)[-1][-1]
            preds.append(pred)

        preds = np.array(preds)
        return preds
