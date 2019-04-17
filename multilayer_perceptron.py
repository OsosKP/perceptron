import numpy as np
from math import exp, pow


class MultiLayerPerceptron:
    def __init__(self, num_inputs, num_hidden, num_outputs):
        self.number_inputs = num_inputs
        self.number_hidden_units = num_hidden
        self.number_outputs = num_outputs
        self.hidden_neuron_values = np.zeros(self.number_hidden_units)
        self.hidden_errors = np.zeros(self.number_hidden_units)
        self.output = np.zeros(self.number_outputs)
        self.randomize()

    def randomize(self):
        # self.weights_lower = np.random.uniform(-0.05, 0.05, size=(
        #     self.number_hidden_units, self.number_inputs))
        # self.weights_upper = np.random.uniform(-0.05, 0.05, size=(
        #     self.number_outputs, self.number_hidden_units))
        self.weights_lower = np.array([0.15, 0.25], [0.2, 0.3])
        self.weights_upper = np.array([0.4, 0.5], [0.45, 0.55])
        self. bias = [0.35, 0.6]
        self.deltas_lower = np.zeros(self.weights_lower.shape)
        self.deltas_upper = np.zeros(self.weights_upper.shape)
        self.activations_lower = np.zeros(self.weights_lower.shape)
        self.activations_upper = np.zeros(self.weights_upper.shape)

    def forward(self, input):
        for (i, j), value in np.ndenumerate(self.weights_lower):
            self.activations_lower[i][j] = input[j] * value

        for index, vector in enumerate(self.activations_lower):
            self.hidden_neuron_values[index] = self.sigmoid_activation(
                np.sum(vector))

        for (i, j), value in np.ndenumerate(self.weights_upper):
            self.activations_upper[i][j] = self.hidden_neuron_values[j] * value

        for index, vector in enumerate(self.activations_upper):
            self.output[index] = self.sigmoid_activation(np.sum(vector))

        return self.output

    # def backward(self, target):
    #     error = 0.5 * pow(target - self.output, 2)
    #     return error

    # def update_weights(self, learning_rate):
    #         # print("Weights Before:\nUpper: {0}\nLower: {1}").format(
    #         #     self.weights_upper, self.weights_lower)
    #     for (i, j), value in np.ndenumerate(self.activations_upper):
    #         self.deltas_upper[i][j] = learning_rate * \
    #             self.output_error[i] * value
    #     for (i, j), value in np.ndenumerate(self.activations_lower):
    #         self.deltas_lower[i][j] = learning_rate * \
    #             self.hidden_errors[i] * value
    #     # print('Output Error:\t{0}\nActivations Upper:\t{1}\nDeltas Upper:\t{2}\nDeltas Lower:\t{3}').format(
    #         # self.output_error, self.activations_upper, self.deltas_upper, self.deltas_lower)
    #     self.weights_upper = np.add(self.weights_upper, self.deltas_upper)
    #     self.weights_lower = np.add(self.weights_lower, self.deltas_lower)
    #     self.deltas_upper = np.zeros(self.deltas_upper.shape)
    #     self.deltas_lower = np.zeros(self.deltas_lower.shape)
    #     # print("Weights After:\nUpper: {0}\nLower: {1}").format(
    #     #     self.weights_upper, self.weights_lower)

    def sigmoid_activation(self, input): return pow(1. + exp(-input), -1.)
