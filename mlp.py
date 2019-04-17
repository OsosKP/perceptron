import numpy as np
from math import exp, pow


class MultiLayerPerceptron:
    def __init__(self, num_inputs, num_hidden, num_outputs):
        self.number_inputs = num_inputs
        self.number_hidden_units = num_hidden
        self.number_outputs = num_outputs
        self.hidden_neuron_values = np.zeros(self.number_hidden_units)
        self.hidden_errors = np.zeros(self.number_hidden_units)
        self.total_net_inputs_lower = np.zeros(
            self.hidden_neuron_values.shape)
        self.output = np.zeros(self.number_outputs)
        self.total_net_inputs_upper = np.zeros(
            self.output.shape)
        self.randomize()

    def randomize(self):
        # self.weights_lower = np.random.uniform(-0.05, 0.05, size=(
        #     self.number_hidden_units, self.number_inputs))
        # self.weights_upper = np.random.uniform(-0.05, 0.05, size=(
        #     self.number_outputs, self.number_hidden_units))
        self.weights_lower = np.ndarray(shape=(2, 2), dtype=float, buffer=np.array([
                                        [0.15, 0.25], [0.2, 0.3]]))
        self.weights_upper = np.ndarray(shape=(2, 2), dtype=float, buffer=np.array([
                                        [0.4, 0.5], [0.45, 0.55]]))
        self.bias = [0.35, 0.6]
        self.deltas_lower = np.zeros(self.weights_lower.shape)
        self.deltas_upper = np.zeros(self.weights_upper.shape)
        self.activations_lower = np.zeros(self.weights_lower.shape)
        self.activations_upper = np.zeros(self.weights_upper.shape)

    def forward(self, input):
        self.inputs = input
        self.total_net_inputs_lower = np.dot(
            input, self.weights_lower) + self.bias[0]
        self.activations_lower = np.array(
            map(lambda x: self.sigmoid_activation(x), self.total_net_inputs_lower))

        self.total_net_inputs_upper = np.dot(
            self.activations_lower, self.weights_upper) + self.bias[1]
        self.activations_upper = np.array(
            map(lambda x: self.sigmoid_activation(x), self.total_net_inputs_upper))

        return self.activations_upper

    def backward(self, target):
        error = np.sum(map(lambda o, t: 0.5 * self.squared_error(o, t),
                           self.activations_upper, target))
        self.upper_error_wrt_weight = np.multiply(self.derivative_error_wrt_output(target, self.activations_upper),
                                                  np.multiply(self.derivative_output_wrt_activations(self.activations_upper),
                                                              self.derivative_input_wrt_weight(self.activations_lower)))

        self.lower_error_wrt_weight = np.multiply(self.derivative_error_wrt_output(target, self.activations_lower),
                                                  np.multiply(self.derivative_output_wrt_activations(self.activations_lower),
                                                              self.derivative_input_wrt_weight(self.inputs)))

        for i in range(0, 2):
            for j in range(0, 2):
                self.deltas_upper[j][i] += self.upper_error_wrt_weight[i]
                self.deltas_lower[j][i] += self.lower_error_wrt_weight[i]

        return error

    def update_weights(self, learning_rate):
        for i in range(0, 2):
            for j in range(0, 2):
                self.weights_upper[i][j] -= learning_rate * \
                    self.deltas_upper[i][j]
                self.weights_lower[i][j] -= learning_rate * \
                    self.deltas_lower[i][j]
        # print(self.weights_upper)
        print(self.weights_lower)

    def derivative_error_wrt_output(self, target, layer):
        return np.subtract(layer, target)

    def derivative_output_wrt_activations(self, layer):
        return np.array(map(lambda x: x * (1 - x), layer))

    def derivative_input_wrt_weight(self, input):
        return input

    def sigmoid_activation(self, input): return pow(1. + exp(-input), -1.)

    def squared_error(self, output, target): return pow(target - output, 2.)
