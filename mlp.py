import numpy as np
from math import exp, pow


class MultiLayerPerceptron:
    def __init__(self, num_inputs, num_hidden, num_outputs, activation_type):
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

        self.activation_type = activation_type
        self.derivative_output_wrt_activations = {
            "sigmoid": self.derivative_output_wrt_sigmoid_activations,
            "relu": self.derivative_output_wrt_relu_activations,
            "tanh": self.derivative_output_wrt_tanh_activations
        }
        self.activation = {
            "sigmoid": self.sigmoid_activation,
            "relu": self.relu_activation,
            "tanh": self.tanh_activation
        }

    def randomize(self):
        self.weights_lower = np.random.uniform(-0.2, 0.2, size=(
            self.number_inputs, self.number_hidden_units))
        self.weights_upper = np.random.uniform(-0.2, 0.2, size=(
            self.number_hidden_units, self.number_outputs))

        # self.weights_lower = np.ndarray(shape=(2, 2), dtype=float, buffer=np.array([
        #     [0.15, 0.25], [0.2, 0.3]]))
        # self.weights_upper = np.ndarray(shape=(2, 2), dtype=float, buffer=np.array([
        #     [0.4, 0.5], [0.45, 0.55]]))

        self.bias = [0.35, 0.6]
        self.deltas_lower = np.zeros(self.weights_lower.shape)
        self.deltas_upper = np.zeros(self.weights_upper.shape)
        self.activations_lower = np.zeros(self.weights_lower.shape)
        self.activations_upper = np.zeros(self.weights_upper.shape)

    def forward(self, input):
        self.inputs = input
        # Net_H
        self.total_net_inputs_lower = np.dot(
            self.inputs, self.weights_lower) + self.bias[0]
        # Out_H
        self.activations_lower = np.array(
            map(lambda x: self.activation[self.activation_type](x), self.total_net_inputs_lower))
        # Net_O
        self.total_net_inputs_upper = np.dot(
            self.activations_lower, self.weights_upper) + self.bias[1]
        # Out_O
        self.activations_upper = np.array(
            map(lambda x: self.activation[self.activation_type](x), self.total_net_inputs_upper))
        return self.activations_upper

    def calculator_error(self, target):
        if (target.size == 1):
            return 0.5 * self.squared_error(self.activations_upper, target)
        else:
            return np.sum(map(lambda o, t: 0.5 * self.squared_error(o, t),
                              self.activations_upper, target))

    def backward(self, target):
        error = self.calculator_error(target)
        # dE / dO_U
        self.d_error_wrt_output_upper = self.derivative_error_wrt_output(
            target, self.activations_upper)
        # dO_U / dNet_U
        self.d_output_wrt_activations_upper = self.derivative_output_wrt_activations[self.activation_type](
            self.activations_upper)
        # # dNet / DW_L
        self.d_input_wrt_weights_lower = self.activations_lower

        # dE / dW
        self.upper_error_wrt_weight = np.zeros(self.deltas_upper.shape)

        for i in range(self.d_output_wrt_activations_upper.size):
            for j in range(self.d_input_wrt_weights_lower.size):
                self.upper_error_wrt_weight[j][i] = self.d_error_wrt_output_upper[i] * \
                    self.d_output_wrt_activations_upper[i] * \
                    self.d_input_wrt_weights_lower[j]

        # dE / dNet_U
        self.d_error_wrt_activations_upper = np.multiply(
            self.d_error_wrt_output_upper, self.d_output_wrt_activations_upper)
        # dNet_U / dOut_l
        self.d_activations_upper_wrt_outputs_lower = self.weights_upper

        # dE / dO_L
        self.d_error_wrt_output_lower = np.sum(np.multiply(
            self.d_error_wrt_activations_upper, self.d_activations_upper_wrt_outputs_lower), 1)
        # dO_L / dNet_L
        self.d_output_wrt_activations_lower = self.derivative_output_wrt_activations[self.activation_type](
            self.activations_lower)
        # dNet_L / dW
        self.d_activations_lower_wrt_weights_lower = np.asarray(self.inputs)
        # dE / dW
        self.lower_error_wrt_weight = np.zeros(self.deltas_lower.shape)

        for i in range(self.d_error_wrt_output_lower.size):
            for j in range(self.d_activations_lower_wrt_weights_lower.size):
                self.lower_error_wrt_weight[j][i] = self.d_error_wrt_output_lower[i] * \
                    self.d_output_wrt_activations_lower[i] * \
                    self.d_activations_lower_wrt_weights_lower[j]

        for i in range(self.deltas_upper.shape[0]):
            for j in range(self.deltas_upper.shape[1]):
                self.deltas_upper[i][j] += self.upper_error_wrt_weight[i][j]

        for i in range(self.deltas_lower.shape[0]):
            for j in range(self.deltas_lower.shape[1]):
                self.deltas_lower[i][j] += self.lower_error_wrt_weight[i][j]

        # print('Input: {0}\tOutput: {1}\tError: {2}').format(
            # self.inputs, self.activations_upper, error)

        return error

    def update_weights(self, learning_rate):
        for i in range(self.deltas_upper.shape[0]):
            for j in range(self.deltas_upper.shape[1]):
                self.weights_upper[i][j] -= learning_rate * \
                    self.deltas_upper[i][j]

        for i in range(self.deltas_lower.shape[0]):
            for j in range(self.deltas_lower.shape[1]):
                self.weights_lower[i][j] -= learning_rate * \
                    self.deltas_lower[i][j]
        self.deltas_upper = np.zeros(self.deltas_upper.shape)
        self.deltas_lower = np.zeros(self.deltas_lower.shape)

    def derivative_error_wrt_output(self, target, layer):
        return np.subtract(layer, target)

    def derivative_output_wrt_sigmoid_activations(self, layer):
        return np.array(map(lambda x: x * (1 - x), layer))

    def derivative_output_wrt_relu_activations(self, layer):
        values = np.zeros(layer.shape)
        for index, value in np.ndenumerate(layer):
            if (value > 0):
                values[index] = 1
        return values

    def derivative_output_wrt_tanh_activations(self, layer):
        return np.array(map(lambda x: 1 - pow(self.tanh_activation(x), 2), layer))

    def sigmoid_activation(self, input): return pow(1. + exp(-input), -1.)

    def relu_activation(self, input): return max(0, input)

    def tanh_activation(self, input):
        e_pos = exp(input)
        e_neg = exp(-input)
        return (e_pos - e_neg) / (e_pos + e_neg)

    def squared_error(self, output, target): return pow(target - output, 2.)
