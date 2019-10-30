from math import exp
import numpy as np

# pylint: disable=too-many-instance-attributes


class MultiLayerPerceptron:
    def __init__(self, options):
        self.number_inputs = options["number_inputs"] + 1
        self.number_hidden_units = options["number_hidden_units"]
        self.number_outputs = options["number_outputs"]
        self.hidden_neuron_values = np.zeros(self.number_hidden_units)
        self.hidden_errors = np.zeros(self.number_hidden_units)
        self.total_net_inputs_lower = np.zeros(
            self.hidden_neuron_values.shape)
        self.output = np.zeros(self.number_outputs)
        self.total_net_inputs_upper = np.zeros(
            self.output.shape)

        self.inputs = None
        self.weights_lower = None
        self.weights_upper = None
        self.deltas_lower = None
        self.deltas_upper = None
        self.activations_lower = None
        self.activations_upper = None
        self.d_error_wrt_output_upper = None
        self.d_output_wrt_activations_upper = None
        self.d_input_wrt_weights_lower = None
        self.upper_error_wrt_weight = None
        self.d_error_wrt_activations_upper = None
        self.d_activations_upper_wrt_outputs_lower = None
        self.d_error_wrt_output_lower = None
        self.d_output_wrt_activations_lower = None
        self.d_activations_lower_wrt_weights_lower = None
        self.lower_error_wrt_weight = None

        self.activation_type = options["activation_type"]
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
        self.input_bias_value = {
            "sigmoid": .45,
            "relu": 0.8,
            "tanh": 0.5
        }

        self.randomize()

    def randomize(self):
        self.weights_lower = np.random.uniform(-0.3, 0.3, size=(
            self.number_inputs, self.number_hidden_units))
        self.weights_upper = np.random.uniform(-0.3, 0.3, size=(
            self.number_hidden_units, self.number_outputs))

        self.deltas_lower = np.zeros(self.weights_lower.shape)
        self.deltas_upper = np.zeros(self.weights_upper.shape)
        self.activations_lower = np.zeros(self.weights_lower.shape)
        self.activations_upper = np.zeros(self.weights_upper.shape)

    def forward(self, inp):
        self.inputs = inp
        self.inputs = np.append(
            self.inputs, self.input_bias_value[self.activation_type])
        # Net_H
        self.total_net_inputs_lower = np.dot(
            self.inputs, self.weights_lower)
        # Out_H
        self.activations_lower = np.array(
            list(map(self.activation[self.activation_type], self.total_net_inputs_lower)))
        # Net_O
        self.total_net_inputs_upper = np.dot(
            self.activations_lower,
            self.weights_upper)
        # Out_O
        self.activations_upper = np.array(
            list(map(self.activation[self.activation_type], self.total_net_inputs_upper)))
        return self.activations_upper

    def calculator_error(self, target):
        result = None
        if target.size == 1:
            result = 0.5 * self.squared_error(self.activations_upper, target)
        else:
            result = np.sum(map(lambda o, t: 0.5 * self.squared_error(o, t),
                                self.activations_upper, target))
        return result

    def backward(self, target):
        # print('Check 2')
        error = self.calculator_error(target)
        # dE / dO_U
        self.d_error_wrt_output_upper = self.derivative_error_wrt_output(
            target, self.activations_upper)
        # dO_U / dNet_U
        self.d_output_wrt_activations_upper = \
            self.derivative_output_wrt_activations[self.activation_type](
                self.activations_upper)
        # # dNet / DW_L
        self.d_input_wrt_weights_lower = self.activations_lower

        # dE / dW
        self.upper_error_wrt_weight = np.zeros(self.deltas_upper.shape)

        for i in range(self.d_output_wrt_activations_upper.size):
            for j in range(self.d_input_wrt_weights_lower.size):
                print(self.d_output_wrt_activations_upper)
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
        self.d_output_wrt_activations_lower = \
            self.derivative_output_wrt_activations[self.activation_type](
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

    @staticmethod
    def derivative_error_wrt_output(target, layer):
        return np.subtract(layer, target)

    @staticmethod
    def derivative_output_wrt_sigmoid_activations(layer):
        return np.array(map(lambda x: x * (1 - x), layer))

    @staticmethod
    def derivative_output_wrt_relu_activations(layer):
        values = np.zeros(layer.shape)
        for index, value in np.ndenumerate(layer):
            if value > 0:
                values[index] = 1
            else:
                values[index] = value / 100
        return values

    def derivative_output_wrt_tanh_activations(self, layer):
        return np.array(list(map(lambda x: 1 - pow(self.tanh_activation(x), 2), layer)))

    @staticmethod
    def sigmoid_activation(inp):
        return pow(1. + exp(-inp), -1.)

    @staticmethod
    def relu_activation(inp):
        return max(0.01 * inp, inp)

    @staticmethod
    def tanh_activation(inp):
        e_pos = exp(inp)
        e_neg = exp(-inp)
        return (e_pos - e_neg) / (e_pos + e_neg)

    @staticmethod
    def squared_error(output, target):
        return pow(target - output, 2.)

    @staticmethod
    def average_miss(target, output):
        return (sum(map(lambda x, y: abs(x - y), target, output))) / target.shape[0]
