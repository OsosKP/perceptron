# from multilayer_perceptron import MultiLayerPerceptron
from mlp import MultiLayerPerceptron
from mlp_factory import MLP_Factory
import numpy as np
from math import exp, pow

# Customization of the network should be done in these lines
# Options for the second network (sin(x1 - x2 + x3 - x4)) set in mlp_factory
# Testing either "xor" or "sin"
testing = "xor"

# Activation type: "tanh", "relu" or "sigmoid"
# tanh has best overall results
activation_type = "tanh"
learning_rate = {
    "sigmoid": 0.8,
    "tanh": 0.5,
    "relu": 0.2
}
learning_rate_change = {
    "sigmoid": 1.1,
    "tanh": 0.9,
    "relu": 0.9
}

max_epochs = 5000


factory = MLP_Factory()
# If testing XOR:
number_inputs = 2
number_hidden_units = 4
number_outputs = 1

input = {
    "xor": np.array([[0, 0], [0, 1], [1, 0], [1, 1]]),
    "sin": np.zeros((200, 4))
}

target = {
    "xor": np.array([0, 1, 1, 0]),
    "sin": np.zeros((200, 1))
}

if (testing == "sin"):
    factory.generate_sin_values(input[testing], target[testing])

output = np.zeros(target[testing].shape)

options = {
    "xor": {
        "number_inputs": number_inputs,
        "number_hidden_units": number_hidden_units,
        "number_outputs": number_outputs,
        "activation_type": activation_type,
    },
    "sin": factory.sin_finder(activation_type)
}

nn = MultiLayerPerceptron(options[testing])

for epoch in range(max_epochs):
    error = 0
    for index, value in enumerate(input[testing]):
        output[index] = nn.forward(value)
        error += nn.backward(target[testing][index])
        nn.update_weights(learning_rate[activation_type])
    if (epoch % 100 == 0):
        print('Epoch:\t{0}\tError:\t{1}').format(epoch, error)
        learning_rate[activation_type] *= learning_rate_change[activation_type]

if (testing == "xor"):
    print('Output: {0}').format(output)
print('Average difference between target and output: {0}').format(
    nn.average_miss(target[testing], output))
