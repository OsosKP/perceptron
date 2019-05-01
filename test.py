# from multilayer_perceptron import MultiLayerPerceptron
from mlp import MultiLayerPerceptron
from mlp_factory import MLP_Factory
import numpy as np
from math import exp, pow

# Testing either "xor" or "sin"
testing = "xor"

# Activation type: "tanh", "relu" or "sigmoid"
# tanh has best overall results
activation_type = "tanh"
learning_rate = {
    "sigmoid": .8,
    "tanh": 0.2,
    "relu": 0.2
}
max_epochs = 10000


factory = MLP_Factory()

# If testing XOR:
number_inputs = 2
number_hidden_units = 4
number_outputs = 1

input = {
    "xor": [[0, 0], [0, 1], [1, 0], [1, 1]],
    "sin": np.zeros((200, 4))
}

target = {
    "xor": np.array([0, 1, 1, 0]),
    "sin": np.zeros((200, 1))
}

if (testing == "sin"):
    factory.generate_sin_values(input[testing], target[testing])


output = np.zeros(target[testing].shape)

options_xor = {
    "number_inputs": number_inputs,
    "number_hidden_units": number_hidden_units,
    "number_outputs": number_outputs,
    "activation_type": activation_type,
}

options_sin = factory.sin_finder(activation_type)

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
        # if (index % 3 == 0):
        nn.update_weights(learning_rate[activation_type])
    print('Epoch:\t{0}\tError:\t{1}').format(epoch, error)

print('Output: {0}\nAverage difference between target and output: {1}').format(
    output, factory.average_miss(target[testing], output))
