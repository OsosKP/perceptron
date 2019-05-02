# from multilayer_perceptron import MultiLayerPerceptron
from mlp import MultiLayerPerceptron
from mlp_factory import MLP_Factory
import numpy as np
from math import exp, pow

# Customization of the network should be done in these lines
# Options for the second network (sin(x1 - x2 + x3 - x4)) set in mlp_factory

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

input_train = np.zeros((150, 4))
target_train = np.zeros((150, 1))
factory.generate_sin_values(input_train, target_train)
input_test = np.zeros((50, 4))
target_test = np.zeros((50, 1))
factory.generate_sin_values(input_test, target_test)

output_train = np.zeros(target_train.shape)
output_test = np.zeros(target_test.shape)
options = factory.sin_finder(activation_type)
nn = MultiLayerPerceptron(options)

# Training
print("----------------------------------\nTrain\n----------------------------------")
for epoch in range(max_epochs):
    error = 0
    for index, value in enumerate(input_train):
        output_train[index] = nn.forward(value)
        error += nn.backward(target_train[index])
        nn.update_weights(learning_rate[activation_type])
    if (epoch % 500 == 0):
        print('Epoch:\t{0}\tError:\t{1}').format(epoch, error)
        learning_rate[activation_type] *= learning_rate_change[activation_type]

print('Training:\tAverage difference between target and output: {0}').format(
    nn.average_miss(target_train, output_train))

# Testing
print("----------------------------------\nTest\n----------------------------------")
for epoch in range(max_epochs):
    error = 0
    for index, value in enumerate(input_test):
        output_test[index] = nn.forward(value)
        error += nn.backward(target_test[index])

print('Testing:\tAverage difference between target and output: {0}').format(
    nn.average_miss(target_test, output_test))
