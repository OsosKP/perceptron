# from multilayer_perceptron import MultiLayerPerceptron
from mlp import MultiLayerPerceptron
import numpy as np
from math import exp, pow

number_inputs = 2
# input = [0.05, 0.1]
# input = [1, 1]
input = [[0, 0], [0, 1], [1, 0], [1, 1]]
number_hidden_units = 2
number_outputs = 1
target = np.array([0, 1, 1, 0])
# target = np.array([0])
# target = np.array([0.01, 0.99, .5])
learning_rate = .2
max_epochs = 10000
activation_type = "tanh"
output = np.zeros(target.shape)

nn = MultiLayerPerceptron(
    number_inputs, number_hidden_units, number_outputs, activation_type)
# for epoch in range(max_epochs):
#     output = nn.forward(input)
#     error = nn.backward(target)
#     nn.update_weights(learning_rate)
#     print('Epoch:\t{0}\tError:\t{1}\nOutput:\t{2}').format(
#         epoch, error, output)

for epoch in range(max_epochs):
    error = 0
    for index, value in enumerate(input):
        output[index] = nn.forward(value)
        error += nn.backward(target[index])
        nn.update_weights(learning_rate)
    print('Epoch:\t{0}\tError:\t{1}').format(epoch, error)

print(output)
