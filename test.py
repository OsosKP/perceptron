# from multilayer_perceptron import MultiLayerPerceptron
from mlp import MultiLayerPerceptron
import numpy as np
from math import exp, pow

# input = [0.05, 0.1]
input = [[0, 0], [0, 1], [1, 0], [1, 1]]
# target = np.array([0.01, 0.99])
target = np.array([0, 1, 1, 0])
learning_rate = 0.9
max_epochs = 100
output = np.zeros(target.shape)

nn = MultiLayerPerceptron(2, 8, 1)
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
    print('Epoch:\t{0}\tError:\t{1}\nOutput:\t{2}').format(
        epoch, error, output)

print(output)
