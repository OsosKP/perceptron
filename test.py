# from multilayer_perceptron import MultiLayerPerceptron
from mlp import MultiLayerPerceptron
import numpy as np
from math import exp, pow

input = [0.05, 0.1]
target = np.array([0.01, 0.99])
learning_rate = 0.5
max_epochs = 10
output = np.zeros(target.shape)

# nn = MultiLayerPerceptron(2, 2, 1)

# for epoch in range(max_epochs):
#     error = 0
#     for index, value in enumerate(input):
#         output[index] = nn.forward(value)
#         error += nn.backward(target[index])
#         if index % 3 == 0:
#             nn.update_weights(learning_rate)
#     # print('Epoch:\t{0}\tError:\t{1}').format(epoch, error)
# print(output)

nn = MultiLayerPerceptron(2, 2, 2)
output = nn.forward(input)
error = nn.backward(target)
nn.update_weights(learning_rate)
