from mlp import MultiLayerPerceptron
from reader import Reader
import numpy as np

activation_type = "tanh"
learning_rate = {
    "sigmoid": 0.8,
    "tanh": 0.2,
    "relu": 0.2
}
learning_rate_change = {
    "sigmoid": 1.1,
    "tanh": 0.9,
    "relu": 0.9
}

max_epochs = 1500

input = []
target = []
reader = Reader()
input = reader.result("input")
target = reader.result("target")

input_train = input[0:16000]
input_test = input[16000:20000]
target_train = target[0:16000]
target_test = target[16000:20000]

output_train = np.zeros(target_train.shape)
output_test = np.zeros(target_test.shape)

options = {
    "number_inputs": input_train.shape[1],
    "number_hidden_units": 10,
    "number_outputs": 1,
    "activation_type": activation_type
}

nn = MultiLayerPerceptron(options)

# Training
print("----------------------------------\nTrain\n----------------------------------")
for epoch in range(max_epochs):
    error = 0
    for index, value in enumerate(input_train):
        output_train[index] = round(nn.forward(value), 2)
        error += nn.backward(target_train[index])
        nn.update_weights(learning_rate[activation_type])
    print('Epoch:\t{0}\tError:\t{1}').format(epoch, error)
    if (epoch % 150 == 0):
        learning_rate[activation_type] *= learning_rate_change[activation_type]

print('Training:\tAverage difference between target and output: {0}').format(
    nn.average_miss(target_train, output_train))

correct = 0
# Testing
print("----------------------------------\nTest\n----------------------------------")
error = 0
for index, value in enumerate(input_test):
    output_test[index] = round(nn.forward(value), 2)
    error += nn.backward(target_test[index])
    if (output_test[index] == target_test[index]):
        correct += 1

print('Testing:\tError: {0}\nTesting:\tDifference between target and output: {1}').format(
    error, nn.average_miss(target_test, output_test))

print('Correct Guesses: {0}').format(correct)
