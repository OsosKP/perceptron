import numpy as np
from math import sin


class MLP_Factory:
    def sin_finder(self, activation_type):
        options = {
            "number_inputs": 4,
            "number_hidden_units": 8,
            "number_outputs": 1,
            "activation_type": activation_type,
        }
        return options

    def generate_sin_values(self, input, output):
        for i in range(input.shape[0]):
            element = np.random.uniform(-1, 1, size=(4))
            output[i] = sin(element[0] - element[1] +
                            element[2] - element[3])
            input[i] = element
