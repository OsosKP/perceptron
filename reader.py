import numpy as np


class Reader:
    def __init__(self):
        read = []
        target = []
        # target_line = [0] * 26
        input_values = []
        with open('letter-recognition.data', 'r') as file:
            for line in file:
                # target_line = [0] * 26
                read = line.replace('\n', '').split(',')
                target.append(float(ord(read.pop(0))) / 100)
                # target_line[ord(read.pop(0)) - 65] = 1
                # target.append(target_line)
                input_values.append(map(lambda x: x / 100, map(float, read)))
        self.values = {
            "target": np.asarray(target),
            "input": np.asarray(input_values)
        }

    def result(self, arg):
        return self.values[arg]

    def alphabet(self, index):
        return chr(index + 65)

    def letter(self, ascii_value):
        return chr(int(round(ascii_value * 100)))
