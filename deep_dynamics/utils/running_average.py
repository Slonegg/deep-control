import numpy as np


class RunningAverage(object):
    def __init__(self, num_average):
        self.x = []
        self.i = 0
        self.num_average = num_average

    def reset(self):
        self.x = []
        self.i = 0

    def add(self, x):
        if len(self.x) < self.num_average:
            self.x.append(x)
        else:
            self.x[self.i] = x
            self.i = (self.i + 1) % len(self.x)

        return np.sum(self.x) / len(self.x)
