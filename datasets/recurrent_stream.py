import numpy as np

class RecurrentStream:

    def __init__(self):
        self.step = 0

    def next_instance(self):

        x = np.random.rand(3)

        period = (self.step // 5000) % 2

        threshold = 0.5 if period == 0 else 0.8

        y = 1 if x[0] + x[1] > threshold else 0

        self.step += 1
        return x, y