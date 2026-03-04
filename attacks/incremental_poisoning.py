import numpy as np

class IncrementalPoisoning:

    def __init__(self):
        self.step = 0

    def poison(self, x, y):

        ratio = min(0.25, self.step * 0.000025)

        if np.random.rand() < ratio:
            y = 1 - y

        self.step += 1
        return x, y