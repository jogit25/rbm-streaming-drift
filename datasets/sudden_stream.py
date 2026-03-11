import numpy as np
#sudden drift
class SuddenStream:

    def __init__(self):
        self.step = 0

    def next_instance(self):

        x = np.random.rand(3)

        if self.step < 5000:
            threshold = 0.5
        else:
            threshold = 0.8

        y = 1 if x[0] + x[1] > threshold else 0

        self.step += 1
        return x, y
