import numpy as np
#gradual drift
class SEAStream:

    def __init__(self):
        self.step = 0

    def next_instance(self):

        x = np.random.rand(3)

        threshold = 0.5 + 0.2*np.sin(self.step/2000)

        y = 1 if x[0] + x[1] > threshold else 0

        self.step += 1
        return x, y
