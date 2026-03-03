import numpy as np

class SlidingWindowRegression:
    def __init__(self, window_size=50):
        self.window_size = window_size
        self.values = []

    def update(self, value):
        self.values.append(value)
        if len(self.values) > self.window_size:
            self.values.pop(0)

    def compute_slope(self):
        if len(self.values) < 2:
            return 0.0

        y = np.array(self.values)
        x = np.arange(len(y))

        x_mean = x.mean()
        y_mean = y.mean()

        numerator = np.sum((x - x_mean) * (y - y_mean))
        denominator = np.sum((x - x_mean) ** 2) + 1e-8

        return numerator / denominator