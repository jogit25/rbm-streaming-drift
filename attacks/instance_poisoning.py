import random

class InstancePoisoning:

    def __init__(self, ratio):
        self.ratio = ratio

    def poison(self, x, y):

        if random.random() < self.ratio:
            y = 1 - y

        return x, y