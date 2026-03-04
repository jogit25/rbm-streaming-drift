class ConceptPoisoning:

    def __init__(self, n_concepts, stream_length):

        self.step = 0
        self.windows = []

        spacing = stream_length // (n_concepts + 1)

        for i in range(n_concepts):

            start = spacing * (i + 1)
            end = start + 500

            self.windows.append((start, end))


    def poison(self, x, y):

        for start, end in self.windows:
            if start <= self.step <= end:
                y = 1 - y

        self.step += 1
        return x, y