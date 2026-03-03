class DriftDecisionEngine:
    def __init__(self, slope_threshold=0.001):
        self.slope_threshold = slope_threshold

    def detect(self, slope, granger_flag):
        return abs(slope) > self.slope_threshold and granger_flag