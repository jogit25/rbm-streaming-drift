import numpy as np
from .sliding_window import SlidingWindowRegression
from .granger_test import GrangerTest
from .decision_engine import DriftDecisionEngine

class RRBMDriftDetector:
    def __init__(self, window_size=50, slope_threshold=0.001):
        self.monitor = SlidingWindowRegression(window_size)
        self.granger = GrangerTest()
        self.decision = DriftDecisionEngine(slope_threshold)
        self.recon_series = []

    def update_and_check(self, recon_error, classifier_errors):
        self.recon_series.append(recon_error)
        self.monitor.update(recon_error)

        slope = self.monitor.compute_slope()

        granger_flag = False
        if len(self.recon_series) > 10 and len(classifier_errors) > 10:
            granger_flag = self.granger.test(
                np.array(self.recon_series),
                np.array(classifier_errors)
            )

        return self.decision.detect(slope, granger_flag)