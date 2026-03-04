import numpy as np
from .sliding_window import SlidingWindowRegression
from .granger_test import GrangerTest
from .decision_engine import DriftDecisionEngine


class RRBMDriftDetector:

    def __init__(self, window_size=50, slope_threshold=0.001):

        self.window_size = window_size
        self.monitor = SlidingWindowRegression(window_size)

        self.granger = GrangerTest()
        self.decision = DriftDecisionEngine(slope_threshold)

        self.recon_series = []
        self.classifier_series = []

        # NEW: counter for periodic Granger
        self.counter = 0


    def update_and_check(self, recon_error, classifier_error):

        self.counter += 1

        # store series
        self.recon_series.append(recon_error)
        self.classifier_series.append(classifier_error)

        # keep only window
        if len(self.recon_series) > self.window_size:
            self.recon_series.pop(0)

        if len(self.classifier_series) > self.window_size:
            self.classifier_series.pop(0)

        # update regression monitor
        self.monitor.update(recon_error)

        # compute slope
        slope = self.monitor.compute_slope()

        granger_flag = False

        # run Granger only periodically
        if (
            len(self.recon_series) >= self.window_size
            and self.counter % 20 == 0
        ):

            granger_flag = self.granger.test(
                np.array(self.recon_series),
                np.array(self.classifier_series)
            )

        return self.decision.detect(slope, granger_flag)