import numpy as np
from statsmodels.tsa.stattools import grangercausalitytests

class GrangerTest:
    def __init__(self, max_lag=2, alpha=0.05):
        self.max_lag = max_lag
        self.alpha = alpha

    def test(self, series_x, series_y):
        if len(series_x) < self.max_lag + 5:
            return False

        data = np.column_stack([series_y, series_x])
        results = grangercausalitytests(data, maxlag=self.max_lag, verbose=False)

        for lag in range(1, self.max_lag + 1):
            p_value = results[lag][0]['ssr_ftest'][1]
            if p_value < self.alpha:
                return True

        return False