import numpy as np
from statsmodels.tsa.stattools import grangercausalitytests


class GrangerTest:

    def test(self, recon_series, classifier_series):

        # variance check
        if np.std(recon_series) == 0 or np.std(classifier_series) == 0:
            return False

        data = np.column_stack([classifier_series, recon_series])

        try:
            results = grangercausalitytests(data, maxlag=1, verbose=False)
            p_value = results[1][0]['ssr_ftest'][1]

            return p_value < 0.05

        except Exception:
            return False