import torch
import math

class RobustGradientMixin:
    def __init__(self, delta=0.95):
        self.delta = delta

    def compute_dispersion(self, losses):
        gamma = losses.mean()
        centered = losses - gamma
        sigma_hat = torch.median(torch.abs(centered)) + 1e-8
        return sigma_hat

    def compute_truncation_factor(self, losses):
        n = len(losses)
        sigma_hat = self.compute_dispersion(losses)
        s_i = sigma_hat * math.sqrt(n / math.log(2 / (1 - self.delta)))

        normalized = (losses - losses.mean()) / (s_i + 1e-8)
        rho_vals = torch.log1p(normalized ** 2)
        theta_hat = rho_vals.mean()

        return theta_hat