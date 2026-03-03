import torch
import torch.nn as nn
import torch.nn.functional as F
from .robust_gradient import RobustGradientMixin

class RobustSupervisedRBM(nn.Module, RobustGradientMixin):
    def __init__(self, n_visible, n_hidden, n_classes, delta=0.95):
        nn.Module.__init__(self)
        RobustGradientMixin.__init__(self, delta)

        self.V = n_visible
        self.H = n_hidden
        self.Z = n_classes

        self.W = nn.Parameter(torch.randn(n_visible, n_hidden) * 0.01)
        self.U = nn.Parameter(torch.randn(n_hidden, n_classes) * 0.01)

        self.a = nn.Parameter(torch.zeros(n_visible))
        self.b = nn.Parameter(torch.zeros(n_hidden))
        self.c = nn.Parameter(torch.zeros(n_classes))

        self.noise_mean = nn.Parameter(torch.zeros(n_visible))
        self.noise_var = nn.Parameter(torch.ones(n_visible))

    def sample_hidden(self, v):
        prob = torch.sigmoid(v @ self.W + self.b)
        return prob, torch.bernoulli(prob)

    def sample_visible(self, h):
        prob = torch.sigmoid(h @ self.W.T + self.a)
        return prob, torch.bernoulli(prob)

    def contrastive_divergence(self, v, z, k=1):
        h_prob, h = self.sample_hidden(v)

        v_k = v
        for _ in range(k):
            v_prob, v_k = self.sample_visible(h)
            h_prob, h = self.sample_hidden(v_k)

        return v_k, h_prob

    def robust_update(self, v, z, lr=0.01):
        v_recon, h_recon = self.contrastive_divergence(v, z)
        h_data_prob, _ = self.sample_hidden(v)

        E_data = torch.einsum("bi,bj->ij", v, h_data_prob)
        E_recon = torch.einsum("bi,bj->ij", v_recon, h_recon)

        logits = h_data_prob @ self.U
        losses = F.cross_entropy(logits, z.argmax(dim=1), reduction="none")

        theta_hat = self.compute_truncation_factor(losses)
        grad_W = E_recon - theta_hat * E_data

        with torch.no_grad():
            self.W -= lr * grad_W