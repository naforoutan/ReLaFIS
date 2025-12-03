import torch
import torch.nn as nn

from .membership import GaussianSigmoidMF, SimpleGaussianMF


class ANFISSimple(nn.Module):
    """Takagi–Sugeno ANFIS with Gaussian MFs (simple FCM init)."""

    def __init__(self, centers_init, spreads_init, n_outputs: int = 1):
        super().__init__()
        self.K = centers_init.shape[0]
        self.n_inputs = centers_init.shape[1]
        self.n_outputs = n_outputs
        self.mf_layer = SimpleGaussianMF(centers_init, spreads_init)
        self.consequents = nn.Parameter(
            torch.randn(self.K, self.n_inputs + 1, self.n_outputs, dtype=torch.float32) * 0.1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.mf_layer(x)
        w_sum = torch.sum(w, dim=1, keepdim=True) + 1e-8
        w_norm = w / w_sum

        weights = self.consequents[:, :-1, :]
        bias = self.consequents[:, -1, :]
        linear = torch.einsum("bi,kio->bko", x, weights)
        rule_outputs = linear + bias.unsqueeze(0)
        y_pred = torch.sum(w_norm.unsqueeze(-1) * rule_outputs, dim=1)
        return y_pred


class ANFISAdvanced(nn.Module):
    """Takagi–Sugeno ANFIS using blended Gaussian+sigmoid MFs."""

    def __init__(self, centers_init, spreads_init, s_mode: str = "alpha_beta"):
        super().__init__()
        self.K = centers_init.shape[0]
        self.n_inputs = centers_init.shape[1]
        self.mf_layer = GaussianSigmoidMF(centers_init, spreads_init, s_mode=s_mode)
        self.consequents = nn.Parameter(torch.randn(self.K, self.n_inputs + 1, dtype=torch.float32) * 0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.mf_layer(x)
        w_sum = torch.sum(w, dim=1, keepdim=True) + 1e-8
        w_norm = w / w_sum

        x_exp = x.unsqueeze(1).expand(-1, self.K, -1)
        cons_w = self.consequents[:, :-1].unsqueeze(0)
        linear_part = torch.sum(x_exp * cons_w, dim=2)
        bias = self.consequents[:, -1].unsqueeze(0)
        y_r = linear_part + bias
        logit = torch.sum(w_norm * y_r, dim=1, keepdim=True)
        return logit

