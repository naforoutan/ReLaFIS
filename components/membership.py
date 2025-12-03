import torch
import torch.nn as nn


class SimpleGaussianMF(nn.Module):
    """Gaussian membership functions initialized from FCM prototypes."""

    def __init__(self, centers_init, spreads_init):
        super().__init__()
        self.K = centers_init.shape[0]
        self.n_inputs = centers_init.shape[1]

        centers_t = (
            torch.tensor(centers_init, dtype=torch.float32)
            if not torch.is_tensor(centers_init)
            else centers_init.float()
        )
        spreads_t = (
            torch.tensor(spreads_init, dtype=torch.float32)
            if not torch.is_tensor(spreads_init)
            else spreads_init.float()
        )

        self.centers = nn.Parameter(centers_t)
        self.spreads = nn.Parameter(spreads_t)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_exp = x.unsqueeze(1)
        c_exp = self.centers.unsqueeze(0)
        s_exp = torch.nn.functional.softplus(self.spreads).unsqueeze(0)
        s_exp = torch.clamp(s_exp, min=1e-6)

        mu = torch.exp(-((x_exp - c_exp) ** 2) / (2.0 * (s_exp ** 2)))
        firing = torch.prod(mu, dim=2)
        return firing


class GaussianSigmoidMF(nn.Module):
    """Blended Gaussian + sigmoid membership functions (mu+)."""

    def __init__(self, centers_init, spreads_init, s_mode: str = "alpha_beta", eps_s: float = 1e-6):
        super().__init__()
        self.K = int(centers_init.shape[0])
        self.n_inputs = int(centers_init.shape[1])
        self.s_mode = s_mode
        self.eps_s = eps_s

        centers_t = (
            torch.tensor(centers_init, dtype=torch.float32)
            if not torch.is_tensor(centers_init)
            else centers_init.float()
        )
        spreads_t = (
            torch.tensor(spreads_init, dtype=torch.float32)
            if not torch.is_tensor(spreads_init)
            else spreads_init.float()
        )

        self.c = nn.Parameter(centers_t)
        self.s = nn.Parameter(spreads_t)

        if s_mode == "C":
            self.C = nn.Parameter(torch.zeros(self.K, self.n_inputs, dtype=torch.float32))
        elif s_mode == "alpha_beta":
            self.alpha = nn.Parameter(torch.ones(self.K, self.n_inputs, dtype=torch.float32))
            self.beta = nn.Parameter(torch.ones(self.K, self.n_inputs, dtype=torch.float32))
        else:
            raise ValueError("s_mode must be 'C' or 'alpha_beta'")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        s_pos = torch.nn.functional.softplus(self.s) + 1e-6

        x_exp = x.unsqueeze(1).expand(batch, self.K, self.n_inputs)
        c_exp = self.c.unsqueeze(0).expand(batch, self.K, self.n_inputs)
        s_exp = s_pos.unsqueeze(0).expand(batch, self.K, self.n_inputs)

        mu_gauss = torch.exp(-((x_exp - c_exp) ** 2) / (2.0 * (s_exp ** 2) + 1e-12))
        mu_sig = torch.sigmoid((x_exp - c_exp) / (s_exp + 1e-12))

        if self.s_mode == "C":
            blend = torch.sigmoid(self.C)
        else:
            alpha2 = self.alpha**2
            beta2 = self.beta**2
            blend = alpha2 / (alpha2 + beta2 + self.eps_s)

        blend_exp = blend.unsqueeze(0).expand(batch, self.K, self.n_inputs)
        mu_blend = blend_exp * mu_gauss + (1.0 - blend_exp) * mu_sig
        w = torch.prod(mu_blend, dim=2)
        return torch.clamp(w, min=1e-8)

