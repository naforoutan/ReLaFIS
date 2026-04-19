import torch
import torch.nn as nn
import torch.nn.functional as F


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

        mu = torch.exp(-((x_exp - c_exp) ** 2) / (2.0 * (s_exp ** 2)))
        firing = torch.exp(torch.sum(torch.log(mu + 1e-6), dim=2))
        return firing


class GIFT(nn.Module):
    """
    Hierarchical GIFT:

    Step 1: (mu_pos vs mu_neg)
    Step 2: (mu_g vs mu_l)
    Step 3: combine both
    Step 4: relaxation
    """

    def __init__(self, centers_init, spreads_init, eps: float = 1e-6):
        super().__init__()
        self.K = int(centers_init.shape[0])
        self.n_inputs = int(centers_init.shape[1])
        self.eps = eps

        m0 = torch.tensor(centers_init, dtype=torch.float32) if not torch.is_tensor(centers_init) else centers_init.float()
        phi0 = torch.tensor(spreads_init, dtype=torch.float32) if not torch.is_tensor(spreads_init) else spreads_init.float()

        # Gaussian params
        self.m = nn.Parameter(m0)        # (K, D)
        self.phi = nn.Parameter(phi0)    # (K, D)
        self.s = nn.Parameter(torch.ones(self.K, self.n_inputs, dtype=torch.float32))

        # hierarchical gates
        self.alpha_logits = nn.Parameter(torch.zeros(self.K, self.n_inputs))  # pos vs neg
        self.beta_logits  = nn.Parameter(torch.zeros(self.K, self.n_inputs))  # g vs l
        self.gamma_logits = nn.Parameter(torch.zeros(self.K, self.n_inputs))  # sym vs dir
        self.delta_logits = nn.Parameter(torch.zeros(self.K, self.n_inputs))  # relaxation

        # smart initialization
        self.alpha_logits.data.fill_(0.0)   # prefer mu_pos
        self.beta_logits.data.fill_(0.0)    # neutral
        self.gamma_logits.data.fill_(0.0)   # prefer Gaussian branch
        self.delta_logits.data.fill_(0.0)   # avoid relaxation early

    def forward(self, x: torch.Tensor, return_parts: bool = False):
        B, D = x.shape
        assert D == self.n_inputs, f"Expected input dim {self.n_inputs}, got {D}"

        # enforce positivity
        sigma = F.softplus(self.phi) + self.eps
        slope = F.softplus(self.s) + self.eps

        # expand
        x_exp = x.unsqueeze(1).expand(B, self.K, D)
        m_exp = self.m.unsqueeze(0).expand(B, self.K, D)
        sig_exp = sigma.unsqueeze(0).expand(B, self.K, D)
        slp_exp = slope.unsqueeze(0).expand(B, self.K, D)

        # ---- base membership functions ----
        mu_pos = torch.exp(-((x_exp - m_exp) ** 2) / (2.0 * (sig_exp ** 2)))
        mu_neg = torch.exp(-mu_pos)
        mu_g   = torch.sigmoid((x_exp - m_exp) / slp_exp)
        mu_l   = torch.sigmoid((m_exp - x_exp) / slp_exp)

        # ---- hierarchical gates ----
        alpha = torch.sigmoid(self.alpha_logits).unsqueeze(0).expand(B, self.K, D)
        beta  = torch.sigmoid(self.beta_logits).unsqueeze(0).expand(B, self.K, D)
        gamma = torch.sigmoid(self.gamma_logits).unsqueeze(0).expand(B, self.K, D)
        delta = torch.sigmoid(self.delta_logits).unsqueeze(0).expand(B, self.K, D)

        # Step 1: symmetric branch
        mu_sym = alpha * mu_pos + (1.0 - alpha) * mu_neg

        # Step 2: directional branch
        mu_dir = beta * mu_g + (1.0 - beta) * mu_l

        # Step 3: combine
        mu_comb = gamma * mu_sym + (1.0 - gamma) * mu_dir

        # Step 4: relaxation (controlled)
        mu = delta * mu_comb + (1.0 - delta)

        # ---- rule firing ----
        w = torch.mean(mu, dim=2)

        if return_parts:
            return w, mu_pos, mu_neg, mu_g, mu_l, mu_sym, mu_dir, mu_comb, mu, alpha, beta, gamma, delta

        return w

