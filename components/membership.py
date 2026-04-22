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
    def __init__(self, centers_init, spreads_init, eps: float = 1e-6):
        super().__init__()
        self.K = int(centers_init.shape[0])
        self.n_inputs = int(centers_init.shape[1])
        self.eps = eps
        
        m0 = torch.tensor(centers_init, dtype=torch.float32) if not torch.is_tensor(centers_init) else centers_init.float()
        phi0 = torch.tensor(spreads_init, dtype=torch.float32) if not torch.is_tensor(spreads_init) else spreads_init.float()
        
        self.m = nn.Parameter(m0)
        self.phi = nn.Parameter(phi0)
        self.s = nn.Parameter(torch.ones(self.K, self.n_inputs))
        
        # hierarchical gates
        self.alpha_logits = nn.Parameter(torch.randn(self.K, self.n_inputs) * 0.5)  # pos vs neg
        self.beta_logits  = nn.Parameter(torch.randn(self.K, self.n_inputs) * 0.5)  # g vs l
        self.gamma_logits = nn.Parameter(torch.randn(self.K, self.n_inputs) * 0.5)  # sym vs dir
        self.rho_logits   = nn.Parameter(torch.zeros(self.K, self.n_inputs))  # relaxation
        # rho_logits get special treatment to start with almost no relaxation
        self.rho_logits.data.fill_(-3.0)
    
    def forward(self, x: torch.Tensor, return_parts: bool = False):
        B, D = x.shape
        assert D == self.n_inputs
        
        sigma = F.softplus(self.phi) + self.eps
        slope = F.softplus(self.s) + self.eps
        
        x_exp = x.unsqueeze(1).expand(B, self.K, D)
        m_exp = self.m.unsqueeze(0).expand(B, self.K, D)
        sig_exp = sigma.unsqueeze(0).expand(B, self.K, D)
        slp_exp = slope.unsqueeze(0).expand(B, self.K, D)
        
        # base membership functions
        mu_pos = torch.exp(-((x_exp - m_exp) ** 2) / (2.0 * (sig_exp ** 2)))
        mu_neg = 1 - mu_pos
        mu_g = torch.sigmoid((x_exp - m_exp) / slp_exp)
        mu_l = torch.sigmoid((m_exp - x_exp) / slp_exp)
        
        # hierarchical gates
        alpha = torch.sigmoid(self.alpha_logits).unsqueeze(0).expand(B, self.K, D)
        beta  = torch.sigmoid(self.beta_logits).unsqueeze(0).expand(B, self.K, D)
        gamma = torch.sigmoid(self.gamma_logits).unsqueeze(0).expand(B, self.K, D)
        rho   = torch.sigmoid(self.rho_logits).unsqueeze(0).expand(B, self.K, D)
        
        # combine branches
        mu_sym = alpha * mu_pos + (1 - alpha) * mu_neg
        mu_dir = beta * mu_g + (1 - beta) * mu_l
        mu_comb = gamma * mu_sym + (1 - gamma) * mu_dir
        
        # relaxation: ω = (1-ρ)·μ + ρ
        mu = (1 - rho) * mu_comb + rho
        
        w = torch.exp(torch.mean(torch.log(mu + 1e-6), dim=2))
        w = w + 1e-8
        
        if return_parts:
            return w, mu_pos, mu_neg, mu_g, mu_l, mu_sym, mu_dir, mu_comb, mu, alpha, beta, gamma, rho
        return w


