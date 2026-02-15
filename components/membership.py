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

    def forward(self, x: torch.Tensor, return_parts: bool = False):
        batch = x.shape[0]
        s_pos = torch.nn.functional.softplus(self.s) + 1e-6

        x_exp = x.unsqueeze(1).expand(batch, self.K, self.n_inputs)
        c_exp = self.c.unsqueeze(0).expand(batch, self.K, self.n_inputs)
        s_exp = s_pos.unsqueeze(0).expand(batch, self.K, self.n_inputs)

        mu_gauss = torch.exp(-((x_exp - c_exp) ** 2) / (2.0 * (s_exp ** 2)))
        mu_sig = torch.sigmoid((x_exp - c_exp) / (s_exp))

        if self.s_mode == "C":
            blend = torch.sigmoid(self.C)
        else:
            alpha2 = self.alpha**2
            beta2 = self.beta**2
            blend = alpha2 / (alpha2 + beta2 + self.eps_s)

        blend_exp = blend.unsqueeze(0).expand(batch, self.K, self.n_inputs)
        mu_blend = blend_exp * mu_gauss + (1.0 - blend_exp) * mu_sig
        w = torch.prod(mu_blend, dim=2)

        if return_parts:
            return w, mu_gauss, mu_sig, blend_exp, mu_blend

        return w


class GIFT(nn.Module):
    """
    GIFT neuron per (rule i, dimension j):
      mu = d+*mu+ + d-*mu- + dg*mug + dl*mul + dr*mur
    where d = softmax(logits) over 5 choices.

    Trainable per (K, n_inputs):
      - m: center
      - phi: gaussian spread (sigma)
      - s: sigmoid slope / temperature (optional but useful)
      - logits: (s+, s-, sg, sl, sr) -> softmax -> (d+, d-, dg, dl, dr)
    """

    def __init__(self, centers_init, spreads_init, eps: float = 1e-6):
        super().__init__()
        self.K = int(centers_init.shape[0])
        self.n_inputs = int(centers_init.shape[1])
        self.eps = eps

        m0 = torch.tensor(centers_init, dtype=torch.float32) if not torch.is_tensor(centers_init) else centers_init.float()
        phi0 = torch.tensor(spreads_init, dtype=torch.float32) if not torch.is_tensor(spreads_init) else spreads_init.float()

        # Gaussian parameters
        self.m = nn.Parameter(m0)        # (K, D)
        self.phi = nn.Parameter(phi0)    # (K, D)  -> sigma via softplus

        # Optional separate slope for sigmoids
        self.s = nn.Parameter(torch.ones(self.K, self.n_inputs, dtype=torch.float32))

        # Gating logits per (K, D, 5): [+, -, g, l, r]
        # Initialize to favor "+" (gaussian) a bit, if you want:
        logits0 = torch.zeros(self.K, self.n_inputs, 5, dtype=torch.float32)
        logits0[..., 0] = 1.0  # mild preference for mu+
        self.logits = nn.Parameter(logits0)

    def forward(self, x: torch.Tensor, return_parts: bool = False):
        """
        x: (B, D)
        returns:
          w: (B, K) rule firing strengths
        optionally returns parts for debugging/plots
        """
        B, D = x.shape
        assert D == self.n_inputs, f"Expected input dim {self.n_inputs}, got {D}"

        # enforce positivity
        sigma = F.softplus(self.phi) + self.eps      # (K, D)
        slope = F.softplus(self.s) + self.eps        # (K, D)

        # expand
        x_exp = x.unsqueeze(1).expand(B, self.K, D)          # (B, K, D)
        m_exp = self.m.unsqueeze(0).expand(B, self.K, D)     # (B, K, D)
        sig_exp = sigma.unsqueeze(0).expand(B, self.K, D)    # (B, K, D)
        slp_exp = slope.unsqueeze(0).expand(B, self.K, D)    # (B, K, D)

        # --- 5 candidate membership functions ---
        # mu+ : Gaussian (0..1)
        mu_pos = torch.exp(-((x_exp - m_exp) ** 2) / (2.0 * (sig_exp ** 2)))

        # mu- : Negation (0..1), "upside-down" gaussian
        mu_neg = 1.0 - mu_pos

        # mug : "greater" sigmoid (0..1), increases with x
        mu_g = torch.sigmoid((x_exp - m_exp) / slp_exp)

        # mul : "less" sigmoid (0..1), decreases with x
        # This is 1 for x << m, 0.5 at x=m, and 0 for x >> m (standard mirrored sigmoid)
        mu_l = torch.sigmoid((m_exp - x_exp) / slp_exp)

        # mur : relaxation constant 1
        mu_r = torch.ones_like(mu_pos)

        # --- softmax gate d over 5 choices ---
        d = torch.softmax(self.logits, dim=-1)               # (K, D, 5)
        d_exp = d.unsqueeze(0).expand(B, self.K, D, 5)       # (B, K, D, 5)

        # stack mus: (B, K, D, 5)
        mus = torch.stack([mu_pos, mu_neg, mu_g, mu_l, mu_r], dim=-1)

        # weighted sum: (B, K, D)
        mu_blend = torch.sum(d_exp * mus, dim=-1)

        # T-norm (product) over dimensions -> firing strength per rule: (B, K)
        w = torch.prod(mu_blend, dim=2)

        if return_parts:
            return w, mu_pos, mu_neg, mu_g, mu_l, mu_r, d, mu_blend

        return w
