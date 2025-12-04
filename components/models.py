import torch
import torch.nn as nn
import numpy as np
import skfuzzy as fuzz 


from .membership import GaussianSigmoidMF, SimpleGaussianMF


def fcm_initialize(X: np.ndarray, K: int, m: float = 2.0, error: float = 1e-5, maxiter: int = 2000):
    """
    Run FCM to get rule antecedent prototypes.
    Args:
        X: (N, n_inputs) numpy array
        K: number of clusters / rules
        m: fuzziness parameter
        error: termination tolerance
        maxiter: maximum number of iterations
    Returns:
        centers: (K, n_inputs)
        spreads: (K, n_inputs)
    """
    X_t = X.T
    cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(
        X_t,
        c=K,
        m=m,
        error=error,
        maxiter=maxiter,
        init=None,
    )
    N = X.shape[0]

    spreads = []
    for k in range(K):
        weights = u[k].reshape(N, 1)
        diff = X - cntr[k]
        denom = np.sum(weights) + 1e-12
        var = np.sum(weights * (diff ** 2), axis=0) / denom
        spreads.append(np.sqrt(np.clip(var, 1e-6, None)))

    return np.array(cntr, dtype=np.float32), np.array(spreads, dtype=np.float32)




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

    def __init__(self, centers_init, spreads_init, s_mode: str = "alpha_beta", n_outputs: int = 1):
        super().__init__()
        self.K = centers_init.shape[0]
        self.n_inputs = centers_init.shape[1]
        self.n_outputs = n_outputs
        self.mf_layer = GaussianSigmoidMF(centers_init, spreads_init, s_mode=s_mode)
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
        y = torch.sum(w_norm.unsqueeze(-1) * rule_outputs, dim=1)
        return y
