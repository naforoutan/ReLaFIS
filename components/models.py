import torch
import torch.nn as nn
import numpy as np

from .membership import GaussianSigmoidMF, SimpleGaussianMF


def fcm_initialize(X: np.ndarray, K: int, m: float = 2.0, error: float = 1e-5, maxiter: int = 2000):
    import skfuzzy as fuzz 
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

def init_mf_params(X_train, K, method="fcm", scale=1.0, s_mode="alpha_beta", seed=42):
    np.random.seed(seed)

    if isinstance(X_train, torch.Tensor):
        X_train = X_train.detach().cpu().numpy()

    n_features = X_train.shape[1]

    if method == "fcm":
        centers, spreads = fcm_initialize(X_train, K)  # your FCM function
    elif method == "random_uniform":
        # Uniformly distributed centers in the range of X_train
        if isinstance(X_train, torch.Tensor):
            X_train = X_train.detach().cpu().numpy()
        X_min = X_train.min(axis=0, keepdims=True)
        X_max = X_train.max(axis=0, keepdims=True)
        centers = np.random.uniform(X_min, X_max, size=(K, n_features)).astype(np.float32)
        # Large spreads uniformly sampled to cover wide area
        spreads = np.random.uniform(0.5*scale, 1.5*scale, size=(K, n_features)).astype(np.float32)
    else:
        raise ValueError(f"Unknown method: {method}")

    return centers, spreads



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
    uses_reconstruction = True

    def __init__(self, centers_init, spreads_init, s_mode: str = "alpha_beta", n_outputs: int = 1):
        super().__init__()
        self.K = centers_init.shape[0]
        self.n_inputs = centers_init.shape[1]
        self.n_outputs = n_outputs
        self.mf_layer = GaussianSigmoidMF(centers_init, spreads_init, s_mode=s_mode)
        self.consequents = nn.Parameter(
            torch.randn(self.K, self.n_inputs + 1, self.n_outputs, dtype=torch.float32) * 0.1
        )
        self.reconstructor = nn.Linear(self.K, self.n_inputs, bias=False)


    def forward(self, x: torch.Tensor, return_recon: bool = False):
        # ---- rule firing
        w = self.mf_layer(x)
        w_sum = torch.sum(w, dim=1, keepdim=True) + 1e-8
        w_norm = w / w_sum            # φ (B, K)

        # ---- TS consequents (unchanged)
        weights = self.consequents[:, :-1, :]
        bias = self.consequents[:, -1, :]
        linear = torch.einsum("bi,kio->bko", x, weights)
        rule_outputs = linear + bias.unsqueeze(0)
        y = torch.sum(w_norm.unsqueeze(-1) * rule_outputs, dim=1)

        if not return_recon:
            return y

        # ---- reconstruction
        x_hat = self.reconstructor(w_norm)

        return y, x_hat


def create_anfis_model(mf_type, centers_init, spreads_init, n_outputs=1, s_mode="alpha_beta"):
    if mf_type == "simple":
        return ANFISSimple(centers_init, spreads_init, n_outputs=n_outputs)
    elif mf_type == "advanced":
        return ANFISAdvanced(centers_init, spreads_init, s_mode=s_mode, n_outputs=n_outputs)
    else:
        raise ValueError(f"Unknown mf_type: {mf_type}")
