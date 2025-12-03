import numpy as np
import skfuzzy as fuzz


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

