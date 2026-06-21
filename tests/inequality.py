"""
Synthetic differential inequality datasets.

GIFTSHIFT vs LitANFIS benchmark — inequality classification tasks.

These datasets are generated programmatically (no local files / network
access required). Labels are derived directly from a closed-form inequality
over the sampled features, so the "ground truth" boundary is known exactly
and can be used to sanity-check learned fuzzy rules / membership functions
against the true decision surface.

Each dataset has 2 000 samples and a fixed random seed, so results are
reproducible across runs:
  - LinearThreshold     : 2x1 + 3x2 > 7                          (sharp linear)
  - ExponentialGrowth    : dy/dx > k*y                            (sharp sigmoid)
  - LogisticGrowth       : dy/dx <= r*y*(1 - y/K)                 (curved nonlinear)
  - DampedOscillator     : d2x + 2*zeta*omega*dx + omega^2*x <= 0 (mixed coupled)
  - PDE (Heat Equation)  : du/dt < alpha * d2u/dx                 (mixed adaptive,
                           comb_weight stress test)

To add a new inequality dataset here:
  - Sample features with numpy using a fixed seed (np.random.default_rng(seed)).
  - Compute the label as a boolean/int column from the inequality formula.
  - Set self.df (features only, no label column) and self.target (the label
    Series), then call super().__init__() as usual.
"""

import numpy as np
import pandas as pd
from .base import Test


class LinearThreshold(Test):
    """Dataset 1 - Linear Threshold.

    Simplest possible classification boundary. Useful as a calibration
    check: both models should reach near-perfect accuracy here. If either
    fails on this dataset, something is wrong with training.

    Formula: 2*x1 + 3*x2 > 7
    Sampling: x1, x2 ~ Uniform(-5, 5)
    2 features, 2 000 samples, sharp linear boundary.
    """

    def __init__(self, *args, n_samples=2000, seed=0, **kwargs) -> None:
        rng = np.random.default_rng(seed)

        x1 = rng.uniform(-5, 5, n_samples)
        x2 = rng.uniform(-5, 5, n_samples)

        label = (2 * x1 + 3 * x2 > 7).astype(int)

        self.df = pd.DataFrame({"x1": x1, "x2": x2})
        self.target = pd.Series(label, name="label")

        print(f"✅ LinearThreshold generated: {self.df.shape[0]} samples, "
              f"{self.df.shape[1]} features, label balance "
              f"{dict(self.target.value_counts())}")

        super().__init__(*args, **kwargs)


class ExponentialGrowth(Test):
    """Dataset 2 - Exponential Growth.

    Tests whether the model can learn a multiplicative threshold. The
    boundary is a line in (y, dy/dx) space but scaled by k, making it
    non-trivial for pure Gaussian models.

    Formula: dy/dx > k*y
    Sampling: y ~ Uniform(0.1, 5), dy/dx ~ Uniform(-2, 4), k = 0.5
    2 features, 2 000 samples, sharp sigmoid boundary (sigmoid MF stress test).
    """

    def __init__(self, *args, n_samples=2000, seed=1, k=0.5, **kwargs) -> None:
        rng = np.random.default_rng(seed)

        y_val = rng.uniform(0.1, 5, n_samples)
        dy_dx = rng.uniform(-2, 4, n_samples)

        label = (dy_dx > k * y_val).astype(int)

        self.df = pd.DataFrame({"y_val": y_val, "dy_dx": dy_dx})
        self.target = pd.Series(label, name="label")

        print(f"✅ ExponentialGrowth generated: {self.df.shape[0]} samples, "
              f"{self.df.shape[1]} features, k={k}, label balance "
              f"{dict(self.target.value_counts())}")

        super().__init__(*args, **kwargs)


class LogisticGrowth(Test):
    """Dataset 3 - Logistic Growth.

    The right-hand side is a quadratic surface in (y, K) space. Tests
    curved, parameter-dependent boundaries. Including r and K as features
    makes the problem harder and more realistic.

    Formula: dy/dx <= r*y*(1 - y/K)
    Sampling: y ~ Uniform(0.1, K), dy/dx ~ Uniform(-2, 4), r=1, K=10
    4 features, 2 000 samples, curved nonlinear boundary.
    """

    def __init__(self, *args, n_samples=2000, seed=2, r=1.0, K=10.0, **kwargs) -> None:
        rng = np.random.default_rng(seed)

        y_val = rng.uniform(0.1, K, n_samples)
        dy_dx = rng.uniform(-2, 4, n_samples)
        r_arr = np.full(n_samples, r)
        K_arr = np.full(n_samples, K)

        label = (dy_dx <= r_arr * y_val * (1 - y_val / K_arr)).astype(int)

        self.df = pd.DataFrame({
            "y_val": y_val, "dy_dx": dy_dx, "r": r_arr, "K": K_arr,
        })
        self.target = pd.Series(label, name="label")

        print(f"✅ LogisticGrowth generated: {self.df.shape[0]} samples, "
              f"{self.df.shape[1]} features, r={r}, K={K}, label balance "
              f"{dict(self.target.value_counts())}")

        super().__init__(*args, **kwargs)


class DampedOscillator(Test):
    """Dataset 4 - Damped Oscillator.

    Most complex synthetic dataset. The boundary couples three physical
    quantities (position, velocity, acceleration) with two parameters
    (damping ratio, natural frequency). Underdamped vs overdamped regions
    create qualitatively different boundary shapes — a direct test of
    local TSK coupling.

    Formula: d2x + 2*zeta*omega*dx + omega^2*x <= 0
    Sampling: x ~ U(-3,3), dx ~ U(-4,4), d2x ~ U(-10,10),
              zeta ~ U(0.1,2), omega ~ U(0.5,3)
    5 features, 2 000 samples, mixed coupled boundary.
    """

    def __init__(self, *args, n_samples=2000, seed=3, **kwargs) -> None:
        rng = np.random.default_rng(seed)

        x = rng.uniform(-3, 3, n_samples)
        dx = rng.uniform(-4, 4, n_samples)
        d2x = rng.uniform(-10, 10, n_samples)
        zeta = rng.uniform(0.1, 2, n_samples)
        omega = rng.uniform(0.5, 3, n_samples)

        label = (d2x + 2 * zeta * omega * dx + (omega ** 2) * x <= 0).astype(int)

        self.df = pd.DataFrame({
            "x": x, "dx": dx, "d2x": d2x, "zeta": zeta, "omega": omega,
        })
        self.target = pd.Series(label, name="label")

        print(f"✅ DampedOscillator generated: {self.df.shape[0]} samples, "
              f"{self.df.shape[1]} features, label balance "
              f"{dict(self.target.value_counts())}")

        super().__init__(*args, **kwargs)


class PDE(Test):
    """Dataset 5 - Heat Equation (PDE).

    Discretised PDE inequality. The diffusivity parameter alpha acts as a
    scaling factor on the spatial term, creating a region boundary that
    shifts with alpha. This is the key stress test for GIFTSHIFT's
    comb_weight: near-equilibrium zones are Gaussian-like, far-from-
    equilibrium zones are sigmoid-like.

    Formula: du/dt < alpha * d2u/dx
    Sampling: u ~ U(-2,2), du/dt ~ U(-3,3), d2u/dx ~ U(-5,5), alpha ~ U(0.1,2)
    4 features, 2 000 samples, mixed adaptive boundary.
    """

    def __init__(self, *args, n_samples=2000, seed=4, **kwargs) -> None:
        rng = np.random.default_rng(seed)

        u = rng.uniform(-2, 2, n_samples)
        du_dt = rng.uniform(-3, 3, n_samples)
        d2u_dx = rng.uniform(-5, 5, n_samples)
        alpha = rng.uniform(0.1, 2, n_samples)

        label = (du_dt < alpha * d2u_dx).astype(int)

        self.df = pd.DataFrame({
            "u": u, "du_dt": du_dt, "d2u_dx": d2u_dx, "alpha": alpha,
        })
        self.target = pd.Series(label, name="label")

        print(f"✅ PDE (Heat Equation) generated: {self.df.shape[0]} samples, "
              f"{self.df.shape[1]} features, label balance "
              f"{dict(self.target.value_counts())}")

        super().__init__(*args, **kwargs)


# ---------------------------------------------------------------------------
# Future slots — add below as you test new inequality formulas
# ---------------------------------------------------------------------------
# class WaveEquation(Test):    ...
# class NavierStokesProxy(Test): ...