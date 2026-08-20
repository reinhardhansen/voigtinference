"""Exact simulation from the Voigt profile."""

from __future__ import annotations

import numpy as np

__all__ = ["rand_voigt"]


def rand_voigt(n, mu, sigma, gamma, rng=None):
    """Draw ``n`` iid variates from the Voigt profile by exact convolution.

    ``Y = mu + sigma * N(0, 1) + gamma * tan(pi (U - 1/2))``.

    Parameters
    ----------
    rng : None, int, or numpy.random.Generator
        Seed or generator.  ``None`` uses fresh entropy.
    """
    if not isinstance(n, (int, np.integer)) or isinstance(n, bool) or n < 0:
        raise ValueError("n must be a nonnegative integer")
    if not (np.isfinite(mu) and np.isfinite(sigma) and np.isfinite(gamma)):
        raise ValueError("mu, sigma, gamma must be finite")
    if sigma < 0.0:
        raise ValueError("sigma must be >= 0")
    if gamma < 0.0:
        raise ValueError("gamma must be >= 0")
    if sigma == 0.0 and gamma == 0.0:
        raise ValueError("sigma and gamma cannot both be 0 (degenerate model)")
    if not isinstance(rng, np.random.Generator):
        rng = np.random.default_rng(rng)
    return (
        mu
        + sigma * rng.standard_normal(n)
        + gamma * np.tan(np.pi * (rng.random(n) - 0.5))
    )
