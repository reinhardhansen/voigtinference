"""Expected Fisher information for the Voigt profile."""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from .core import (
    _SQRT2PI,
    _far_mask,
    _kl,
    _score_arrays,
    _score_tail,
)

__all__ = ["voigt_fisher"]


@lru_cache(maxsize=8)
def _leggauss(nodes: int):
    """Gauss-Legendre nodes and weights, cached.

    NumPy builds these from a companion-matrix eigendecomposition, which at
    ``nodes = 400`` costs more than the entire rest of the quadrature -- and
    they do not depend on the parameters, so recomputing them on every call
    (as a naive implementation does) dominates the cost of a small fit.  The
    returned arrays are marked read-only so a caller cannot corrupt the cache.
    """
    t, w = np.polynomial.legendre.leggauss(nodes)
    t.flags.writeable = False
    w.flags.writeable = False
    return t, w


def voigt_fisher(mu, sigma, gamma, nodes: int = 400):
    """Expected Fisher information ``I(theta) = E[s s']``, ``theta = (mu, sigma, gamma)``.

    Computed by Gauss-Legendre quadrature after the substitution
    ``y = mu + (sigma + gamma) tan(t)``, ``t in (-pi/2, pi/2)``, which maps the
    Cauchy-tailed integrand to a bounded one.  By the information-matrix
    equality ``I(theta) = -E[H]``; by symmetry ``I`` is block diagonal, with
    ``I[mu, sigma] = I[mu, gamma] = 0``.

    The quadrature is vectorised over nodes and shares a single Faddeeva
    evaluation between the density and the score.
    """
    if not (np.isfinite(mu) and np.isfinite(sigma) and np.isfinite(gamma)):
        raise ValueError("mu, sigma, gamma must be finite")
    if sigma <= 0.0 or gamma <= 0.0:
        raise ValueError("sigma and gamma must be > 0 for the Fisher information")
    if not isinstance(nodes, (int, np.integer)) or isinstance(nodes, bool) \
            or nodes < 2:
        raise ValueError("nodes must be an integer >= 2")

    t, wq = _leggauss(nodes)
    u = (np.pi / 2.0) * t
    scale = sigma + gamma
    ytil = scale * np.tan(u)
    jac = (np.pi / 2.0) * scale / np.cos(u) ** 2

    K, L = _kl(ytil, sigma, gamma)
    f = K / (sigma * _SQRT2PI)

    smu, ssig, sgam, _ = _score_arrays(ytil, sigma, gamma, K, L)
    far = _far_mask(ytil, sigma, gamma)
    if far.any():
        idx = np.flatnonzero(far)
        tmu, tsig, tgam = _score_tail(ytil[idx], sigma, gamma)
        smu[idx], ssig[idx], sgam[idx] = tmu, tsig, tgam

    s = np.stack((smu, ssig, sgam), axis=-1)
    c = wq * jac * f
    info = np.einsum("i,ij,ik->jk", c, s, s)
    return 0.5 * (info + info.T)
