"""Likelihood identities of the DISPATCHED derivatives.

Pointwise accuracy is not sufficient for the extreme-ratio branches: the
leading term of the Cauchy-limit expansion cancels under the expectation
while a truncation error need not, so a branch can be pointwise accurate
yet fail E[s] = 0 and E[ss'] = -E[H].  The shipped branches are derivatives
of one truncated log density, which restores both identities to the
retained order; these tests verify that under the EXACT model by
tangent-transformed Gauss-Legendre quadrature, across width ratios that
span exact-only, mixed, and branch-only dispatch.

The failure mode these tests guard against was real: leading-order branches
gave E[s_sigma] * gamma^4 / sigma^3 -> 1/2 and made -E[H] indefinite
(leading eigenvalue -1/2 in the sigma direction).
"""
import numpy as np
import pytest

from voigtinference import voigt_hessian, voigt_loglik, voigt_pdf, voigt_score

SIGMA = 1.0
# Centering is checked at all ratios.  The information identity and the PSD
# check are restricted to ratios <= 1e4: the sigma-sigma slot integrates a
# pointwise integrand ~(gamma/sigma)^2 times larger than its expectation, so
# double-precision quadrature noise is ~45*eps*(gamma/sigma)^2 and swamps the
# identity beyond ~1e5 (the identity there is established analytically: the
# branches are derivatives of one expansion with O(r^3) truncation).
RATIOS = [10.0, 50.0, 1.0e2, 1.0e3, 1.0e4, 1.0e6, 1.0e8]
RATIOS_IDENTITY = [10.0, 50.0, 1.0e2, 1.0e3, 1.0e4]


def _quad_nodes(gamma, n=4000):
    """y = gamma * tan(pi t / 2): maps the Lorentzian tails to [-1, 1]."""
    t, w = np.polynomial.legendre.leggauss(n)
    y = gamma * np.tan(np.pi * t / 2)
    dy = gamma * (np.pi / 2) / np.cos(np.pi * t / 2) ** 2
    return y, w * dy


def _scaled_moments(gamma):
    """Return scaled E[s], scaled E[ss'] + E[H] under the exact model.

    Each component is scaled by the inverse of its Fisher-information order
    (I_mm, I_gg ~ 1/(2 gamma^2); I_ss ~ sigma^2/gamma^4; I_sg ~ sigma/gamma^3)
    BEFORE summation, so the quadrature works with O(1) integrands and the
    scaled statistics are comparable across ratios.
    """
    y, w = _quad_nodes(gamma)
    f = voigt_pdf(y, 0.0, SIGMA, gamma)
    s = voigt_score(y, 0.0, SIGMA, gamma)
    H = voigt_hessian(y, 0.0, SIGMA, gamma)
    scale = np.array([gamma, gamma * gamma / SIGMA, gamma])
    wf = w * f
    Es = np.array([np.sum(wf * s[:, i] * scale[i]) for i in range(3)])
    M = np.empty((3, 3))
    for i in range(3):
        for j in range(i, 3):
            M[i, j] = M[j, i] = np.sum(
                wf * (s[:, i] * s[:, j] + H[:, i, j]) * scale[i] * scale[j]
            )
    return Es, M


@pytest.mark.parametrize("ratio", RATIOS)
def test_score_is_centered(ratio):
    Es, _ = _scaled_moments(ratio * SIGMA)
    # old leading-order branches gave |scaled E[s_sigma]| -> 0.5
    assert np.all(np.abs(Es) < 1e-3), Es


@pytest.mark.parametrize("ratio", RATIOS_IDENTITY)
def test_information_identity_and_psd(ratio):
    _, M = _scaled_moments(ratio * SIGMA)
    # E[ss'] + E[H] = 0: leading-order branches gave -1.5 in the
    # (sigma, sigma) slot, and sigma^4-truncated branches combined with
    # large exact-branch zones gave ~0.14 at gamma/sigma ~ 50-100
    assert np.max(np.abs(M)) < 1e-2, M
    # -E[H] positive definite within tolerance: -E[H] = E[ss'] - M, and
    # E[ss'] (scaled) has eigenvalues bounded away from 0
    y, w = _quad_nodes(ratio * SIGMA)
    f = voigt_pdf(y, 0.0, SIGMA, ratio * SIGMA)
    H = voigt_hessian(y, 0.0, SIGMA, ratio * SIGMA)
    g = ratio * SIGMA
    scale = np.array([g, g * g / SIGMA, g])
    EH = np.einsum("k,kij->ij", w * f, H) * np.outer(scale, scale)
    eig = np.linalg.eigvalsh(-EH)
    assert eig.min() > -1e-3, eig


def test_dispatched_score_differentiates_the_loglik():
    """FD of the exact sample log likelihood vs the aggregated dispatched
    score, in the all-branch regime."""
    rng = np.random.default_rng(5)
    gamma = 2.0e3 * SIGMA                      # every point dispatches
    y = gamma * np.tan(np.pi * (rng.random(2000) - 0.5))
    theta = np.array([0.1, SIGMA, gamma])

    s = voigt_score(y, *theta).sum(axis=0)
    for i, h in enumerate([1e-4, 1e-5 * SIGMA, 1e-2 * gamma]):
        tp = theta.copy(); tp[i] += h
        tm = theta.copy(); tm[i] -= h
        fd = (voigt_loglik(y, *tp) - voigt_loglik(y, *tm)) / (2 * h)
        # tolerance is FD-limited: the sigma direction differentiates a tiny
        # signal (~1e-4) riding on a log likelihood of magnitude ~1e4
        assert abs(s[i] - fd) / (abs(fd) + 1.0) < 1e-3, (i, s[i], fd)


def test_extreme_ratio_reference_points():
    """Independently computed high-precision reference values (sigma = 1);
    leading-order branches returned +1.14e-8, +4.56e-8, 2.30e-6 for the
    first three."""
    from voigtinference import voigt_condmean, voigt_condvar

    s0 = voigt_score(0.0, 0.0, 1.0, 1.0e4)
    assert abs(s0[1] / -1.9999999000e-8 - 1) < 1e-6
    s1 = voigt_score(1.0e4, 0.0, 1.0, 1.0e4)
    assert abs(s1[1] / 9.99999965e-9 - 1) < 1e-6
    s2 = voigt_score(1.0e5, 0.0, 1.0, 1.0e4)
    assert abs(s2[1] / 5.86217038e-10 - 1) < 1e-6
    cm = voigt_condmean(41.0e6, 0.0, 1.0, 1.0e6)
    assert abs(cm / 4.87514863e-8 - 1) < 1e-6
    cv = voigt_condvar(41.0e6, 0.0, 1.0, 1.0e6)
    assert abs(cv - 1.0) < 1e-3
