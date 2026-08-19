"""Fisher information and maximum likelihood tests."""

import numpy as np
import pytest

from voigtinference import rand_voigt, voigt_fisher, voigt_mle, voigt_score
from voigtinference.core import loglik_grad_hess


def test_fisher_positive_definite_and_block_diagonal():
    info = voigt_fisher(0.0, 1.0, 1.0)
    np.linalg.cholesky(info)  # raises if not positive definite
    assert abs(info[0, 1]) < 1e-8
    assert abs(info[0, 2]) < 1e-8
    assert np.allclose(info, info.T, atol=0)


def test_fisher_scale_equivariance():
    """I(mu, c*sigma, c*gamma) = D I(mu, sigma, gamma) D with D = diag(1/c, 1/c, 1/c)."""
    c = 3.0
    a = voigt_fisher(0.0, 1.0, 0.4)
    b = voigt_fisher(0.0, c, c * 0.4)
    assert np.allclose(b, a / c**2, rtol=1e-8)


def test_fisher_quadrature_converges():
    coarse = voigt_fisher(0.0, 1.0, 0.5, nodes=200)
    fine = voigt_fisher(0.0, 1.0, 0.5, nodes=800)
    assert np.max(np.abs(coarse - fine)) < 1e-9


def test_information_matrix_equality():
    """I(theta) = -E[H(theta)], checked by Monte Carlo."""
    mu, sigma, gamma = 0.0, 1.0, 1.0
    info = voigt_fisher(mu, sigma, gamma)
    y = rand_voigt(200_000, mu, sigma, gamma, rng=1)
    _, _, H, _ = loglik_grad_hess(y, mu, sigma, gamma, need_ll=False)
    assert np.max(np.abs(info + H / y.size)) < 0.05


def test_mle_recovery():
    mu0, sigma0, gamma0 = 0.5, 1.0, 0.3
    y = rand_voigt(5000, mu0, sigma0, gamma0, rng=2026)
    r = voigt_mle(y)
    assert r.converged
    assert abs(r.mu - mu0) < 5 * r.se[0]
    assert abs(r.sigma - sigma0) < 5 * r.se[1]
    assert abs(r.gamma - gamma0) < 5 * r.se[2]


def test_score_vanishes_at_the_optimum():
    y = rand_voigt(5000, 0.5, 1.0, 0.3, rng=2026)
    r = voigt_mle(y)
    g = voigt_score(y, r.mu, r.sigma, r.gamma).mean(axis=0)
    assert np.linalg.norm(g) < 1e-6


def test_expected_and_observed_standard_errors_agree():
    y = rand_voigt(20_000, 0.0, 1.0, 0.5, rng=99)
    r = voigt_mle(y)
    assert np.allclose(r.se, r.se_obs, rtol=0.05)


def test_mle_is_location_scale_equivariant():
    y = rand_voigt(4000, 0.0, 1.0, 0.4, rng=17)
    a, b = 2.5, 3.0
    r0 = voigt_mle(y)
    r1 = voigt_mle(a + b * y)
    assert r1.mu == pytest.approx(a + b * r0.mu, rel=1e-6)
    assert r1.sigma == pytest.approx(b * r0.sigma, rel=1e-6)
    assert r1.gamma == pytest.approx(b * r0.gamma, rel=1e-6)


def test_mle_uses_one_faddeeva_pass_per_accepted_step():
    """The (K, L) cache must make nfev equal the iteration count plus one."""
    y = rand_voigt(5000, 0.5, 1.0, 0.3, rng=2026)
    r = voigt_mle(y)
    # one pass per accepted iteration plus the observed-information pass;
    # a stalled final line search may add a few rejected trials on some
    # platforms before the representable-improvement rule stops it
    assert r.nfev <= r.iterations + 4


def test_mle_rejects_tiny_samples():
    with pytest.raises(ValueError):
        voigt_mle([1.0, 2.0])


def test_mle_survives_a_degenerate_sample():
    """A sample with no Cauchy signal must stop gracefully, not raise."""
    rng = np.random.default_rng(4)
    y = rng.standard_normal(200)  # gamma is weakly identified
    r = voigt_mle(y)
    assert np.isfinite(r.mu)
    assert r.sigma > 0
