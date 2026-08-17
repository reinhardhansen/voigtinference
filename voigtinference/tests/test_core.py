"""Density, score, Hessian and conditional-moment tests.

Ports the Julia test suite of VoigtInference.jl and adds checks that the
vectorised paths agree with the scalar ones.
"""

import numpy as np
import pytest
from numpy.polynomial.legendre import leggauss

from voigtinference import (
    rand_voigt,
    voigt_condmean,
    voigt_condvar,
    voigt_hessian,
    voigt_logpdf,
    voigt_pdf,
    voigt_score,
)
from voigtinference.core import _kl, loglik_grad_hess

MU, SIGMA, GAMMA = 0.3, 1.2, 0.7


def fd(f, x, h=1e-5):
    return (f(x + h) - f(x - h)) / (2 * h)


def fd2(f, x, h=1e-4):
    return (f(x + h) - 2 * f(x) + f(x - h)) / h**2


def fdmix(f, x, y, h=1e-4):
    return (f(x + h, y + h) - f(x + h, y - h) - f(x - h, y + h) + f(x - h, y - h)) / (
        4 * h**2
    )


# ----------------------------------------------------------------- density


def test_density_integrates_to_one():
    s = SIGMA + GAMMA
    t, wq = leggauss(200)
    u = (np.pi / 2) * t
    mass = np.sum(
        wq * (np.pi / 2) * s / np.cos(u) ** 2 * voigt_pdf(MU + s * np.tan(u), MU, SIGMA, GAMMA)
    )
    assert mass == pytest.approx(1.0, abs=1e-8)


def test_density_limits():
    assert voigt_pdf(0.5, 0.0, 0.0, 1.0) == pytest.approx(1 / (np.pi * 1.25))
    assert voigt_pdf(0.5, 0.0, 1.0, 0.0) == pytest.approx(np.exp(-0.125) / np.sqrt(2 * np.pi))


def test_density_symmetry():
    assert voigt_pdf(MU + 1.3, MU, SIGMA, GAMMA) == pytest.approx(
        voigt_pdf(MU - 1.3, MU, SIGMA, GAMMA)
    )


def test_negative_widths_rejected():
    with pytest.raises(ValueError):
        voigt_pdf(0.0, 0.0, -1.0, 1.0)
    with pytest.raises(ValueError):
        voigt_pdf(0.0, 0.0, 1.0, -1.0)


# ------------------------------------------------------- score and Hessian


@pytest.mark.parametrize("y", [-3.0, 0.0, 0.31, 2.1, 15.0])
def test_score_matches_finite_differences(y):
    s = voigt_score(y, MU, SIGMA, GAMMA)
    assert s[0] == pytest.approx(fd(lambda m: voigt_logpdf(y, m, SIGMA, GAMMA), MU), abs=1e-6)
    assert s[1] == pytest.approx(fd(lambda t: voigt_logpdf(y, MU, t, GAMMA), SIGMA), abs=1e-6)
    assert s[2] == pytest.approx(fd(lambda t: voigt_logpdf(y, MU, SIGMA, t), GAMMA), abs=1e-6)


@pytest.mark.parametrize("y", [-3.0, 0.0, 0.31, 2.1, 15.0])
def test_hessian_matches_finite_differences(y):
    H = voigt_hessian(y, MU, SIGMA, GAMMA)
    assert H[0, 0] == pytest.approx(fd2(lambda m: voigt_logpdf(y, m, SIGMA, GAMMA), MU), abs=1e-4)
    assert H[1, 1] == pytest.approx(fd2(lambda t: voigt_logpdf(y, MU, t, GAMMA), SIGMA), abs=1e-4)
    assert H[2, 2] == pytest.approx(fd2(lambda t: voigt_logpdf(y, MU, SIGMA, t), GAMMA), abs=1e-4)
    assert H[0, 1] == pytest.approx(
        fdmix(lambda m, t: voigt_logpdf(y, m, t, GAMMA), MU, SIGMA), abs=1e-4
    )
    assert H[0, 2] == pytest.approx(
        fdmix(lambda m, t: voigt_logpdf(y, m, SIGMA, t), MU, GAMMA), abs=1e-4
    )
    assert H[1, 2] == pytest.approx(
        fdmix(lambda t, u: voigt_logpdf(y, MU, t, u), SIGMA, GAMMA), abs=1e-4
    )


def test_hessian_is_symmetric():
    y = rand_voigt(50, MU, SIGMA, GAMMA, rng=7)
    H = voigt_hessian(y, MU, SIGMA, GAMMA)
    assert np.allclose(H, np.swapaxes(H, 1, 2), rtol=0, atol=0)


# ------------------------------------------------- conditional moments


@pytest.mark.parametrize("y", [-2.0, 0.3, 2.1, 8.0])
def test_tweedie_identities(y):
    # E[Z|y] = -sigma^2 d/dy log f
    assert voigt_condmean(y, MU, SIGMA, GAMMA) == pytest.approx(
        -SIGMA**2 * fd(lambda t: voigt_logpdf(t, MU, SIGMA, GAMMA), y), abs=1e-6
    )
    # V(Z|y) = sigma^2 + sigma^4 d2/dy2 log f
    assert voigt_condvar(y, MU, SIGMA, GAMMA) == pytest.approx(
        SIGMA**2 + SIGMA**4 * fd2(lambda t: voigt_logpdf(t, MU, SIGMA, GAMMA), y), abs=1e-3
    )


@pytest.mark.parametrize("y", [-2.0, 0.3, 2.1, 8.0])
def test_components_sum_to_deviation(y):
    K, L = _kl(np.array([y - MU]), SIGMA, GAMMA)
    assert voigt_condmean(y, MU, SIGMA, GAMMA) + GAMMA * L[0] / K[0] == pytest.approx(y - MU)


def test_condmean_redescends():
    assert abs(voigt_condmean(1e6, MU, SIGMA, GAMMA)) < 1e-2


def test_condvar_tends_to_sigma_squared():
    assert voigt_condvar(1e8, MU, SIGMA, GAMMA) == pytest.approx(SIGMA**2, rel=1e-8)


# -------------------------------------------------- vectorisation parity


def test_vectorised_matches_scalar():
    y = rand_voigt(200, MU, SIGMA, GAMMA, rng=11)
    y = np.concatenate([y, [1e5, -1e5, 1e9]])  # include far-tail points

    pv = voigt_pdf(y, MU, SIGMA, GAMMA)
    sv = voigt_score(y, MU, SIGMA, GAMMA)
    hv = voigt_hessian(y, MU, SIGMA, GAMMA)
    cm = voigt_condmean(y, MU, SIGMA, GAMMA)
    cvv = voigt_condvar(y, MU, SIGMA, GAMMA)

    for i, yi in enumerate(y):
        assert pv[i] == voigt_pdf(float(yi), MU, SIGMA, GAMMA)
        assert np.array_equal(sv[i], voigt_score(float(yi), MU, SIGMA, GAMMA))
        assert np.array_equal(hv[i], voigt_hessian(float(yi), MU, SIGMA, GAMMA))
        assert cm[i] == voigt_condmean(float(yi), MU, SIGMA, GAMMA)
        assert cvv[i] == voigt_condvar(float(yi), MU, SIGMA, GAMMA)


def test_loglik_grad_hess_matches_elementwise_sums():
    y = np.concatenate([rand_voigt(500, MU, SIGMA, GAMMA, rng=3), [2e5, -3e5]])
    ll, g, H, _ = loglik_grad_hess(y, MU, SIGMA, GAMMA)
    assert ll == pytest.approx(float(voigt_logpdf(y, MU, SIGMA, GAMMA).sum()), rel=1e-14)
    assert np.allclose(g, voigt_score(y, MU, SIGMA, GAMMA).sum(axis=0), rtol=1e-12)
    assert np.allclose(H, voigt_hessian(y, MU, SIGMA, GAMMA).sum(axis=0), rtol=1e-12)


def test_kl_cache_reuse_is_exact():
    y = rand_voigt(300, MU, SIGMA, GAMMA, rng=5)
    ll, g, H, cache = loglik_grad_hess(y, MU, SIGMA, GAMMA)
    ll2, g2, H2, _ = loglik_grad_hess(y, MU, SIGMA, GAMMA, _kl_cache=cache)
    assert ll2 == ll
    assert np.array_equal(g, g2)
    assert np.array_equal(H, H2)
