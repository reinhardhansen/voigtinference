"""Boundary diagnostics: boundary flags, suppressed Wald SEs, submodels,
termination reasons, multistart."""

import numpy as np
import pytest

from voigtinference import rand_voigt, voigt_mle


def test_interior_fit_full_diagnostics():
    y = rand_voigt(4000, 0.5, 1.0, 0.4, rng=3)
    r = voigt_mle(y)
    assert r.converged and r.termination == "gradient_converged"
    assert not (r.sigma_boundary or r.gamma_boundary or r.upper_boundary)
    assert np.all(np.isfinite(r.se)) and np.all(np.isfinite(r.se_obs))
    assert r.expected_info_posdef and r.observed_info_posdef
    # the Voigt fit must beat both boundary submodels on likelihood
    assert r.loglik > r.loglik_gaussian
    assert r.loglik > r.loglik_cauchy
    assert r.projected_gradient_norm <= r.gradient_norm + 1e-15


# Submodel-true data: at the Cauchy boundary the local parameter is
# tau = sigma^2, so sigma-hat sits at an interior value of order n**-0.25 in
# about half of all samples (one-sided boundary asymptotics); at the Gaussian
# boundary the gamma-score has infinite Fisher information, no standard rate
# is claimed, and empirically gamma-hat also often sits at a small interior
# value. The reliable diagnostic is the submodel likelihood comparison; the
# flag fires only when the fit is numerically the submodel.


def test_closed_family_boundary_lr_nonnegative():
    """boundary_lr is a likelihood ratio against the CLOSED family (the
    full likelihood is the max of the interior and both submodel fits), so
    it is nonnegative for every sample.  The raw interior-only difference
    2 (ll_int - ll_sub) can be negative near a boundary by a clamp
    residual; this test is the regression guard against reverting to it."""
    from voigtinference import boundary_lr, rand_voigt

    gens = [
        lambda rng, n: rng.standard_normal(n),                    # Gaussian null
        lambda rng, n: np.tan(np.pi * (rng.random(n) - 0.5)),     # Cauchy null
        lambda rng, n: rand_voigt(n, 0.0, 1.0, 0.02, rng),        # near-boundary
    ]
    for g, gen in enumerate(gens):
        nzero_g = 0
        for s in range(25):
            rng = np.random.default_rng(18500902 + 97 * g + s)
            r = voigt_mle(gen(rng, 150))
            lrg, lrc = boundary_lr(r)
            assert lrg >= 0.0 and np.isfinite(lrg)
            assert lrc >= 0.0 and np.isfinite(lrc)
            nzero_g += lrg == 0.0
        if g == 0:
            # Gaussian null: the closed-family maximum is usually the
            # Gaussian fit itself -> atom at exactly zero
            assert nzero_g >= 1


def test_gaussian_data_diagnosed():
    rng = np.random.default_rng(11)
    y = 0.3 + 1.2 * rng.standard_normal(3000)     # no Cauchy component at all
    r = voigt_mle(y)
    assert r.loglik - r.loglik_gaussian < 4.0
    assert r.gamma_boundary or r.gamma < 0.05 * r.sigma
    assert not r.sigma_boundary
    if r.gamma_boundary:
        # Wald SEs are suppressed at the boundary, not clipped to zero
        assert np.all(np.isnan(r.se)) and np.all(np.isnan(r.se_obs))
    mg, sg = r.gaussian_fit
    assert abs(mg - 0.3) < 0.1 and abs(sg - 1.2) < 0.1


def test_cauchy_data_diagnosed():
    rng = np.random.default_rng(12)
    y = 0.1 + 0.7 * np.tan(np.pi * (rng.random(3000) - 0.5))   # pure Cauchy
    r = voigt_mle(y)
    assert r.loglik - r.loglik_cauchy < 4.0
    assert r.sigma_boundary or r.sigma < 0.5 * r.gamma
    assert not r.gamma_boundary
    if r.sigma_boundary:
        assert np.all(np.isnan(r.se))
    mc, gc = r.cauchy_fit
    assert abs(mc - 0.1) < 0.1 and abs(gc - 0.7) < 0.1


def test_multistart_never_worse():
    y = rand_voigt(1500, 0.0, 1.0, 0.05, rng=5)
    r1 = voigt_mle(y)
    r3 = voigt_mle(y, starts=4)
    assert r3.starts == 4
    assert r3.loglik >= r1.loglik - 1e-8


def test_nonfinite_data_raises():
    with pytest.raises(ValueError):
        voigt_mle(np.array([1.0, np.nan, 2.0, 3.0]))
