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


# Submodel-true data: on the boundary the local parameter is the width
# (gamma) or its square (sigma^2), so the MLE sits at an interior estimate of
# order n**-0.5 or n**-0.25 in about half of all samples (half-normal boundary
# asymptotics). The reliable diagnostic is the submodel likelihood comparison;
# the flag fires only when the fit is numerically the submodel.


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
