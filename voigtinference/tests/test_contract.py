"""The public input contract: shapes, width domains, and option validation.

The density permits the boundary limits (sigma = 0 pure Lorentzian,
gamma = 0 pure Gaussian, not both); the likelihood calculus (log-density,
score, Hessian, conditional moments) is interior-only and rejects boundary
or nonfinite widths; observations are scalar or 1-d; optimizer options are
validated rather than silently coerced.
"""
import numpy as np
import pytest

from voigtinference import (
    rand_voigt,
    voigt_condmean,
    voigt_condvar,
    voigt_hessian,
    voigt_logpdf,
    voigt_loglik,
    voigt_mle,
    voigt_pdf,
    voigt_pdf_score,
    voigt_score,
)

INTERIOR_ONLY = [
    voigt_logpdf,
    voigt_loglik,
    voigt_score,
    voigt_pdf_score,
    voigt_hessian,
    voigt_condmean,
    voigt_condvar,
]


def test_density_allows_each_boundary_but_not_both():
    assert voigt_pdf(0.5, 0.0, 0.0, 1.0) == pytest.approx(1 / (np.pi * 1.25))
    assert voigt_pdf(0.5, 0.0, 1.0, 0.0) > 0
    with pytest.raises(ValueError):
        voigt_pdf(0.0, 0.0, 0.0, 0.0)
    with pytest.raises(ValueError):
        voigt_pdf(0.0, 0.0, -1.0, 1.0)
    with pytest.raises(ValueError):
        voigt_pdf(0.0, 0.0, np.nan, 1.0)


@pytest.mark.parametrize("fn", INTERIOR_ONLY)
def test_interior_calculus_rejects_boundary_widths(fn):
    y = np.array([0.1, 0.5, 1.0])
    with pytest.raises(ValueError):
        fn(y, 0.0, 0.0, 1.0)
    with pytest.raises(ValueError):
        fn(y, 0.0, 1.0, 0.0)
    with pytest.raises(ValueError):
        fn(y, 0.0, 1.0, np.inf)


def test_observations_must_be_scalar_or_1d():
    with pytest.raises(ValueError):
        voigt_pdf(np.zeros((2, 2)), 0.0, 1.0, 1.0)
    with pytest.raises(ValueError):
        voigt_hessian(np.zeros((2, 2)), 0.0, 1.0, 1.0)
    # scalar and 1-d remain fine
    assert np.isfinite(voigt_pdf(0.3, 0.0, 1.0, 1.0))
    assert voigt_score(np.array([0.3, 1.0]), 0.0, 1.0, 1.0).shape == (2, 3)


def test_mle_option_validation():
    y = rand_voigt(200, 0.0, 1.0, 0.5, rng=1)
    for kw in ({"starts": 0}, {"gtol": 0.0}, {"maxiter": 0}, {"nodes": 1}):
        with pytest.raises(ValueError):
            voigt_mle(y, **kw)


def test_simulation_width_validation():
    with pytest.raises(ValueError):
        rand_voigt(10, 0.0, -1.0, 1.0, rng=1)
    with pytest.raises(ValueError):
        rand_voigt(10, 0.0, 1.0, -1.0, rng=1)
