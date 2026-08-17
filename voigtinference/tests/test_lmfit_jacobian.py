"""Tests for the lmfit-compatible analytic Voigt Jacobian.

Run:  PYTHONPATH=src python -m pytest tests/test_lmfit_jacobian.py -q
The lmfit integration test is skipped automatically if lmfit is absent.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "examples"))

from lmfit_voigt_jacobian import voigt_lineshape, voigt_lineshape_jacobian

X = np.array([-3.0, 0.0, 0.7, 2.1, 4.9, 5.0, 5.1, 8.0, 40.0])
PARS = dict(amplitude=120.0, center=5.0, sigma=0.8, gamma=0.4)


def _fd(fun, pars, name, h):
    up = dict(pars); up[name] += h
    dn = dict(pars); dn[name] -= h
    return (fun(X, **up) - fun(X, **dn)) / (2 * h)


@pytest.mark.parametrize("name,col,h", [
    ("amplitude", 0, 1e-4),
    ("center",    1, 1e-6),
    ("sigma",     2, 1e-6),
    ("gamma",     3, 1e-6),
])
def test_jacobian_vs_finite_differences(name, col, h):
    _, J = voigt_lineshape_jacobian(X, **PARS)
    fd = _fd(voigt_lineshape, PARS, name, h)
    scale = np.maximum(np.abs(fd), 1e-3)
    assert np.max(np.abs(J[:, col] - fd) / scale) < 1e-6


def test_tied_gamma_chain_rule():
    # lmfit's default ties gamma = sigma; d/d(shared sigma) must be the sum
    pars = dict(amplitude=120.0, center=5.0, sigma=0.8)
    f, J = voigt_lineshape_jacobian(X, gamma=None, **pars)
    assert J.shape == (X.size, 3)
    h = 1e-6
    fd = (voigt_lineshape(X, 120.0, 5.0, 0.8 + h, None)
          - voigt_lineshape(X, 120.0, 5.0, 0.8 - h, None)) / (2 * h)
    scale = np.maximum(np.abs(fd), 1e-3)
    assert np.max(np.abs(J[:, 2] - fd) / scale) < 1e-6
    # and equals the untied sum evaluated at gamma = sigma
    _, Ju = voigt_lineshape_jacobian(X, 120.0, 5.0, 0.8, 0.8)
    assert np.allclose(J[:, 2], Ju[:, 2] + Ju[:, 3], rtol=1e-12, atol=0)


def test_value_matches_lmfit_lineshape():
    lmfit = pytest.importorskip("lmfit")
    from lmfit.lineshapes import voigt as lmfit_voigt
    ours = voigt_lineshape(X, **PARS)
    theirs = lmfit_voigt(X, PARS["amplitude"], PARS["center"], PARS["sigma"],
                         PARS["gamma"])
    assert np.allclose(ours, theirs, rtol=1e-12, atol=0)


def test_least_squares_same_solution_fewer_evals():
    # NB: scipy's least_squares does NOT charge finite-difference Jacobian
    # evaluations to r.nfev (they hide under njev), so count calls ourselves.
    from scipy.optimize import least_squares
    rng = np.random.default_rng(7)
    x = np.linspace(0.0, 10.0, 401)
    data = voigt_lineshape(x, **PARS) + 0.5 * rng.standard_normal(x.size)
    p0 = np.array([80.0, 5.3, 1.2, 0.2])

    calls = {"n": 0}

    def resid(p):
        calls["n"] += 1
        return voigt_lineshape(x, *p) - data

    jac = lambda p: voigt_lineshape_jacobian(x, *p)[1]

    calls["n"] = 0
    r_num = least_squares(resid, p0)
    n_num = calls["n"]
    calls["n"] = 0
    r_ana = least_squares(resid, p0, jac=jac)
    n_ana = calls["n"]

    assert np.allclose(r_ana.x, r_num.x, rtol=1e-6)
    # numeric '2-point' Jacobian costs ~4 extra residual sweeps per iteration
    assert n_ana < n_num / 3


def test_lmfit_dfun_agrees_and_saves_evals():
    lmfit = pytest.importorskip("lmfit")
    rng = np.random.default_rng(11)
    x = np.linspace(0.0, 10.0, 401)
    data = voigt_lineshape(x, **PARS) + 0.5 * rng.standard_normal(x.size)

    params = lmfit.Parameters()
    for name, v0 in zip(("amplitude", "center", "sigma", "gamma"),
                        (80.0, 5.3, 1.2, 0.2)):
        params.add(name, value=v0)
    order = [n for n in params if params[n].vary]

    def residual(pars, x, data):
        return voigt_lineshape(x, *[pars[n].value for n in order]) - data

    def dfun(pars, x, data):
        return voigt_lineshape_jacobian(
            x, *[pars[n].value for n in order])[1].T

    out0 = lmfit.minimize(residual, params, args=(x, data), method="leastsq")
    out1 = lmfit.minimize(residual, params, args=(x, data), method="leastsq",
                          Dfun=dfun, col_deriv=1)
    for n in order:
        assert abs(out1.params[n].value / out0.params[n].value - 1) < 1e-6
    assert out1.nfev < out0.nfev / 2
