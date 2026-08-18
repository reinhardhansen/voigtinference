"""Analytic Jacobians for lmfit's Voigt lineshape, from voigtinference.

lmfit's built-in ``VoigtModel`` wraps the lineshape

    voigt(x; amplitude, center, sigma, gamma)
        = amplitude * Re[w(z)] / (sigma*sqrt(2*pi)),
    z = (x - center + i*gamma) / (sigma*sqrt(2)),

which is ``amplitude * voigt_pdf(x, center, sigma, gamma)`` in this package's
notation.  Its parameter derivatives are therefore available analytically from
the likelihood score s_theta = d log pdf / d(center, sigma, gamma):

    dI/d(amplitude, center, sigma, gamma) = (pdf, A*pdf*s_mu, A*pdf*s_sigma,
                                             A*pdf*s_gamma),

reusing the same Faddeeva evaluation as the model itself.  One subtlety: by
default lmfit *ties* gamma to sigma (parameter hint ``gamma`` with
``expr='sigma'``), in which case the derivative with respect to the shared
width is the chain-rule sum ``A*pdf*(s_sigma + s_gamma)``.

This module provides the Jacobian in both parameterizations, plus a demo that
fits a synthetic Voigt peak three ways and reports function-evaluation counts:

  1. scipy.optimize.least_squares, numerical Jacobian (baseline);
  2. scipy.optimize.least_squares, analytic Jacobian;
  3. lmfit.minimize(method='leastsq') with ``Dfun`` (the pattern documented in
     lmfit's "Fit Specifying a Function to Compute the Jacobian" example),
     with and without the analytic Jacobian.

Run:  PYTHONPATH=src python examples/lmfit_voigt_jacobian.py
(needs lmfit for part 3: pip install lmfit)
"""

from __future__ import annotations

import numpy as np

from voigtinference import voigt_pdf, voigt_pdf_score

__all__ = ["voigt_lineshape", "voigt_lineshape_jacobian"]

_S2PI = np.sqrt(2.0 * np.pi)


def voigt_lineshape(x, amplitude=1.0, center=0.0, sigma=1.0, gamma=None):
    """lmfit-compatible Voigt lineshape (identical parameterization)."""
    g = sigma if gamma is None else gamma
    return amplitude * voigt_pdf(np.asarray(x, dtype=float), center, sigma, g)


def voigt_lineshape_jacobian(x, amplitude=1.0, center=0.0, sigma=1.0,
                             gamma=None):
    """Value and analytic Jacobian of lmfit's Voigt lineshape.

    Returns ``(f, J)`` where ``f`` has shape ``(n,)`` and ``J`` has shape
    ``(n, 4)`` with columns ``d/d(amplitude, center, sigma, gamma)`` when
    ``gamma`` is a number (untied), or shape ``(n, 3)`` with columns
    ``d/d(amplitude, center, sigma)`` when ``gamma is None`` (lmfit's default
    tie ``gamma = sigma``, handled by the chain rule).

    All columns reuse the single Faddeeva evaluation made for the value; the
    far-tail branches of :func:`voigtinference.voigt_score` apply.
    """
    xv = np.asarray(x, dtype=float)
    tied = gamma is None
    g = sigma if tied else gamma
    pdf, s = voigt_pdf_score(xv, center, sigma, g)  # one Faddeeva pass;
    #                                       s is (n, 3): d/d(mu, sigma, gamma)
    f = amplitude * pdf
    d_amp = pdf
    d_cen = f * s[:, 0]
    if tied:
        d_sig = f * (s[:, 1] + s[:, 2])
        J = np.stack((d_amp, d_cen, d_sig), axis=-1)
    else:
        d_sig = f * s[:, 1]
        d_gam = f * s[:, 2]
        J = np.stack((d_amp, d_cen, d_sig, d_gam), axis=-1)
    return f, J


# ----------------------------------------------------------------------
# demo
# ----------------------------------------------------------------------

def _demo():
    rng = np.random.default_rng(2026)
    true = dict(amplitude=120.0, center=5.0, sigma=0.8, gamma=0.4)
    x = np.linspace(0.0, 10.0, 401)
    data = voigt_lineshape(x, **true) + 0.5 * rng.standard_normal(x.size)
    p0 = np.array([80.0, 5.3, 1.2, 0.2])   # (amplitude, center, sigma, gamma)

    from scipy.optimize import least_squares

    calls = {"n": 0}

    def resid(p):
        calls["n"] += 1
        return voigt_lineshape(x, *p) - data

    def jac(p):
        return voigt_lineshape_jacobian(x, *p)[1]

    calls["n"] = 0
    r_num = least_squares(resid, p0)
    n_num = calls["n"]
    calls["n"] = 0
    r_ana = least_squares(resid, p0, jac=jac)
    n_ana = calls["n"]
    dpar = np.max(np.abs(r_ana.x / r_num.x - 1.0))
    print("scipy.optimize.least_squares on a synthetic Voigt peak "
          f"(n = {x.size}; residual evaluations counted directly, since\n"
          "  scipy hides finite-difference evaluations from r.nfev):")
    print(f"  numerical Jacobian: {n_num:3d} residual evaluations")
    print(f"  analytic  Jacobian: {n_ana:3d} residual evaluations "
          f"+ {r_ana.njev} Jacobian evaluations")
    print(f"  parameter agreement: {dpar:.2e}\n")

    try:
        import lmfit
    except ImportError:
        print("lmfit not installed; skipping the lmfit demo "
              "(pip install lmfit).")
        return

    params = lmfit.Parameters()
    for name, v0 in zip(("amplitude", "center", "sigma", "gamma"), p0):
        params.add(name, value=v0)

    order = [name for name in params if params[name].vary]

    def residual(pars, x, data):
        v = [pars[n].value for n in order]
        return voigt_lineshape(x, *v) - data

    def dfun(pars, x, data):
        v = [pars[n].value for n in order]
        return voigt_lineshape_jacobian(x, *v)[1].T   # (nvarys, n), col_deriv=1

    out0 = lmfit.minimize(residual, params, args=(x, data), method="leastsq")
    out1 = lmfit.minimize(residual, params, args=(x, data), method="leastsq",
                          Dfun=dfun, col_deriv=1)
    agree = max(abs(out1.params[n].value / out0.params[n].value - 1.0)
                for n in order)
    print(f"lmfit.minimize (leastsq), lmfit {lmfit.__version__}:")
    print(f"  without Dfun: nfev = {out0.nfev:3d}")
    print(f"  with    Dfun: nfev = {out1.nfev:3d}")
    print(f"  parameter agreement: {agree:.2e}")
    print("\nfitted (with Dfun) vs truth:")
    for n, t in true.items():
        p = out1.params[n]
        print(f"  {n:10s} {p.value:10.4f} +/- {p.stderr:.4f}   (true {t})")


if __name__ == "__main__":
    _demo()
