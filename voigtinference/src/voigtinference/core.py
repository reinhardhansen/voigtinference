"""Density, score, Hessian, and conditional moments for the Voigt profile.

Model
-----
``Y = mu + Z + X`` with ``Z ~ N(0, sigma**2)`` and ``X ~ Cauchy(0, gamma)``
(Lorentzian with HWHM ``gamma``), so ``Y`` follows the Voigt profile

    f(y) = K(x, a) / (sigma * sqrt(2*pi)),
    x = (y - mu) / (sigma * sqrt(2)),   a = gamma / (sigma * sqrt(2)),

where ``K = Re w(z)`` and ``L = Im w(z)`` are the absorption and dispersion
parts of the Faddeeva function ``w(z) = exp(-z**2) erfc(-i z)`` at ``z = x + i a``.

Because ``w'(z) = -2 z w(z) + 2i/sqrt(pi)``, every derivative of the
log-likelihood is algebraic in ``(K, L)``.  One Faddeeva evaluation per data
point therefore delivers the density, the score, the Hessian, and the
conditional moments of the Gaussian component.

Reference: P. R. Hansen and C. Tong, "Exact likelihood inference and robust
filtering for Gauss-Cauchy convolution models", arXiv:2605.01665.

Implementation notes
--------------------
Everything is vectorised over the sample.  ``wofz`` is a ufunc and costs
roughly 150 ns per element, which dominates the elementwise algebra that
follows it, so the routines here are organised to call it *once* per
(sample, parameter) pair and to share the result across the density, the
score and the Hessian.  In particular :func:`loglik_grad_hess` -- the
workhorse used by the optimiser -- makes a single Faddeeva pass and never
materialises a per-observation Hessian array.
"""

from __future__ import annotations

import numpy as np
from scipy.special import wofz

__all__ = [
    "faddeeva",
    "voigt_pdf",
    "voigt_logpdf",
    "voigt_loglik",
    "voigt_score",
    "voigt_hessian",
    "voigt_condmean",
    "voigt_condvar",
    "loglik_grad_hess",
    "loglik_only",
]

_SQRT2 = np.sqrt(2.0)
_SQRT2PI = np.sqrt(2.0 * np.pi)
_SQRT2OVERPI = np.sqrt(2.0 / np.pi)
_LOG_SQRT2PI = 0.5 * np.log(2.0 * np.pi)

#: Branch switches, expressed through the Cauchy-limit expansion parameter
#:
#:     r = sigma**2 / (ytil**2 + gamma**2).
#:
#: The closed-form score and Hessian are algebraically exact but lose digits to
#: floating-point cancellation whenever the Gaussian component is a small
#: perturbation of the Cauchy component -- which happens BOTH deep in the tail
#: (|ytil| large) AND at any |ytil| when gamma >> sigma (large imaginary
#: Faddeeva argument; digits lost grow like 4*log10(gamma/sigma) at the
#: center).  In exactly that regime the moment expansion
#: ``f(y) = c(ytil) + (sigma**2/2) c''(ytil) + O(sigma**4 c4)``, with ``c``
#: the Cauchy(0, gamma) density and ``c4`` its fourth derivative, is accurate:
#: its relative error is O(r).  Gating on r therefore covers both failure
#: modes with one criterion; the earlier |ytil|-based switch missed the
#: large-gamma/sigma center entirely.
#:
#: The branches carry three expansion orders and are derivatives of one
#: truncated log density (see _score_tail), so their truncation error is
#: O(r**3) pointwise and the likelihood identities hold to the retained
#: order after integration.  That makes the branch MORE accurate than the
#: exact formulas well before the cancellation becomes visible.  The
#: thresholds are the minimax optima from the certify.jl tune scan
#: (worst pointwise error: score ~1e-10 at r_s = 1e-4, Hessian ~6e-7 at
#: r_h = 5e-4), and the small exact zones also remove the integrated
#: information-identity violation that large exact-branch zones caused
#: at gamma/sigma ~ 50-100.  Certified by the Julia
#: package's examples/certify.jl over gamma/sigma in [1e-8, 1e8]; see its
#: output for current bounds.  Values must match Julia bit for bit.
_R_SCORE = 1.0e-4    # score and conditional moments switch where r < _R_SCORE
_R_HESS = 5.0e-4     # Hessian switches where r < _R_HESS


def _asarray(y):
    """Return ``(1-d float array, was_scalar)``.

    Public contract: observations are a scalar or a one-dimensional
    array-like; higher-dimensional input is rejected rather than silently
    broadcast (different functions would broadcast differently).
    """
    arr = np.asarray(y, dtype=np.float64)
    if arr.ndim == 0:
        return arr.reshape(1), True
    if arr.ndim > 1:
        raise ValueError(
            f"observations must be scalar or 1-d, got shape {arr.shape}"
        )
    return arr, False


def _check(sigma, gamma):
    """Width contract for the density: finite, nonnegative, not both zero."""
    if not (np.isfinite(sigma) and np.isfinite(gamma)):
        raise ValueError("sigma and gamma must be finite")
    if sigma < 0.0:
        raise ValueError("sigma must be >= 0")
    if gamma < 0.0:
        raise ValueError("gamma must be >= 0")
    if sigma == 0.0 and gamma == 0.0:
        raise ValueError("sigma and gamma cannot both be 0 (degenerate model)")


def _check_interior(sigma, gamma):
    """Width contract for the likelihood calculus: strictly interior.

    The score, Hessian, and conditional-moment formulas are the interior
    calculus of the Voigt family; at ``sigma == 0`` or ``gamma == 0`` they
    are not defined (the boundary submodels have their own, different
    calculus, handled by :func:`voigt_mle`'s submodel fits).
    """
    if not (np.isfinite(sigma) and np.isfinite(gamma)):
        raise ValueError("sigma and gamma must be finite")
    if sigma <= 0.0 or gamma <= 0.0:
        raise ValueError(
            "sigma and gamma must be > 0 for the interior likelihood calculus"
        )


def _kl(ytil, sigma, gamma):
    """Absorption and dispersion parts ``(K, L)`` at ``z = x + i a``.

    One complex Faddeeva evaluation; every other routine in this module is
    algebra on the returned pair.  The complex argument is assembled in place
    to avoid intermediate temporaries.

    Note the *division* by ``sigma*sqrt(2)``.  Multiplying by the reciprocal is
    marginally faster but differs by an ulp, and the score formulas amplify
    that ulp to a relative 1e-9 by ``|ytil| = 100*scale``.  Dividing matches
    VoigtInference.jl exactly; see ``_score_arrays`` for why that matters.
    """
    den = sigma * _SQRT2
    z = np.empty(ytil.shape, dtype=np.complex128)
    np.divide(ytil, den, out=z.real)
    z.imag = gamma / den
    w = wofz(z)
    return w.real, w.imag


def _far_mask(ytil, sigma, gamma, r_thresh=_R_SCORE):
    # Cauchy-limit branch used where sigma**2 < r_thresh * (ytil**2 + gamma**2)
    return sigma * sigma < r_thresh * (ytil * ytil + gamma * gamma)


# ----------------------------------------------------------------------
# Faddeeva function
# ----------------------------------------------------------------------

def faddeeva(z):
    """Faddeeva function ``w(z) = exp(-z**2) erfc(-i z)``.

    Thin wrapper over :func:`scipy.special.wofz`, provided so that the
    primitive used throughout the package is explicit and swappable.
    """
    return wofz(np.asarray(z, dtype=np.complex128))


# ----------------------------------------------------------------------
# Density
# ----------------------------------------------------------------------

def voigt_pdf(y, mu, sigma, gamma):
    """Voigt density ``f(y) = K(x, a) / (sigma * sqrt(2*pi))``.

    The boundary cases ``sigma == 0`` (pure Lorentzian) and ``gamma == 0``
    (pure Gaussian) are handled by their limits.
    """
    _check(sigma, gamma)
    yv, scalar = _asarray(y)
    ytil = yv - mu
    if sigma == 0.0:
        out = gamma / (np.pi * (ytil * ytil + gamma * gamma))
    elif gamma == 0.0:
        out = np.exp(-0.5 * (ytil / sigma) ** 2) / (sigma * _SQRT2PI)
    else:
        K, _ = _kl(ytil, sigma, gamma)
        out = K / (sigma * _SQRT2PI)
    return out[0] if scalar else out


def voigt_logpdf(y, mu, sigma, gamma):
    """Log density, interior case ``sigma > 0``, ``gamma > 0``.

    ``K`` is computed accurately everywhere (the cancellation that motivates
    the far-tail branch afflicts *differences* built from ``K`` and ``L``, not
    ``K`` itself), so no tail switch is needed here.
    """
    _check_interior(sigma, gamma)
    yv, scalar = _asarray(y)
    K, _ = _kl(yv - mu, sigma, gamma)
    out = np.log(K) - np.log(sigma) - _LOG_SQRT2PI
    return out[0] if scalar else out


def voigt_loglik(y, mu, sigma, gamma):
    """Sample log-likelihood ``sum_i log f(y_i)``."""
    _check_interior(sigma, gamma)
    yv, _ = _asarray(y)
    K, _ = _kl(yv - mu, sigma, gamma)
    return float(np.log(K).sum() - yv.size * (np.log(sigma) + _LOG_SQRT2PI))


# ----------------------------------------------------------------------
# Score
# ----------------------------------------------------------------------

def _score_arrays(ytil, sigma, gamma, K, L):
    """Score components as three 1-d arrays, exact branch.

        s_mu    = (ytil - gamma*L/K) / sigma**2
        s_sigma = ((ytil**2 - gamma**2 - sigma**2)*K - 2*gamma*ytil*L
                   + sqrt(2/pi)*sigma*gamma) / (sigma**3 * K)
        s_gamma = (gamma*K + ytil*L - sqrt(2/pi)*sigma) / (sigma**2 * K)

    The operations below are written to match the published formulas -- and
    VoigtInference.jl -- *token for token*, including association order.

    This is deliberate and it costs about 0.4% of a score evaluation.  An
    algebraically equivalent rearrangement (dividing by ``K`` once up front and
    working with ``r = L/K``) is very slightly faster and no less accurate, but
    it rounds differently; and because these expressions cancel catastrophically
    in the tail, "rounds differently" turns into relative disagreements of
    1e-8 by ``|ytil| = 10*scale`` and 1e-2 by ``40*scale``.  Matching the
    arrangement makes the two implementations agree bit for bit wherever they
    both take the exact branch, which is what lets ``bench/crosscheck.jl``
    assert agreement to 1e-12 without a fudge factor.
    """
    s2 = sigma * sigma
    r = L / K
    smu = (ytil - gamma * L / K) / s2
    ssig = (
        (ytil * ytil - gamma * gamma - s2) * K
        - 2 * gamma * ytil * L
        + _SQRT2OVERPI * sigma * gamma
    ) / (s2 * sigma * K)
    sgam = (gamma * K + ytil * L - _SQRT2OVERPI * sigma) / (s2 * K)
    return smu, ssig, sgam, r


def _score_tail(ytil, sigma, gamma):
    """Cauchy-limit score, used where ``_far_mask`` is set.

    Every entry is the exact derivative of ONE truncated expansion of the
    log density,

        log f = log c + (sigma^2/2) A + (sigma^4/8) (B - A^2)
                + sigma^6 (C/48 - A B/16 + A^3/24),

    with c the Lorentzian density and A = c2/c, B = c4/c, C = c6/c the
    ratios of its even derivatives.  Deriving all entries from the same truncated
    log density keeps the dispatched score centered (``E[s] = 0``) and
    makes the dispatched Hessian its derivative to the retained order.
    Leading-order formulas are pointwise accurate but violate these
    identities after integration: the leading term cancels under the
    expectation while the truncation error does not (for instance
    ``E[sigma A] = sigma^3/(2 gamma^4) != 0``).  The ratio variables
    ``w, v, rs2`` all lie in ``[0, 1]`` on the dispatch region, so the
    evaluation cannot overflow.
    """
    den = ytil * ytil + gamma * gamma
    w = (ytil * ytil) / den
    v = (gamma * gamma) / den
    ur = ytil / den
    gr = gamma / den
    rs2 = (sigma * sigma) / den
    smu = ur * (2.0 + rs2 * (6.0 * w - 10.0 * v)
                + rs2 * rs2 * (42.0 * w * w - 204.0 * w * v + 74.0 * v * v)
                + rs2 * rs2 * rs2 * (414.0 * w * w * w - 3846.0 * w * w * v
                                     + 4506.0 * w * v * v - 706.0 * v * v * v))
    ssig = (sigma / den) * ((6.0 * w - 2.0 * v)
                            + rs2 * (42.0 * w * w - 108.0 * w * v
                                     + 10.0 * v * v)
                            + rs2 * rs2 * (414.0 * w * w * w - 2574.0 * w * w * v
                                           + 1674.0 * w * v * v
                                           - 74.0 * v * v * v))
    sgam = (w - v) / gamma + gr * rs2 * ((2.0 * v - 14.0 * w)
                                         + rs2 * (-138.0 * w * w
                                                  + 172.0 * w * v
                                                  - 10.0 * v * v)
                                         + rs2 * rs2 * (-1686.0 * w * w * w
                                                        + 5406.0 * w * w * v
                                                        - 2306.0 * w * v * v
                                                        + 74.0 * v * v * v))
    return smu, ssig, sgam


def voigt_score(y, mu, sigma, gamma):
    """Score ``d log f(y; theta) / d theta`` for ``theta = (mu, sigma, gamma)``.

    Returns shape ``(3,)`` for scalar ``y`` and ``(n, 3)`` for array ``y``.
    """
    _check_interior(sigma, gamma)
    yv, scalar = _asarray(y)
    ytil = yv - mu
    K, L = _kl(ytil, sigma, gamma)
    smu, ssig, sgam, _ = _score_arrays(ytil, sigma, gamma, K, L)

    far = _far_mask(ytil, sigma, gamma)
    if far.any():
        idx = np.flatnonzero(far)
        tmu, tsig, tgam = _score_tail(ytil[idx], sigma, gamma)
        smu[idx], ssig[idx], sgam[idx] = tmu, tsig, tgam

    out = np.stack((smu, ssig, sgam), axis=-1)
    return out[0] if scalar else out


def voigt_pdf_score(y, mu, sigma, gamma):
    """Density and score from ONE Faddeeva evaluation.

    Returns ``(pdf, score)`` where ``pdf`` has the shape of
    :func:`voigt_pdf` and ``score`` the shape of :func:`voigt_score`; both
    are bit-identical to calling the two functions separately (same ``(K, L)``
    algebra, same far-tail branch for the score), at half the special-function
    cost.  This is the natural primitive for least-squares Jacobians of the
    lineshape, ``df/dtheta = f * s_theta``, as used by
    ``examples/lmfit_voigt_jacobian.py`` and the Julia ``examples/raman.jl``.
    """
    _check_interior(sigma, gamma)
    yv, scalar = _asarray(y)
    ytil = yv - mu
    K, L = _kl(ytil, sigma, gamma)
    pdf = K / (sigma * _SQRT2PI)
    smu, ssig, sgam, _ = _score_arrays(ytil, sigma, gamma, K, L)

    far = _far_mask(ytil, sigma, gamma)
    if far.any():
        idx = np.flatnonzero(far)
        tmu, tsig, tgam = _score_tail(ytil[idx], sigma, gamma)
        smu[idx], ssig[idx], sgam[idx] = tmu, tsig, tgam

    score = np.stack((smu, ssig, sgam), axis=-1)
    if scalar:
        return pdf[0], score[0]
    return pdf, score


# ----------------------------------------------------------------------
# Hessian
# ----------------------------------------------------------------------

def _hessian_arrays(ytil, sigma, smu, ssig, sgam, r, gamma):
    """The six unique Hessian entries, assembled recursively from the score.

    No additional Faddeeva evaluations: the recursion below uses only the
    score components and ``r = L/K``.
    """
    s2 = sigma * sigma
    Hmm = ssig / sigma - smu * smu
    Hgg = -ssig / sigma - sgam * sgam
    Hmg = (ytil * sgam + gamma * smu - r) / s2 - smu * sgam
    Hms = -(smu + gamma * Hmg - ytil * Hmm) / sigma
    Hgs = -(sgam + gamma * Hgg - ytil * Hmg) / sigma
    Hss = -(ssig + gamma * Hgs - ytil * Hms) / sigma
    return Hmm, Hms, Hmg, Hss, Hgs, Hgg


def _hessian_tail(ytil, sigma, gamma):
    """Cauchy-limit Hessian: second derivatives of the same truncated log
    density as :func:`_score_tail` (see its docstring), so the dispatched
    Hessian is the derivative of the dispatched score and the integrated
    information identity ``E[ss'] = -E[H]`` holds to the retained order.
    In particular ``H_ss = A + (3 sigma^2/2)(B - A^2)``: the leading term
    ``A`` alone has expectation ``+sigma^2/(2 gamma^4)``, the wrong sign
    versus ``-I_ss = -sigma^2/gamma^4``."""
    den = ytil * ytil + gamma * gamma
    w = (ytil * ytil) / den
    v = (gamma * gamma) / den
    ur = ytil / den
    gr = gamma / den
    rs2 = (sigma * sigma) / den
    Hmm = ((2.0 * w - 2.0 * v)
           + rs2 * (18.0 * w * w - 68.0 * w * v + 10.0 * v * v)
           + rs2 * rs2 * (210.0 * w * w * w - 1638.0 * w * w * v
                          + 1278.0 * w * v * v - 74.0 * v * v * v)
           + rs2 * rs2 * rs2 * (2898.0 * w * w * w * w - 37512.0 * w * w * w * v
                                + 68796.0 * w * w * v * v
                                - 22696.0 * w * v * v * v + 706.0 * v * v * v * v)) / den
    Hms = (sigma / den) * ur * ((12.0 * w - 20.0 * v)
                                + rs2 * (168.0 * w * w - 816.0 * w * v
                                         + 296.0 * v * v)
                                + rs2 * rs2 * (2484.0 * w * w * w
                                               - 23076.0 * w * w * v
                                               + 27036.0 * w * v * v
                                               - 4236.0 * v * v * v))
    Hmg = ur * gr * (-4.0 + rs2 * (40.0 * v - 56.0 * w)
                     + rs2 * rs2 * (-828.0 * w * w + 1928.0 * w * v
                                    - 444.0 * v * v)
                     + rs2 * rs2 * rs2 * (-13488.0 * w * w * w
                                          + 64176.0 * w * w * v
                                          - 49296.0 * w * v * v
                                          + 5648.0 * v * v * v))
    Hss = ((6.0 * w - 2.0 * v)
           + rs2 * (126.0 * w * w - 324.0 * w * v + 30.0 * v * v)
           + rs2 * rs2 * (2070.0 * w * w * w - 12870.0 * w * w * v
                          + 8370.0 * w * v * v - 370.0 * v * v * v)) / den
    Hgs = (sigma / den) * gr * ((4.0 * v - 28.0 * w)
                                + rs2 * (-552.0 * w * w + 688.0 * w * v
                                         - 40.0 * v * v)
                                + rs2 * rs2 * (-10116.0 * w * w * w
                                               + 32436.0 * w * w * v
                                               - 13836.0 * w * v * v
                                               + 444.0 * v * v * v))
    Hgg = ((v - 4.0 * w)
           + rs2 * (-14.0 * w * w + 76.0 * w * v - 6.0 * v * v)
           + rs2 * rs2 * (-138.0 * w * w * w + 1758.0 * w * w * v
                          - 1254.0 * w * v * v + 50.0 * v * v * v)
           + rs2 * rs2 * rs2 * (-1686.0 * w * w * w * w + 38136.0 * w * w * w * v
                                - 70996.0 * w * w * v * v
                                + 21272.0 * w * v * v * v - 518.0 * v * v * v * v)) / den \
          - (w * w) / (gamma * gamma)
    return Hmm, Hms, Hmg, Hss, Hgs, Hgg


def voigt_hessian(y, mu, sigma, gamma):
    """Hessian ``d^2 log f / d theta d theta'``, parameter order ``(mu, sigma, gamma)``.

    Returns shape ``(3, 3)`` for scalar ``y`` and ``(n, 3, 3)`` for array ``y``.
    For the sample sum, prefer :func:`loglik_grad_hess`, which avoids building
    the ``(n, 3, 3)`` array altogether.
    """
    _check_interior(sigma, gamma)
    yv, scalar = _asarray(y)
    ytil = yv - mu
    K, L = _kl(ytil, sigma, gamma)
    smu, ssig, sgam, r = _score_arrays(ytil, sigma, gamma, K, L)
    Hmm, Hms, Hmg, Hss, Hgs, Hgg = _hessian_arrays(ytil, sigma, smu, ssig, sgam, r, gamma)

    far = _far_mask(ytil, sigma, gamma, _R_HESS)
    if far.any():
        idx = np.flatnonzero(far)
        t = _hessian_tail(ytil[idx], sigma, gamma)
        Hmm[idx], Hms[idx], Hmg[idx], Hss[idx], Hgs[idx], Hgg[idx] = t

    n = ytil.size
    out = np.empty((n, 3, 3), dtype=np.float64)
    out[:, 0, 0] = Hmm
    out[:, 1, 1] = Hss
    out[:, 2, 2] = Hgg
    out[:, 0, 1] = out[:, 1, 0] = Hms
    out[:, 0, 2] = out[:, 2, 0] = Hmg
    out[:, 1, 2] = out[:, 2, 1] = Hgs
    return out[0] if scalar else out


# ----------------------------------------------------------------------
# Conditional moments of the Gaussian component (deconvolution)
# ----------------------------------------------------------------------

def voigt_condmean(y, mu, sigma, gamma):
    """``E[Z | Y = y] = ytil - gamma * L/K`` (Tweedie's formula).

    Redescending: it tends to 0 as ``|ytil| -> inf``, so extreme deviations are
    attributed to the Lorentzian component.  The Lorentzian counterpart is
    ``E[X | Y = y] = gamma * L/K``, and the two sum to ``ytil``.
    """
    _check_interior(sigma, gamma)
    yv, scalar = _asarray(y)
    ytil = yv - mu
    K, L = _kl(ytil, sigma, gamma)
    out = ytil - gamma * L / K

    far = _far_mask(ytil, sigma, gamma)
    if far.any():
        # E[Z|y] = sigma^2 s_mu (Tweedie), order by order in the expansion
        idx = np.flatnonzero(far)
        tmu, _, _ = _score_tail(ytil[idx], sigma, gamma)
        out[idx] = sigma * sigma * tmu
    return out[0] if scalar else out


def voigt_condvar(y, mu, sigma, gamma):
    """``V(Z | Y = y) = sqrt(2/pi)*sigma*gamma/K - gamma**2 (1 + L**2/K**2)``.

    Equals ``sigma**2`` at the extrema of the conditional mean and tends to
    ``sigma**2`` as ``|y - mu| -> inf``.
    """
    _check_interior(sigma, gamma)
    yv, scalar = _asarray(y)
    ytil = yv - mu
    K, L = _kl(ytil, sigma, gamma)
    r = L / K
    out = _SQRT2OVERPI * sigma * gamma / K - gamma * gamma * (1 + r * r)

    far = _far_mask(ytil, sigma, gamma)
    if far.any():
        # V(Z|y) = sigma^2 (1 + sigma^2 H_mumu) (Tweedie), order by order
        idx = np.flatnonzero(far)
        thmm = _hessian_tail(ytil[idx], sigma, gamma)[0]
        out[idx] = sigma * sigma * (1.0 + sigma * sigma * thmm)
    return out[0] if scalar else out


# ----------------------------------------------------------------------
# The workhorse: log-likelihood, gradient and Hessian in one Faddeeva pass
# ----------------------------------------------------------------------

def loglik_only(y, mu, sigma, gamma):
    """Log-likelihood plus the ``(K, L, ytil)`` triple, for reuse by the gradient.

    Used by the line search: the Faddeeva evaluation made here is handed to
    :func:`loglik_grad_hess` at the accepted step, so an accepted Newton
    iteration costs exactly one Faddeeva pass over the sample.
    """
    yv, _ = _asarray(y)
    ytil = yv - mu
    K, L = _kl(ytil, sigma, gamma)
    ll = float(np.log(K).sum() - ytil.size * (np.log(sigma) + _LOG_SQRT2PI))
    return ll, (K, L, ytil)


def loglik_grad_hess(y, mu, sigma, gamma, need_hess=True, need_ll=True, _kl_cache=None):
    """Sample log-likelihood, score sum, and Hessian sum from one Faddeeva pass.

    This is the routine an optimiser should call.  It evaluates ``w(z)`` once
    for the whole sample and accumulates the three-vector gradient and the
    3x3 Hessian directly, without ever forming per-observation arrays of shape
    ``(n, 3)`` or ``(n, 3, 3)``.

    Parameters
    ----------
    need_hess : bool
        If ``False``, skip the Hessian recursion and return ``None`` for it.
    _kl_cache : tuple or None
        Optional ``(K, L, ytil)`` computed at these exact parameter values by a
        previous call, reused to skip the Faddeeva evaluation.

    Returns
    -------
    (loglik, grad, hess, cache)
        ``grad`` has shape ``(3,)``, ``hess`` shape ``(3, 3)`` or ``None``, and
        ``cache`` is the ``(K, L, ytil)`` triple for reuse.
    """
    if _kl_cache is not None:
        K, L, ytil = _kl_cache
    else:
        yv, _ = _asarray(y)
        ytil = yv - mu
        K, L = _kl(ytil, sigma, gamma)

    n = ytil.size
    ll = (
        float(np.log(K).sum() - n * (np.log(sigma) + _LOG_SQRT2PI))
        if need_ll
        else float("nan")
    )

    smu, ssig, sgam, r = _score_arrays(ytil, sigma, gamma, K, L)

    # the Hessian switches over much earlier than the score, so two masks;
    # both gate on the expansion parameter r = sigma^2/(ytil^2+gamma^2)
    s2m = sigma * sigma
    den_r = ytil * ytil + gamma * gamma
    far_s = s2m < _R_SCORE * den_r
    has_far_s = bool(far_s.any())
    if has_far_s:
        idx_s = np.flatnonzero(far_s)
        tmu, tsig, tgam = _score_tail(ytil[idx_s], sigma, gamma)
        smu[idx_s], ssig[idx_s], sgam[idx_s] = tmu, tsig, tgam

    grad = np.array([smu.sum(), ssig.sum(), sgam.sum()], dtype=np.float64)

    hess = None
    if need_hess:
        Hmm, Hms, Hmg, Hss, Hgs, Hgg = _hessian_arrays(
            ytil, sigma, smu, ssig, sgam, r, gamma
        )
        far_h = s2m < _R_HESS * den_r
        if far_h.any():
            idx_h = np.flatnonzero(far_h)
            t = _hessian_tail(ytil[idx_h], sigma, gamma)
            Hmm[idx_h], Hms[idx_h], Hmg[idx_h], Hss[idx_h], Hgs[idx_h], Hgg[idx_h] = t
        hmm, hms, hmg = Hmm.sum(), Hms.sum(), Hmg.sum()
        hss, hgs, hgg = Hss.sum(), Hgs.sum(), Hgg.sum()
        hess = np.array(
            [[hmm, hms, hmg], [hms, hss, hgs], [hmg, hgs, hgg]], dtype=np.float64
        )

    return ll, grad, hess, (K, L, ytil)
