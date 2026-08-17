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

#: Far-tail switches, as squared multiples of ``sqrt(sigma**2 + gamma**2)``.
#:
#: The closed-form score and Hessian are algebraically exact but lose precision
#: to floating-point cancellation in the tail: ``ytil*L`` and
#: ``sqrt(2/pi)*sigma`` agree to many digits while their *difference* is the
#: answer.  Out there the profile is Cauchy-dominated and the moment expansion
#: ``f(y) = c(ytil) + (sigma**2/2) c''(ytil) + O(sigma**4/ytil**6)``, with ``c``
#: the Cauchy(0, gamma) density, gives limits with relative error
#: ``O((sigma**2 + gamma**2)/ytil**2)``.  Each switch is placed where the two
#: error curves cross.
#:
#: The score and the Hessian need *different* switches.  The Hessian recursion
#: divides the cancelling score components by ``sigma`` and multiplies by
#: ``ytil`` at every step, so it loses digits far sooner: its crossover sits at
#: ``|ytil| ~ 40 * sqrt(sigma**2 + gamma**2)``, against ~500 for the score.
#: Using one switch for both -- as an earlier version of this code did, and as
#: VoigtInference.jl still does -- leaves the Hessian with no correct digits at
#: all over roughly a decade and a half of the tail.
#:
#: Both constants were chosen by minimising the worst-case relative error
#: against 60-digit ground truth over ``|ytil|/scale`` in ``[1, 1e6]`` and over
#: a spread of ``(sigma, gamma)``; see ``tests/test_accuracy.py``, which both
#: verifies the resulting bounds and prints the crossover tables.
_FAR_TAIL_SQ = 2.5e5        # score and conditional moments: |ytil| > 500 * scale
_FAR_TAIL_HESS_SQ = 1.6e3   # Hessian:                       |ytil| >  40 * scale


def _asarray(y):
    """Return ``(1-d float array, was_scalar)``."""
    arr = np.asarray(y, dtype=np.float64)
    if arr.ndim == 0:
        return arr.reshape(1), True
    return arr, False


def _check(sigma, gamma):
    if sigma < 0.0:
        raise ValueError("sigma must be >= 0")
    if gamma < 0.0:
        raise ValueError("gamma must be >= 0")


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


def _far_mask(ytil, sigma, gamma, thresh_sq=_FAR_TAIL_SQ):
    return ytil * ytil > thresh_sq * (sigma * sigma + gamma * gamma)


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
    yv, scalar = _asarray(y)
    K, _ = _kl(yv - mu, sigma, gamma)
    out = np.log(K) - np.log(sigma) - _LOG_SQRT2PI
    return out[0] if scalar else out


def voigt_loglik(y, mu, sigma, gamma):
    """Sample log-likelihood ``sum_i log f(y_i)``."""
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
    """Cauchy-limit score, used where ``_far_mask`` is set."""
    den = ytil * ytil + gamma * gamma
    smu = 2 * ytil / den
    ssig = sigma * (6 * (ytil * ytil) - 2 * (gamma * gamma)) / (den * den)
    sgam = 1.0 / gamma - 2 * gamma / den
    return smu, ssig, sgam


def voigt_score(y, mu, sigma, gamma):
    """Score ``d log f(y; theta) / d theta`` for ``theta = (mu, sigma, gamma)``.

    Returns shape ``(3,)`` for scalar ``y`` and ``(n, 3)`` for array ``y``.
    """
    _check(sigma, gamma)
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
    den = ytil * ytil + gamma * gamma
    den2 = den * den
    den3 = den2 * den
    q = (6 * (ytil * ytil) - 2 * (gamma * gamma)) / den2
    d = 2 * (ytil * ytil - gamma * gamma) / den2
    Hmm = d
    Hgg = -1 / (gamma * gamma) - d
    Hmg = -4 * ytil * gamma / den2
    Hms = sigma * (-12 * ytil / den2 + 4 * ytil * (6 * (ytil * ytil) - 2 * (gamma * gamma)) / den3)
    Hgs = sigma * (-4 * gamma / den2 - 4 * gamma * (6 * (ytil * ytil) - 2 * (gamma * gamma)) / den3)
    Hss = q
    return Hmm, Hms, Hmg, Hss, Hgs, Hgg


def voigt_hessian(y, mu, sigma, gamma):
    """Hessian ``d^2 log f / d theta d theta'``, parameter order ``(mu, sigma, gamma)``.

    Returns shape ``(3, 3)`` for scalar ``y`` and ``(n, 3, 3)`` for array ``y``.
    For the sample sum, prefer :func:`loglik_grad_hess`, which avoids building
    the ``(n, 3, 3)`` array altogether.
    """
    _check(sigma, gamma)
    yv, scalar = _asarray(y)
    ytil = yv - mu
    K, L = _kl(ytil, sigma, gamma)
    smu, ssig, sgam, r = _score_arrays(ytil, sigma, gamma, K, L)
    Hmm, Hms, Hmg, Hss, Hgs, Hgg = _hessian_arrays(ytil, sigma, smu, ssig, sgam, r, gamma)

    far = _far_mask(ytil, sigma, gamma, _FAR_TAIL_HESS_SQ)
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
    _check(sigma, gamma)
    yv, scalar = _asarray(y)
    ytil = yv - mu
    K, L = _kl(ytil, sigma, gamma)
    out = ytil - gamma * L / K

    far = _far_mask(ytil, sigma, gamma)
    if far.any():
        idx = np.flatnonzero(far)
        yt = ytil[idx]
        out[idx] = sigma * sigma * (2 * yt) / (yt * yt + gamma * gamma)
    return out[0] if scalar else out


def voigt_condvar(y, mu, sigma, gamma):
    """``V(Z | Y = y) = sqrt(2/pi)*sigma*gamma/K - gamma**2 (1 + L**2/K**2)``.

    Equals ``sigma**2`` at the extrema of the conditional mean and tends to
    ``sigma**2`` as ``|y - mu| -> inf``.
    """
    _check(sigma, gamma)
    yv, scalar = _asarray(y)
    ytil = yv - mu
    K, L = _kl(ytil, sigma, gamma)
    r = L / K
    out = _SQRT2OVERPI * sigma * gamma / K - gamma * gamma * (1 + r * r)

    far = _far_mask(ytil, sigma, gamma)
    if far.any():
        idx = np.flatnonzero(far)
        yt = ytil[idx]
        den = yt * yt + gamma * gamma
        out[idx] = sigma * sigma * (1 + sigma * sigma * 2 * (yt * yt - gamma * gamma) / (den * den))
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

    # the Hessian switches over much earlier than the score, so two masks
    ysq = ytil * ytil
    thresh = sigma * sigma + gamma * gamma
    far_s = ysq > _FAR_TAIL_SQ * thresh
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
        far_h = ysq > _FAR_TAIL_HESS_SQ * thresh
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
