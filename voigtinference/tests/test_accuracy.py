"""High-precision accuracy validation against mpmath.

These tests replace the "two implementations agree" style of check with
something stronger: agreement with 60-digit ground truth.  They validate

1. the Faddeeva primitive (``scipy.special.wofz``) across the ``(x, a)`` plane;
2. the exact closed-form score and Hessian in the bulk;
3. the Cauchy-limit expansions used in the far tail; and
4. the placement of the two switches between them -- the one genuinely delicate
   numerical choice in the package.

The score and the Hessian switch at *different* points.  The Hessian recursion
divides the cancelling score components by ``sigma`` and multiplies by ``ytil``
at each step, so it loses digits far sooner: its dispatch criterion is
``r = sigma^2/(ytil^2 + gamma^2) < 5e-4`` against ``r < 1e-4`` for the score
(for the moderate-ratio regimes here that is roughly ``|ytil| = 45 * scale``
versus ``100 * scale``; at ``gamma >> sigma`` both criteria hold for every
``ytil``, including the line center).  Sharing one switch would cost the
Hessian about two orders of magnitude at its crossover, which is what these
tests exist to prevent regressing.

Run this module as a script to print both crossover tables.
"""

import numpy as np
import pytest

from voigtinference import voigt_hessian, voigt_score
from voigtinference.core import (
    _R_HESS,
    _R_SCORE,
    _hessian_arrays,
    _hessian_tail,
    _kl,
    _score_arrays,
    _score_tail,
)

mpmath = pytest.importorskip("mpmath")
from mpmath import erfc, exp, mp, mpc, mpf, pi, sqrt  # noqa: E402

#: Working precision for the reference, in decimal digits.
#:
#: A fixed precision is NOT enough.  The Hessian recursion forms
#: ``smu - ytil*hmm``, a difference of two quantities of size ``2/ytil`` whose
#: value is of size ``1/ytil**3``; the reference therefore loses digits in
#: proportion to ``log10(|ytil|)``, and at 60 digits it silently degrades past
#: ``|ytil| ~ 1e5 * scale`` -- which looks exactly like a failure of the tail
#: expansion.  ``test_reference_precision_is_sufficient`` guards against this.
_BASE_DPS = 60
_DPS_PER_DECADE = 12


def _set_dps(mult):
    mp.dps = int(_BASE_DPS + _DPS_PER_DECADE * np.log10(max(mult, 1.0)))


mp.dps = _BASE_DPS

REGIMES = [(1.0, 0.3), (1.2, 0.7), (1.0, 2.0), (0.5, 0.05)]
SIGMA, GAMMA = REGIMES[0]


def _scale(sigma, gamma):
    return np.sqrt(sigma**2 + gamma**2)


def _switch_mult(r_thresh, sigma, gamma):
    """The r-criterion sigma^2 < r_thresh*(ytil^2 + gamma^2), expressed as the
    |ytil|/scale multiple at which the branch flips. For the moderate-ratio
    REGIMES here the criterion is |ytil|-driven; at large gamma/sigma it holds
    for every ytil (the large-ratio cancellation regime), covered by certify.jl."""
    arg = sigma * sigma / r_thresh - gamma * gamma
    return np.sqrt(max(arg, 0.0)) / _scale(sigma, gamma)


def _score_switch(sigma=SIGMA, gamma=GAMMA):
    return _switch_mult(_R_SCORE, sigma, gamma)


def _hess_switch(sigma=SIGMA, gamma=GAMMA):
    return _switch_mult(_R_HESS, sigma, gamma)


SCORE_SWITCH = _score_switch()   # default regime, for dispatch and reports
HESS_SWITCH = _hess_switch()


def _w_mp(x, a):
    z = mpc(mpf(x), mpf(a))
    return exp(-z * z) * erfc(mpc(0, -1) * z)


def _truth(ytil, sigma, gamma):
    """Ground-truth score and Hessian from the exact formulas, in high precision."""
    _set_dps(abs(ytil) / _scale(sigma, gamma))
    yt, sg, gm = mpf(ytil), mpf(sigma), mpf(gamma)
    w = _w_mp(yt / (sg * sqrt(2)), gm / (sg * sqrt(2)))
    K, L = w.real, w.imag
    s2 = sg * sg
    r = L / K
    smu = (yt - gm * r) / s2
    ssg = ((yt * yt - gm * gm - s2) * K - 2 * gm * yt * L + sqrt(2 / pi) * sg * gm) / (s2 * sg * K)
    sgm = (gm * K + yt * L - sqrt(2 / pi) * sg) / (s2 * K)
    hmm = ssg / sg - smu * smu
    hgg = -ssg / sg - sgm * sgm
    hmg = (yt * sgm + gm * smu - r) / s2 - smu * sgm
    hms = -(smu + gm * hmg - yt * hmm) / sg
    hgs = -(sgm + gm * hgg - yt * hmg) / sg
    hss = -(ssg + gm * hgs - yt * hms) / sg
    score = np.array([float(smu), float(ssg), float(sgm)])
    hess = np.array([float(v) for v in (hmm, hms, hmg, hss, hgs, hgg)])
    return score, hess


def _exact_branch(ytil, sigma, gamma):
    a = np.array([ytil], dtype=float)
    K, L = _kl(a, sigma, gamma)
    smu, ssig, sgam, r = _score_arrays(a, sigma, gamma, K, L)
    H = _hessian_arrays(a, sigma, smu, ssig, sgam, r, gamma)
    return (
        np.array([smu[0], ssig[0], sgam[0]]),
        np.array([float(np.atleast_1d(c)[0]) for c in H]),
    )


def _tail_branch(ytil, sigma, gamma):
    a = np.array([ytil], dtype=float)
    s = _score_tail(a, sigma, gamma)
    H = _tail_h(a, sigma, gamma)
    return np.array([float(np.atleast_1d(c)[0]) for c in s]), H


def _tail_h(a, sigma, gamma):
    return np.array([float(np.atleast_1d(c)[0]) for c in _hessian_tail(a, sigma, gamma)])


def _relerr(approx, truth):
    return float(np.max(np.abs(approx - truth) / np.abs(truth)))


def _normerr(approx, truth):
    """Normwise block error: max component error over the block's largest
    true component. This is the certified metric (see examples/certify.jl in
    the Julia package): componentwise ratios are meaningless at zeros of the
    leading-order expansion, and Newton steps / Wald matrices are perturbed
    at the level of the block norm."""
    scale = float(np.max(np.abs(truth)))
    return float(np.max(np.abs(approx - truth))) / scale


# ------------------------------------------------------------- the primitive


@pytest.mark.parametrize("a", [1e-3, 0.05, 0.3, 1.0, 5.0, 50.0])
def test_wofz_matches_high_precision(a):
    """scipy's Faddeeva routine, checked against 60-digit ground truth."""
    _set_dps(1.0)  # `_truth` leaves mp.dps wherever it last needed it
    xs = np.array([0.0, 0.1, 0.7, 1.5, 4.0, 12.0, 60.0, 300.0])
    K, L = _kl(xs * (SIGMA * np.sqrt(2)), SIGMA, a * SIGMA * np.sqrt(2))
    got = K + 1j * L
    want = np.array([complex(_w_mp(x, a)) for x in xs])
    assert _relerr(got, want) < 1e-13


# --------------------------------------------------------- branch behaviour


def test_exact_branch_is_accurate_in_the_bulk():
    scale = _scale(SIGMA, GAMMA)
    for mult in (0.5, 1.0, 3.0, 10.0):
        yt = mult * scale
        s_t, h_t = _truth(yt, SIGMA, GAMMA)
        s_e, h_e = _exact_branch(yt, SIGMA, GAMMA)
        assert _relerr(s_e, s_t) < 1e-9, f"score, mult={mult}"
        assert _relerr(h_e, h_t) < 1e-6, f"hessian, mult={mult}"


def test_tail_expansions_converge():
    scale = _scale(SIGMA, GAMMA)
    prev_s = prev_h = np.inf
    for mult in (1e3, 1e4, 1e5, 1e6):
        yt = mult * scale
        s_t, h_t = _truth(yt, SIGMA, GAMMA)
        s_l, h_l = _tail_branch(yt, SIGMA, GAMMA)
        es, eh = _relerr(s_l, s_t), _relerr(h_l, h_t)
        # strictly falling until the double-precision rounding floor
        assert es < prev_s or es < 1e-13, f"expansion error must fall, mult={mult}"
        assert eh < prev_h or eh < 1e-13, f"expansion error must fall, mult={mult}"
        prev_s, prev_h = es, eh
    assert prev_s < 1e-10
    assert prev_h < 1e-10


def test_reference_precision_is_sufficient():
    """The reference must be converged wherever it is used as ground truth.

    Recomputing at double the working precision must not move it.  Without the
    |ytil|-dependent precision this fails from mult ~ 1e6 upward, and the
    symptom is indistinguishable from a broken tail expansion.
    """
    global _DPS_PER_DECADE
    for sigma, gamma in REGIMES[:2]:
        scale = _scale(sigma, gamma)
        for mult in (1e2, 1e4, 1e6, 1e8):
            _, h_a = _truth(mult * scale, sigma, gamma)
            saved = _DPS_PER_DECADE
            _DPS_PER_DECADE = 2 * saved + _BASE_DPS
            try:
                _, h_b = _truth(mult * scale, sigma, gamma)
            finally:
                _DPS_PER_DECADE = saved
            assert _relerr(h_a, h_b) < 1e-20, f"reference not converged at mult={mult:.0e}"


def test_hessian_switches_no_later_than_the_score():
    """The Hessian's exact branch is never more accurate than the score's,
    so it must never switch later (r_h >= r_s, i.e. smaller |y| switch)."""
    assert HESS_SWITCH <= SCORE_SWITCH + 1e-12


# --------------------------------------- worst case over the whole real line


def _worst_case(kind, override_hess_switch=None):
    worst, where, regime = 0.0, 0.0, None
    for sigma, gamma in REGIMES:
        scale = _scale(sigma, gamma)
        hsw = _hess_switch(sigma, gamma) if override_hess_switch is None \
            else override_hess_switch
        # decade grid plus both sides of each dispatch crossing, where the
        # exact-branch error peaks -- grid placement must not decide the worst
        crossings = [m for sw in (_score_switch(sigma, gamma), hsw)
                     for m in (0.999 * sw, 1.001 * sw) if m > 0]
        for mult in np.concatenate([np.logspace(0, 6, 40), crossings]):
            yt = mult * scale
            s_t, h_t = _truth(yt, sigma, gamma)
            if kind == "score":
                got = _tail_branch(yt, sigma, gamma)[0] if mult > _score_switch(sigma, gamma) \
                    else _exact_branch(yt, sigma, gamma)[0]
                err = _normerr(got, s_t)
            else:
                got = _tail_branch(yt, sigma, gamma)[1] if mult > hsw \
                    else _exact_branch(yt, sigma, gamma)[1]
                err = _normerr(got, h_t)
            if err > worst:
                worst, where, regime = err, mult, (sigma, gamma)
    return worst, where, regime


def test_score_worst_case_relative_error():
    # validated normwise bound 1.4e-10 over the full certify.jl grid
    # (worst case at the r_s crossing); wide margin for platform variation
    worst, where, regime = _worst_case("score")
    assert worst < 1e-8, f"{worst:.2e} at mult={where:.0f}, (sigma, gamma)={regime}"


def test_hessian_worst_case_relative_error():
    # validated normwise bound 6.3e-7 over the full certify.jl grid
    # (worst case at the r_h crossing); margin as above
    worst, where, regime = _worst_case("hessian")
    assert worst < 1e-5, f"{worst:.2e} at mult={where:.0f}, (sigma, gamma)={regime}"


def test_sharing_the_score_switch_would_degrade_the_hessian():
    """The regression the r_s/r_h split exists to prevent.  After the
    minimax retune the switches sit a factor five apart in r (r_s = 1e-4,
    r_h = 5e-4), so forcing the Hessian to use the score threshold costs
    about two orders of magnitude (7.1e-5 vs 6.3e-7 on the certify grid);
    assert a factor 30 for platform robustness."""
    split, _, _ = _worst_case("hessian")
    shared, _, _ = _worst_case("hessian", override_hess_switch=SCORE_SWITCH)
    assert shared > 30 * split, (shared, split)


# ------------------------------------------- the dispatched public interface


def test_public_score_and_hessian_are_uniformly_accurate():
    scale = _scale(SIGMA, GAMMA)
    worst_s = worst_h = 0.0
    for mult in np.logspace(-1, 8, 40):
        yt = mult * scale
        s_t, h_t = _truth(yt, SIGMA, GAMMA)
        s = voigt_score(yt, 0.0, SIGMA, GAMMA)
        H = voigt_hessian(yt, 0.0, SIGMA, GAMMA)
        h = np.array([H[0, 0], H[0, 1], H[0, 2], H[1, 1], H[1, 2], H[2, 2]])
        worst_s = max(worst_s, _normerr(s, s_t))
        worst_h = max(worst_h, _normerr(h, h_t))
    assert worst_s < 5e-6, f"score worst {worst_s:.2e}"
    assert worst_h < 5e-3, f"hessian worst {worst_h:.2e}"


# ------------------------------------------------------------------ reporting


def _table(sigma, gamma):
    rows = []
    scale = _scale(sigma, gamma)
    for mult in (1, 10, 30, 40, 1e2, 3e2, 5e2, 1e3, 1e4, 1e6):
        yt = mult * scale
        s_t, h_t = _truth(yt, sigma, gamma)
        s_e, h_e = _exact_branch(yt, sigma, gamma)
        s_l, h_l = _tail_branch(yt, sigma, gamma)
        rows.append(
            (mult, _relerr(s_e, s_t), _relerr(s_l, s_t), _relerr(h_e, h_t), _relerr(h_l, h_t))
        )
    return rows


if __name__ == "__main__":
    for sigma, gamma in REGIMES[:2]:
        print(f"\nsigma = {sigma}, gamma = {gamma}   "
              f"(score switch {SCORE_SWITCH:.0f}x, Hessian switch {HESS_SWITCH:.0f}x scale)\n")
        print(f"{'|ytil|/scale':>14}{'score exact':>14}{'score tail':>14}"
              f"{'HESS exact':>14}{'HESS tail':>14}")
        for mult, se, sl, he, hl in _table(sigma, gamma):
            mark = ""
            if mult == HESS_SWITCH:
                mark = "  <- Hessian switch"
            elif mult == SCORE_SWITCH:
                mark = "  <- score switch"
            print(f"{mult:14.0f}{se:14.2e}{sl:14.2e}{he:14.2e}{hl:14.2e}{mark}")
    print("\nRelative error against 60-digit ground truth.")
    for kind in ("score", "hessian"):
        worst, where, regime = _worst_case(kind)
        print(f"worst-case {kind:8s}: {worst:.2e} at mult={where:.0f}, "
              f"(sigma, gamma)={regime}")
