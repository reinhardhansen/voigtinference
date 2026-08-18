"""Maximum likelihood estimation for the Voigt profile.

Newton-based local maximization of the exact Voigt log-likelihood in
``eta = (mu, log sigma, log gamma)`` coordinates, with analytic,
machine-stabilized derivatives, a ridge safeguard, and a projected
backtracking line search (the Armijo condition is tested against the executed,
clamped step). With ``starts > 1`` a deterministic set of dispersed starting
values is tried and the best local maximum returned; otherwise the result is a
local stationary or boundary-clamped candidate, not a certified global
maximum.

The Faddeeva evaluation made by the line search at an accepted step is carried
forward and reused for the gradient and Hessian at the next iterate, so an
accepted Newton iteration costs exactly one Faddeeva pass over the sample plus
one pass per rejected trial step.

Standard errors are asymptotic likelihood-based (Wald) quantities from the
interior asymptotic-normality theorem for the iid Voigt MLE in the companion
paper (arXiv:2605.01665). They are deliberately NOT reported at parameter
boundaries, where that theorem does not apply; the Gaussian and Cauchy
boundary-submodel fits returned alongside are the appropriate comparison
there.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .core import loglik_grad_hess, loglik_only
from .fisher import voigt_fisher

__all__ = ["voigt_mle", "VoigtMLEResult"]


@dataclass
class VoigtMLEResult:
    """Result of :func:`voigt_mle`."""

    mu: float
    sigma: float
    gamma: float
    se: np.ndarray = field(repr=False)
    se_obs: np.ndarray = field(repr=False)
    vcov: np.ndarray = field(repr=False)
    loglik: float = float("nan")
    converged: bool = False
    termination: str = "max_iterations"
    sigma_boundary: bool = False
    gamma_boundary: bool = False
    upper_boundary: bool = False
    gradient_norm: float = float("nan")
    projected_gradient_norm: float = float("nan")
    expected_info_posdef: bool = False
    observed_info_posdef: bool = False
    loglik_gaussian: float = float("nan")
    loglik_cauchy: float = float("nan")
    gaussian_fit: tuple = (float("nan"), float("nan"))
    cauchy_fit: tuple = (float("nan"), float("nan"))
    starts: int = 1
    iterations: int = 0
    nfev: int = 0

    @property
    def theta(self) -> np.ndarray:
        """Estimates as an array ``[mu, sigma, gamma]``."""
        return np.array([self.mu, self.sigma, self.gamma])

    @property
    def at_boundary(self) -> bool:
        return self.sigma_boundary or self.gamma_boundary or self.upper_boundary

    def summary(self) -> str:
        names = ("mu", "sigma", "gamma")
        lines = [
            f"Voigt MLE   n-iterations={self.iterations}  "
            f"converged={self.converged}  termination={self.termination}  "
            f"loglik={self.loglik:.6f}"
        ]
        lines.append(f"{'':>7}{'estimate':>14}{'std. error':>14}{'obs. s.e.':>14}")
        for i, nm in enumerate(names):
            lines.append(
                f"{nm:>7}{self.theta[i]:14.6f}{self.se[i]:14.6f}{self.se_obs[i]:14.6f}"
            )
        if self.gamma_boundary:
            lines.append(
                "note: gamma at its lower clamp (fitted model numerically "
                "Gaussian); Wald standard errors suppressed"
            )
        if self.sigma_boundary:
            lines.append(
                "note: sigma at its lower clamp (fitted model numerically "
                "Cauchy); Wald standard errors suppressed"
            )
        if self.upper_boundary:
            lines.append("note: a width reached its upper clamp")
        lines.append(
            f"boundary submodels: loglik(Gaussian) = {self.loglik_gaussian:.6f},"
            f" loglik(Cauchy) = {self.loglik_cauchy:.6f}"
        )
        return "\n".join(lines)


def _startvalues(y: np.ndarray):
    """Moment-free starting values (the Voigt profile has no finite moments)."""
    mu0 = float(np.median(y))
    d = np.abs(y - mu0)
    # tails are Lorentzian: P(|Y - mu| > c) ~ 2 gamma / (pi c), so at c = q90(|d|)
    # the implied scale is gamma ~ 0.05 pi c
    gamma0 = max(0.05 * np.pi * float(np.quantile(d, 0.90)), 1e-8)
    q75, q25 = np.quantile(y, [0.75, 0.25])
    iqr = float(q75 - q25)
    # the core is Gaussian plus Lorentzian: subtract the Lorentzian IQR (= 2 gamma)
    sigma0 = max((iqr - 2.0 * gamma0) / 1.349, 0.05 * iqr, 1e-8)
    return mu0, sigma0, gamma0


def _chol_solve(A: np.ndarray, b: np.ndarray):
    """Solve ``A x = b`` if ``A`` is positive definite, else return ``None``."""
    try:
        c = np.linalg.cholesky(A)
    except np.linalg.LinAlgError:
        return None
    z = np.linalg.solve(c, b)
    return np.linalg.solve(c.T, z)


def _is_posdef(A: np.ndarray) -> bool:
    try:
        np.linalg.cholesky(0.5 * (A + A.T))
        return True
    except np.linalg.LinAlgError:
        return False


# ----------------------------------------------------------------------
# boundary submodels (likelihood-based boundary diagnosis)
# ----------------------------------------------------------------------

def _gaussian_fit(y: np.ndarray):
    """Gaussian-submodel MLE (the gamma = 0 boundary): closed form."""
    n = y.size
    m = float(np.mean(y))
    s2 = float(np.mean((y - m) ** 2))
    ll = -0.5 * n * (np.log(2.0 * np.pi * s2) + 1.0)
    return m, float(np.sqrt(s2)), float(ll)


def _cauchy_fit(y: np.ndarray, maxiter: int = 100, gtol: float = 1e-9):
    """Cauchy-submodel MLE (the sigma = 0 boundary), by safeguarded Newton in
    (mu, log gamma) with the Fisher-scoring direction as fallback. The Cauchy
    expected information is diagonal and known (I(mu) = n/(2 gamma^2),
    I(log gamma) = n/2), so the scoring direction is always an ascent
    direction; Newton supplies the quadratic tail. Cost is deterministic and
    small (typically < 15 iterations)."""
    n = y.size
    mu = float(np.median(y))
    gamma = max(float(np.quantile(np.abs(y - mu), 0.5)), 1e-12)

    def cll(m, g):
        # errstate: degenerate inputs (coincident points, gamma underflow) can
        # produce -inf trial values; the line search rejects them, so the
        # warnings are noise
        with np.errstate(divide="ignore", invalid="ignore"):
            return float(np.sum(np.log(g) - np.log(np.pi)
                                - np.log((y - m) ** 2 + g * g)))

    ll = cll(mu, gamma)
    converged = False
    for _ in range(maxiter):
        d = y - mu
        den = d * d + gamma * gamma
        g2 = gamma * gamma
        gmu = float(np.sum(2.0 * d / den))
        ggam = float(np.sum(1.0 / gamma - 2.0 * gamma / den))
        gs = gamma * ggam                                   # d/d(log gamma)
        if np.hypot(gmu, gs) < gtol * n:
            converged = True
            break
        # observed information in (mu, log gamma)
        hmm = float(np.sum(2.0 * (d * d - g2) / den ** 2))
        hgg = -n / g2 - float(np.sum(2.0 * (d * d - g2) / den ** 2))
        hmg = float(np.sum(-4.0 * d * gamma / den ** 2))
        hss = g2 * hgg + gs                                 # chain rule to log gamma
        hms = gamma * hmg
        det = hmm * hss - hms * hms
        use_newton = det > 0.0 and hmm < 0.0                # negative definite
        if use_newton:
            dmu = -(hss * gmu - hms * gs) / det
            ds = -(hmm * gs - hms * gmu) / det
            use_newton = dmu * gmu + ds * gs > 0.0          # ascent check
        if not use_newton:                                  # Fisher scoring
            dmu = 2.0 * g2 * gmu / n
            ds = 2.0 * gs / n
        gp0 = dmu * gmu + ds * gs
        t = 1.0
        accepted = False
        for _ in range(30):
            mun = mu + t * dmu
            gamn = gamma * np.exp(t * ds)
            lln = cll(mun, gamn)
            if np.isfinite(lln) and lln >= ll + 1e-4 * t * gp0:
                accepted = True
                stalled = abs(lln - ll) <= 1e-13 * (abs(ll) + 1.0)
                mu, gamma, ll = mun, gamn, lln
                break
            t *= 0.5
        if not accepted:
            break                                           # numerically stationary
        if stalled:
            converged = True                                # no double-precision headroom left
            break
    return mu, gamma, ll, converged


# ----------------------------------------------------------------------
# Newton core
# ----------------------------------------------------------------------

def _projnorm(g: np.ndarray, eta: np.ndarray, lb: float, ub: float) -> float:
    """Projected gradient norm: components pushing into an active clamp are
    zeroed."""
    g2, g3 = g[1], g[2]
    if (eta[1] <= lb and g2 < 0.0) or (eta[1] >= ub and g2 > 0.0):
        g2 = 0.0
    if (eta[2] <= lb and g3 < 0.0) or (eta[2] >= ub and g3 > 0.0):
        g3 = 0.0
    return float(np.sqrt(g[0] ** 2 + g2 * g2 + g3 * g3))


def _newton_core(y, eta0, lb, ub, maxiter, gtol, verbose):
    n = y.size
    eta = np.array(eta0, dtype=np.float64)
    ll, cache = loglik_only(y, eta[0], np.exp(eta[1]), np.exp(eta[2]))
    nfev = 1
    if not np.isfinite(ll):
        return eta, -np.inf, False, 0, "nonfinite_start", np.nan, np.nan, nfev
    ll /= n

    converged = False
    reason = "max_iterations"
    it = 0
    eye = np.eye(3)
    g = np.full(3, np.nan)

    while it < maxiter:
        it += 1
        mu, sigma, gamma = eta[0], np.exp(eta[1]), np.exp(eta[2])

        _, g_raw, H_raw, _ = loglik_grad_hess(
            y, mu, sigma, gamma, need_hess=True, need_ll=False, _kl_cache=cache
        )
        g_raw = g_raw / n
        H_raw = H_raw / n
        if not (np.all(np.isfinite(g_raw)) and np.all(np.isfinite(H_raw))):
            reason = "nonfinite_derivatives"
            break

        # chain rule to eta = (mu, log sigma, log gamma)
        D = np.array([1.0, sigma, gamma])
        g = D * g_raw
        H = (D[:, None] * H_raw * D[None, :]) + np.diag(
            [0.0, sigma * g_raw[1], gamma * g_raw[2]]
        )
        H = 0.5 * (H + H.T)

        if _projnorm(g, eta, lb, ub) < gtol:
            converged = True
            reason = "gradient_converged"
            break

        # ascent direction: solve (-H + lam I) delta = g with the smallest ridge
        lam = 0.0
        delta = None
        while lam <= 1e12:
            sol = _chol_solve(-H + lam * eye, g)
            if sol is not None and float(sol @ g) > 0.0:
                delta = sol
                break
            lam = 1e-6 if lam == 0.0 else 10.0 * lam
        if delta is None:
            reason = "no_ascent_direction"
            break

        # projected backtracking line search: the Armijo condition is tested
        # against the EXECUTED (clamped) step p, not the proposed t*delta
        t = 1.0
        improved = False
        for _ in range(40):
            eta_new = eta + t * delta
            eta_new[1] = min(max(eta_new[1], lb), ub)
            eta_new[2] = min(max(eta_new[2], lb), ub)
            p = eta_new - eta
            gp = float(g @ p)
            if gp > 0.0:
                ll_new, cache_new = loglik_only(
                    y, eta_new[0], np.exp(eta_new[1]), np.exp(eta_new[2])
                )
                nfev += 1
                ll_new /= n
                if np.isfinite(ll_new) and ll_new >= ll + 1e-4 * gp:
                    eta, ll, cache = eta_new, ll_new, cache_new
                    improved = True
                    break
            t *= 0.5

        if not improved:
            converged = _projnorm(g, eta, lb, ub) < 1e-4
            reason = "gradient_converged" if converged else "line_search_failed"
            break

        if verbose:
            print(f"iter {it:3d}  loglik {n * ll:.8f}")

    gnorm = float(np.linalg.norm(g)) if np.all(np.isfinite(g)) else np.nan
    pgnorm = _projnorm(g, eta, lb, ub) if np.all(np.isfinite(g)) else np.nan
    return eta, ll, converged, it, reason, gnorm, pgnorm, nfev


def voigt_mle(
    y,
    maxiter: int = 200,
    gtol: float = 1e-8,
    nodes: int = 400,
    starts: int = 1,
    verbose: bool = False,
) -> VoigtMLEResult:
    """Maximum likelihood estimation of the Voigt profile from an iid sample.

    Parameters
    ----------
    y : array_like
        The sample (finite values; raises otherwise).
    maxiter, gtol : int, float
        Newton iteration cap and projected-gradient tolerance (on the
        *average* log-likelihood, so the tolerance is sample-size free).
    nodes : int
        Gauss-Legendre nodes for the expected information behind ``se``.
    starts : int
        Number of deterministic starting values (multistart). Default 1.

    Returns
    -------
    VoigtMLEResult
        Estimates; asymptotic Fisher-information standard errors ``se`` and
        observed-information standard errors ``se_obs`` (NaN unless the
        corresponding information matrix is positive definite AND the estimate
        is interior); ``vcov``; ``loglik``; convergence and termination
        diagnostics; separated ``sigma_boundary``/``gamma_boundary``/
        ``upper_boundary`` clamp flags; gradient norms; the Gaussian and
        Cauchy boundary-submodel fits and log-likelihoods; ``iterations`` and
        ``nfev`` (Faddeeva passes over the sample).

    Notes
    -----
    The standard errors are asymptotic likelihood-based (Wald) quantities from
    the interior asymptotic-normality theorem for the iid Voigt MLE in the
    companion paper (arXiv:2605.01665); asymptotic normality holds at rate
    ``sqrt(n)`` although the Voigt profile has no finite positive integer
    moments. Wald standard errors are not valid at parameter boundaries and
    are suppressed there; compare ``loglik`` with ``loglik_gaussian`` and
    ``loglik_cauchy`` instead. The optimiser does not raise on finite data.
    """
    y = np.asarray(y, dtype=np.float64).ravel()
    n = y.size
    if n < 3:
        raise ValueError("need at least 3 observations")
    if not np.all(np.isfinite(y)):
        raise ValueError("data contain non-finite values")

    mu0, sigma0, gamma0 = _startvalues(y)
    # keep the log-widths in a sane range relative to the data scale; the true
    # boundary cases sigma -> 0 / gamma -> 0 lie outside the regular family
    s0 = max(sigma0, gamma0)
    lb, ub = np.log(1e-8 * s0), np.log(1e8 * s0)

    startlist = [np.array([mu0, np.log(sigma0), np.log(gamma0)])]
    if starts > 1:
        q75, q25 = np.quantile(y, [0.75, 0.25])
        iqr = float(q75 - q25)
        for fs, fg, dm in ((4.0, 0.25, 0.0), (0.25, 4.0, 0.0),
                           (1.0, 1.0, 0.5 * iqr), (1.0, 1.0, -0.5 * iqr),
                           (8.0, 8.0, 0.0), (0.125, 0.125, 0.0)):
            if len(startlist) >= starts:
                break
            startlist.append(np.array([
                mu0 + dm,
                min(max(np.log(sigma0 * fs), lb), ub),
                min(max(np.log(gamma0 * fg), lb), ub),
            ]))

    best = None
    nfev = 0
    for eta0 in startlist:
        res = _newton_core(y, eta0, lb, ub, maxiter, gtol, verbose)
        nfev += res[7]
        if best is None or res[1] > best[1]:
            best = res
    eta, ll, converged, it, reason, gnorm, pgnorm, _ = best

    mu_hat = float(eta[0])
    sigma_hat = float(np.exp(eta[1]))
    gamma_hat = float(np.exp(eta[2]))
    # boundary flags: clamp active OR width negligible relative to the other
    # width (the gradient in a log-width vanishes proportionally to the width,
    # so the optimiser can satisfy the gradient criterion at a tiny width
    # without touching the literal clamp)
    sigma_boundary = bool(eta[1] <= lb + 1e-10 or sigma_hat < 1e-6 * gamma_hat)
    gamma_boundary = bool(eta[2] <= lb + 1e-10 or gamma_hat < 1e-6 * sigma_hat)
    upper_boundary = bool(eta[1] >= ub - 1e-10 or eta[2] >= ub - 1e-10)
    at_boundary = sigma_boundary or gamma_boundary or upper_boundary

    # Wald standard errors require a positive-definite information matrix and
    # an interior estimate; otherwise NaN plus diagnostics -- a clipped zero
    # would misrepresent an invalid covariance as certainty
    se = np.full(3, np.nan)
    vcov = np.full((3, 3), np.nan)
    expected_pd = False
    try:
        info = voigt_fisher(mu_hat, sigma_hat, gamma_hat, nodes=nodes)
        expected_pd = _is_posdef(np.asarray(info))
        if expected_pd and not at_boundary:
            vcov = np.linalg.inv(info) / n
            se = np.sqrt(np.diag(vcov))
    except (np.linalg.LinAlgError, ValueError):
        pass

    se_obs = np.full(3, np.nan)
    observed_pd = False
    try:
        _, _, H_sum, _ = loglik_grad_hess(
            y, mu_hat, sigma_hat, gamma_hat, need_hess=True, need_ll=False
        )
        nfev += 1
        M = -0.5 * (H_sum + H_sum.T)
        observed_pd = _is_posdef(M)
        if observed_pd and not at_boundary:
            se_obs = np.sqrt(np.diag(np.linalg.inv(M)))
    except (np.linalg.LinAlgError, ValueError):
        pass

    mg, sg, llg = _gaussian_fit(y)
    mc, gc, llc, _ = _cauchy_fit(y)

    return VoigtMLEResult(
        mu=mu_hat,
        sigma=sigma_hat,
        gamma=gamma_hat,
        se=se,
        se_obs=se_obs,
        vcov=vcov,
        loglik=float(n * ll),
        converged=converged,
        termination=reason,
        sigma_boundary=sigma_boundary,
        gamma_boundary=gamma_boundary,
        upper_boundary=upper_boundary,
        gradient_norm=gnorm,
        projected_gradient_norm=pgnorm,
        expected_info_posdef=expected_pd,
        observed_info_posdef=observed_pd,
        loglik_gaussian=llg,
        loglik_cauchy=llc,
        gaussian_fit=(mg, sg),
        cauchy_fit=(mc, gc),
        starts=len(startlist),
        iterations=it,
        nfev=nfev,
    )
