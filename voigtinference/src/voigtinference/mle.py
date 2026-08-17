"""Maximum likelihood estimation for the Voigt profile.

Newton's method with the analytic score and Hessian, in
``eta = (mu, log sigma, log gamma)`` coordinates, with a ridge safeguard and an
Armijo backtracking line search.

The Faddeeva evaluation made by the line search at an accepted step is carried
forward and reused for the gradient and Hessian at the next iterate, so an
accepted Newton iteration costs exactly one Faddeeva pass over the sample plus
one pass per rejected trial step.
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
    boundary: bool = False
    iterations: int = 0
    nfev: int = 0

    @property
    def theta(self) -> np.ndarray:
        """Estimates as an array ``[mu, sigma, gamma]``."""
        return np.array([self.mu, self.sigma, self.gamma])

    def summary(self) -> str:
        names = ("mu", "sigma", "gamma")
        lines = [
            f"Voigt MLE   n-iterations={self.iterations}  "
            f"converged={self.converged}  loglik={self.loglik:.6f}"
        ]
        lines.append(f"{'':>7}{'estimate':>14}{'std. error':>14}{'obs. s.e.':>14}")
        for i, nm in enumerate(names):
            lines.append(
                f"{nm:>7}{self.theta[i]:14.6f}{self.se[i]:14.6f}{self.se_obs[i]:14.6f}"
            )
        if self.boundary:
            lines.append("warning: optimiser stopped at a parameter boundary")
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


def voigt_mle(
    y,
    maxiter: int = 200,
    gtol: float = 1e-8,
    nodes: int = 400,
    verbose: bool = False,
) -> VoigtMLEResult:
    """Exact maximum likelihood estimation of the Voigt profile from an iid sample.

    Parameters
    ----------
    y : array_like
        The sample.
    maxiter, gtol : int, float
        Newton iteration cap and gradient-norm tolerance (on the *average*
        log-likelihood, so the tolerance is sample-size free).
    nodes : int
        Gauss-Legendre nodes for the expected information behind ``se``.

    Returns
    -------
    VoigtMLEResult
        ``mu, sigma, gamma`` estimates; ``se`` from the expected information,
        ``se_obs`` from the observed information; ``vcov``, ``loglik``,
        ``converged``, ``boundary``, ``iterations``, and ``nfev`` (the number
        of Faddeeva passes over the sample).

    Notes
    -----
    Standard errors are exact and conventional: the MLE is consistent and
    asymptotically normal at rate ``sqrt(n)`` despite the Voigt profile having
    no finite moments (Hansen & Tong 2026, Thm. 4).

    The optimiser does not raise on valid data.  If the likelihood pushes a
    width to the boundary of the clamped parameter space -- for instance
    ``gamma -> 0`` when the Cauchy component is weakly identified in small
    samples -- estimation stops there, ``boundary`` is set, and any standard
    error that cannot be computed is returned as ``nan``.
    """
    y = np.asarray(y, dtype=np.float64).ravel()
    n = y.size
    if n < 3:
        raise ValueError("need at least 3 observations")

    mu0, sigma0, gamma0 = _startvalues(y)
    # keep the log-widths in a sane range relative to the data scale; the true
    # boundary cases sigma -> 0 / gamma -> 0 lie outside the regular family
    s0 = max(sigma0, gamma0)
    lb, ub = np.log(1e-8 * s0), np.log(1e8 * s0)

    eta = np.array([mu0, np.log(sigma0), np.log(gamma0)], dtype=np.float64)
    ll, cache = loglik_only(y, eta[0], np.exp(eta[1]), np.exp(eta[2]))
    ll /= n
    nfev = 1

    converged = False
    it = 0
    eye = np.eye(3)

    while it < maxiter:
        it += 1
        mu, sigma, gamma = eta[0], np.exp(eta[1]), np.exp(eta[2])

        _, g_raw, H_raw, _ = loglik_grad_hess(
            y, mu, sigma, gamma, need_hess=True, need_ll=False, _kl_cache=cache
        )
        g_raw = g_raw / n
        H_raw = H_raw / n
        if not (np.all(np.isfinite(g_raw)) and np.all(np.isfinite(H_raw))):
            break

        # chain rule to eta = (mu, log sigma, log gamma)
        D = np.array([1.0, sigma, gamma])
        g = D * g_raw
        H = (D[:, None] * H_raw * D[None, :]) + np.diag([0.0, sigma * g_raw[1], gamma * g_raw[2]])
        H = 0.5 * (H + H.T)

        gnorm = float(np.linalg.norm(g))
        if gnorm < gtol:
            converged = True
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
            break

        # Armijo backtracking, log-widths clamped
        t = 1.0
        improved = False
        armijo = 1e-4 * float(g @ delta)
        for _ in range(40):
            eta_new = eta + t * delta
            eta_new[1] = min(max(eta_new[1], lb), ub)
            eta_new[2] = min(max(eta_new[2], lb), ub)
            ll_new, cache_new = loglik_only(
                y, eta_new[0], np.exp(eta_new[1]), np.exp(eta_new[2])
            )
            nfev += 1
            ll_new /= n
            if np.isfinite(ll_new) and ll_new >= ll + t * armijo:
                eta, ll, cache = eta_new, ll_new, cache_new
                improved = True
                break
            t *= 0.5

        if not improved:
            # no ascent step: at a (near-)boundary or a numerical optimum
            if gnorm < 1e-4:
                converged = True
            break

        if verbose:
            print(f"iter {it:3d}  loglik {n * ll:.8f}  |g| {gnorm:.3e}")

    mu_hat, sigma_hat, gamma_hat = eta[0], float(np.exp(eta[1])), float(np.exp(eta[2]))
    boundary = bool(eta[1] <= lb + 1e-10 or eta[2] <= lb + 1e-10)

    se = np.full(3, np.nan)
    vcov = np.full((3, 3), np.nan)
    try:
        info = voigt_fisher(mu_hat, sigma_hat, gamma_hat, nodes=nodes)
        vcov = np.linalg.inv(info) / n
        se = np.sqrt(np.maximum(np.diag(vcov), 0.0))
    except (np.linalg.LinAlgError, ValueError):
        pass

    se_obs = np.full(3, np.nan)
    try:
        _, _, H_sum, _ = loglik_grad_hess(
            y, mu_hat, sigma_hat, gamma_hat, need_hess=True, need_ll=False
        )
        se_obs = np.sqrt(np.maximum(np.diag(np.linalg.inv(-H_sum)), 0.0))
    except (np.linalg.LinAlgError, ValueError):
        pass

    return VoigtMLEResult(
        mu=float(mu_hat),
        sigma=sigma_hat,
        gamma=gamma_hat,
        se=se,
        se_obs=se_obs,
        vcov=vcov,
        loglik=float(n * ll),
        converged=converged,
        boundary=boundary,
        iterations=it,
        nfev=nfev,
    )
