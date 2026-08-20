"""Exact likelihood inference for the Voigt profile (Gauss-Cauchy convolution).

The Voigt profile is the convolution of a Gaussian and a Lorentzian.  Its
*evaluation* is a solved problem; this package supplies the *inference*: the
score, Hessian, Fisher information, maximum likelihood estimator, and the
conditional moments of the latent Gaussian component, all in closed form and
all obtained from a single Faddeeva evaluation per data point.

    >>> import numpy as np
    >>> from voigtinference import rand_voigt, voigt_mle
    >>> y = rand_voigt(5000, 0.5, 1.0, 0.3, rng=1)
    >>> r = voigt_mle(y)
    >>> round(r.mu, 3), round(r.sigma, 3), round(r.gamma, 3)  # doctest: +SKIP
    (0.49, 1.004, 0.297)

Reference: P. R. Hansen and C. Tong, "Exact likelihood inference and robust
filtering for Gauss-Cauchy convolution models", arXiv:2605.01665.
"""

from .core import (
    faddeeva,
    loglik_grad_hess,
    loglik_only,
    voigt_condmean,
    voigt_condvar,
    voigt_hessian,
    voigt_loglik,
    voigt_logpdf,
    voigt_pdf,
    voigt_pdf_score,
    voigt_score,
)
from .fisher import voigt_fisher
from .mle import VoigtMLEResult, boundary_lr, voigt_mle
from .simulate import rand_voigt

__version__ = "1.1.0"

__all__ = [
    "faddeeva",
    "voigt_pdf",
    "voigt_pdf_score",
    "voigt_logpdf",
    "voigt_loglik",
    "voigt_score",
    "voigt_hessian",
    "voigt_condmean",
    "voigt_condvar",
    "voigt_fisher",
    "voigt_mle",
    "boundary_lr",
    "VoigtMLEResult",
    "rand_voigt",
    "loglik_grad_hess",
    "loglik_only",
    "__version__",
]
