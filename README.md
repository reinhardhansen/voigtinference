# voigtinference

Exact likelihood calculus for the Voigt profile from a single Faddeeva
evaluation: density, score, Hessian, Fisher information, maximum likelihood
estimation, and the conditional moments of the Gaussian component
(deconvolution) — with numerically robust far-tail behavior.

Two implementations with a common interface, which agree **bit for bit** on
every evaluation-level quantity:

- **`voigtinference/`** — Python (NumPy + SciPy only). The reference and
  archived implementation.
- **`VoigtInference.jl/`** — Julia (SpecialFunctions.jl only). The companion
  implementation, in which the software was developed.

Companion papers:

> P. R. Hansen and C. Tong, *Exact likelihood calculus for the Voigt profile
> from a single Faddeeva evaluation* (software note; submitted to Computer
> Physics Communications).
>
> P. R. Hansen and C. Tong, *Exact likelihood inference and robust filtering
> for Gauss–Cauchy convolution models*, arXiv:2605.01665 (theory).

## Model and conventions

`Y = mu + Z + X` with `Z ~ N(0, sigma^2)` and `X ~ Cauchy(0, gamma)`
(Lorentzian HWHM `gamma`), so `Y` follows the Voigt profile
`f(y) = K(x, a) / (sigma sqrt(2 pi))` with `K = Re w(z)`, `L = Im w(z)` the
absorption and dispersion parts of the Faddeeva function at
`z = x + i a`, `x = (y - mu)/(sigma sqrt 2)`, `a = gamma/(sigma sqrt 2)`.
Because `w'(z) = -2 z w(z) + 2i/sqrt(pi)`, the score, Hessian, and conditional
moments are algebraic in `(K, L)`; one Faddeeva evaluation per observation
delivers everything.

## Quick start (Python)

```python
import numpy as np
from voigtinference import rand_voigt, voigt_mle, voigt_condmean

y = rand_voigt(5_000, 0.5, 1.0, 0.3, rng=2026)
r = voigt_mle(y)          # safeguarded Newton, one Faddeeva pass per iteration
print(r.summary())        # estimates with expected/observed-information s.e.
voigt_condmean(2.0, r.mu, r.sigma, r.gamma)   # E[Z | Y = y], redescending
```

Tests: `PYTHONPATH=src python -m pytest tests/ -q` (from `voigtinference/`).

## Quick start (Julia)

```julia
using VoigtInference, Random
y = rand_voigt(MersenneTwister(2026), 5_000, 0.5, 1.0, 0.3)
r = voigt_mle(y)
r.se
```

Tests: `julia --project=. -e 'using Pkg; Pkg.test()'` (from
`VoigtInference.jl/`).

## What is in here

| path | contents |
| --- | --- |
| `voigtinference/src/` | the Python package |
| `voigtinference/tests/` | 53 package tests + lmfit-Jacobian tests |
| `voigtinference/examples/` | demo, MINUIT-style fit, **analytic Jacobians for lmfit's VoigtModel** |
| `voigtinference/bench/` | two-language benchmark + bit-identity cross-check (`run_bench.sh`), far-tail validation table (`tailtable.py`) |
| `VoigtInference.jl/src/` | the Julia package |
| `VoigtInference.jl/test/` | 71 tests (finite differences, Tweedie identities, information equality, MLE recovery) |
| `VoigtInference.jl/examples/` | demo, Monte Carlo study, figures, Raman-spectrum fit, 256/512-bit tail validation |

The Raman example uses the red-ochre spectrum distributed with the CRAN
`voigt` package (GPL-2, Cannas & Piras; data of Pisu et al., Spectrochim.
Acta A 329 (2025) 125581), which is **not** redistributed here; the script
header explains how to fetch and convert it (no R required).

## Numerical robustness

The exact score/Hessian formulas suffer catastrophic floating-point
cancellation deep in the Lorentzian tail. Both implementations switch to
Cauchy-limit expansions at `|y - mu| > 500 sqrt(sigma^2 + gamma^2)` for the
score and conditional moments and at `40 sqrt(sigma^2 + gamma^2)` for the
Hessian (which loses digits an order of magnitude sooner). The switches are
validated against 256-/512-bit references in `bench/tailtable.py`; the naive
formulas reach relative errors of order `1e47` where the packaged ones are at
machine precision.

The two implementations match association order, not just algebra; the
cross-check (`bench/crosscheck.py` + `bench/crosscheck.jl`) asserts exact
bitwise agreement and is the alarm that should fire if either side is
"tidied."

## License

MIT (both packages).
