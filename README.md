# voigtinference

Exact likelihood calculus and conditional attribution for the Voigt profile,
from a single Faddeeva evaluation per observation: density, score, Hessian,
Fisher information, maximum likelihood estimation, and the conditional
moments of the Gaussian component (deconvolution) — with numerically robust
far-tail behavior.

Two implementations with a common interface, cross-validated to within
`1e-12` on the distributed validation grid (in the recorded reference
environment every difference was in fact zero):

- **`voigtinference/`** — Python (NumPy + SciPy only). The reference and
  archived implementation.
- **`VoigtInference.jl/`** — Julia (SpecialFunctions.jl only). The companion
  implementation, in which the software was developed.

Companion papers:

> P. R. Hansen and C. Tong, *voigtinference: Exact likelihood calculus and
> conditional attribution for the Voigt profile* (software note; prepared for
> Computer Physics Communications).
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
| `voigtinference/tests/` | the pytest suite: 60-digit accuracy references, likelihood identities, input contracts, boundary diagnostics, closed-family LR, MLE recovery, lmfit-Jacobian integration |
| `voigtinference/examples/` | demo, MINUIT-style fit, **analytic Jacobians for lmfit's VoigtModel** |
| `voigtinference/bench/` | two-language benchmark + `1e-12` cross-check (`run_bench.sh`), far-tail validation table (`tailtable.py`) |
| `VoigtInference.jl/src/` | the Julia package |
| `VoigtInference.jl/test/` | the Julia suite (finite differences, Tweedie identities, information equality, boundaries, closed-family LR, MLE recovery) |
| `VoigtInference.jl/examples/` | demo, Monte Carlo study + boundary-LR calibration, figures, Raman-spectrum fit, 256/512-bit tail validation |

The Raman example uses the red-ochre spectrum distributed with the CRAN
`voigt` package (GPL-2, Cannas & Piras; data of Pisu et al., Spectrochim.
Acta A 329 (2025) 125581), which is **not** redistributed here;
`VoigtInference.jl/examples/get_raman.R` fetches the pinned upstream
version (with CRAN-Archive fallback), verifies its checksum, and converts
it (a one-time step that requires R).

## Reproducing the paper's numbers

All scripts are deterministic (fixed integer seeds; thread-count
invariant). From the repository root:

```sh
python -m pip install -e "voigtinference[test]" && python -m pytest voigtinference/tests -q
julia --project=VoigtInference.jl -e 'using Pkg; Pkg.instantiate(); Pkg.test()'
julia --project=VoigtInference.jl VoigtInference.jl/examples/certify.jl          # validation grid
bash voigtinference/bench/run_bench.sh                                           # timings + cross-check
python voigtinference/bench/tailtable.py                                         # far-tail table
REPS=5000 CALIB_B=9999 julia -t auto --project=VoigtInference.jl VoigtInference.jl/examples/montecarlo.jl   # Monte Carlo + LR calibration + validation
Rscript VoigtInference.jl/examples/get_raman.R                                   # fetch Raman data (one-time)
julia --project=VoigtInference.jl/examples -e 'using Pkg; Pkg.develop(path="VoigtInference.jl"); Pkg.instantiate()'   # plotting project (one-time)
julia --project=VoigtInference.jl/examples VoigtInference.jl/examples/figures.jl # figure PDFs
julia --project=VoigtInference.jl/examples VoigtInference.jl/examples/raman.jl   # Raman fit + figure
```

The computational examples (certify, Monte Carlo, benchmark) run in the
package project; only the figure scripts need the plotting project in
`VoigtInference.jl/examples/`. Frozen outputs behind the published tables
are in `paper-results/` with their own README.

## Numerical robustness

The exact score/Hessian formulas suffer catastrophic floating-point
cancellation deep in the Lorentzian tail and, at large `gamma/sigma`, even
at the line center. Both implementations dispatch to third-order
Cauchy-limit expansions wherever `r = sigma^2/((y-mu)^2 + gamma^2)` is
small (`r < 1e-4` for the score and conditional moments, `r < 5e-4` for the
Hessian). The branches are derivatives of one truncated log density, so the
likelihood identities `E[s] = 0` and `E[ss'] = -E[H]` survive the dispatch;
the implementation is validated against high-precision references over
`gamma/sigma` in `[1e-8, 1e8]` (worst cases: score `1.4e-10`, Hessian `6.3e-7`),
while the naive formulas reach relative errors of order `1e16` on the same
grid (`bench/tailtable.py`, `examples/certify.jl`).

The two implementations match association order, not just algebra; the
cross-check (`bench/crosscheck.py` + `bench/crosscheck.jl`) requires
agreement within `1e-12` on the distributed grid and fails the build
otherwise (in the authors' recorded environment every evaluation-level
difference is exactly zero). It is the alarm that should fire if either
side is "tidied."

## License

MIT (both packages).
