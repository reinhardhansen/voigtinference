# VoigtInference.jl

Exact likelihood inference for the Voigt profile (Gauss–Cauchy convolution) from a
single Faddeeva-function evaluation per data point.

Companion software to:

> P. R. Hansen and C. Tong, *Exact likelihood inference and robust filtering for
> Gauss–Cauchy convolution models*, arXiv:2605.01665.

## Model and conventions

`Y = μ + Z + X` with `Z ~ N(0, σ²)` and `X ~ Cauchy(0, γ)` (Lorentzian HWHM `γ`),
so `Y ~ 𝒱(μ, σ, γ)` with density `f(y) = K(x,a)/(σ√(2π))`, where `K = Re w(z)` and
`L = Im w(z)` are the absorption and dispersion parts of the Faddeeva function
`w(z)`, at `z = x + ia`, `x = (y−μ)/(σ√2)`, `a = γ/(σ√2)`.

Because `w′(z) = −2z w(z) + 2i/√π`, the score, Hessian, Fisher information, and
the conditional moments of the Gaussian component are all algebraic in `(K, L)`.
Within this package's model — the normalized, non-relativistic, constant-width
Voigt distribution — no numerical convolution, finite differences, automatic
differentiation, or pseudo-Voigt approximations are used.

## Usage

```julia
using VoigtInference, Random

y = rand_voigt(MersenneTwister(1), 5_000, 0.5, 1.0, 0.3)

r = voigt_mle(y)             # safeguarded-Newton likelihood optimizer
r.μ, r.σ, r.γ                # estimates
r.se                         # asymptotic Wald standard errors (expected information)

voigt_pdf(2.0, r.μ, r.σ, r.γ)
voigt_score(2.0, r.μ, r.σ, r.γ)      # ∂ log f / ∂(μ, σ, γ)
voigt_pdf_score(2.0, r.μ, r.σ, r.γ)  # both from ONE Faddeeva evaluation
voigt_hessian(2.0, r.μ, r.σ, r.γ)
voigt_fisher(r.μ, r.σ, r.γ)          # expected information, quadrature

voigt_condmean(2.0, r.μ, r.σ, r.γ)   # E[Z | Y=y] = ỹ − γ L/K  (redescending)
voigt_condvar(2.0, r.μ, r.σ, r.γ)    # V(Z | Y=y)
```

For a Breit–Wigner resonance of mass `M` and width `Γ` under Gaussian resolution
`σ`, the event-level mass density is `𝒱(M, σ, Γ/2)`.

## Contents

- `src/VoigtInference.jl` — the package (single file; depends only on
  SpecialFunctions, Statistics, LinearAlgebra)
- `test/runtests.jl` — finite-difference validation of all formulas, Tweedie
  identities, information-matrix equality, MLE recovery
- `examples/demo.jl` — simulate, fit, and deconvolve

Run tests with `julia --project=. -e 'using Pkg; Pkg.test()'`.

## License

MIT
