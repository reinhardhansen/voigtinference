# voigtinference

Exact likelihood inference for the **Voigt profile** (Gauss–Cauchy convolution) from a
single Faddeeva-function evaluation per data point.

Companion software to:

> P. R. Hansen and C. Tong, *Exact likelihood inference and robust filtering for
> Gauss–Cauchy convolution models*, arXiv:2605.01665.

Evaluating the Voigt profile is a solved problem. **Inference** for its parameters has not
kept pace: applied work still relies on pseudo-Voigt approximations, finite-difference
Jacobians, or numerical convolution. None of that is necessary. Because

```
w'(z) = -2 z w(z) + 2i/√π,
```

every derivative of the Voigt log-likelihood is an algebraic function of the absorption and
dispersion parts `K = Re w(z)` and `L = Im w(z)` — obtained from the *same* single complex
evaluation that delivers the profile itself.

## Model and conventions

`Y = μ + Z + X` with `Z ~ N(0, σ²)` and `X ~ Cauchy(0, γ)` (Lorentzian with HWHM `γ`), so

```
f(y) = K(x, a) / (σ√(2π)),    x = (y − μ)/(σ√2),    a = γ/(σ√2).
```

For a Breit–Wigner resonance of mass `M` and width `Γ` observed under Gaussian detector
resolution `σ`, the event-level mass density is the Voigt profile with `μ = M` and
`γ = Γ/2`.

## Install

From the repository root:

```bash
pip install -e voigtinference            # or: pip install ./voigtinference
```

Dependencies: NumPy and SciPy. Nothing else. Optional extras:
`[test]` (pytest, mpmath) for the test suite and
`[examples]` (matplotlib, iminuit, lmfit) for the example scripts.

## Usage

```python
import numpy as np
from voigtinference import (
    rand_voigt, voigt_mle, voigt_pdf, voigt_score, voigt_hessian,
    voigt_fisher, voigt_condmean, voigt_condvar,
)

y = rand_voigt(5_000, mu=0.5, sigma=1.0, gamma=0.3, rng=1)

r = voigt_mle(y)              # safeguarded Newton with analytic derivatives
print(r.summary())
r.mu, r.sigma, r.gamma        # estimates
r.se                          # asymptotic Fisher-information standard errors
r.vcov                        # covariance matrix

voigt_pdf(2.0, r.mu, r.sigma, r.gamma)
voigt_score(2.0, r.mu, r.sigma, r.gamma)     # ∂ log f / ∂(μ, σ, γ)
voigt_hessian(2.0, r.mu, r.sigma, r.gamma)
voigt_fisher(r.mu, r.sigma, r.gamma)         # expected information

voigt_condmean(2.0, r.mu, r.sigma, r.gamma)  # E[Z | Y=y] = ỹ − γ L/K  (redescending)
voigt_condvar(2.0, r.mu, r.sigma, r.gamma)   # V(Z | Y=y)
```

Every function is vectorised: pass an array of `y` and get an array back
(shape `(n, 3)` for the score, `(n, 3, 3)` for the Hessian).

### Plugging analytic gradients into an optimiser

`loglik_grad_hess` returns the log-likelihood, the score sum and the Hessian sum from a
**single** Faddeeva pass over the sample — this is what you want inside MINUIT, `scipy.optimize`,
or any other optimiser that accepts a gradient:

```python
from voigtinference import loglik_grad_hess

def cost(mu, sigma, gamma):
    ll, g, _, _ = loglik_grad_hess(y, mu, sigma, gamma, need_hess=False)
    return -ll

def cost_grad(mu, sigma, gamma):
    _, g, _, _ = loglik_grad_hess(y, mu, sigma, gamma, need_hess=False, need_ll=False)
    return -g
```

See `examples/minuit_fit.py` for a complete `iminuit` fit and
`examples/demo.py` for simulation, estimation and deconvolution.

## Design notes

`scipy.special.wofz` costs roughly 150 ns per element and dominates the elementwise algebra
that follows it. The package is therefore organised around minimising the number of
Faddeeva passes rather than around micro-optimising the arithmetic:

* the score and the Hessian share one evaluation (the Hessian is assembled recursively from
  the score components and `L/K`, with no additional special-function calls);
* `loglik_grad_hess` never materialises `(n, 3)` or `(n, 3, 3)` intermediates — it
  accumulates the gradient and Hessian sums directly;
* the MLE carries the `(K, L)` pair computed by the line search forward to the next
  iterate, so an *accepted* Newton step costs exactly one Faddeeva pass over the sample;
* `voigt_pdf_score` returns the density and the score from one pass — the natural
  primitive for least-squares Jacobians of the lineshape (`df/dtheta = f * s_theta`),
  used by `examples/lmfit_voigt_jacobian.py`.

No automatic differentiation, finite differences, numerical convolution, or pseudo-Voigt
approximations are used anywhere.

### Far-tail branches

The closed-form score and Hessian are algebraically exact but lose floating-point precision
in the tail, where terms that agree to many digits are subtracted. Past the crossover the
package switches to a Cauchy-limit expansion.

**The score and the Hessian need different switches.** The Hessian recursion forms
`smu − ỹ·hmm` — a difference of two quantities of size `2/ỹ` whose value is of size `1/ỹ³` —
so it loses digits far sooner than the score does. Measured against high-precision ground
truth, the crossovers sit at

| quantity | switch | worst-case relative error |
| --- | ---: | ---: |
| score, conditional moments | `\|y − μ\| > 500 √(σ² + γ²)` | 2.0e-05 |
| Hessian | `\|y − μ\| > 40 √(σ² + γ²)` | 8.6e-03 |

Sharing one switch is not a marginal loss: with the score's threshold applied to the
Hessian, the Hessian has **no correct digits at all** from about `100 √(σ² + γ²)` outward
(relative error 1.9 at 100×, 4.0e+07 at 1000×). Individual far-tail Hessian entries are
`O(1/ỹ²)` so this does not disturb a summed Hessian or the standard errors, but it makes
`voigt_hessian` at a single tail point meaningless — and it is invisible unless you check
against something better than double precision.

Both constants were chosen by minimising worst-case relative error over `|ỹ|/scale ∈ [1, 1e6]`
across a spread of `(σ, γ)`. `tests/test_accuracy.py` verifies the resulting bounds, guards
the split against regression, and checks that the reference itself is converged — the
reference needs precision growing with `log|ỹ|` for the same reason the Hessian does, and a
fixed 60 digits silently degrades past `1e5 √(σ² + γ²)` in a way that looks exactly like a
broken expansion. Run the module as a script to print the crossover tables.

## Tests

```bash
pip install -e ".[test]"
pytest
```

The suite covers finite-difference validation of every formula, the Tweedie identities,
the information-matrix equality, MLE recovery and equivariance, vectorised-versus-scalar
parity, and high-precision accuracy of the primitive and of both tail branches.

## License

MIT
