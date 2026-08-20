# Voigt inference: Python vs Julia

- **Python**: python 3.14.7, numpy 2.5.2, scipy 1.18.0 on arm64 (arm)
- **Julia**: julia 1.12.7, SpecialFunctions 2.9.0 on arm64-apple-darwin25.5.0 (apple-m1)

## 1. The Faddeeva primitive

This is the floor. It is not a language difference but a difference between two special-function implementations; everything below is downstream of it.

| implementation | ns/obs |
| --- | ---: |
| Python — `scipy.special.wofz` | 71.9 |
| Julia — `SpecialFunctions.erfcx` | 85.1 |

Julia / Python = **1.18x**

## 2. Per-observation cost, and the ratio to the density

The **ratio** column is the language-independent quantity and the one the paper claims: analytic derivatives cost almost nothing over the density, while finite differences cost multiples of it.

| task | Python ns/obs | Python ratio | Julia ns/obs | Julia ratio | Julia/Python |
| --- | ---: | ---: | ---: | ---: | ---: |
| log-density | 78.4 | 1.00x | 97.0 | 1.00x | 1.24x |
| analytic score | 83.1 | 1.06x | 98.3 | 1.01x | 1.18x |
| analytic Hessian | 97.6 | 1.25x | 171.5 | 1.77x | 1.76x |
| fused ll+grad+hess | 97.0 | 1.24x | 97.2 | 1.00x | 1.00x |
| FD score (6 evals) | 470.5 | 6.00x | 558.6 | 5.76x | 1.19x |
| FD Hessian (24 evals) | 1893.2 | 24.15x | 2127.8 | 21.94x | 1.12x |
| ll+score+hess (3 evals) | -- | --x | 358.3 | 3.69x | --x |

## 3. End-to-end maximum likelihood

Same data, same starting values, same tolerance. Compare the iteration counts before comparing the times: if they differ, the two implementations are not doing the same amount of work.

| n | Python s | its | passes | Julia s | its | Julia/Python |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1000 | 0.0023 | 5 | 6 | 0.0014 | 5 | 0.59x |
| 10000 | 0.0096 | 6 | 7 | 0.0145 | 6 | 1.51x |
| 100000 | 0.0791 | 6 | 7 | 0.1458 | 6 | 1.84x |

`passes` counts Faddeeva evaluations over the whole sample. The Python implementation fuses the gradient and Hessian into one pass and reuses the line search's evaluation at the accepted step, so an accepted Newton iteration costs one pass there and two in Julia (the accepted trial's log-likelihood plus the fused gradient/Hessian pass); both use the fused per-point kernel.

### Do the two agree?

| n | max rel. diff in (mu, sigma, gamma) | rel. diff in loglik |
| ---: | ---: | ---: |
| 1000 | 3.00e-15 | 5.82e-16 |
| 10000 | 3.71e-14 | 2.41e-15 |
| 100000 | 4.10e-14 | 5.18e-15 |

## 4. Fisher quadrature and fit throughput

| quantity | Python | Julia |
| --- | ---: | ---: |
| Fisher information, 400 nodes (ms) | 0.10 | 0.07 |
| fits/s at n = 10000 | 103.2 | 67.8 |
