# Voigt inference: Python vs Julia

- **Python**: python 3.14.7, numpy 2.5.2, scipy 1.18.0 on arm64 (arm)
- **Julia**: julia 1.12.7 on arm64-apple-darwin25.5.0 (apple-m1)

## 1. The Faddeeva primitive

This is the floor. It is not a language difference but a difference between two special-function implementations; everything below is downstream of it.

| implementation | ns/obs |
| --- | ---: |
| Python — `scipy.special.wofz` | 71.0 |
| Julia — `SpecialFunctions.erfcx` | 84.8 |

Julia / Python = **1.19x**

## 2. Per-observation cost, and the ratio to the density

The **ratio** column is the language-independent quantity and the one the paper claims: analytic derivatives cost almost nothing over the density, while finite differences cost multiples of it.

| task | Python ns/obs | Python ratio | Julia ns/obs | Julia ratio | Julia/Python |
| --- | ---: | ---: | ---: | ---: | ---: |
| log-density | 77.1 | 1.00x | 93.7 | 1.00x | 1.22x |
| analytic score | 81.5 | 1.06x | 98.3 | 1.05x | 1.21x |
| analytic Hessian | 96.0 | 1.25x | 169.7 | 1.81x | 1.77x |
| fused ll+grad+hess | 95.5 | 1.24x | 96.0 | 1.03x | 1.01x |
| FD score (6 evals) | 464.8 | 6.03x | 539.0 | 5.75x | 1.16x |
| FD Hessian (24 evals) | 1866.5 | 24.22x | 2038.5 | 21.76x | 1.09x |
| ll+score+hess (3 evals) | -- | --x | 352.7 | 3.76x | --x |

## 3. End-to-end maximum likelihood

Same data, same starting values, same tolerance. Compare the iteration counts before comparing the times: if they differ, the two implementations are not doing the same amount of work.

| n | Python s | its | passes | Julia s | its | Julia/Python |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1000 | 0.0024 | 5 | 6 | 0.0013 | 5 | 0.55x |
| 10000 | 0.0100 | 6 | 7 | 0.0143 | 6 | 1.43x |
| 100000 | 0.0755 | 6 | 7 | 0.1437 | 6 | 1.90x |

`passes` counts Faddeeva evaluations over the whole sample. The Python implementation fuses the gradient and Hessian into one pass and reuses the line search's evaluation at the accepted step, so an accepted Newton iteration costs one pass; the shipped Julia package evaluates `w(z)` separately for the score and the Hessian.

### Do the two agree?

| n | max rel. diff in (mu, sigma, gamma) | rel. diff in loglik |
| ---: | ---: | ---: |
| 1000 | 3.87e-14 | 1.16e-16 |
| 10000 | 5.20e-14 | 2.23e-15 |
| 100000 | 4.77e-14 | 5.33e-15 |

## 4. Fisher quadrature and fit throughput

| quantity | Python | Julia |
| --- | ---: | ---: |
| Fisher information, 400 nodes (ms) | 0.10 | 0.07 |
| fits/s at n = 10000 | 108.2 | 69.7 |
