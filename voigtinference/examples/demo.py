"""Simulate, fit, and deconvolve.

    python examples/demo.py
"""

import numpy as np

from voigtinference import (
    rand_voigt,
    voigt_condmean,
    voigt_condvar,
    voigt_fisher,
    voigt_mle,
)

MU, SIGMA, GAMMA = 0.5, 1.0, 0.3
N = 5_000

y = rand_voigt(N, MU, SIGMA, GAMMA, rng=2026)

print(f"Voigt sample: n = {N}, true (mu, sigma, gamma) = ({MU}, {SIGMA}, {GAMMA})\n")

r = voigt_mle(y)
print(r.summary())
print(f"\nFaddeeva passes over the sample: {r.nfev} "
      f"({r.iterations} Newton iterations)")

t = (r.theta - np.array([MU, SIGMA, GAMMA])) / r.se
print("t-statistics against the truth: " + "  ".join(f"{v:+.2f}" for v in t))

print("\nExpected Fisher information at the estimate:")
print(voigt_fisher(r.mu, r.sigma, r.gamma))

# ---------------------------------------------------------------- deconvolution
print("\nDeconvolution: E[Z | Y = y], the Gaussian part of an observation.")
print("The map is redescending -- moderate deviations are attributed to the")
print("Gaussian (Doppler/resolution) component, extreme ones to the Lorentzian tail.\n")

print(f"{'y - mu':>10}{'E[Z|y]':>12}{'E[X|y]':>12}{'V(Z|y)':>12}")
for d in (0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 100.0):
    yy = r.mu + d
    ez = voigt_condmean(yy, r.mu, r.sigma, r.gamma)
    vz = voigt_condvar(yy, r.mu, r.sigma, r.gamma)
    print(f"{d:10.2f}{ez:12.4f}{d - ez:12.4f}{vz:12.4f}")

peak = float(np.abs(voigt_condmean(r.mu + np.linspace(0, 10, 20_001), r.mu, r.sigma, r.gamma)).max())
print(f"\nMaximum attributed Gaussian deviation: {peak:.4f} "
      f"(= {peak / r.sigma:.2f} estimated sigma)")
