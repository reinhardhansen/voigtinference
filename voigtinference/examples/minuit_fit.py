"""Unbinned Breit-Wigner (x) Gaussian resonance fit with analytic gradients.

The event-level mass density of a Breit-Wigner resonance of mass M and width
Gamma, smeared by Gaussian detector resolution sigma, is the Voigt profile with
mu = M and gamma = Gamma/2.  This is the "Voigtian" of RooFit.

The point of this example is the `grad=` argument: MINUIT is handed the exact
analytic score, so no finite differences are taken anywhere in the fit.

    pip install iminuit
    python examples/minuit_fit.py
"""

import numpy as np

from voigtinference import loglik_grad_hess, rand_voigt, voigt_mle

try:
    from iminuit import Minuit
except ImportError:  # pragma: no cover
    Minuit = None

M_TRUE, GAMMA_TRUE, SIGMA_TRUE = 91.19, 2.50, 1.20  # a Z-like resonance, GeV
N = 20_000

mass = rand_voigt(N, M_TRUE, SIGMA_TRUE, GAMMA_TRUE / 2.0, rng=7)


def nll(M, sigma, Gamma):
    ll, _, _, _ = loglik_grad_hess(mass, M, sigma, Gamma / 2.0, need_hess=False)
    return -ll


def nll_grad(M, sigma, Gamma):
    _, g, _, _ = loglik_grad_hess(
        mass, M, sigma, Gamma / 2.0, need_hess=False, need_ll=False
    )
    # chain rule for the Gamma -> gamma = Gamma/2 reparameterisation
    return np.array([-g[0], -g[1], -0.5 * g[2]])


if __name__ == "__main__":
    print(f"Unbinned resonance fit, N = {N} events")
    print(f"truth: M = {M_TRUE}, Gamma = {GAMMA_TRUE}, sigma = {SIGMA_TRUE}\n")

    if Minuit is None:
        print("iminuit is not installed; showing the built-in Newton fit only.\n")
    else:
        m = Minuit(nll, M=90.0, sigma=1.0, Gamma=2.0, grad=nll_grad)
        m.errordef = Minuit.LIKELIHOOD
        m.limits["sigma"] = (1e-6, None)
        m.limits["Gamma"] = (1e-6, None)
        m.migrad()
        m.hesse()
        print("iminuit (MIGRAD with analytic gradient):")
        for name in ("M", "Gamma", "sigma"):
            print(f"  {name:>6} = {m.values[name]:9.4f} +/- {m.errors[name]:.4f}")
        print(f"  function calls: {m.fmin.nfcn}, gradient calls: {m.fmin.ngrad}\n")

    r = voigt_mle(mass)
    print("voigt_mle (analytic Newton, expected-information errors):")
    print(f"  {'M':>6} = {r.mu:9.4f} +/- {r.se[0]:.4f}")
    print(f"  {'Gamma':>6} = {2 * r.gamma:9.4f} +/- {2 * r.se[2]:.4f}")
    print(f"  {'sigma':>6} = {r.sigma:9.4f} +/- {r.se[1]:.4f}")
    print(f"  Faddeeva passes: {r.nfev}")
