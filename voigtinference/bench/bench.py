"""Python side of the Julia-vs-Python benchmark.

Measures, in five layers:

  1. the Faddeeva primitive alone -- the floor, and not a language difference
     at all but a difference between two special-function implementations;
  2. per-observation cost of the density, the analytic score and Hessian, the
     fused workhorse, and their finite-difference counterparts, each reported
     as a *ratio to the density* (the ratio is the language-independent
     quantity and the one the paper claims);
  3. end-to-end MLE at three sample sizes, with iteration and Faddeeva-pass
     counts so that the comparison is like-for-like;
  4. the Fisher-information quadrature;
  5. fit throughput, which is what decides whether a large Monte Carlo is
     practical.

Run `python bench/gendata.py` first so both languages read identical samples.

    OMP_NUM_THREADS=1 python bench/bench.py -o bench/results_python.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import platform
import sys
import time

import numpy as np
import scipy
from scipy.special import wofz

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from voigtinference import (  # noqa: E402
    loglik_grad_hess,
    voigt_fisher,
    voigt_hessian,
    voigt_logpdf,
    voigt_mle,
    voigt_score,
)

HERE = pathlib.Path(__file__).resolve().parent
DATA = HERE / "data"

TARGET_SECONDS = 0.25
MIN_REPEATS = 5


def timeit(fn, target=TARGET_SECONDS, repeats=MIN_REPEATS):
    """Best-of-`repeats` wall time for one call of `fn`, after a warm-up call.

    The number of inner iterations is chosen so each timed block runs for about
    `target` seconds.  Returns ``(best, median)`` seconds per call.
    """
    fn()  # warm-up (page faults, lazy imports, any first-call cost)
    n = 1
    while True:
        t0 = time.perf_counter()
        for _ in range(n):
            fn()
        dt = time.perf_counter() - t0
        if dt >= target or n >= 1 << 20:
            break
        n = max(n * 2, int(n * target / max(dt, 1e-9)))
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        for _ in range(n):
            fn()
        times.append((time.perf_counter() - t0) / n)
    return float(min(times)), float(np.median(times))


# ------------------------------------------------------------------ layers


def fd_score(y, mu, sigma, gamma, h=1e-6):
    """Central-difference score: 6 log-density evaluations."""
    return np.stack(
        (
            (voigt_logpdf(y, mu + h, sigma, gamma) - voigt_logpdf(y, mu - h, sigma, gamma)) / (2 * h),
            (voigt_logpdf(y, mu, sigma + h, gamma) - voigt_logpdf(y, mu, sigma - h, gamma)) / (2 * h),
            (voigt_logpdf(y, mu, sigma, gamma + h) - voigt_logpdf(y, mu, sigma, gamma - h)) / (2 * h),
        ),
        axis=-1,
    )


def fd_hessian(y, mu, sigma, gamma, h=1e-4):
    """Central-difference Hessian: 6 pairs x 4 evaluations = 24."""
    p = np.array([mu, sigma, gamma])
    n = np.asarray(y).size
    H = np.empty((n, 3, 3))
    for j in range(3):
        for k in range(j, 3):
            ej = np.zeros(3)
            ek = np.zeros(3)
            ej[j] = h
            ek[k] = h
            v = (
                voigt_logpdf(y, *(p + ej + ek))
                - voigt_logpdf(y, *(p + ej - ek))
                - voigt_logpdf(y, *(p - ej + ek))
                + voigt_logpdf(y, *(p - ej - ek))
            ) / (4 * h * h)
            H[:, j, k] = H[:, k, j] = v
    return H


def run(data_dir=DATA):
    manifest = json.loads((data_dir / "manifest.json").read_text())
    mu, sigma, gamma = manifest["mu"], manifest["sigma"], manifest["gamma"]

    def load(n):
        return np.fromfile(data_dir / manifest["files"][str(n)], dtype="<f8")

    sizes = sorted(int(k) for k in manifest["files"])
    n_big = max(sizes)
    y = load(n_big)

    out = {
        "language": "python",
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "platform": {
            "system": platform.system(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "n_per_obs": n_big,
    }

    # --- layer 1: the primitive -------------------------------------------
    z = ((y - mu) / (sigma * np.sqrt(2))).astype(np.complex128)
    z.imag = gamma / (sigma * np.sqrt(2))
    best, med = timeit(lambda: wofz(z))
    out["primitive"] = {
        "name": "scipy.special.wofz",
        "ns_per_obs": 1e9 * best / n_big,
        "ns_per_obs_median": 1e9 * med / n_big,
    }

    # --- layer 2: per-observation cost ------------------------------------
    tasks = [
        ("log-density", lambda: voigt_logpdf(y, mu, sigma, gamma)),
        ("analytic score", lambda: voigt_score(y, mu, sigma, gamma)),
        ("analytic Hessian", lambda: voigt_hessian(y, mu, sigma, gamma)),
        ("fused ll+grad+hess", lambda: loglik_grad_hess(y, mu, sigma, gamma)),
        ("FD score (6 evals)", lambda: fd_score(y, mu, sigma, gamma)),
        ("FD Hessian (24 evals)", lambda: fd_hessian(y, mu, sigma, gamma)),
    ]
    per_obs = {}
    base = None
    for name, fn in tasks:
        best, med = timeit(fn)
        if base is None:
            base = best
        per_obs[name] = {
            "ns_per_obs": 1e9 * best / n_big,
            "ns_per_obs_median": 1e9 * med / n_big,
            "ratio_to_density": best / base,
        }
    out["per_obs"] = per_obs

    # --- layer 3: end-to-end MLE ------------------------------------------
    mle = []
    for n in sizes:
        yn = load(n)
        voigt_mle(yn)  # warm-up
        t0 = time.perf_counter()
        r = voigt_mle(yn)
        dt = time.perf_counter() - t0
        mle.append(
            {
                "n": n,
                "seconds": dt,
                "iterations": r.iterations,
                "faddeeva_passes": r.nfev,
                "seconds_per_iteration": dt / max(r.iterations, 1),
                "converged": bool(r.converged),
                "mu": r.mu,
                "sigma": r.sigma,
                "gamma": r.gamma,
                "loglik": r.loglik,
            }
        )
    out["mle"] = mle

    # --- layer 4: Fisher quadrature ---------------------------------------
    best, med = timeit(lambda: voigt_fisher(mu, sigma, gamma, nodes=400))
    out["fisher"] = {"nodes": 400, "seconds": best, "seconds_median": med}

    # --- layer 5: throughput ----------------------------------------------
    y_mid = load(sizes[len(sizes) // 2])
    t0 = time.perf_counter()
    reps = 0
    while time.perf_counter() - t0 < 2.0:
        voigt_mle(y_mid)
        reps += 1
    dt = time.perf_counter() - t0
    out["throughput"] = {
        "n": int(y_mid.size),
        "fits": reps,
        "seconds": dt,
        "fits_per_sec": reps / dt,
    }

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--out", default=str(HERE / "results_python.json"))
    ap.add_argument("--data", default=str(DATA))
    args = ap.parse_args()

    if not pathlib.Path(args.data, "manifest.json").exists():
        raise SystemExit("run `python bench/gendata.py` first")

    res = run(pathlib.Path(args.data))
    pathlib.Path(args.out).write_text(json.dumps(res, indent=2))

    print(f"{'primitive':<24}{res['primitive']['ns_per_obs']:>10.1f} ns/obs")
    print()
    print(f"{'task':<24}{'ns/obs':>10}{'ratio':>9}")
    for k, v in res["per_obs"].items():
        print(f"{k:<24}{v['ns_per_obs']:>10.1f}{v['ratio_to_density']:>9.2f}")
    print()
    for m in res["mle"]:
        print(
            f"MLE n = {m['n']:>7}: {m['seconds']:7.4f} s   "
            f"{m['iterations']:2d} iterations, {m['faddeeva_passes']:2d} Faddeeva passes"
        )
    print(f"\nFisher (400 nodes): {res['fisher']['seconds'] * 1e3:.2f} ms")
    print(
        f"throughput at n = {res['throughput']['n']}: "
        f"{res['throughput']['fits_per_sec']:.1f} fits/s"
    )
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
