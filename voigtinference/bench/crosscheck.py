"""Write Python reference values for the Julia implementation to check against.

    python bench/crosscheck.py            # writes bench/reference_python.json
    julia --project=. ../bench/crosscheck.jl   # reads it, reports differences

The grid deliberately spans the bulk, the neighbourhood of the far-tail switch,
and several decades beyond it, so that any disagreement in the tail branch shows
up rather than being averaged away.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from voigtinference import (  # noqa: E402
    voigt_condmean,
    voigt_condvar,
    voigt_fisher,
    voigt_hessian,
    voigt_logpdf,
    voigt_mle,
    voigt_pdf,
    voigt_score,
)

HERE = pathlib.Path(__file__).resolve().parent

# The primary case plus the two width-ratio extremes: gamma/sigma ~ 1e4 puts
# the CENTER of the profile in the Cauchy-limit branch (the regime of the
# regime of the extreme-ratio regression), gamma/sigma ~ 1e-4 approaches the Gaussian
# limit.  The Fisher quadrature is compared for the primary case only (at
# extreme width ratios the two node generators' last-ulp differences are
# amplified by the integrand, which is a property of quadrature, not of the
# formulas this file checks).
THETAS = [
    (0.3, 1.2, 0.7, True),
    (0.0, 1.0, 1.0e4, False),
    (0.0, 1.0, 1.0e-4, False),
]


def grid(mu, sigma, gamma):
    scale = np.sqrt(sigma**2 + gamma**2)
    bulk = scale * np.array(
        [-8.0, -3.0, -1.0, -0.25, 0.0, 0.25, 1.0, 3.0, 8.0, 40.0]
    )
    tail = scale * np.array([3e2, 7e2, 1e3, 1.5e3, 1e4, 1e6, 1e8])
    return np.concatenate([mu + bulk, mu + tail, mu - tail])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--out", default=str(HERE / "reference_python.json"))
    args = ap.parse_args()

    cases = []
    for mu, sigma, gamma, with_fisher in THETAS:
        y = grid(mu, sigma, gamma)
        case = {
            "theta": {"mu": mu, "sigma": sigma, "gamma": gamma},
            "y": y.tolist(),
            "pdf": voigt_pdf(y, mu, sigma, gamma).tolist(),
            "logpdf": voigt_logpdf(y, mu, sigma, gamma).tolist(),
            "score": voigt_score(y, mu, sigma, gamma).tolist(),
            "hessian": voigt_hessian(y, mu, sigma, gamma).tolist(),
            "condmean": voigt_condmean(y, mu, sigma, gamma).tolist(),
            "condvar": voigt_condvar(y, mu, sigma, gamma).tolist(),
        }
        if with_fisher:
            case["fisher"] = voigt_fisher(mu, sigma, gamma, nodes=400).tolist()
        cases.append(case)
    ref = {"cases": cases}

    data_dir = HERE / "data"
    if (data_dir / "manifest.json").exists():
        manifest = json.loads((data_dir / "manifest.json").read_text())
        fits = {}
        for key, name in manifest["files"].items():
            yn = np.fromfile(data_dir / name, dtype="<f8")
            r = voigt_mle(yn)
            fits[key] = {
                "mu": r.mu,
                "sigma": r.sigma,
                "gamma": r.gamma,
                "loglik": r.loglik,
                "se": r.se.tolist(),
                "iterations": r.iterations,
            }
        ref["mle"] = fits

    pathlib.Path(args.out).write_text(json.dumps(ref, indent=2))

    # dependency-free twin for the Julia side (no JSON parser in the stdlib);
    # %.17g round-trips float64 exactly
    def g(x):
        return format(float(x), ".17g")

    lines = [
        "# Voigt reference values written by bench/crosscheck.py",
        "# theta <mu> <sigma> <gamma>   starts a case; the fisher and point",
        "# lines that follow belong to it",
        "# fisher <9 entries, row-major>  (primary case only)",
        "# point <y> <pdf> <logpdf> <s1 s2 s3> <H11 H12 H13 H22 H23 H33> <condmean> <condvar>",
    ]
    for case in cases:
        th = case["theta"]
        lines.append(
            "theta " + " ".join(g(th[k]) for k in ("mu", "sigma", "gamma"))
        )
        if "fisher" in case:
            lines.append(
                "fisher " + " ".join(g(v) for v in np.ravel(case["fisher"]))
            )
        H = np.asarray(case["hessian"])
        for i, yi in enumerate(case["y"]):
            vals = [
                yi,
                case["pdf"][i],
                case["logpdf"][i],
                *case["score"][i],
                H[i, 0, 0], H[i, 0, 1], H[i, 0, 2],
                H[i, 1, 1], H[i, 1, 2], H[i, 2, 2],
                case["condmean"][i],
                case["condvar"][i],
            ]
            lines.append("point " + " ".join(g(v) for v in vals))
    if "mle" in ref:
        lines.append("# mle <n> <mu> <sigma> <gamma> <loglik> <iterations>")
        for key, f in sorted(ref["mle"].items(), key=lambda kv: int(kv[0])):
            lines.append(
                "mle " + " ".join(
                    [key] + [g(f[k]) for k in ("mu", "sigma", "gamma", "loglik")]
                    + [str(f["iterations"])]
                )
            )

    txt_path = pathlib.Path(args.out).with_suffix(".txt")
    txt_path.write_text("\n".join(lines) + "\n")

    npts = sum(len(c["y"]) for c in cases)
    print(f"wrote {args.out}")
    print(f"wrote {txt_path}")
    print(f"  {len(cases)} theta cases, {npts} evaluation points, "
          f"{'with' if 'mle' in ref else 'without'} MLE fits")


if __name__ == "__main__":
    main()
