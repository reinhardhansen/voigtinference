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

MU, SIGMA, GAMMA = 0.3, 1.2, 0.7


def grid():
    scale = np.sqrt(SIGMA**2 + GAMMA**2)
    bulk = np.array([-8.0, -3.0, -1.0, -0.25, 0.0, 0.25, 1.0, 3.0, 8.0, 40.0])
    tail = scale * np.array([3e2, 7e2, 1e3, 1.5e3, 1e4, 1e6, 1e8])
    return np.concatenate([bulk, MU + tail, MU - tail])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--out", default=str(HERE / "reference_python.json"))
    args = ap.parse_args()

    y = grid()
    ref = {
        "theta": {"mu": MU, "sigma": SIGMA, "gamma": GAMMA},
        "y": y.tolist(),
        "pdf": voigt_pdf(y, MU, SIGMA, GAMMA).tolist(),
        "logpdf": voigt_logpdf(y, MU, SIGMA, GAMMA).tolist(),
        "score": voigt_score(y, MU, SIGMA, GAMMA).tolist(),
        "hessian": voigt_hessian(y, MU, SIGMA, GAMMA).tolist(),
        "condmean": voigt_condmean(y, MU, SIGMA, GAMMA).tolist(),
        "condvar": voigt_condvar(y, MU, SIGMA, GAMMA).tolist(),
        "fisher": voigt_fisher(MU, SIGMA, GAMMA, nodes=400).tolist(),
    }

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
        "# theta <mu> <sigma> <gamma>",
        "theta " + " ".join(g(v) for v in (MU, SIGMA, GAMMA)),
        "# fisher <9 entries, row-major>",
        "fisher " + " ".join(g(v) for v in np.ravel(ref["fisher"])),
        "# point <y> <pdf> <logpdf> <s1 s2 s3> <H11 H12 H13 H22 H23 H33> <condmean> <condvar>",
    ]
    H = np.asarray(ref["hessian"])
    for i, yi in enumerate(y):
        vals = [
            yi,
            ref["pdf"][i],
            ref["logpdf"][i],
            *ref["score"][i],
            H[i, 0, 0], H[i, 0, 1], H[i, 0, 2], H[i, 1, 1], H[i, 1, 2], H[i, 2, 2],
            ref["condmean"][i],
            ref["condvar"][i],
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

    print(f"wrote {args.out}")
    print(f"wrote {txt_path}")
    print(f"  {len(y)} evaluation points, "
          f"{'with' if 'mle' in ref else 'without'} MLE fits")


if __name__ == "__main__":
    main()
