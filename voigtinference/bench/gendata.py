"""Generate the benchmark samples that both implementations read.

Writes raw little-endian float64 arrays so that Python and Julia time the
*same* numbers -- otherwise different samples give different Newton paths and
the end-to-end comparison is meaningless.

    python bench/gendata.py
"""

import json
import pathlib

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
DATA = HERE / "data"

MU, SIGMA, GAMMA = 0.5, 1.0, 0.3
SIZES = (1_000, 10_000, 100_000)
SEED = 20260816


def main():
    DATA.mkdir(exist_ok=True)
    rng = np.random.default_rng(SEED)
    manifest = {
        "mu": MU,
        "sigma": SIGMA,
        "gamma": GAMMA,
        "seed": SEED,
        "dtype": "float64",
        "byte_order": "little",
        "files": {},
    }
    for n in SIZES:
        # exact convolution, drawn here once so both languages see identical data
        y = (
            MU
            + SIGMA * rng.standard_normal(n)
            + GAMMA * np.tan(np.pi * (rng.random(n) - 0.5))
        )
        name = f"y_{n}.f64"
        y.astype("<f8").tofile(DATA / name)
        manifest["files"][str(n)] = name
        print(f"wrote {name}  n = {n}  median = {np.median(y):+.6f}")

    (DATA / "manifest.json").write_text(json.dumps(manifest, indent=2))
    # plain-text twin so the Julia side needs no JSON dependency
    (DATA / "params.txt").write_text(
        f"mu {MU}\nsigma {SIGMA}\ngamma {GAMMA}\nsizes {' '.join(str(n) for n in SIZES)}\n"
    )
    print("wrote manifest.json and params.txt")


if __name__ == "__main__":
    main()
