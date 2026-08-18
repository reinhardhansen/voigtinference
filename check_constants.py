#!/usr/bin/env python3
"""Guard the far-tail switch constants across their four duplication sites.

The Cauchy-limit branch thresholds r_s (score/moments) and r_h (Hessian) are
certified by VoigtInference.jl/examples/certify.jl and must be numerically
identical in:

    1. VoigtInference.jl/src/VoigtInference.jl   (_far_tail / _far_tail_hess)
    2. voigtinference/src/voigtinference/core.py (_R_SCORE / _R_HESS)
    3. voigtinference/bench/bench.jl             (fused benchmark kernel)
    4. VoigtInference.jl/examples/certify.jl     (ypoints switch thresholds)

Run from the repository root (exit code 1 on any mismatch):

    python check_constants.py
"""
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent
if not (ROOT / "voigtinference").is_dir():        # staged in repo/ in the
    ROOT = pathlib.Path.cwd()                     # Drive tree; run from CPC/

R_S = "5.0e-7"
R_H = "6.25e-5"


def numbers(pattern, path):
    text = (ROOT / path).read_text()
    out = []
    for m in re.finditer(pattern, text):
        out.extend(g for g in m.groups() if g is not None)
    return out


def close(a, b):
    return abs(float(a) - float(b)) <= 1e-20 * max(abs(float(a)), 1.0)


def main():
    checks = [
        ("VoigtInference.jl/src/VoigtInference.jl",
         r"_far_tail\(ỹ, σ, γ\)\s*=\s*σ\^2 <\s*([0-9.e+-]+)", [R_S]),
        ("VoigtInference.jl/src/VoigtInference.jl",
         r"_far_tail_hess\(ỹ, σ, γ\)\s*=\s*σ\^2 <\s*([0-9.e+-]+)", [R_H]),
        ("voigtinference/src/voigtinference/core.py",
         r"_R_SCORE\s*=\s*([0-9.e+-]+)", [R_S]),
        ("voigtinference/src/voigtinference/core.py",
         r"_R_HESS\s*=\s*([0-9.e+-]+)", [R_H]),
        ("voigtinference/bench/bench.jl",
         r"r_s\s*=\s*([0-9.e+-]+)", [R_S]),
        ("voigtinference/bench/bench.jl",
         r"r_h\s*=\s*([0-9.e+-]+)", [R_H]),
        ("VoigtInference.jl/examples/certify.jl",
         r"ypoints[\s\S]{0,200}?\(([0-9.e+-]+),\s*([0-9.e+-]+)\)", [R_S, R_H]),
    ]
    ok = True
    for path, pattern, expected in checks:
        found = numbers(pattern, path)
        if not found:
            print(f"MISSING  {path}: pattern {pattern!r} not found")
            ok = False
            continue
        got = found[0] if len(expected) == 1 else found[: len(expected)]
        got = [got] if isinstance(got, str) else list(got)
        for g, e in zip(got, expected):
            status = "ok " if close(g, e) else "FAIL"
            if status == "FAIL":
                ok = False
            print(f"{status}  {path}: {g}  (expected {e})")
    print("PASS: switch constants agree at all sites." if ok
          else "FAIL: switch constants disagree.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
