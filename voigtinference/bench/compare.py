"""Render the Julia-vs-Python comparison table.

    python bench/compare.py bench/results_python.json bench/results_julia.json \
        -o bench/comparison.md

Either file may be omitted, in which case only the available one is shown.
"""

from __future__ import annotations

import argparse
import json
import pathlib


def _fmt(x, spec=".2f"):
    return "--" if x is None else format(x, spec)


def _ratio(a, b):
    """b / a, i.e. how many times slower `b` is than `a`."""
    if a is None or b is None or a == 0:
        return None
    return b / a


def render(py, jl):
    L = []
    have_both = py is not None and jl is not None

    L.append("# Voigt inference: Python vs Julia\n")

    for res, label in ((py, "Python"), (jl, "Julia")):
        if res is None:
            continue
        v = ", ".join(f"{k} {w}" for k, w in res["versions"].items())
        p = res["platform"]
        L.append(
            f"- **{label}**: {v} on {p.get('machine', '?')} "
            f"({p.get('processor') or p.get('system', '?')})"
        )
    L.append("")

    # ---------------------------------------------------------- primitive
    L.append("## 1. The Faddeeva primitive\n")
    L.append(
        "This is the floor. It is not a language difference but a difference "
        "between two special-function implementations; everything below is "
        "downstream of it.\n"
    )
    L.append("| implementation | ns/obs |")
    L.append("| --- | ---: |")
    for res, label in ((py, "Python"), (jl, "Julia")):
        if res is None:
            continue
        pr = res["primitive"]
        L.append(f"| {label} — `{pr['name']}` | {pr['ns_per_obs']:.1f} |")
    if have_both:
        r = _ratio(py["primitive"]["ns_per_obs"], jl["primitive"]["ns_per_obs"])
        L.append("")
        L.append(f"Julia / Python = **{r:.2f}x**")
    L.append("")

    # ------------------------------------------------------- per-observation
    L.append("## 2. Per-observation cost, and the ratio to the density\n")
    L.append(
        "The **ratio** column is the language-independent quantity and the one "
        "the paper claims: analytic derivatives cost almost nothing over the "
        "density, while finite differences cost multiples of it.\n"
    )
    # union, preserving order, so a row present in only one language still shows
    keys = list(
        dict.fromkeys(
            list((py or {}).get("per_obs", {})) + list((jl or {}).get("per_obs", {}))
        )
    )
    if have_both:
        L.append("| task | Python ns/obs | Python ratio | Julia ns/obs | Julia ratio | Julia/Python |")
        L.append("| --- | ---: | ---: | ---: | ---: | ---: |")
        for k in keys:
            a, b = py["per_obs"].get(k), jl["per_obs"].get(k)
            rr = _ratio(a["ns_per_obs"] if a else None, b["ns_per_obs"] if b else None)
            L.append(
                f"| {k} | {_fmt(a and a['ns_per_obs'], '.1f')} | "
                f"{_fmt(a and a['ratio_to_density'])}x | "
                f"{_fmt(b and b['ns_per_obs'], '.1f')} | "
                f"{_fmt(b and b['ratio_to_density'])}x | {_fmt(rr)}x |"
            )
    else:
        res = py or jl
        L.append("| task | ns/obs | ratio to density |")
        L.append("| --- | ---: | ---: |")
        for k in keys:
            v = res["per_obs"][k]
            L.append(f"| {k} | {v['ns_per_obs']:.1f} | {v['ratio_to_density']:.2f}x |")
    L.append("")

    # ------------------------------------------------------------- the MLE
    L.append("## 3. End-to-end maximum likelihood\n")
    L.append(
        "Same data, same starting values, same tolerance. Compare the iteration "
        "counts before comparing the times: if they differ, the two "
        "implementations are not doing the same amount of work.\n"
    )
    if have_both:
        L.append("| n | Python s | its | passes | Julia s | its | Julia/Python |")
        L.append("| ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        jl_by_n = {m["n"]: m for m in jl["mle"]}
        for m in py["mle"]:
            j = jl_by_n.get(m["n"])
            rr = _ratio(m["seconds"], j and j["seconds"])
            L.append(
                f"| {m['n']} | {m['seconds']:.4f} | {m['iterations']} | "
                f"{m.get('faddeeva_passes', '--')} | "
                f"{_fmt(j and j['seconds'], '.4f')} | {j['iterations'] if j else '--'} | "
                f"{_fmt(rr)}x |"
            )
        L.append("")
        L.append(
            "`passes` counts Faddeeva evaluations over the whole sample. The "
            "Python implementation fuses the gradient and Hessian into one pass "
            "and reuses the line search's evaluation at the accepted step, so an "
            "accepted Newton iteration costs one pass there and two in Julia "
            "(the accepted trial's log-likelihood plus the fused "
            "gradient/Hessian pass); both use the fused per-point kernel."
        )
    else:
        res = py or jl
        L.append("| n | seconds | iterations | passes |")
        L.append("| ---: | ---: | ---: | ---: |")
        for m in res["mle"]:
            L.append(
                f"| {m['n']} | {m['seconds']:.4f} | {m['iterations']} | "
                f"{m.get('faddeeva_passes', '--')} |"
            )
    L.append("")

    # ------------------------------------------- agreement of the estimates
    if have_both:
        L.append("### Do the two agree?\n")
        jl_by_n = {m["n"]: m for m in jl["mle"]}
        L.append("| n | max rel. diff in (mu, sigma, gamma) | rel. diff in loglik |")
        L.append("| ---: | ---: | ---: |")
        for m in py["mle"]:
            j = jl_by_n.get(m["n"])
            if j is None:
                continue
            d = max(
                abs(m[k] - j[k]) / max(abs(j[k]), 1e-300)
                for k in ("mu", "sigma", "gamma")
            )
            dl = abs(m["loglik"] - j["loglik"]) / max(abs(j["loglik"]), 1e-300)
            L.append(f"| {m['n']} | {d:.2e} | {dl:.2e} |")
        L.append("")

    # -------------------------------------------------- quadrature, throughput
    L.append("## 4. Fisher quadrature and fit throughput\n")
    L.append("| quantity | Python | Julia |")
    L.append("| --- | ---: | ---: |")
    L.append(
        f"| Fisher information, 400 nodes (ms) | "
        f"{_fmt(py and py['fisher']['seconds'] * 1e3)} | "
        f"{_fmt(jl and jl['fisher']['seconds'] * 1e3)} |"
    )
    n_tp = (py or jl)["throughput"]["n"]
    L.append(
        f"| fits/s at n = {n_tp} | "
        f"{_fmt(py and py['throughput']['fits_per_sec'], '.1f')} | "
        f"{_fmt(jl and jl['throughput']['fits_per_sec'], '.1f')} |"
    )
    L.append("")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("python_json", nargs="?", default="bench/results_python.json")
    ap.add_argument("julia_json", nargs="?", default="bench/results_julia.json")
    ap.add_argument("-o", "--out", default=None)
    args = ap.parse_args()

    def load(p):
        path = pathlib.Path(p)
        return json.loads(path.read_text()) if path.exists() else None

    py, jl = load(args.python_json), load(args.julia_json)
    if py is None and jl is None:
        raise SystemExit("no results files found; run bench.py and/or bench.jl first")
    if jl is None:
        print("note: no Julia results found, showing Python only\n")

    md = render(py, jl)
    if args.out:
        pathlib.Path(args.out).write_text(md)
        print(f"wrote {args.out}\n")
    print(md)


if __name__ == "__main__":
    main()
