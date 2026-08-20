# Benchmark and cross-check harness

Two things live here:

* **`bench.py` / `bench.jl`** — the timing comparison, layer for layer, on identical data.
* **`crosscheck.py` / `crosscheck.jl`** — numerical agreement between the two
  implementations, evaluation point by evaluation point.

Both halves read the *same* samples, written once by `gendata.py`. That matters: different
samples give different Newton paths, and an end-to-end comparison on different data measures
nothing.

## Running it

Everything must run on **one machine, natively**, in one sitting. Ratios travel across
machines; absolute times do not.

```bash
# 1. shared data (once)
python bench/gendata.py

# 2. Python side
OMP_NUM_THREADS=1 python bench/bench.py -o bench/results_python.json

# 3. Julia side, from the VoigtInference.jl package directory
julia -t 1 --project=. /path/to/bench/bench.jl \
      --data /path/to/bench/data -o /path/to/bench/results_julia.json

# 4. render the table
python bench/compare.py bench/results_python.json bench/results_julia.json \
       -o bench/comparison.md
```

Numerical agreement:

```bash
python bench/crosscheck.py                    # writes reference_python.{json,txt}
julia -t 1 --project=. /path/to/bench/crosscheck.jl /path/to/bench/reference_python.txt
```

`crosscheck.jl` exits non-zero if any evaluation-level quantity disagrees by more than
`1e-12` relative, and prints the worst offender and the `y` at which it occurred.

## What is being measured, and why in this shape

**Layer 1, the primitive.** `scipy.special.wofz` versus `SpecialFunctions.erfcx`. This is the
floor, and it is not a language difference — it is a difference between two special-function
algorithms. Most of any gap in the layers below traces back here, and saying so is the honest
reading.

**Layer 2, per-observation cost.** Reported both in ns/obs and as a **ratio to the density**.
The ratio is the language-independent quantity, and it is what the paper actually claims:
analytic derivatives cost almost nothing over the density, finite differences cost multiples
of it. If the ratio comes out the same in both languages, the claim is stronger than a
single-language table can make it.

**Layer 3, end-to-end MLE.** Same data, same starting values, same tolerance. *Compare the
iteration counts before comparing the times.* If they differ, the two implementations are not
doing the same amount of work and the wall-clock ratio is not interpretable. The Python side
also reports `faddeeva_passes`, which is the quantity that actually drives the cost.

**Layer 4, the Fisher quadrature**, and **layer 5, fit throughput** — the latter being the
number that decides whether a large Monte Carlo is an afternoon or a week.

## Fairness notes

Julia is scalar-loop-natural; NumPy is array-natural. A literal transliteration — a Python
loop calling `wofz` per observation — would be dramatically slower and would measure
programming style, not the problem. Each side here is written the way a competent
practitioner in that language would write it: scalar loops with `@inbounds` in Julia, fully
vectorised over the sample in NumPy.

The timing protocol is deliberately identical on both sides (warm-up call, inner-loop count
chosen for a ~0.25 s block, best of five blocks) rather than using `BenchmarkTools` on one
side and `timeit` on the other. The Julia warm-up call also absorbs JIT compilation, which is
the classic way to accidentally make Julia look slow.

Run single-threaded on both sides (`OMP_NUM_THREADS=1`, `julia -t 1`) so BLAS threading on
the 3x3 solves does not leak into the comparison, and record every version — special-function
performance moves between releases.

## What the fused rows measure

Both shipped optimisers now use a fused internal kernel: the log-likelihood,
score, and Hessian come from ONE Faddeeva evaluation per observation, and
Python additionally hands the accepted line-search evaluation forward so an
accepted Newton iteration costs a single pass over the sample (Julia
re-evaluates derivatives at the accepted point, so an accepted iteration may
cost two passes). The benchmark times three variants to keep those effects
separate: the stand-alone public routines (three passes, the price of calling
`voigt_logpdf`/`voigt_score`/`voigt_hessian` independently), the fused
one-pass kernel (mirroring what the optimisers run internally), and the full
MLE.
