# Frozen outputs behind the paper's numerical tables and figures

Environment: Apple M1 Max (arm64), macOS; Python 3.14.7 with NumPy 2.5.2 and
SciPy 1.18.0; Julia 1.12.7 with SpecialFunctions 2.9.0. All runs 19-20 Aug
2026 from the source tree at the corresponding commit. The Monte Carlo ran
with `julia -t 8`; every replication seeds its own generator (seedbase
18500902), so the output is identical for any thread count.

| file | produces | script |
| --- | --- | --- |
| montecarlo_reps5000_calib9999_20260820.txt | Monte Carlo table + closed-family boundary-LR calibration (atom/deficit/termination diagnostics) + independent size validation | VoigtInference.jl/examples/montecarlo.jl (REPS=5000, CALIB_B=9999, starts=7) |
| certify_20260819.txt | validated worst-case bounds quoted in the software section | VoigtInference.jl/examples/certify.jl |
| certify_tune_20260819.txt | threshold scan behind r_s = 1e-4, r_h = 5e-4 | VoigtInference.jl/examples/certify.jl tune |
| tailtable_20260819.txt | far-tail validation table | voigtinference/bench/tailtable.py |
| demo_20260819.txt | the verbatim session in the software section | voigtinference/examples/demo.py |
| results_python.json / results_julia.json / comparison.md | timing table and cross-language comparison | voigtinference/bench/run_bench.sh |

The Monte Carlo, calibration, validation, and benchmark seeds are fixed
integers in the scripts; reproducibility is deterministic under the recorded
environment (random streams and last-ulp rounding may differ across library
versions). The frozen filenames use the .txt extension so no ignore pattern
can exclude them from the repository.
