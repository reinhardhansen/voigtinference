# Frozen outputs behind the paper's numerical tables and figures

Environment: Apple M1 Max (arm64), macOS; Python 3.14.7 with NumPy 2.5.2 and
SciPy 1.18.0; Julia 1.12.7 with SpecialFunctions 2.9.0. All runs 19 Aug 2026
from the source tree at the corresponding commit.

| file | produces | script |
| --- | --- | --- |
| montecarlo_reps5000_20260819.log | Table "Monte Carlo" + boundary-LR calibration | VoigtInference.jl/examples/montecarlo.jl (REPS=5000, CALIB_B=999, starts=7) |
| certify_20260819.txt | certified worst-case bounds quoted in the software section | VoigtInference.jl/examples/certify.jl |
| certify_tune_20260819.txt | threshold minimax scan behind r_s = 1e-5, r_h = 5e-4 | VoigtInference.jl/examples/certify.jl tune |
| tailtable_20260819.txt | far-tail validation table | voigtinference/bench/tailtable.py |
| demo_20260819.txt | the verbatim session in the software section | voigtinference/examples/demo.py |
| results_python.json / results_julia.json / comparison.md | timing table and cross-language comparison | voigtinference/bench/run_bench.sh |

The Monte Carlo, calibration, and benchmark seeds are fixed integers in the
scripts; reproducibility is deterministic under the recorded environment
(random streams and last-ulp rounding may differ across library versions).
