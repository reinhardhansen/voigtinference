#!/usr/bin/env bash
#
# Run the full Python-vs-Julia comparison on this machine.
#
#   bash bench/run_bench.sh              # timing + cross-check
#   bash bench/run_bench.sh --cross      # numerical cross-check only
#   bash bench/run_bench.sh --time       # timing only
#
# Override the interpreters or the Julia package location if needed:
#   PYTHON=python3.12 JULIA=/usr/local/bin/julia JL_PKG=/path/to/VoigtInference.jl \
#       bash bench/run_bench.sh

set -uo pipefail

PKG="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BENCH="$PKG/bench"
PY="${PYTHON:-python3}"
JULIA="${JULIA:-julia}"
JL_PKG="${JL_PKG:-$PKG/../VoigtInference.jl}"

DO_TIME=1
DO_CROSS=1
case "${1:-}" in
    --time)  DO_CROSS=0 ;;
    --cross) DO_TIME=0 ;;
    "")      ;;
    *)       echo "unknown option: $1" >&2; exit 2 ;;
esac

hr()  { printf '\n\033[1m%s\033[0m\n' "$*"; }
die() { printf '\033[31merror:\033[0m %s\n' "$*" >&2; exit 1; }

hr "Environment"
command -v "$PY"    >/dev/null || die "python not found: $PY"
command -v "$JULIA" >/dev/null || die "julia not found: $JULIA  (set JULIA=/path/to/julia)"
[ -d "$JL_PKG" ]               || die "Julia package not found: $JL_PKG  (set JL_PKG=...)"
JL_PKG="$(cd "$JL_PKG" && pwd)"

"$PY" - <<'EOF' || exit 1
import sys
missing = []
for m in ("numpy", "scipy"):
    try:
        __import__(m)
    except ImportError:
        missing.append(m)
if missing:
    sys.exit(f"error: python is missing {', '.join(missing)} -- pip install {' '.join(missing)}")
import numpy, scipy, platform
print(f"  python {platform.python_version()}  numpy {numpy.__version__}  scipy {scipy.__version__}")
print(f"  machine {platform.machine()}")
EOF

echo "  julia   $("$JULIA" --version | awk '{print $3}')"
echo "  package $PKG"
echo "  julia pkg $JL_PKG"

hr "Resolving the Julia project"
( cd "$JL_PKG" && "$JULIA" --project=. -e 'using Pkg; Pkg.instantiate()' ) \
    || die "Pkg.instantiate failed in $JL_PKG"

hr "Generating shared benchmark data"
"$PY" "$BENCH/gendata.py" || die "gendata.py failed"

if [ "$DO_TIME" = 1 ]; then
    hr "Python timings"
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
        "$PY" "$BENCH/bench.py" -o "$BENCH/results_python.json" || die "bench.py failed"

    hr "Julia timings"
    ( cd "$JL_PKG" && "$JULIA" -t 1 --project=. "$BENCH/bench.jl" \
        -o "$BENCH/results_julia.json" ) || die "bench.jl failed"

    hr "Comparison table"
    "$PY" "$BENCH/compare.py" "$BENCH/results_python.json" "$BENCH/results_julia.json" \
        -o "$BENCH/comparison.md" || die "compare.py failed"
fi

if [ "$DO_CROSS" = 1 ]; then
    hr "Python reference values"
    "$PY" "$BENCH/crosscheck.py" || die "crosscheck.py failed"

    hr "Julia cross-check"
    ( cd "$JL_PKG" && "$JULIA" -t 1 --project=. "$BENCH/crosscheck.jl" )
    CROSS_STATUS=$?
fi

hr "Done"
[ "$DO_TIME" = 1 ]  && echo "  table:     $BENCH/comparison.md"
[ "$DO_CROSS" = 1 ] && {
    if [ "${CROSS_STATUS:-1}" -eq 0 ]; then
        echo "  crosscheck: PASS"
    else
        echo "  crosscheck: FAIL -- see the table above for which quantity and at what y"
        echo "              (pdf/logpdf at the most extreme grid points would be a"
        echo "               primitive-accuracy difference, not a formula disagreement;"
        echo "               'const TOL' at the top of bench/crosscheck.jl is the knob)"
    fi
}
# propagate the cross-check result: CI and reviewer scripts must see failures
exit "${CROSS_STATUS:-0}"
