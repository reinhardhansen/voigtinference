#!/bin/bash
# Build the versioned release artifact from a clean, tagged repository tree.
# This is the reviewer-facing release workflow: it verifies the guarded
# constants, runs both test suites and the numerical certification, builds
# the Python distributions, and produces the exact archives to be submitted,
# with SHA-256 checksums.
#
# Run from the repository root (a git checkout, not the working folder):
#
#   bash release.sh v1.0.0
#
set -euo pipefail
TAG="${1:?usage: bash release.sh <tag, e.g. v1.0.0>}"
test -d .git || { echo "error: run from the repository root"; exit 1; }
git rev-parse --verify "$TAG" > /dev/null
git diff --quiet && git diff --cached --quiet || {
    echo "error: working tree not clean"; exit 1; }

echo "== 1/6 constants guard =="
python3 check_constants.py

echo "== 2/6 Python tests =="
python3 -m pip install -q -e "voigtinference[test]"
python3 -m pytest voigtinference/tests -q

echo "== 3/6 Julia tests =="
( cd VoigtInference.jl && julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.test()' )

echo "== 4/6 numerical certification =="
( cd VoigtInference.jl && julia --project=. examples/certify.jl )

echo "== 5/6 Python build (wheel + sdist) =="
python3 -m pip install -q build
python3 -m build voigtinference --outdir dist/

echo "== 6/6 archives from the tag =="
NAME="voigtinference-${TAG#v}"
git archive --format=zip    --prefix="$NAME/" -o "dist/$NAME.zip"    "$TAG"
git archive --format=tar.gz --prefix="$NAME/" -o "dist/$NAME.tar.gz" "$TAG"
( cd dist && shasum -a 256 * > SHA256SUMS && cat SHA256SUMS )

echo
echo "Release artifacts in dist/ — submit dist/$NAME.zip"
