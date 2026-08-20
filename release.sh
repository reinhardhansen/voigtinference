#!/usr/bin/env bash
# Tag-exact release build.
#
# Every check, test, certification, build, and archive below runs inside a
# PRISTINE DETACHED WORKTREE of the requested tag, so the artifacts provably
# correspond to the tagged tree -- never to the possibly-diverged current
# checkout (testing one commit and archiving another is the failure mode
# this layout eliminates).
#
#   bash release.sh vX.Y.Z
#
# Artifacts:
#   dist/python/voigtinference-X.Y.Z-py3-none-any.whl   (wheel)
#   dist/python/voigtinference-X.Y.Z.tar.gz             (Python sdist)
#   dist/cpc/voigtinference-source-X.Y.Z.zip            (repository, submit this)
#   dist/cpc/voigtinference-source-X.Y.Z.tar.gz         (repository)
#   dist/SHA256SUMS
# The two tar.gz files have distinct names and directories: the Python sdist
# and the repository archive must never overwrite each other.
set -euo pipefail

TAG="${1:?usage: bash release.sh vX.Y.Z}"
test -d .git || { echo "error: run from the repository root"; exit 1; }
git rev-parse --verify "$TAG^{commit}" > /dev/null

ROOT="$(pwd)"
WT="$ROOT/.release-worktree"
OUT="$ROOT/dist"
rm -rf "$WT" "$OUT"
mkdir -p "$OUT/python" "$OUT/cpc"
git worktree add --detach "$WT" "$TAG"
trap 'cd "$ROOT"; git worktree remove --force "$WT" 2>/dev/null || true' EXIT

cd "$WT"

echo "== 1/6 constants guard (tag tree) =="
python3 check_constants.py

echo "== 2/6 Python tests (ephemeral venv, tag tree) =="
python3 -m venv .release-venv
# shellcheck disable=SC1091
source .release-venv/bin/activate
python -m pip install -q --upgrade pip
python -m pip install -q -e "voigtinference[test]"
python -m pytest voigtinference/tests -q

echo "== 3/6 Julia tests (tag tree) =="
( cd VoigtInference.jl && julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.test()' )

echo "== 4/6 numerical certification (tag tree) =="
( cd VoigtInference.jl && julia --project=. examples/certify.jl )

echo "== 5/6 Python build (wheel + sdist) =="
python -m pip install -q build
python -m build voigtinference --outdir "$OUT/python"
deactivate

echo "== 6/6 repository archives from the tag =="
VER="${TAG#v}"
NAME="voigtinference-source-$VER"
git archive --format=zip    --prefix="voigtinference-$VER/" -o "$OUT/cpc/$NAME.zip"    "$TAG"
git archive --format=tar.gz --prefix="voigtinference-$VER/" -o "$OUT/cpc/$NAME.tar.gz" "$TAG"

cd "$OUT"
required=("python/voigtinference-$VER-py3-none-any.whl"
          "python/voigtinference-$VER.tar.gz"
          "cpc/$NAME.zip"
          "cpc/$NAME.tar.gz")
for f in "${required[@]}"; do
    [ -s "$f" ] || { echo "error: missing or empty artifact $f"; exit 1; }
done
shasum -a 256 "${required[@]}" > SHA256SUMS
cat SHA256SUMS

echo
echo "Release artifacts in dist/ -- submit dist/cpc/$NAME.zip"
echo "Publish all four artifacts plus SHA256SUMS as GitHub Release assets."
