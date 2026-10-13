#!/usr/bin/env bash
# Runs the auto-exposure study (tools/eval/README.md) on the local real files and on the synthetic set.
#
#   tools/eval/run.sh            # unit tests, then both sets
#   tools/eval/run.sh real       # only testdata/real/files
#   tools/eval/run.sh synthetic  # only testdata/synthetic/arw (one raw variant per scene)
#   tools/eval/run.sh test       # only the unit tests
#
# Builds the dev `ym` (never --release: a second profile costs gigabytes) unless YM points at a binary. Reports go to
# testdata/eval/<set>/ (git-ignored). The real files are local evaluation data: never copy or commit them.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$root"
what="${1:-all}"
export CARGO_INCREMENTAL="${CARGO_INCREMENTAL:-0}"

python3 -I -m unittest discover -s tools/eval -p 'test_*.py'
[ "$what" = test ] && exit 0

ym="${YM:-${CARGO_TARGET_DIR:-$root/target}/debug/ym}"
if [ -z "${YM:-}" ]; then
  cargo build -p ym-cli
fi

if [ "$what" = all ] || [ "$what" = real ]; then
  if ls testdata/real/files/*.ARW >/dev/null 2>&1; then
    python3 -I tools/eval/autoexp.py testdata/real/files --ym "$ym" --label real
  else
    echo "no ARW files in testdata/real/files; skipping the real set" >&2
  fi
fi
if [ "$what" = all ] || [ "$what" = synthetic ]; then
  if ls testdata/synthetic/arw/*.ARW >/dev/null 2>&1; then
    python3 -I tools/eval/autoexp.py testdata/synthetic/arw --ym "$ym" --label synthetic --one-per-scene
  else
    echo "no synthetic set; run tools/fetch/synthetic.sh first" >&2
  fi
fi
