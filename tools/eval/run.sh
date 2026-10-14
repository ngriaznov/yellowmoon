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
#
# Both sets run as the M1-08 gate (`autoexp.py --gate`: G3 within 0.5 EV for at least 80 % of the files; the outliers are
# printed for the M1-H5 review). The script runs every set it can, then exits with the worst code: 0 all met, 1 a file could
# not be analysed, 3 a gate is not met. (Since 2026-10-14 EV_auto is the camera's gain: the gate is met on the twelve local files
# in-sample and on the synthetic set by construction; see docs/02, Auto exposure.)
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

status=0
if [ "$what" = all ] || [ "$what" = real ]; then
  if ls testdata/real/files/*.ARW >/dev/null 2>&1; then
    code=0
    # the nine public files and, when present, the three A1 photos (all local evaluation data)
    inputs=(testdata/real/files)
    [ -d testdata/readme-photos/src ] && inputs+=(testdata/readme-photos/src)
    python3 -I tools/eval/autoexp.py "${inputs[@]}" --ym "$ym" --label real --gate || code=$?
    [ "$code" -gt "$status" ] && status=$code
  else
    echo "no ARW files in testdata/real/files; skipping the real set" >&2
  fi
fi
if [ "$what" = all ] || [ "$what" = synthetic ]; then
  if ls testdata/synthetic/arw/*.ARW >/dev/null 2>&1; then
    code=0
    python3 -I tools/eval/autoexp.py testdata/synthetic/arw --ym "$ym" --label synthetic --one-per-scene --gate || code=$?
    [ "$code" -gt "$status" ] && status=$code
  else
    echo "no synthetic set; run tools/fetch/synthetic.sh first" >&2
  fi
fi
exit "$status"
