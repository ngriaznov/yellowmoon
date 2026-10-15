#!/usr/bin/env bash
# The whole lens oracle in one command: builds the synthetic Sony files, renders them with
# darktable-cli (lens module off / on), checks our maths against the renders and, with --record,
# rewrites the snapshot crates/ym-gpu/tests/cpu_data/darktable_oracle.json.
#
# usage: tools/lens/run_oracle_suite.sh [--record] SUITE_DIR
#
# Needs: darktable-cli >= 4.4 (Ubuntu 24.04: sudo apt-get install -y darktable), cargo, python3 with
# numpy and scipy. Disk: about 60 MB in SUITE_DIR (four 12 MB raw files and the gzipped dumps), and
# about 220 MB of temporary files per render while it runs. Exit status is that of the gates in
# compare_with_darktable.py (0: all pass).
set -euo pipefail

RECORD=0
if [ "${1:-}" = "--record" ]; then RECORD=1; shift; fi
[ $# -eq 1 ] || { echo "usage: $0 [--record] SUITE_DIR" >&2; exit 2; }
SUITE="$1"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
LENSGEN=(cargo run --release --quiet --manifest-path "$ROOT/tools/lens/lensgen/Cargo.toml" --)
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$ROOT/target/lensgen}"

# Everything below runs `python3 -I`, which ignores PYTHONPATH and the per-user site (where `pip3 install --user`, and plain
# `pip3 install` as a non-root user, put packages). Say so now rather than after the renders.
python3 -I -c 'import numpy, scipy' 2>/dev/null || {
  echo "python3 -I cannot import numpy and scipy. Install them where -I sees them: apt-get install python3-numpy python3-scipy," >&2
  echo "or a venv put first on PATH (python3 -m venv V && V/bin/pip install numpy scipy && PATH=V/bin:\$PATH $0 ...)." >&2
  exit 2
}
command -v darktable-cli >/dev/null || { echo "darktable-cli not found; on Ubuntu: sudo apt-get install -y darktable" >&2; exit 2; }
# Record what the numbers below were produced with: the gates were calibrated on darktable 4.6.1, and a CI runner or a Mac
# may have another build.
echo "darktable-cli: $(darktable-cli --version 2>&1 | head -n 1)"
python3 -I -c 'import sys, numpy, scipy; print("python3 %s, numpy %s, scipy %s" % (sys.version.split()[0], numpy.__version__, scipy.__version__))'

mkdir -p "$SUITE"
# CA x1 for the vignetting / distortion / combined renders, CA x8 for the CA-only measurement (a CA
# that is large against the sub-pixel noise of the line tracker).
"${LENSGEN[@]}" "$SUITE/fixtures" 1
"${LENSGEN[@]}" "$SUITE/fixtures_ca" 8
rm -f "$SUITE/fixtures_ca/flat.ARW" "$SUITE/fixtures_ca/hlines.ARW"

oracle() { # oracle <out> <flags> <file>
  "$HERE/darktable_oracle.sh" --require-lens --lens-flags "$2" --out "$1" "$3"
}
# One correction per render: darktable zooms the corrected image, so combined corrections interact.
oracle "$SUITE/vignetting" vignetting "$SUITE/fixtures/flat.ARW"
oracle "$SUITE/distortion" distortion "$SUITE/fixtures/vlines.ARW"
oracle "$SUITE/distortion" distortion "$SUITE/fixtures/hlines.ARW"
oracle "$SUITE/ca" ca "$SUITE/fixtures_ca/vlines.ARW"

status=0
if [ "$RECORD" = 1 ]; then
  python3 -I "$HERE/record_oracle.py" "$SUITE" || status=1
fi
python3 -I "$HERE/compare_with_darktable.py" vignette --dir "$SUITE/vignetting/flat" --tags "$SUITE/fixtures/tags.json" || status=1
python3 -I "$HERE/compare_with_darktable.py" distortion --ignore-ca --dir "$SUITE/distortion/vlines" --tags "$SUITE/fixtures/tags.json" || status=1
python3 -I "$HERE/compare_with_darktable.py" distortion --ignore-ca --horizontal --dir "$SUITE/distortion/hlines" --tags "$SUITE/fixtures/tags.json" || status=1
python3 -I "$HERE/compare_with_darktable.py" ca --dir "$SUITE/ca/vlines" --tags "$SUITE/fixtures_ca/tags.json" || status=1
exit $status
