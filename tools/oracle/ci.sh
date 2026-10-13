#!/usr/bin/env bash
# Decoder oracle job (M0-05): compares ym-raw's pixels, levels and white balance with rawler and
# LibRaw. Test-only: rawler (LGPL) and LibRaw (LGPL/CDDL) are never linked into a shipped crate.
#
#   tools/oracle/ci.sh              synthetic corpus, then testdata/arw when it holds files
#   tools/oracle/ci.sh --synthetic  synthetic corpus only
#
# Needs: Rust, python3 with `pip install rawpy numpy pylibjpeg-libjpeg`, and ExifTool (`apt install libimage-exiftool-perl`, or
# YM_EXIFTOOL=/path/to/exiftool for a clone of github.com/exiftool/exiftool). Writes oracle-synthetic.json and
# oracle-report.json (one per corpus) into $ORACLE_OUT (default: the folder the script was started
# from). Exit codes follow `oracle`: 0 identical, 1 a mismatch, 2 an oracle could not run.
set -euo pipefail
out="$(realpath -m "${ORACLE_OUT:-$PWD}")"
mkdir -p "$out"
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

export YM_REQUIRE_ORACLE=1

# libjpeg is the third oracle for the Lossless L tiles; the oracle binary skips it when missing, but
# in CI a comparison that cannot run is a failure, so it is required here too.
python3 -c "import rawpy, numpy, libjpeg" 2>/dev/null || {
  echo "error: python3 needs rawpy, numpy and libjpeg (pip install rawpy numpy pylibjpeg-libjpeg)" >&2
  exit 2
}

# ExifTool is the second metadata oracle; without it oracle_exiftool would print SKIP and pass, so refuse to start.
if ! command -v exiftool >/dev/null 2>&1 && [ -z "${YM_EXIFTOOL:-}" ]; then
  echo "error: ExifTool is required (apt install libimage-exiftool-perl, or set YM_EXIFTOOL)" >&2
  exit 2
fi

# 1. The synthetic corpus: every variant on several bodies, written by the test-only encoders.
rm -rf testdata/oracle-synthetic
cargo run -p ym-raw --features testgen --release --example gen_corpus -- testdata/oracle-synthetic
python3 -c 'import json, sys; json.load(open(sys.argv[1]))' testdata/oracle-synthetic/manifest.json
cargo run -p oracle --release -- --corpus testdata/oracle-synthetic --report "$out/oracle-synthetic.json"

# 2. The Rust tests that call LibRaw and ExifTool on synthetic files (M0-04: metadata equals LibRaw and ExifTool), and the tests
#    of the oracle program itself (tools/oracle/tests/report.rs: LibRaw end to end, a LibRaw that refuses every file). No other
#    job runs the latter against LibRaw: the macOS `rust` job has no rawpy and prints SKIP.
cargo test -p ym-raw --test oracle_decode --test oracle_libraw --test oracle_exiftool --release
cargo test -p oracle --release

# 3. The real corpus, when tools/fetch/samples.sh has filled it.
if [ "${1:-}" != "--synthetic" ] && find testdata/arw -iname '*.ARW' -print -quit 2>/dev/null | grep -q .; then
  cargo run -p oracle --release -- --corpus testdata/arw --report "$out/oracle-report.json"
fi
