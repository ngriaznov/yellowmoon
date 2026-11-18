#!/usr/bin/env bash
# The baseline of docs/29-raw-constants.md (task RC-06): every objective and gate of section 2.3 at Tuning::ORACLE,
# the floor that a later fit is held to. One command, release builds, about 2 minutes on 4 cores.
#
#   tools/fit/baseline.sh OUTDIR          # run everything, print the tables (markdown), keep the raw output in OUTDIR
#   tools/fit/baseline.sh OUTDIR cheap    # only the parts under 15 seconds (the fit harness lines, the spec 03 and spec 04 gates)
#   tools/fit/baseline.sh OUTDIR tables   # format the tables again from the raw output already in OUTDIR
#
# Needs testdata/demosaic/{kodak,mcmaster} (tools/fetch/demosaic-sets.sh; YM_TESTDATA moves the root). The real-sensor
# group P reads testdata/demosaic/pixelshift200 when the folder has images and says "not run" when it has not. The data
# is for evaluation and local fitting and is never committed; nothing here writes a picture.
#
# Raw output in OUTDIR: demosaic_gate/{gate.json,numbers.md}, fit_demosaic.jsonl, capture_sharpen_ab/{results.json,table.md},
# fit_diffraction.jsonl, fit_sharpen.jsonl, spec04_edge.txt, and baseline.md (the tables). The tables are a function of
# those files and the run is deterministic, so a second run into another folder gives identical files (apart from the
# PNG strips of capture_sharpen_ab, which are not read).
set -euo pipefail
cd "$(dirname "$0")/../.."

out=${1:?an output folder}
mode=${2:-all}
mkdir -p "$out"

ex() { # ex EXAMPLE ARGS...
  local e=$1; shift
  cargo run -q --release -p ym-gpu --example "$e" -- "$@"
}

tables() { python3 -I tools/fit/baseline.py "$out" | tee "$out/baseline.md"; }

if [[ $mode == tables ]]; then tables; exit 0; fi

# The baseline is Tuning::ORACLE. demosaic_gate and capture_sharpen_ab run the shipped Tuning::DEFAULT, so the two
# must be equal while the tables are recorded; this fails when a fit has moved a default.
for e in fit_demosaic fit_diffraction fit_sharpen; do
  if ! diff <(ex "$e" --from oracle --list) <(ex "$e" --from default --list) >/dev/null; then
    echo "baseline.sh: Tuning::DEFAULT differs from Tuning::ORACLE in $e; the gates below would not be at the oracle" >&2
    exit 1
  fi
done

ex fit_demosaic --held --artifacts --edge --ca >"$out/fit_demosaic.jsonl"
ex fit_diffraction --held >"$out/fit_diffraction.jsonl"
ex fit_sharpen --held --edge >"$out/fit_sharpen.jsonl"
ex capture_sharpen_ab "$out/capture_sharpen_ab" >"$out/capture_sharpen_ab.txt"
cargo test -q --release -p ym-gpu --test cpu_output_sharpen edge_profile -- --nocapture --test-threads 1 \
  >"$out/spec04_edge.txt" 2>&1

if [[ $mode == all ]]; then
  ex demosaic_gate all --out "$out/demosaic_gate" >"$out/demosaic_gate.txt"
elif [[ ! -f $out/demosaic_gate/gate.json ]]; then
  echo "baseline.sh: cheap mode needs demosaic_gate/gate.json from a full run; the tables are not printed" >&2
  exit 0
fi
tables
