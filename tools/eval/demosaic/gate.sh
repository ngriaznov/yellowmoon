#!/usr/bin/env bash
# RA-05: the numbers of the R6 demosaic gate (docs/05 R6, docs/algorithms/TASKS.md), measurement only.
#
#   tools/eval/demosaic/gate.sh            # PSNR, zipper, false colour, noisy inputs, then the 50 MP timing
#   tools/eval/demosaic/gate.sh psnr       # the tables only (about 2 minutes on four cores)
#   tools/eval/demosaic/gate.sh timing     # the 50 MP timing only (about 5 minutes)
#   tools/eval/demosaic/gate.sh blind      # build `ym` with the test-only switch and make the blind-look set
#
# Output goes to testdata/ra05/ (git-ignored): numbers.md, gate.json, timing.md, timing.json, and for
# `blind` key.json, index.html and images/. Kodak and McMaster come from tools/fetch/demosaic-sets.sh.
#
# The Rust parts run in the workspace's dev profile with the ym-gpu crate compiled like a release build
# (opt-level 3, no debug assertions, no overflow checks, 16 codegen units; only the thin LTO is missing).
# That costs one extra build of ym-gpu instead of a second profile of the whole dependency tree, which
# already builds at opt-level 3 in the dev profile. Use `cargo run --release` if you have the disk and want
# the thin LTO as well.
set -euo pipefail
cd "$(dirname "$0")/../../.."
export CARGO_INCREMENTAL=0

RELEASE_LIKE=(
  --config 'profile.dev.package.ym-gpu.opt-level=3'
  --config 'profile.dev.package.ym-gpu.debug-assertions=false'
  --config 'profile.dev.package.ym-gpu.overflow-checks=false'
  --config 'profile.dev.package.ym-gpu.codegen-units=16'
)

what="${1:-all}"
case "$what" in
  all | psnr | timing | blind) ;;
  *)
    echo "usage: $0 [all|psnr|timing|blind [blind_look.py render options]]" >&2
    exit 2
    ;;
esac
if [ "$what" = all ] || [ "$what" = psnr ]; then
  cargo run -p ym-gpu --example ra05_gate "${RELEASE_LIKE[@]}" -- all
fi
if [ "$what" = all ] || [ "$what" = timing ]; then
  cargo run -p ym-gpu --example ra05_timing "${RELEASE_LIKE[@]}"
fi
if [ "$what" = blind ]; then
  cargo build -p ym-cli --features force-demosaic "${RELEASE_LIKE[@]}"
  python3 -I tools/eval/demosaic/blind_look.py render --ym target/debug/ym --out testdata/ra05 "${@:2}"
fi
