#!/usr/bin/env bash
# The DNG gates of docs/09 (task M3-12), the same locally and in CI:
#
#   tools/dng/run_gates.sh                  the default synthetic golden set (61 files)
#   tools/dng/run_gates.sh --filter TEXT    only golden entries whose id contains TEXT (repeatable)
#   tools/dng/run_gates.sh --png            also compare with the 16-bit PNG render (no JPEG coding or dither)
#   tools/dng/run_gates.sh --files ARW...   check these ARW files instead of the golden set (local evaluation of real files,
#                                           for example testdata/real/files/*.ARW, relative to where you run this; step 1 is
#                                           skipped)
#   tools/dng/run_gates.sh --reprofile      only for a `ym` that still writes the identity profile: write a DNG with the
#                                           identity profile again with data/dng/profile.json first (a DNG that already
#                                           carries the fitted profile is still checked as it is)
#   tools/dng/run_gates.sh --help
#
# By default the DNGs are checked exactly as `ym` writes them (`dngfit gate --as-written`): `ym` embeds
# data/dng/profile.json, so the gates measure what a user gets, and a `ym` that wrote the identity profile would fail them
# (--as-written is accepted and means the default).
#
# Steps: dng_validate is built if it is missing (tools/dng/build_dng_validate.sh, about 10 minutes the first time, then
# cached in testdata/dng_sdk); `ym` and `dngfit` are built; the synthetic set is generated if it is missing
# (tools/fetch/synthetic.sh); then
#   1. `dngfit eval`: how close Adobe's renderer with the fitted profile is to ACES 2 (neutral ramp, hue and chroma
#      sweeps, random colours, the chart patches of the synthetic scenes);
#   2. `dngfit gate`: every DNG passes dng_validate with no warning and no error, and the picture dng_validate renders
#      from it is within a median CIEDE2000 of 1.5 of our JPEG of the same photo (every file; 2.5 at ISO 12800 and
#      above). Without --filter every
#      file of the default synthetic set has to be on disk (--require-all), so a partly generated set cannot pass.
# Exit code 0 when both gates pass, 1 when one fails, 2 on usage or setup errors.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
invoked_from="$PWD"
cd "$repo_root"
export CARGO_INCREMENTAL="${CARGO_INCREMENTAL:-0}"

gate_args=()
filtered=0
reprofile=0
files=()
while [ $# -gt 0 ]; do
  case "$1" in
    --filter) shift; [ $# -gt 0 ] || { echo "run_gates.sh: --filter needs a value" >&2; exit 2; }
              gate_args+=(--filter "$1"); filtered=1 ;;
    --files) shift
             # relative paths are the caller's: the script works from the repository root
             while [ $# -gt 0 ] && [ "${1#--}" = "$1" ]; do
               case "$1" in /*) files+=("$1") ;; *) files+=("$invoked_from/$1") ;; esac
               shift
             done
             [ ${#files[@]} -gt 0 ] || { echo "run_gates.sh: --files needs at least one ARW file" >&2; exit 2; }
             filtered=1
             continue ;;
    --reprofile) reprofile=1 ;;
    --as-written) reprofile=0 ;;
    --png) gate_args+=("$1") ;;
    -h|--help) sed -n '2,/^set -euo/p' "${BASH_SOURCE[0]}" | sed '$d' | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "run_gates.sh: unknown option $1 (see --help)" >&2; exit 2 ;;
  esac
  shift
done
if [ "$reprofile" = 0 ]; then gate_args+=(--as-written); fi
if [ ${#files[@]} -gt 0 ]; then gate_args+=(--files "${files[@]}"); fi

dng_validate="${DNG_VALIDATE:-$repo_root/testdata/dng_sdk/dng_validate}"
if [ ! -x "$dng_validate" ]; then
  tools/dng/build_dng_validate.sh "$repo_root/testdata/dng_sdk"
  dng_validate="$repo_root/testdata/dng_sdk/dng_validate"
fi
export DNG_VALIDATE="$dng_validate"

if [ ${#files[@]} -eq 0 ] && [ ! -d "${YM_TESTDATA:-$repo_root/testdata}/synthetic/arw" ]; then
  tools/fetch/synthetic.sh
fi

# Two builds, not one with both -p: dngfit (through golden) turns on ym-raw's test-only `testgen` feature, and cargo gives every
# package of one build the union of the features, so the `ym` under test would carry the test writer the shipped one lacks.
cargo build --release -p ym-cli
cargo build --release -p dngfit
target_dir="${CARGO_TARGET_DIR:-$repo_root/target}"
case "$target_dir" in /*) ;; *) target_dir="$repo_root/$target_dir" ;; esac
export YM_BIN="${YM_BIN:-$target_dir/release/ym}"
# a release build looks for its data folder next to the executable (docs/01); here it is the workspace's
export YELLOWMOON_DATA="${YELLOWMOON_DATA:-$repo_root/data}"

if [ ${#files[@]} -eq 0 ]; then
  echo "== profile against ACES 2 (Adobe's renderer)"
  "$target_dir/release/dngfit" eval
  echo "== DNG gates on the golden set"
else
  echo "== DNG gates on ${#files[@]} given file(s)"
fi
status=0
if [ "$filtered" = 0 ]; then gate_args+=(--require-all); fi
"$target_dir/release/dngfit" gate ${gate_args[@]+"${gate_args[@]}"} || status=$?
exit "$status"
