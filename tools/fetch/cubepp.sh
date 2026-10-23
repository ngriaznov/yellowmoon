#!/usr/bin/env bash
# Fetch the Cube++ colour-constancy dataset (Ershov et al., CC BY 4.0, Zenodo record 4153431) into
# testdata/cubepp/ for the illuminant gate of M1-06 (`python3 -m training.illum.eval_cubepp`).
#
# Cube++ is used for the Cube++ gate (median angular error <= 3 degrees, `docs/13`) and, as CC BY 4.0, may
# also train the illuminant estimator (`docs/14`); attribution goes into NOTICE when weights trained on it ship.
# V1 uses it as an evaluation set only: the test split.
#
# zenodo.org is not reachable from the agents' cloud containers (`curl: (56) CONNECT tunnel failed, response 403`); the
# Cube++ check runs on GitHub Actions instead (`.github/workflows/cubepp.yml`). The whole record is about 200 GB; the
# check needs only `SimpleCube++.zip` (about 2 GB: `SimpleCube++/{train,test}/PNG/*.png`, `gt.csv`, `auxiliary/`),
# so pass `--only=SimpleCube++.zip`. On a machine that can reach Zenodo:
#
#   tools/fetch/cubepp.sh --list                  # show the record's files and sizes (no download)
#   tools/fetch/cubepp.sh --accept-new --only=SimpleCube++.zip
#                                                 # download, print the SHA-256, store it in testdata/cubepp/SHA256SUMS;
#                                                 # copy the line into PINS below
#   tools/fetch/cubepp.sh --only=SimpleCube++.zip # later runs refuse a file whose SHA-256 differs from PINS
#
# While PINS is empty a plain run stops with exit code 3 and says so (nothing is downloaded unpinned by accident).
#
# Usage: tools/fetch/cubepp.sh [--list] [--accept-new] [--only=NAME]      (YM_TESTDATA=/path moves testdata/)
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
testdata="${YM_TESTDATA:-$repo_root/testdata}"
dest="$testdata/cubepp"
RECORD=4153431
API="https://zenodo.org/api/records/$RECORD"

# --- pins: "<sha256>  <file name on Zenodo>", one per line. Empty until someone has run --accept-new. -----------------
PINS=""
# ---------------------------------------------------------------------------------------------------------------------

mode="fetch"
only=""
for arg in "$@"; do
  case "$arg" in
    --list) mode="list" ;;
    --accept-new) mode="accept" ;;
    --only=*) only="${arg#--only=}" ;;
    -h|--help) sed -n '2,24p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "cubepp.sh: unknown argument $arg" >&2; exit 2 ;;
  esac
done

need() { command -v "$1" >/dev/null 2>&1 || { echo "cubepp.sh: $1 is required" >&2; exit 2; }; }
need curl; need python3
# Portable SHA-256 and file size (GNU coreutils on Linux, BSD tools on macOS).
sha256_of() { if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | cut -d' ' -f1; else shasum -a 256 "$1" | cut -d' ' -f1; fi; }
size_of() { wc -c <"$1" | tr -d ' '; }

meta="$(curl -fsS --retry 3 "$API")" || {
  echo "cubepp.sh: cannot reach $API (zenodo.org may be blocked from this network)" >&2
  exit 4
}

files="$(printf '%s' "$meta" | python3 -I -c '
import json, sys
rec = json.load(sys.stdin)
for f in rec.get("files", []):
    print(f"{f['"'"'key'"'"']}\t{f['"'"'size'"'"']}\t{f['"'"'links'"'"']['"'"'self'"'"']}")
')"
[ -n "$files" ] || { echo "cubepp.sh: the record lists no files" >&2; exit 1; }

if [ "$mode" = "list" ]; then
  printf '%s\n' "$files" | awk -F'\t' '{printf "%12s  %s\n", $2, $1}'
  exit 0
fi

if [ -z "$PINS" ] && [ "$mode" != "accept" ]; then
  echo "cubepp.sh: no SHA-256 pins in this script yet. Run it once with --accept-new on a machine with access," >&2
  echo "           review testdata/cubepp/SHA256SUMS and copy the lines into PINS." >&2
  exit 3
fi

mkdir -p "$dest"
sums="$dest/SHA256SUMS"
: > "$sums"
found=0
while IFS=$'\t' read -r name size url; do
  [ -n "$name" ] || continue
  if [ -n "$only" ] && [ "$name" != "$only" ]; then continue; fi
  found=1
  out="$dest/$name"
  if [ ! -f "$out" ] || [ "$(size_of "$out" 2>/dev/null || echo 0)" != "$size" ]; then
    echo "cubepp.sh: downloading $name ($size bytes)"
    curl -fL --retry 3 -o "$out.part" "$url"
    mv "$out.part" "$out"
  fi
  got="$(sha256_of "$out")"
  printf '%s  %s\n' "$got" "$name" >> "$sums"
  if [ "$mode" = "fetch" ]; then
    want="$(printf '%s\n' "$PINS" | awk -v n="$name" '$2 == n {print $1}')"
    if [ -z "$want" ]; then
      echo "cubepp.sh: $name is not pinned" >&2; exit 1
    fi
    if [ "$got" != "$want" ]; then
      echo "cubepp.sh: $name has SHA-256 $got, pinned $want" >&2; exit 1
    fi
  fi
done <<< "$files"
if [ "$found" = 0 ]; then
  echo "cubepp.sh: ${only:-no file} is not in the record (see --list)" >&2; exit 1
fi

# Unpack archives next to the files (the test split is selected by the evaluation script).
for z in "$dest"/*.zip; do
  [ -e "$z" ] || continue
  marker="$z.unpacked"
  if [ ! -f "$marker" ]; then
    need unzip
    unzip -q -o "$z" -d "$dest"
    touch "$marker"
  fi
done
echo "cubepp.sh: dataset ready in $dest (checksums in $sums)"
