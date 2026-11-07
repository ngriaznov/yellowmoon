#!/usr/bin/env bash
# Fetch the Depth Anything V2-Small weights, and for D-03's tests the oracle fixtures, from Yellowmoon's own GitHub release
# `models-depth-anything-v2-small-1` (a prerelease that is never "latest": the in-app updater reads releases/latest, docs/12).
# The release is made by .github/workflows/models.yml (job `depth`) from training/depth/export.py and oracle.py; the weights
# are the Depth Anything V2-Small checkpoint converted to f16 safetensors (Apache-2.0, data/models/depth/MODEL_CARD.md).
#
#   tools/fetch/depth.sh                  the weights -> data/models/depth/   (what the build needs: release.yml, build.yml)
#   tools/fetch/depth.sh --fixtures       the weights and the oracle fixtures -> testdata/depth/oracle-1/   (D-03's tests)
#   tools/fetch/depth.sh --fixtures-only  only the fixtures
#   tools/fetch/depth.sh --verify         check the files already on disk against their pins; download nothing
#   tools/fetch/depth.sh --accept-new     download both assets, print their SHA-256 lines and check nothing (the pins are
#                                         not there yet); the lines go to the two files below
#
# Every download is checked against a SHA-256 pin, and a file that does not match is removed:
#   the weights   data/models/depth/SHA256SUMS    "<sha256>  depth_anything_v2_vits.f16.safetensors"  (also read by the runtime and by
#                                                 tools/licences/check_bundle.py --require-weights)
#   the fixtures  tools/fetch/depth.sha256        "<sha256>  depth-oracle-fixtures-1.tar.zst"
# While a pin is not 64 hex digits (the placeholder "TBD-after-first-run" is written before the workflow has run) the script
# stops with exit code 3 and downloads nothing unpinned by accident.
#
# The app and `ym` never download anything (AGENTS.md rule 7); this is build and test tooling. YM_TESTDATA=/path moves
# testdata/; YM_DEPTH_BASE_URL replaces the release URL (the unit tests point it at a local directory).
# Exit codes: 0 ok, 1 a download or a file failed its check, 2 usage, 3 a pin is missing, 127 a required tool is missing.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
tag="models-depth-anything-v2-small-1"
base="${YM_DEPTH_BASE_URL:-https://github.com/ngriaznov/yellowmoon/releases/download/$tag}"
weights_name="depth_anything_v2_vits.f16.safetensors"
fixtures_name="depth-oracle-fixtures-1.tar.zst"
model_dir="$root/data/models/depth"
testdata="${YM_TESTDATA:-$root/testdata}"
fixtures_dir="$testdata/depth/oracle-1"
weights_sums="$model_dir/SHA256SUMS"
fixtures_sums="$root/tools/fetch/depth.sha256"

want_weights=1
want_fixtures=0
mode="fetch"
for arg in "$@"; do
  case "$arg" in
    --fixtures) want_fixtures=1 ;;
    --fixtures-only) want_fixtures=1; want_weights=0 ;;
    --verify) mode="verify" ;;
    --accept-new) mode="accept"; want_fixtures=1 ;;
    -h|--help) sed -n '2,23p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "depth.sh: unknown argument $arg" >&2; exit 2 ;;
  esac
done

need() { command -v "$1" >/dev/null 2>&1 || { echo "depth.sh: $1 is required" >&2; exit 127; }; }
# Portable SHA-256 (GNU coreutils on Linux, BSD tools on macOS).
sha256_of() { if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | cut -d' ' -f1; else shasum -a 256 "$1" | cut -d' ' -f1; fi; }
# The pin of NAME in the sums file FILE (empty when the file or the line is missing).
pin_of() { if [ -f "$1" ]; then awk -v n="$2" '$2 == n { print $1 }' "$1"; fi; }
is_hash() { printf '%s' "$1" | grep -Eq '^[0-9a-f]{64}$'; }

# Stops with exit code 3 unless NAME has a real pin in FILE.
require_pin() {
  local pin
  pin="$(pin_of "$1" "$2")"
  if ! is_hash "$pin"; then
    echo "depth.sh: $2 has no SHA-256 pin in ${1#"$root"/} yet (found '${pin:-nothing}')." >&2
    echo "         Run the models workflow (a push to a branch models/depth-...), copy the two lines of its SHA256SUMS asset into" >&2
    echo "         data/models/depth/SHA256SUMS and tools/fetch/depth.sha256, then run this again." >&2
    exit 3
  fi
}

# Downloads NAME to DEST unless DEST already has the pinned hash; a download with another hash is removed and is an error.
fetch() {
  local name="$1" pin="$2" dest="$3" got
  if [ -f "$dest" ] && [ "$(sha256_of "$dest")" = "$pin" ]; then
    echo "depth.sh: $name is on disk and matches its pin"
    return 0
  fi
  mkdir -p "$(dirname "$dest")"
  echo "depth.sh: downloading $name from $base"
  if ! curl -fL --retry 3 --retry-delay 5 -o "$dest.part" "$base/$name"; then
    rm -f "$dest.part"
    echo "depth.sh: could not download $base/$name" >&2
    return 1
  fi
  got="$(sha256_of "$dest.part")"
  if [ "$got" != "$pin" ]; then
    rm -f "$dest.part"
    echo "depth.sh: $name has SHA-256 $got, the pin is $pin; nothing was kept" >&2
    return 1
  fi
  mv "$dest.part" "$dest"
}

# Checks DEST against the pin of NAME in FILE; prints one line.
verify() {
  local file="$1" name="$2" dest="$3" pin got
  require_pin "$file" "$name"
  pin="$(pin_of "$file" "$name")"
  [ -f "$dest" ] || { echo "depth.sh: ${dest#"$root"/} is missing" >&2; return 1; }
  got="$(sha256_of "$dest")"
  [ "$got" = "$pin" ] || { echo "depth.sh: ${dest#"$root"/} has SHA-256 $got, the pin is $pin" >&2; return 1; }
  echo "depth.sh: ${dest#"$root"/} matches its pin"
}

need curl
if [ "$mode" = "accept" ]; then
  tmp="$(mktemp -d)"
  trap 'rm -rf "$tmp"' EXIT
  for name in "$weights_name" "$fixtures_name"; do
    curl -fL --retry 3 --retry-delay 5 -o "$tmp/$name" "$base/$name"
    printf '%s  %s\n' "$(sha256_of "$tmp/$name")" "$name"
  done
  exit 0
fi

status=0
if [ "$want_weights" = 1 ]; then
  if [ "$mode" = "verify" ]; then
    verify "$weights_sums" "$weights_name" "$model_dir/$weights_name" || status=1
  else
    require_pin "$weights_sums" "$weights_name"
    fetch "$weights_name" "$(pin_of "$weights_sums" "$weights_name")" "$model_dir/$weights_name" || status=1
  fi
fi

if [ "$want_fixtures" = 1 ]; then
  archive="$testdata/depth/$fixtures_name"
  if [ "$mode" = "verify" ]; then
    verify "$fixtures_sums" "$fixtures_name" "$archive" || status=1
  else
    require_pin "$fixtures_sums" "$fixtures_name"
    pin="$(pin_of "$fixtures_sums" "$fixtures_name")"
    if fetch "$fixtures_name" "$pin" "$archive"; then
      marker="$fixtures_dir/.unpacked-$pin"
      if [ ! -f "$marker" ]; then
        need zstd
        need tar
        rm -rf "$fixtures_dir"
        mkdir -p "$fixtures_dir"
        zstd -dc "$archive" | tar -x -C "$fixtures_dir"
        : > "$marker"
      fi
      echo "depth.sh: fixtures ready in ${fixtures_dir#"$root"/}"
    else
      status=1
    fi
  fi
fi
exit "$status"
