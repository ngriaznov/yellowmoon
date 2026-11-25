#!/usr/bin/env bash
# Fetch the EoMT small (COCO panoptic) weights from Yellowmoon's own GitHub release `models-eomt-small-1` (a prerelease that is never
# "latest": the in-app updater reads releases/latest, docs/12). The release is made by .github/workflows/models.yml (job `eomt`) from
# training/eomt/export.py; the weights are the authors' checkpoint tue-mps/coco_panoptic_eomt_small_640_2x converted to f16
# safetensors (MIT, with DINOv2's Apache-2.0 backbone; data/models/eomt/MODEL_CARD.md).
#
# STATUS (RG-03, 2026-11-25): the release is published, and the download matches the pin (the file is byte-identical on every
# export.py run). CI (the `eomt` job of ci.yml) and the build and release workflows run this script.
#
#   tools/fetch/eomt.sh            the weights -> data/models/eomt/   (what the build and the tests need)
#   tools/fetch/eomt.sh --verify   check the file already on disk against its pin; download nothing
#
# Every download is checked against the SHA-256 pin, and a file that does not match is removed:
#   data/models/eomt/SHA256SUMS    "<sha256>  eomt_coco_panoptic_small.f16.safetensors"  (also read by the runtime and by
#                                  tools/licences/check_bundle.py --require-eomt)
# While the pin is not 64 hex digits the script stops with exit code 3 and downloads nothing.
#
# The parity fixture of the tests is small and committed (crates/ym-ml/tests/data/eomt/): there is no fixture asset to fetch.
# The app and `ym` never download anything (AGENTS.md rule 7); this is build and test tooling. YM_EOMT_BASE_URL replaces the release
# URL (the unit tests point it at a local directory).
# Exit codes: 0 ok, 1 a download or a file failed its check, 2 usage, 3 a pin is missing, 127 a required tool is missing.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
tag="models-eomt-small-1"
base="${YM_EOMT_BASE_URL:-https://github.com/ngriaznov/yellowmoon/releases/download/$tag}"
weights_name="eomt_coco_panoptic_small.f16.safetensors"
model_dir="$root/data/models/eomt"
weights_sums="$model_dir/SHA256SUMS"

mode="fetch"
for arg in "$@"; do
  case "$arg" in
    --verify) mode="verify" ;;
    -h|--help) awk 'NR > 1 && /^#/ { print; next } NR > 1 { exit }' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "eomt.sh: unknown argument $arg" >&2; exit 2 ;;
  esac
done

need() { command -v "$1" >/dev/null 2>&1 || { echo "eomt.sh: $1 is required" >&2; exit 127; }; }
# Portable SHA-256 (GNU coreutils on Linux, BSD tools on macOS).
sha256_of() { if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | cut -d' ' -f1; else shasum -a 256 "$1" | cut -d' ' -f1; fi; }
# The pin of NAME in the sums file FILE (empty when the file or the line is missing).
pin_of() { if [ -f "$1" ]; then awk -v n="$2" '$2 == n { print $1 }' "$1"; fi; }
is_hash() { printf '%s' "$1" | grep -Eq '^[0-9a-f]{64}$'; }

pin="$(pin_of "$weights_sums" "$weights_name")"
if ! is_hash "$pin"; then
  echo "eomt.sh: $weights_name has no SHA-256 pin in ${weights_sums#"$root"/} (found '${pin:-nothing}')." >&2
  echo "         The pin is the line of the models workflow's SHA256SUMS asset (job eomt) or of training/eomt/export.py's output." >&2
  exit 3
fi

dest="$model_dir/$weights_name"
if [ "$mode" = "verify" ]; then
  [ -f "$dest" ] || { echo "eomt.sh: ${dest#"$root"/} is missing" >&2; exit 1; }
  got="$(sha256_of "$dest")"
  [ "$got" = "$pin" ] || { echo "eomt.sh: ${dest#"$root"/} has SHA-256 $got, the pin is $pin" >&2; exit 1; }
  echo "eomt.sh: ${dest#"$root"/} matches its pin"
  exit 0
fi

need curl
if [ -f "$dest" ] && [ "$(sha256_of "$dest")" = "$pin" ]; then
  echo "eomt.sh: $weights_name is on disk and matches its pin"
  exit 0
fi
mkdir -p "$model_dir"
echo "eomt.sh: downloading $weights_name from $base"
if ! curl -fL --retry 3 --retry-delay 5 -o "$dest.part" "$base/$weights_name"; then
  rm -f "$dest.part"
  echo "eomt.sh: could not download $base/$weights_name (is the release $tag published? see the header of this script)" >&2
  exit 1
fi
got="$(sha256_of "$dest.part")"
if [ "$got" != "$pin" ]; then
  rm -f "$dest.part"
  echo "eomt.sh: $weights_name has SHA-256 $got, the pin is $pin; nothing was kept" >&2
  exit 1
fi
mv "$dest.part" "$dest"
