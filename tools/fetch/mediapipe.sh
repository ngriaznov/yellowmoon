#!/usr/bin/env bash
# Fetch the MediaPipe face networks (BlazeFace full range, Face Mesh V2, Selfie Multiclass 256) from Yellowmoon's own GitHub
# release `models-mediapipe-faces-1` (a prerelease that is never "latest": the in-app updater reads releases/latest, docs/12).
# The release is made by .github/workflows/models.yml (job `mediapipe`) from tools/mediapipe/convert.py, which converts the
# TFLite files Google publishes (Apache-2.0, data/models/mediapipe/MODEL_CARD.md) to safetensors.
#
#   tools/fetch/mediapipe.sh                  the three networks -> data/models/mediapipe/   (what the build needs: release.yml, build.yml)
#   tools/fetch/mediapipe.sh --only NAME      only NAME: blazeface, face_mesh or selfie_multiclass (repeatable)
#   tools/fetch/mediapipe.sh --photos         also the test photos -> testdata/mediapipe/photos/   (tests/blazeface.rs)
#   tools/fetch/mediapipe.sh --photos-only    only the photos
#   tools/fetch/mediapipe.sh --verify         check the files already on disk against their pins; download nothing
#
# Every download is checked against a SHA-256 pin, and a file that does not match is removed:
#   the networks  data/models/mediapipe/SHA256SUMS      "<sha256>  blaze_face_full_range.f16.safetensors"  (also read by the runtime and by
#                                                       tools/licences/check_bundle.py). The converter is deterministic, so the pins are
#                                                       committed before the release exists and the workflow refuses to publish other bytes.
#   the photos    tools/fetch/mediapipe-photos.tsv      "<id>  <sha256>  <url>  <licence>" per line; the photos are CC0 files on Wikimedia
#                                                       Commons (the same pins as training/depth/oracle_photos.json) and one test photo of
#                                                       MediaPipe's own repository. They are never committed.
#
# The app and `ym` never download anything (AGENTS.md rule 7); this is build and test tooling. YM_TESTDATA=/path moves testdata/;
# YM_MEDIAPIPE_BASE_URL replaces the release URL (the unit tests point it at a local directory); YM_MEDIAPIPE_RETRIES and
# YM_MEDIAPIPE_RETRY_DELAY set the retries of a failed download (5 and 20 s: Wikimedia answers a burst with 429).
# Exit codes: 0 ok, 1 a download or a file failed its check, 2 usage, 3 a pin is missing, 127 a required tool is missing.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
tag="models-mediapipe-faces-1"
base="${YM_MEDIAPIPE_BASE_URL:-https://github.com/ngriaznov/yellowmoon/releases/download/$tag}"
model_dir="$root/data/models/mediapipe"
testdata="${YM_TESTDATA:-$root/testdata}"
photos_dir="$testdata/mediapipe/photos"
weights_sums="$model_dir/SHA256SUMS"
photos_list="$root/tools/fetch/mediapipe-photos.tsv"

# name -> file
file_of() {
  case "$1" in
    blazeface) echo "blaze_face_full_range.f16.safetensors" ;;
    face_mesh) echo "face_mesh_v2.f16.safetensors" ;;
    selfie_multiclass) echo "selfie_multiclass_256.f32.safetensors" ;;
    *) return 1 ;;
  esac
}

only=()
want_weights=1
want_photos=0
mode="fetch"
while [ $# -gt 0 ]; do
  case "$1" in
    --only)
      [ $# -ge 2 ] || { echo "mediapipe.sh: --only needs a name" >&2; exit 2; }
      file_of "$2" >/dev/null || { echo "mediapipe.sh: unknown network $2 (blazeface, face_mesh, selfie_multiclass)" >&2; exit 2; }
      only+=("$2"); shift ;;
    --photos) want_photos=1 ;;
    --photos-only) want_photos=1; want_weights=0 ;;
    --verify) mode="verify" ;;
    -h|--help) sed -n '2,21p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "mediapipe.sh: unknown argument $1" >&2; exit 2 ;;
  esac
  shift
done
if [ ${#only[@]} -eq 0 ]; then only=(blazeface face_mesh selfie_multiclass); fi

need() { command -v "$1" >/dev/null 2>&1 || { echo "mediapipe.sh: $1 is required" >&2; exit 127; }; }
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
    echo "mediapipe.sh: $2 has no SHA-256 pin in ${1#"$root"/} (found '${pin:-nothing}')." >&2
    exit 3
  fi
}

# Downloads URL to DEST unless DEST already has the pinned hash; a download with another hash is removed and is an error.
fetch() {
  local name="$1" url="$2" pin="$3" dest="$4" got
  if [ -f "$dest" ] && [ "$(sha256_of "$dest")" = "$pin" ]; then
    echo "mediapipe.sh: $name is on disk and matches its pin"
    return 0
  fi
  mkdir -p "$(dirname "$dest")"
  echo "mediapipe.sh: downloading $name from $url"
  # Wikimedia asks automated clients to say who they are and answers 429 to a burst, hence the agent and the slow retries.
  if ! curl -fL --retry "${YM_MEDIAPIPE_RETRIES:-5}" --retry-delay "${YM_MEDIAPIPE_RETRY_DELAY:-20}" --retry-all-errors -A "yellowmoon-test-fetch/1.0 (+https://github.com/ngriaznov/yellowmoon)" -o "$dest.part" "$url"; then
    rm -f "$dest.part"
    echo "mediapipe.sh: could not download $url" >&2
    return 1
  fi
  got="$(sha256_of "$dest.part")"
  if [ "$got" != "$pin" ]; then
    rm -f "$dest.part"
    echo "mediapipe.sh: $name has SHA-256 $got, the pin is $pin; nothing was kept" >&2
    return 1
  fi
  mv "$dest.part" "$dest"
}

# Checks DEST against PIN; prints one line.
verify() {
  local name="$1" pin="$2" dest="$3" got
  [ -f "$dest" ] || { echo "mediapipe.sh: ${dest#"$root"/} is missing" >&2; return 1; }
  got="$(sha256_of "$dest")"
  [ "$got" = "$pin" ] || { echo "mediapipe.sh: ${dest#"$root"/} has SHA-256 $got, the pin is $pin" >&2; return 1; }
  echo "mediapipe.sh: ${dest#"$root"/} matches its pin"
}

need curl
status=0
if [ "$want_weights" = 1 ]; then
  for name in "${only[@]}"; do
    file="$(file_of "$name")"
    require_pin "$weights_sums" "$file"
    pin="$(pin_of "$weights_sums" "$file")"
    if [ "$mode" = "verify" ]; then
      verify "$file" "$pin" "$model_dir/$file" || status=1
    else
      fetch "$file" "$base/$file" "$pin" "$model_dir/$file" || status=1
    fi
  done
fi

if [ "$want_photos" = 1 ]; then
  [ -f "$photos_list" ] || { echo "mediapipe.sh: ${photos_list#"$root"/} is missing" >&2; exit 3; }
  while IFS=$'\t' read -r id pin url _licence; do
    case "$id" in ''|'#'*) continue ;; esac
    is_hash "$pin" || { echo "mediapipe.sh: photo $id has no SHA-256 pin" >&2; exit 3; }
    if [ "$mode" = "verify" ]; then
      verify "$id.jpg" "$pin" "$photos_dir/$id.jpg" || status=1
    else
      fetch "$id.jpg" "$url" "$pin" "$photos_dir/$id.jpg" || status=1
    fi
  done < "$photos_list"
fi
exit "$status"
