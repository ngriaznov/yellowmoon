#!/usr/bin/env bash
# Fetch the public portraits of the teeth tests (docs/27 S-17) -> testdata/teeth/photos/<id>.jpg   (tests/teeth_real.rs)
#
#   tools/fetch/teeth-photos.sh            download what is missing
#   tools/fetch/teeth-photos.sh --verify   check the files already on disk against their pins; download nothing
#
# Every photo is listed in tools/fetch/teeth-photos.tsv as "<id>  <sha256>  <url>  <licence>": CC0 or public domain by its provider,
# found with Openverse. The photos are never committed. A download with another SHA-256 is kept as <file>.rejected (never as the
# photo) and is an error. The app and `ym` never download anything (AGENTS.md rule 7); this is test tooling.
# YM_TESTDATA=/path moves testdata/; YM_TEETH_RETRIES and YM_TEETH_RETRY_DELAY set the retries of a failed download (5 and 20 s).
# Exit codes: 0 ok, 1 a download or a file failed its check, 2 usage, 3 a pin is missing, 127 curl is missing.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
testdata="${YM_TESTDATA:-$root/testdata}"
photos_dir="$testdata/teeth/photos"
photos_list="$root/tools/fetch/teeth-photos.tsv"

mode="fetch"
while [ $# -gt 0 ]; do
  case "$1" in
    --verify) mode="verify" ;;
    -h|--help) sed -n '2,9p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "teeth-photos.sh: unknown argument $1" >&2; exit 2 ;;
  esac
  shift
done

command -v curl >/dev/null 2>&1 || { echo "teeth-photos.sh: curl is required" >&2; exit 127; }
sha256_of() { if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | cut -d' ' -f1; else shasum -a 256 "$1" | cut -d' ' -f1; fi; }
is_hash() { printf '%s' "$1" | grep -Eq '^[0-9a-f]{64}$'; }

[ -f "$photos_list" ] || { echo "teeth-photos.sh: ${photos_list#"$root"/} is missing" >&2; exit 3; }
status=0
while IFS=$'\t' read -r id pin url _licence; do
  case "$id" in ''|'#'*) continue ;; esac
  is_hash "$pin" || { echo "teeth-photos.sh: photo $id has no SHA-256 pin" >&2; exit 3; }
  dest="$photos_dir/$id.jpg"
  if [ -f "$dest" ] && [ "$(sha256_of "$dest")" = "$pin" ]; then
    echo "teeth-photos.sh: $id.jpg is on disk and matches its pin"
    continue
  fi
  if [ "$mode" = "verify" ]; then
    echo "teeth-photos.sh: $id.jpg is missing or does not match its pin" >&2
    status=1
    continue
  fi
  mkdir -p "$photos_dir"
  echo "teeth-photos.sh: downloading $id.jpg from $url"
  if ! curl -fL --retry "${YM_TEETH_RETRIES:-5}" --retry-delay "${YM_TEETH_RETRY_DELAY:-20}" --retry-all-errors -A "test-fetch/1.0" -o "$dest.part" "$url"; then
    echo "teeth-photos.sh: could not download $url" >&2
    status=1
    continue
  fi
  got="$(sha256_of "$dest.part")"
  if [ "$got" != "$pin" ]; then
    mv -f "$dest.part" "$dest.rejected"
    echo "teeth-photos.sh: $id.jpg has SHA-256 $got, the pin is $pin; kept as $id.jpg.rejected, not as the photo" >&2
    status=1
    continue
  fi
  mv -f "$dest.part" "$dest"
done < "$photos_list"
exit "$status"
