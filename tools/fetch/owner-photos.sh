#!/usr/bin/env bash
# Fetch Mykyta Griaznov's own photos (the readme-photos branch, raw files in Git LFS) into
# testdata/readme-photos/src/ and check every file's SHA-256 against tools/fetch/owner-photos.sha256.
#
#   tools/fetch/owner-photos.sh            clone or update the branch, pull the LFS files, verify
#   tools/fetch/owner-photos.sh --verify   check the files already on disk (no network)
#
# The photos are © Mykyta Griaznov, all rights reserved, not Apache-2.0. He permits them in this
# repository (that branch), as test and evaluation data, in CI, and in images made from them that the
# repository may publish (NOTICE item 9). They are never bundled with the app.
# Exit codes: 0 ok, 1 verification failure, 2 usage, 127 a required tool is missing.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
dest="$root/testdata/readme-photos/src"
sums="$root/tools/fetch/owner-photos.sha256"

verify() {
  if command -v sha256sum >/dev/null 2>&1; then
    (cd "$dest" && sha256sum --check --quiet "$sums")
  else
    (cd "$dest" && shasum -a 256 --check --quiet "$sums")   # macOS
  fi
}

case "${1:-}" in
  --verify) verify || exit 1; echo "owner photos: $(wc -l < "$sums" | tr -d ' ') files verified"; exit 0 ;;
  "") ;;
  *) sed -n '2,7p' "$0" >&2; exit 2 ;;
esac

git lfs version >/dev/null 2>&1 || { echo "error: Git LFS is required (https://git-lfs.com)" >&2; exit 127; }
if [ -d "$dest/.git" ]; then
  git -C "$dest" fetch --quiet --depth 1 origin readme-photos
  git -C "$dest" checkout --quiet --force FETCH_HEAD
else
  git clone --quiet --depth 1 --branch readme-photos --single-branch "$(git -C "$root" remote get-url origin)" "$dest"
fi
git -C "$dest" lfs pull
verify || exit 1
echo "owner photos: $(wc -l < "$sums" | tr -d ' ') files verified"
