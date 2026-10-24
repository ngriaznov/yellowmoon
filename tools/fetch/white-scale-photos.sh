#!/usr/bin/env bash
# The Play Raw files behind the white scales of docs/03 (17 ARW files of discuss.pixls.us posts) into
# testdata/white-scale/play-raw/, each file pinned by SHA-256 (tools/fetch/white-scale-photos.json).
#
#   tools/fetch/white-scale-photos.sh              download (or verify) the files and list them
#   tools/fetch/white-scale-photos.sh --list       verify what is already in testdata/white-scale/play-raw against the table; no network
#   tools/fetch/white-scale-photos.sh --force      also replace a file whose content differs from the pinned one
#   tools/fetch/white-scale-photos.sh --help       the long text: why these files, licences, exit codes
#
# EVALUATION ONLY: the files are never committed (testdata/ is git-ignored) and never shipped. They are the source of the
# white scales of the stand-in bodies (python3 -m training.white_scale.fit --check, docs/03).
#
# Exit codes: 0 ok, 1 a file does not match the pin (or is missing, with --list), 2 usage error or invalid table, 3 nothing found
# (--list), 4 a host cannot be reached. Needs python3 (standard library only, 3.8 or newer).
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if ! command -v python3 >/dev/null 2>&1; then
  echo "white-scale-photos.sh: python3 is required (standard library only, 3.8 or newer)" >&2
  exit 127
fi

exec python3 -I "$here/white_scale_photos.py" "$@"
