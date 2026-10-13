#!/usr/bin/env bash
# The "local-eval" tier: nine real Sony ARW files for evaluation on THIS machine only.
#
#   tools/fetch/real-eval.sh              fetch (or verify) the files into testdata/real/ and list them
#   tools/fetch/real-eval.sh --list       list what is in testdata/real/files (model, size, source); no network
#   tools/fetch/real-eval.sh --force      also replace a file in testdata/real/files whose content differs from the pinned one
#   tools/fetch/real-eval.sh --help       the long text: sources, layout, licences
#
# Sources (each at one pinned commit, each file at one pinned git blob id): drewnoakes/metadata-extractor-images arw/ (six
# files), AcademySoftwareFoundation/OpenImageIO-images raw/RAW_SONY_A300.ARW, immich-app/test-assets formats/raw/Sony/ (two
# files). The links in testdata/real/files carry the names the local-only tests read (DSLR-A300.ARW, ILCE-6300_craw.ARW, ...).
#
# LICENCES ARE UNCLEAR OR RESTRICTED: local evaluation only. Never committed (testdata/ is git-ignored), never used in CI (the
# script refuses to run there), never uploaded, cached or redistributed. docs/validation-data.md has the reasons per source.
#
# Exit codes: 0 ok, 1 a file does not match the pin (or a link was left alone), 2 usage error, invalid table or running in CI,
# 3 nothing found (--list), 4 a source cannot be fetched. Needs git and python3 (standard library only, 3.8 or newer).
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if ! command -v python3 >/dev/null 2>&1; then
  echo "real-eval.sh: python3 is required (standard library only, 3.8 or newer)" >&2
  exit 127
fi

exec python3 -I "$here/real_eval.py" "$@"
