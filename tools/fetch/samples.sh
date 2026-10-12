#!/usr/bin/env bash
# Download the CC0 Sony ARW corpus into testdata/arw/<model>/ and check every file's SHA-256.
#
#   tools/fetch/samples.sh                      download whatever tools/fetch/manifest.json lists
#   tools/fetch/samples.sh --verify             check files already on disk (no network)
#   tools/fetch/samples.sh --body ILCE-7RM4     one camera model (repeatable)
#   tools/fetch/samples.sh --refresh            query RawDB and raw.pixls.us, rewrite the manifest
#   tools/fetch/samples.sh --coverage           body x variant table (Markdown) for docs/STATUS.md
#   tools/fetch/samples.sh --help               all options
#
# Sources: RawDB (rawdb.dnglab.org JSON API) and raw.pixls.us. CC0 files only, licence checked per file.
# Exit codes: 0 ok, 1 verification or licence failure, 2 usage, 3 manifest empty or nothing matches,
# 4 network unreachable or refused. Details: tools/fetch/README.md.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if ! command -v python3 >/dev/null 2>&1; then
  echo "error: python3 is required (standard library only, 3.8 or newer)" >&2
  exit 127
fi

exec python3 -I "$here/ymfetch.py" samples "$@"
