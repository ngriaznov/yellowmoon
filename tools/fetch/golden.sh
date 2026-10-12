#!/usr/bin/env bash
# Download the golden photo set into testdata/ and check every file's SHA-256.
#
#   tools/fetch/golden.sh                       tier 1 (CC0) and tier 2 (free licence, with attribution)
#   tools/fetch/golden.sh --tier 1              CC0 files only
#   tools/fetch/golden.sh --verify              check files already on disk (no network)
#   tools/fetch/golden.sh --tier3               print the local-only evaluation list (never fetched)
#   tools/fetch/golden.sh --index-tier3         record SHA-256 of the tier 3 files you downloaded by hand
#   tools/fetch/golden.sh --candidates [MODEL]  search the Play Raw forum for UNREVIEWED tier 2 candidates
#   tools/fetch/golden.sh --add-tier2 URL ...   pin one reviewed Play Raw file (see README)
#   tools/fetch/golden.sh --help                all options
#
# Tier 1 and tier 2 come from tools/fetch/manifest.json. Tier 3 (Photography Life, Imaging Resource) is
# copyrighted: it is a documented local-only list, never downloaded, committed, uploaded or cached in CI.
# Exit codes as for samples.sh: 0 ok, 1 verification or licence failure, 2 usage, 3 nothing to fetch, 4 network.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if ! command -v python3 >/dev/null 2>&1; then
  echo "error: python3 is required (standard library only, 3.8 or newer)" >&2
  exit 127
fi

exec python3 -I "$here/ymfetch.py" golden "$@"
