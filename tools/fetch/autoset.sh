#!/usr/bin/env bash
# Build the autoset training dataset (docs/18 section 6): CC0 Sony ARW files, streamed one at a time.
#
#   tools/fetch/autoset.sh refresh [--source pixls|rawdb|all] [--dry-run]   rewrite tools/fetch/autoset/manifest.json
#   tools/fetch/autoset.sh list [--body M]                                  one line per manifest entry
#   tools/fetch/autoset.sh build --out testdata/autoset/rows.jsonl [--limit N] [--body M]... [--keep] [--dump CMD]
#   tools/fetch/autoset.sh verify-rows testdata/autoset/rows.jsonl          check rows against the manifest (no network)
#   tools/fetch/autoset.sh --help                                           all options
#
# Exit codes: 0 ok, 1 failures, 2 usage, 3 empty manifest or nothing matched, 4 network refused,
# 5 not enough disk, 130 interrupted. Details: tools/fetch/autoset/README.md.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if ! command -v python3 >/dev/null 2>&1; then
  echo "error: python3 is required (standard library only, 3.8 or newer)" >&2
  exit 127
fi

exec python3 -I "$here/autoset/autoset.py" "$@"
