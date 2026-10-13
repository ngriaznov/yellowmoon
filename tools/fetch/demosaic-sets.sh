#!/usr/bin/env bash
# The demosaic evaluation sets: Kodak (24 images) and McMaster (18 images) into testdata/demosaic/{kodak,mcmaster}, each file
# pinned by commit and SHA-256 (tools/fetch/demosaic-sets.json).
#
#   tools/fetch/demosaic-sets.sh              download (or verify) the files and list them
#   tools/fetch/demosaic-sets.sh --list       verify what is already in testdata/demosaic against the table; no network
#   tools/fetch/demosaic-sets.sh --force      also replace a file whose content differs from the pinned one
#   tools/fetch/demosaic-sets.sh --help       the long text: sources, licences, exit codes
#
# EVALUATION ONLY: the files are never committed (testdata/ is git-ignored) and never shipped. They feed the ignored test
# `psnr_on_kodak_and_mcmaster` of crates/ym-gpu/tests/cpu_demosaic.rs (docs/05) and the demosaic gate of docs/13.
#
# Exit codes: 0 ok, 1 a file does not match the pin (or is missing, with --list), 2 usage error or invalid table, 3 nothing found
# (--list), 4 a host cannot be reached. Needs python3 (standard library only, 3.8 or newer).
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if ! command -v python3 >/dev/null 2>&1; then
  echo "demosaic-sets.sh: python3 is required (standard library only, 3.8 or newer)" >&2
  exit 127
fi

exec python3 -I "$here/demosaic_sets.py" "$@"
