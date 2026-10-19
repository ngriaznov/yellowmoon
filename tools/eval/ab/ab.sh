#!/usr/bin/env bash
# The blind A/B of Apple Auto (docs/19 section 13, AA-1): make the test on your photos, then score your answers.
#
#   tools/eval/ab/ab.sh prepare PHOTOS_DIR --ym PATH [--out DIR] [--long-edge 2000] [--seed N] [--jobs N]
#                                          [--apple-jobs 1] [--with-camera] [--resume] [--data DIR]
#                                          [--instructions FILE]
#   tools/eval/ab/ab.sh score RESULTS_JSON [--plan PLAN_JSON]
#
# Python 3.9 or newer, standard library only; nothing is uploaded. See tools/eval/ab/README.md.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if ! command -v python3 >/dev/null 2>&1; then
  echo "ab.sh: python3 is needed (3.9 or newer; on a Mac, the command line tools or python.org)" >&2
  exit 2
fi
exec python3 -I "$here/ab.py" "$@"
