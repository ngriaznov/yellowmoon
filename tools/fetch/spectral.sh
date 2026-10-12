#!/usr/bin/env bash
# Fetch the public spectral inputs of the colour training and validation scripts
# into testdata/sources/ (git-ignored). Idempotent: a checkout that is already at
# the pinned commit and unmodified is left alone.
#
#   rawtoaces-data  Apache-2.0  52 measured camera curves (incl. the 11 Sony bodies),
#                               the 190 training reflectances, illuminants
#   aces-dev v1.3.1 AMPAS       ARRI's 17-CCT ALEXA matrices (sparse: only
#                               transforms/ctl/idt/vendorSupplied/arri)
#
# Both are pinned to a commit. A moved tag or a rewritten branch fails the script
# instead of silently changing the training inputs.
#
# Usage: tools/fetch/spectral.sh            (from anywhere; paths are repo-relative)
#        YM_TESTDATA=/path tools/fetch/spectral.sh   (alternative testdata root)
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
testdata="${YM_TESTDATA:-$repo_root/testdata}"
src="$testdata/sources"

# --- pins -------------------------------------------------------------------
RTA_URL="https://github.com/AcademySoftwareFoundation/rawtoaces-data.git"
RTA_SHA="e9b8503cf8a0641f40e5345d9757bde60c15f423"

ACES_URL="https://github.com/ampas/aces-dev.git"
ACES_TAG="v1.3.1"                                      # for the log only
ACES_SHA="fc189808208e28019a8c5a27cdb2d50312939232"    # commit the tag points at
ACES_SPARSE="transforms/ctl/idt/vendorSupplied/arri"
# ---------------------------------------------------------------------------

# fetch_pinned <name> <url> <sha> [sparse-path ...]
fetch_pinned() {
  local name="$1" url="$2" sha="$3"
  shift 3
  local dir="$src/$name"

  if [ -d "$dir/.git" ] \
     && [ "$(git -C "$dir" rev-parse HEAD 2>/dev/null || true)" = "$sha" ] \
     && [ -z "$(git -C "$dir" status --porcelain 2>/dev/null)" ]; then
    echo "spectral.sh: $name already at ${sha:0:12}"
    return 0
  fi

  echo "spectral.sh: fetching $name @ ${sha:0:12}"
  rm -rf "$dir"
  mkdir -p "$dir"
  git -C "$dir" init -q
  git -C "$dir" remote add origin "$url"
  git -C "$dir" config remote.origin.promisor true
  git -C "$dir" config remote.origin.partialclonefilter blob:none
  if [ "$#" -gt 0 ]; then
    git -C "$dir" sparse-checkout init --no-cone
    git -C "$dir" sparse-checkout set --no-cone "$@"
  fi
  git -C "$dir" fetch -q --depth 1 --filter=blob:none origin "$sha"
  git -C "$dir" checkout -q --detach FETCH_HEAD

  local got
  got="$(git -C "$dir" rev-parse HEAD)"
  if [ "$got" != "$sha" ]; then
    echo "spectral.sh: $name is at $got, expected $sha" >&2
    rm -rf "$dir"
    exit 1
  fi
}

mkdir -p "$src"
fetch_pinned rawtoaces-data "$RTA_URL" "$RTA_SHA"
fetch_pinned aces-dev "$ACES_URL" "$ACES_SHA" "$ACES_SPARSE"
echo "spectral.sh: aces-dev $ACES_TAG ok"

# --- sanity checks ------------------------------------------------------------
rta="$src/rawtoaces-data"
arri="$src/aces-dev/$ACES_SPARSE/alexa/AlexaParameters-2013-Nov-13/Alexa_aces_matrix.txt"

n_cams="$(find "$rta/data/camera" -name '*_380_780_5.json' | wc -l | tr -d ' ')"
if [ "$n_cams" != "52" ]; then
  echo "spectral.sh: expected 52 camera curves in rawtoaces-data, found $n_cams" >&2
  exit 1
fi
[ -f "$rta/data/training/training_spectral.json" ] || { echo "spectral.sh: training_spectral.json missing" >&2; exit 1; }
[ -f "$arri" ] || { echo "spectral.sh: ARRI matrix file missing: $arri" >&2; exit 1; }

# The shipped copy of the ARRI table must equal the fetched original.
if [ -f "$repo_root/data/arri/Alexa_aces_matrix.txt" ]; then
  if ! cmp -s "$arri" "$repo_root/data/arri/Alexa_aces_matrix.txt"; then
    echo "spectral.sh: data/arri/Alexa_aces_matrix.txt differs from aces-dev $ACES_TAG" >&2
    exit 1
  fi
fi

# The 11 Sony curves shipped in data/cameras/sony/ must equal the fetched originals.
if [ -d "$repo_root/data/cameras/sony" ]; then
  for f in "$repo_root"/data/cameras/sony/Sony_*_380_780_5.json; do
    base="$(basename "$f")"
    if ! cmp -s "$f" "$rta/data/camera/$base"; then
      echo "spectral.sh: data/cameras/sony/$base differs from rawtoaces-data @ ${RTA_SHA:0:12}" >&2
      exit 1
    fi
  done
fi

echo "spectral.sh: sources ready in $src"
