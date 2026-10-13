#!/usr/bin/env bash
# Proves that tools/ci/check_serde_order.sh does what it says, without touching the real workspace: it builds a throwaway
# workspace in a temporary folder (path crates only, no network, nothing compiled) whose `serde_json` is a stand-in with
# the same feature names, and runs the guard against it.
#
#   1. nobody asks for preserve_order                                                  -> must pass
#   2. a dependency of ym-cli asks for serde_json/preserve_order (what a stray crate
#      in the graph would do); ym-core does not depend on it                           -> ym-cli fails, ym-core passes,
#                                                                                        the workspace check fails
#   3. a package has an optional Cargo feature that switches it on, off by default     -> must fail (--all-features)
#   4. a dev-dependency switches it on                                                 -> must fail
#   5. a package that does not use serde_json at all                                   -> must pass
#
# Usage: tools/ci/check_serde_order_selftest.sh        (needs cargo; the folder is deleted on exit)
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
guard="$repo/tools/ci/check_serde_order.sh"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
export CARGO_TARGET_DIR="$work/target"   # nothing is built, but never let cargo write into the repository's target

# crate <dir under $work> <name> <extra Cargo.toml text> : a local library crate
crate() {
  mkdir -p "$work/$1/src"
  printf '[package]\nname = "%s"\nversion = "0.0.1"\nedition = "2021"\n%s\n' "$2" "${3:-}" >"$work/$1/Cargo.toml"
  : >"$work/$1/src/lib.rs"
}

# workspace <members...> : the root manifest
workspace() {
  local members=""
  for m in "$@"; do members="$members\"$m\", "; done
  printf '[workspace]\nresolver = "2"\nmembers = [%s]\n' "${members%, }" >"$work/ws/Cargo.toml"
}

# a stand-in serde_json with the real feature names (the guard only reads the resolved feature graph). It lives outside
# the workspace, like the real one: a member would get all its features from --workspace --all-features.
crate deps/serde_json serde_json '
[features]
default = ["std"]
std = []
alloc = []
preserve_order = ["std"]
'

fails=0
# expect <pass|fail> <description> <guard arguments...>
expect() {
  local want="$1" what="$2" status=0
  shift 2
  "$guard" --manifest-path "$work/ws/Cargo.toml" "$@" >"$work/out.txt" 2>&1 || status=$?
  if { [ "$want" = pass ] && [ "$status" = 0 ]; } || { [ "$want" = fail ] && [ "$status" = 1 ]; }; then
    echo "ok    $what (guard exit $status)"
  else
    echo "WRONG $what: wanted $want, the guard exited $status" >&2
    cat "$work/out.txt" >&2
    fails=1
  fi
}

# ---- 1. clean ------------------------------------------------------------------------------------------------------
crate ws/ym-core ym-core '
[dependencies]
serde_json = { path = "../../deps/serde_json" }
'
crate ws/ym-cli ym-cli '
[dependencies]
serde_json = { path = "../../deps/serde_json" }
ym-core = { path = "../ym-core" }
'
crate ws/yellowmoon-app yellowmoon-app '
[dependencies]
serde_json = { path = "../../deps/serde_json" }
ym-core = { path = "../ym-core" }
'
workspace ym-core ym-cli yellowmoon-app
expect pass "nothing asks for preserve_order" -p ym-core -p ym-cli -p yellowmoon-app --workspace
expect pass "the default package list and the workspace, in one run"

# ---- 2. a stray dependency of ym-cli -------------------------------------------------------------------------------
crate ws/stray stray '
[dependencies]
serde_json = { path = "../../deps/serde_json", features = ["preserve_order"] }
'
crate ws/ym-cli ym-cli '
[dependencies]
serde_json = { path = "../../deps/serde_json" }
ym-core = { path = "../ym-core" }
stray = { path = "../stray" }
'
workspace ym-core ym-cli yellowmoon-app stray
expect fail "ym-cli depends on a crate that enables preserve_order" -p ym-cli
if ! grep -Fq 'serde_json feature "preserve_order"' "$work/out.txt"; then
  echo "WRONG the failure output does not show the feature tree" >&2
  fails=1
fi
expect pass "ym-core does not depend on that crate" -p ym-core
expect fail "the workspace check sees it (cargo test --workspace would build ym-core with it)" -p ym-core --workspace
expect fail "the default run (all three packages and the workspace)"

# ---- 3. an optional feature of a package ---------------------------------------------------------------------------
crate ws/stray stray ''
crate ws/ym-cli ym-cli '
[dependencies]
serde_json = { path = "../../deps/serde_json" }
ym-core = { path = "../ym-core" }
[features]
ordered = ["serde_json/preserve_order"]
'
expect fail "an optional feature of ym-cli turns it on" -p ym-cli

# ---- 4. a dev-dependency -------------------------------------------------------------------------------------------
crate ws/ym-cli ym-cli '
[dependencies]
serde_json = { path = "../../deps/serde_json" }
ym-core = { path = "../ym-core" }
[dev-dependencies]
serde_json = { path = "../../deps/serde_json", features = ["preserve_order"] }
'
expect fail "a dev-dependency of ym-cli turns it on" -p ym-cli

# ---- 5. no serde_json at all ---------------------------------------------------------------------------------------
crate ws/ym-cli ym-cli ''
crate ws/ym-core ym-core ''
crate ws/yellowmoon-app yellowmoon-app ''
workspace ym-core ym-cli yellowmoon-app stray
expect pass "packages that do not use serde_json" -p ym-core -p ym-cli

if [ "$fails" != 0 ]; then
  echo "check_serde_order_selftest.sh: the guard misbehaves" >&2
  exit 1
fi
echo "check_serde_order_selftest.sh: the guard passes clean graphs and fails on every way of enabling preserve_order"
