#!/usr/bin/env bash
# Prove that deny.toml does what docs/14 says, without touching the real workspace (M0-02 acceptance:
# "a test PR adding a GPL crate fails cargo deny").
#
# Builds throwaway workspaces from local path crates (no network) and runs `cargo deny check licenses bans`
# with the repository's deny.toml:
#   1. oracle (excluded) may depend on an LGPL crate called rawler                       -> must pass
#   2. a shipped crate depending on a GPL-3.0 crate                                       -> must fail (licence)
#   3. a shipped crate depending on an LGPL crate                                         -> must fail (licence)
#   4. a shipped crate depending on rawler, dng or jpegxl-rs, whatever their licence says -> must fail (ban)
#   5. a shipped crate depending on an MIT crate                                          -> must pass
#
# Usage: .github/scripts/deny-selftest.sh      (needs cargo-deny on PATH)
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
config="$repo/deny.toml"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
export CARGO_TARGET_DIR="$work/target"

if ! command -v cargo-deny >/dev/null 2>&1; then
  echo "error: cargo-deny is not installed (cargo install cargo-deny --locked)" >&2
  exit 127
fi

# fixture <dir> <crate name> <licence> : a local crate with the given name and SPDX licence
fixture() {
  mkdir -p "$1/src"
  cat >"$1/Cargo.toml" <<TOML
[package]
name = "$2"
version = "1.0.0"
edition = "2021"
license = "$3"
publish = false
TOML
  echo "pub fn f() {}" >"$1/src/lib.rs"
}

# workspace <dir> <shipped dep name|-> <oracle dep name|-> : members "shipped" and "oracle". The fixture crates live
# in $work/deps, outside the workspace root, so they are plain path dependencies and not implicit members.
workspace() {
  local dir="$1" shipped_dep="$2" oracle_dep="$3"
  mkdir -p "$dir/shipped/src" "$dir/oracle/src"
  printf '[workspace]\nresolver = "2"\nmembers = ["shipped", "oracle"]\n' >"$dir/Cargo.toml"
  for member in shipped oracle; do
    local dep="$shipped_dep"
    [ "$member" = oracle ] && dep="$oracle_dep"
    {
      printf '[package]\nname = "%s"\nversion = "0.0.0"\nedition = "2021"\nlicense = "Apache-2.0"\npublish = false\n\n[dependencies]\n' "$member"
      if [ "$dep" != "-" ]; then printf '%s = { path = "../../deps/%s" }\n' "$dep" "$dep"; fi
    } >"$dir/$member/Cargo.toml"
    echo "pub fn f() {}" >"$dir/$member/src/lib.rs"
  done
}

failures=0

# run_case <name> <expect pass|fail> <expected text in the output, or -> <shipped dep> <oracle dep>
run_case() {
  local name="$1" expect="$2" needle="$3" shipped_dep="$4" oracle_dep="$5"
  local dir="$work/$name"
  workspace "$dir" "$shipped_dep" "$oracle_dep"
  local out status=0
  out="$(cd "$dir" && cargo deny --config "$config" --manifest-path "$dir/Cargo.toml" check licenses bans 2>&1)" || status=$?
  if [ "$expect" = pass ] && [ "$status" -eq 0 ]; then
    echo "ok   $name: passed as expected"
  elif [ "$expect" = fail ] && [ "$status" -ne 0 ] && { [ "$needle" = - ] || grep -q "$needle" <<<"$out"; }; then
    echo "ok   $name: failed as expected (exit $status)"
    grep -E "^error" -A3 <<<"$out" | head -8 | sed 's/^/       /'
  else
    echo "FAIL $name: expected $expect, got exit $status" >&2
    printf '%s\n' "$out" | sed 's/^/       /' >&2
    failures=$((failures + 1))
  fi
}

mkdir -p "$work/deps"
fixture "$work/deps/gpl-fixture" gpl-fixture "GPL-3.0-only"
fixture "$work/deps/lgpl-fixture" lgpl-fixture "LGPL-2.1-only"
fixture "$work/deps/mit-fixture" mit-fixture "MIT"
fixture "$work/deps/rawler" rawler "LGPL-2.1"
fixture "$work/deps/dng" dng "MIT"            # banned by name even if the licence looked fine
fixture "$work/deps/jpegxl-rs" jpegxl-rs "MIT" # same

run_case oracle-may-use-lgpl pass - - rawler
run_case shipped-mit pass - mit-fixture -
run_case shipped-gpl fail "GPL-3.0" gpl-fixture -
run_case shipped-lgpl fail "LGPL-2.1" lgpl-fixture -
run_case shipped-rawler fail "explicitly banned" rawler -
run_case shipped-dng fail "explicitly banned" dng -
run_case shipped-jpegxl fail "explicitly banned" jpegxl-rs -

if [ "$failures" -ne 0 ]; then
  echo "$failures deny.toml self-test case(s) failed" >&2
  exit 1
fi
echo "deny.toml self-test passed"
