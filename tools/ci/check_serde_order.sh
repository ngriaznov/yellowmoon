#!/usr/bin/env bash
# Guard: serde_json's `preserve_order` feature must never be switched on for the code that writes files.
#
# Why: a serde_json `Map` keeps its keys sorted (a BTreeMap) unless the `preserve_order` feature is on, and the order of
# the keys decides the bytes of the `ym:Settings` document inside every exported file. The app and the CLI have to write
# the same bytes (AGENTS.md, rule 6; docs/10 "JSON"; crates/ym-cli/src/report.rs). Cargo unifies features across a build,
# so ONE dependency anywhere in the graph that asks for `serde_json/preserve_order` reorders the keys for all of it, and
# nothing in the code would show it. This script reads the resolved feature graph and fails if the feature is there.
#
#   tools/ci/check_serde_order.sh                    ym-core, ym-cli, yellowmoon-app, then the whole workspace
#   tools/ci/check_serde_order.sh -p NAME [-p NAME]  only these packages (replaces the default list; the workspace
#                                                    check is then left out unless --workspace is given)
#   tools/ci/check_serde_order.sh --workspace        add the whole-workspace check (what `cargo test --workspace` builds)
#   tools/ci/check_serde_order.sh --manifest-path F  look at another workspace (tools/ci/check_serde_order_selftest.sh does)
#   tools/ci/check_serde_order.sh --verbose          print each feature tree, not only the failing ones
#   tools/ci/check_serde_order.sh --help
#
# Each check is `cargo tree -e features -i serde_json --all-features --target all`: every feature of every package, for
# every platform (the shipped app is built on macOS, CI runs on Linux), build and dev dependencies included. That is
# stricter than any one real build on purpose; a feature that only some configuration turns on is still a trap. It
# compiles nothing; it takes a few seconds once cargo has the dependency sources (it needs every manifest of the graph, so a cold
# cache downloads them first). Exit code 0 when the feature is off everywhere checked, 1 when it is on (the
# tree is printed: look for the line `serde_json feature "preserve_order"` and read down to the crate that asks for it),
# 2 on usage errors or when cargo itself fails.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
start_dir="$PWD"

packages=()
workspace=0
workspace_requested=0
verbose=0
manifest=()
locked=(--locked)
while [ $# -gt 0 ]; do
  case "$1" in
    -p|--package) shift; [ $# -gt 0 ] || { echo "check_serde_order.sh: -p needs a package name" >&2; exit 2; }
                  packages+=("$1") ;;
    --workspace) workspace_requested=1 ;;
    --manifest-path) shift; [ $# -gt 0 ] || { echo "check_serde_order.sh: --manifest-path needs a file" >&2; exit 2; }
                     case "$1" in /*) manifest=(--manifest-path "$1") ;; *) manifest=(--manifest-path "$start_dir/$1") ;; esac
                     locked=() ;;   # a scratch workspace has no Cargo.lock to hold to
    --verbose) verbose=1 ;;
    -h|--help) sed -n '2,/^set -euo/p' "${BASH_SOURCE[0]}" | sed '$d' | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "check_serde_order.sh: unknown option $1 (see --help)" >&2; exit 2 ;;
  esac
  shift
done
if [ "${#packages[@]}" -eq 0 ]; then
  packages=(ym-core ym-cli yellowmoon-app)
  workspace=1
fi
[ "$workspace_requested" = 0 ] || workspace=1

command -v cargo >/dev/null 2>&1 || { echo "check_serde_order.sh: cargo is required" >&2; exit 127; }
cd "$repo_root"
failed=0
errfile="$(mktemp)"
trap 'rm -f "$errfile"' EXIT

# check <label> <cargo tree selection...>
check() {
  local label="$1"
  shift
  local out status=0
  out="$(cargo tree ${manifest[@]+"${manifest[@]}"} ${locked[@]+"${locked[@]}"} -e features -i serde_json --all-features --target all "$@" 2>"$errfile")" || status=$?
  if [ "$status" -ne 0 ]; then
    # Not depending on serde_json at all is fine (nothing to reorder). Any other cargo failure is a setup problem.
    if grep -Eq 'did not match any packages|is not a dependency|nothing to print' "$errfile"; then
      echo "ok    $label: does not use serde_json"
      return 0
    fi
    echo "error $label: cargo tree failed:" >&2
    cat "$errfile" >&2
    exit 2
  fi
  if grep -Fq 'serde_json feature "preserve_order"' <<<"$out"; then
    echo "FAIL  $label: serde_json feature \"preserve_order\" is enabled" >&2
    printf '%s\n' "$out" >&2
    failed=1
  else
    echo "ok    $label: serde_json has no preserve_order ($(grep -o 'serde_json feature "[a-z_0-9]*"' <<<"$out" | sed 's/serde_json feature //' | sort -u | tr '\n' ' ' | sed 's/ $//'))"
    [ "$verbose" = 0 ] || printf '%s\n' "$out"
  fi
}

for p in "${packages[@]}"; do
  check "$p" -p "$p"
done
if [ "$workspace" = 1 ]; then
  check "the whole workspace" --workspace
fi

if [ "$failed" != 0 ]; then
  cat >&2 <<'MSG'

serde_json/preserve_order changes the key order of every serde_json::Map in the build, and with it the bytes of the
ym:Settings document the app and the CLI write into their files. Remove the dependency feature (or the dependency), or
keep the ordered map out of serde_json (as crates/ym-cli/src/report.rs does). If an order-preserving JSON map is truly
needed somewhere, use another type there; do not switch the feature on.
MSG
  exit 1
fi
