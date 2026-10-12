#!/usr/bin/env bash
# Builds the `ym-cli` binary and copies it to src-tauri/binaries/yellowmoon-cli-<host triple>, where
# Tauri expects the sidecar declared in bundle.externalBin (docs/01-architecture.md, "Building the
# app crate"). tauri-build checks that this file exists on every build of src-tauri.
#
# Usage: tools/dev/prepare-sidecar.sh [--release]
#
# Builds in debug mode by default. Respects CARGO_TARGET_DIR (and any other cargo configuration of
# the target directory). Works from any working directory.
set -euo pipefail

usage() {
  echo "Usage: ${0##*/} [--release]" >&2
}

profile_dir=debug
cargo_args=()
for arg in "$@"; do
  case "$arg" in
    --release)
      profile_dir=release
      cargo_args+=(--release)
      ;;
    -h | --help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $arg" >&2
      usage
      exit 2
      ;;
  esac
done

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
host_triple="$(rustc -vV | sed -n 's/^host: //p')"
if [[ -z "$host_triple" ]]; then
  echo "Could not read the host triple from 'rustc -vV'." >&2
  exit 1
fi

cd "$repo_root"
cargo build -p ym-cli ${cargo_args[@]+"${cargo_args[@]}"}

# Ask cargo where it put the build, so CARGO_TARGET_DIR and config files are honoured.
target_dir="$(cargo metadata --format-version 1 --no-deps | sed -n 's/.*"target_directory":"\([^"]*\)".*/\1/p')"
target_dir="${target_dir:-${CARGO_TARGET_DIR:-$repo_root/target}}"

source_bin="$target_dir/$profile_dir/yellowmoon"
if [[ ! -x "$source_bin" ]]; then
  echo "Expected the CLI binary at $source_bin, but it is missing." >&2
  exit 1
fi

dest_dir="$repo_root/src-tauri/binaries"
dest_bin="$dest_dir/yellowmoon-cli-$host_triple"
mkdir -p "$dest_dir"
cp -f "$source_bin" "$dest_bin"
echo "Sidecar ready: ${dest_bin#"$repo_root"/}"
