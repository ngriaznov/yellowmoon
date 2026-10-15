#!/usr/bin/env bash
# Sets the one version of the app, ym and the workspace (docs/12, Versioning): the root Cargo.toml
# [workspace.package] version, package.json and package-lock.json (which mirror it), and the workspace
# entries of Cargo.lock. A push of the bump to main publishes the release (release.yml).
#
#   tools/release/bump-version.sh 0.1.1      # an explicit version
#   tools/release/bump-version.sh patch      # 0.1.0 -> 0.1.1
#
# It edits files only: no commit, no tag. Then add a "## [x.y.z] - date" section to CHANGELOG.md.
set -euo pipefail
cd "$(dirname "$0")/../.."

current=$(sed -n '/^\[workspace.package\]/,/^\[/{s/^version = "\(.*\)"/\1/p;}' Cargo.toml | head -n 1)
[ -n "$current" ] || { echo "no [workspace.package] version in Cargo.toml" >&2; exit 2; }

case "${1:-}" in
  patch) IFS=. read -r major minor patch <<< "$current"; next="$major.$minor.$((patch + 1))" ;;
  [0-9]*.[0-9]*.[0-9]*) next=$1 ;;
  *) echo "usage: $0 <x.y.z | patch>   (now $current)" >&2; exit 2 ;;
esac
[[ $next =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || { echo "not a plain x.y.z version: $next" >&2; exit 2; }

# The version line of [workspace.package], and the versions of the workspace's own crates in
# [workspace.dependencies] (`ym-core = { path = "crates/ym-core", version = "x.y.z" }`).
sed -i.bak \
  -e "/^\[workspace.package\]/,/^\[/{s/^version = \"$current\"/version = \"$next\"/;}" \
  -e "/path = \"crates\//s/version = \"$current\"/version = \"$next\"/" \
  Cargo.toml && rm -f Cargo.toml.bak
npm version "$next" --no-git-tag-version --allow-same-version >/dev/null
cargo update --workspace --offline >/dev/null 2>&1 || cargo update --workspace >/dev/null
# The lockfiles of the folders outside the workspace that depend on its crates by path (fuzz targets,
# tools/lens/lensgen): their entries for the ym-* crates.
git ls-files '*Cargo.lock' | grep -v '^Cargo.lock$' | while read -r lock; do
  perl -0pi -e "s/(name = \"ym-[a-z-]+\"\nversion = )\"\Q$current\E\"/\$1\"$next\"/g" "$lock"
done
echo "$current -> $next (Cargo.toml, package.json, package-lock.json, Cargo.lock)"
