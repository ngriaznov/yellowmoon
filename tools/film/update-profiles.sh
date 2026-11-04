#!/usr/bin/env bash
# Takes spektrafilm's film and paper profiles from one commit of github.com/andreavolpato/spektrafilm into
# data/film/spektrafilm/ (docs/24 section 6.1 and 6.6). Build tooling: it needs the network; the app and `ym` never run it.
#
#   tools/film/update-profiles.sh <commit> [--accept-licence-change] [--dest DIR]
#
# <commit> is a 40-digit commit id, or a branch name such as main (resolved with git ls-remote and written down as the id).
#
# What it does: fetches that commit into a temporary folder, copies src/spektrafilm/data/profiles/*.json and
# SPEKTRAFILM_LICENSE.txt byte for byte (git show; nothing is converted or reformatted), checks that every profile parses as
# JSON and names its own stock, prints what changed against the folder as it is, then rewrites profiles/, LICENSE,
# ATTRIBUTION.md and SHA256SUMS. It never edits a profile. If the licence text differs from the one in the folder it stops
# before writing anything, because a new licence is the owner's to read; --accept-licence-change goes on.
#
# CHANGELOG.txt (the licence file asks for one when anything is modified) is not written here: nothing is modified.
set -euo pipefail

REPO_URL="https://github.com/andreavolpato/spektrafilm.git"
PROFILE_DIR="src/spektrafilm/data/profiles"
LICENCE_FILE="SPEKTRAFILM_LICENSE.txt"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="$HERE/../../data/film/spektrafilm"
ACCEPT=0
COMMIT=""

while [ $# -gt 0 ]; do
  case "$1" in
    --accept-licence-change) ACCEPT=1 ;;
    --dest) shift; DEST="$1" ;;
    -h|--help) sed -n '2,16p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    -*) echo "error: unknown option $1" >&2; exit 2 ;;
    *) if [ -n "$COMMIT" ]; then echo "error: one commit only" >&2; exit 2; fi; COMMIT="$1" ;;
  esac
  shift
done
[ -n "$COMMIT" ] || { echo "usage: update-profiles.sh <commit> [--accept-licence-change] [--dest DIR]" >&2; exit 2; }

if ! [[ "$COMMIT" =~ ^[0-9a-f]{40}$ ]]; then
  RESOLVED="$(git ls-remote "$REPO_URL" "refs/heads/$COMMIT" | cut -f1)"
  [[ "$RESOLVED" =~ ^[0-9a-f]{40}$ ]] || { echo "error: '$COMMIT' is neither a 40-digit commit id nor a branch of $REPO_URL" >&2; exit 2; }
  echo "$COMMIT is $RESOLVED"
  COMMIT="$RESOLVED"
fi

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
mkdir -p "$TMP/git" "$TMP/new/profiles"
git -C "$TMP/git" init -q .
git -C "$TMP/git" remote add origin "$REPO_URL"
git -C "$TMP/git" fetch -q --depth 1 origin "$COMMIT"
COMMIT_DATE="$(git -C "$TMP/git" log -1 --format=%cs FETCH_HEAD)"
COMMIT_SUBJECT="$(git -C "$TMP/git" log -1 --format=%s FETCH_HEAD)"

git -C "$TMP/git" show "FETCH_HEAD:$LICENCE_FILE" > "$TMP/new/LICENSE"
git -C "$TMP/git" ls-tree --name-only FETCH_HEAD "$PROFILE_DIR/" | while IFS= read -r path; do
  case "$path" in *.json) git -C "$TMP/git" show "FETCH_HEAD:$path" > "$TMP/new/profiles/$(basename "$path")" ;; esac
done

# Parse every profile; the stock must be the file's name; one profile version across the set is expected.
python3 -I - "$TMP/new/profiles" > "$TMP/versions" <<'PY'
import json, pathlib, sys
folder = pathlib.Path(sys.argv[1])
files = sorted(folder.glob("*.json"))
if not files:
    sys.exit("error: the commit has no profiles in the profiles folder")
versions = set()
for path in files:
    try:
        data = json.loads(path.read_bytes())
        stock = data["info"]["stock"]
        version = data["metadata"]["version"]
    except (ValueError, KeyError, TypeError) as error:
        sys.exit(f"error: {path.name} is not a profile: {error!r}")
    if stock != path.stem:
        sys.exit(f"error: {path.name} says its stock is {stock!r}")
    versions.add(version)
print(", ".join(sorted(versions)))
PY
VERSIONS="$(cat "$TMP/versions")"
COUNT="$(find "$TMP/new/profiles" -name '*.json' | wc -l | tr -d ' ')"
BYTES="$(cat "$TMP"/new/profiles/*.json | wc -c | tr -d ' ')"

# What changed against the folder as it is.
echo "commit $COMMIT ($COMMIT_DATE, \"$COMMIT_SUBJECT\")"
echo "profile version $VERSIONS, $COUNT profiles, $BYTES bytes"
CHANGED=0
if [ -d "$DEST/profiles" ]; then
  for f in "$TMP"/new/profiles/*.json; do
    name="$(basename "$f")"
    if [ ! -f "$DEST/profiles/$name" ]; then echo "added:     $name"; CHANGED=1
    elif ! cmp -s "$f" "$DEST/profiles/$name"; then echo "changed:   $name"; CHANGED=1; fi
  done
  for f in "$DEST"/profiles/*.json; do
    [ -e "$f" ] || continue
    [ -f "$TMP/new/profiles/$(basename "$f")" ] || { echo "removed:   $(basename "$f")"; CHANGED=1; }
  done
  OLD_VERSIONS="$(python3 -I -c 'import json,pathlib,sys; print(", ".join(sorted({json.loads(p.read_bytes())["metadata"]["version"] for p in pathlib.Path(sys.argv[1]).glob("*.json")})))' "$DEST/profiles" 2>/dev/null || true)"
  [ -z "$OLD_VERSIONS" ] || [ "$OLD_VERSIONS" = "$VERSIONS" ] || { echo "profile version: $OLD_VERSIONS -> $VERSIONS"; CHANGED=1; }
else
  echo "(no earlier copy in $DEST: everything is new)"
  CHANGED=1
fi
if [ -f "$DEST/LICENSE" ]; then
  if cmp -s "$TMP/new/LICENSE" "$DEST/LICENSE"; then
    echo "licence text: unchanged"
  else
    echo "licence text: CHANGED"
    diff -u "$DEST/LICENSE" "$TMP/new/LICENSE" | head -60 || true
    if [ "$ACCEPT" -ne 1 ]; then
      echo "error: the licence text differs. Read it, tell the owner, then run again with --accept-licence-change. Nothing was written." >&2
      exit 3
    fi
    CHANGED=1
  fi
fi
[ "$CHANGED" -eq 1 ] || echo "no file changed"

# Write: profiles byte for byte, the licence, the attribution, the sums.
mkdir -p "$DEST/profiles"
find "$DEST/profiles" -maxdepth 1 -name '*.json' -delete
cp "$TMP"/new/profiles/*.json "$DEST/profiles/"
cp "$TMP/new/LICENSE" "$DEST/LICENSE"

cat > "$DEST/ATTRIBUTION.md" <<ATTR
# spektrafilm profiles

\`\`\`
spektrafilm by Andrea Volpato
https://github.com/andreavolpato/spektrafilm
Licensed CC BY-SA 4.0 (the text is in LICENSE, in this folder).

Taken from: commit $COMMIT
of the main branch, $COMMIT_DATE ("$COMMIT_SUBJECT"). Profile version $VERSIONS.
Files: $COUNT profiles in profiles/, not modified. SHA-256 in SHA256SUMS.
Modified by: nobody.
The profiles were made by the author from Kodak and Fujifilm data sheets and other
published material. The original data belong to their holders.
Yellowmoon is not affiliated with Andrea Volpato, Kodak or Fujifilm.
\`\`\`

- Source: \`src/spektrafilm/data/profiles/*.json\` at that commit. \`LICENSE\` is \`SPEKTRAFILM_LICENSE.txt\` at the same commit, byte for byte.
- The files are plain data, read at run time by the Film node. They are never compiled into the program, packed, encrypted or changed.
- Each profile also carries its own notice in \`metadata.license\` and \`metadata.copyright\`.
- If a profile is ever modified, the change is described in \`CHANGELOG.txt\` beside this file, with who made it, and this file says so.
- Written by \`tools/film/update-profiles.sh\`; do not edit by hand.
ATTR

(
  cd "$DEST"
  { find profiles -maxdepth 1 -name '*.json' | sort; echo LICENSE; } | LC_ALL=C sort | xargs sha256sum > SHA256SUMS
)
echo "wrote $DEST (SHA256SUMS: $(wc -l < "$DEST/SHA256SUMS") files)"
