#!/usr/bin/env bash
# Takes the CIE tables the Film node needs from the CIE's own dataset pages into data/film/cie/ (docs/24 section 6.4).
# Build tooling: it needs the network; the app and `ym` never run it.
#
#   tools/film/fetch-cie.sh [--accept-licence-change] [--dest DIR]
#
# The tables, exactly as published (files.cie.co.at, linked from each dataset page on cie.co.at):
#   CIE_xyz_1931_2deg.csv   CIE 1931 colour-matching functions, 2 degree observer, 360-830 nm, 1 nm
#   CIE_std_illum_D50.csv   CIE standard illuminant D50, 300-830 nm, 1 nm
#   CIE_illum_D55.csv       CIE illuminant D55, 300-780 nm, 5 nm
# each with its *_metadata.json. The dataset pages' own text names no licence; the metadata file linked from each page does
# ("rightsList": CC BY-SA 4.0). The script stops if a metadata file says anything else, if a file's MD5 or SHA-256 differs
# from the one the metadata file lists, or if the Creative Commons text it takes for LICENSE changed since the folder was made
# (--accept-licence-change goes on). It never edits a table.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="$HERE/../../data/film/cie"
FILES_URL="https://files.cie.co.at/Publications-datasets"
LEGALCODE_URL="https://creativecommons.org/licenses/by-sa/4.0/legalcode.txt"
ACCEPT=0
while [ $# -gt 0 ]; do
  case "$1" in
    --accept-licence-change) ACCEPT=1 ;;
    --dest) shift; DEST="$1" ;;
    -h|--help) sed -n '2,14p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "error: unknown argument $1" >&2; exit 2 ;;
  esac
  shift
done

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
mkdir -p "$TMP/new"

# file | dataset page | cite line as the page prints it
TABLES=(
  "CIE_xyz_1931_2deg.csv|https://cie.co.at/datatable/cie-1931-colour-matching-functions-2-degree-observer|CIE 2019, Colour-matching functions of CIE 1931 standard colorimetric observer, International Commission on Illumination (CIE), Vienna, AT, DOI: 10.25039/CIE.DS.xvudnb9b"
  "CIE_std_illum_D50.csv|https://cie.co.at/datatable/cie-standard-illuminant-d50|CIE 2022, CIE standard illuminant D50, International Commission on Illumination (CIE), Vienna, AT, DOI: 10.25039/CIE.DS.etgmuqt5"
  "CIE_illum_D55.csv|https://cie.co.at/datatable/relative-spectral-power-distributions-cie-illuminant-d55|CIE 2018, Relative spectral power distributions of CIE illuminant D55, International Commission on Illumination (CIE), Vienna, AT, DOI: 10.25039/CIE.DS.qewfb3kp"
)

for row in "${TABLES[@]}"; do
  file="${row%%|*}"
  curl --fail -sS --retry 3 -o "$TMP/new/$file" "$FILES_URL/$file"
  curl --fail -sS --retry 3 -o "$TMP/new/$file.csv_metadata.json" "$FILES_URL/$file.csv_metadata.json" 2>/dev/null \
    || curl --fail -sS --retry 3 -o "$TMP/new/${file}_metadata.json" "$FILES_URL/${file}_metadata.json"
done
curl --fail -sS --retry 3 -o "$TMP/new/LICENSE" "$LEGALCODE_URL"

# The metadata must say CC BY-SA 4.0 and carry the checksums of the file; the file must match them.
python3 -I - "$TMP/new" <<'PY'
import hashlib, json, pathlib, sys
folder = pathlib.Path(sys.argv[1])
for csv in sorted(folder.glob("*.csv")):
    meta_path = folder / f"{csv.name}_metadata.json"
    meta = json.loads(meta_path.read_bytes())
    rights = meta.get("rightsList") or []
    if len(rights) != 1 or rights[0].get("rightsIdentifier") != "CC BY-SA 4.0" or rights[0].get("rightsURI") != "https://creativecommons.org/licenses/by-sa/4.0/":
        sys.exit(f"error: {meta_path.name} does not state CC BY-SA 4.0 (rightsList: {rights!r}). Stop and ask the owner; nothing was written.")
    listed = {c["hashMethod"]: c["checksum"] for c in meta["checksums"]}
    data = csv.read_bytes()
    for method, digest in (("md5", hashlib.md5(data).hexdigest()), ("sha256", hashlib.sha256(data).hexdigest())):
        if listed.get(method) != digest:
            sys.exit(f"error: {csv.name} has {method} {digest}, the metadata lists {listed.get(method)}")
    print(f"{csv.name}: {rights[0]['rightsIdentifier']}, md5 and sha256 match the metadata")
text = (folder / "LICENSE").read_text(encoding="utf-8")
if not text.startswith("Attribution-ShareAlike 4.0 International") or "Section 6 -- Term and Termination" not in text:
    sys.exit("error: the legalcode download is not the CC BY-SA 4.0 text")
PY

if [ -f "$DEST/LICENSE" ] && ! cmp -s "$TMP/new/LICENSE" "$DEST/LICENSE"; then
  echo "licence text: CHANGED"
  diff -u "$DEST/LICENSE" "$TMP/new/LICENSE" | head -60 || true
  if [ "$ACCEPT" -ne 1 ]; then
    echo "error: the Creative Commons text differs. Read it, tell the owner, then run again with --accept-licence-change. Nothing was written." >&2
    exit 3
  fi
fi

CHANGED=0
mkdir -p "$DEST"
for f in "$TMP"/new/*; do
  name="$(basename "$f")"
  if [ ! -f "$DEST/$name" ]; then echo "added:   $name"; CHANGED=1
  elif ! cmp -s "$f" "$DEST/$name"; then echo "changed: $name"; CHANGED=1; fi
done
for f in "$DEST"/*; do
  [ -e "$f" ] || continue
  case "$(basename "$f")" in ATTRIBUTION.md|SHA256SUMS) continue ;; esac
  [ -f "$TMP/new/$(basename "$f")" ] || { echo "removed: $(basename "$f")"; CHANGED=1; }
done
[ "$CHANGED" -eq 1 ] || echo "no file changed"

# An unchanged set keeps its ATTRIBUTION.md (and so its fetch date); only a changed one is written again.
if [ "$CHANGED" -eq 1 ] || [ ! -f "$DEST/ATTRIBUTION.md" ]; then
  find "$DEST" -maxdepth 1 -type f ! -name ATTRIBUTION.md -delete
  cp "$TMP"/new/* "$DEST/"
  TODAY="$(date -u +%F)"
  {
    cat <<'HEAD'
# CIE tables

```
CIE colour-science tables, from the International Commission on Illumination (CIE), Vienna.
Licensed CC BY-SA 4.0: https://creativecommons.org/licenses/by-sa/4.0/ (the text is in LICENSE, in this folder).
Not modified: the CSV and metadata files are byte for byte as published.
```

The Film node reads these files at run time (the observer for stages 1 and 9, D50 and D55 as lights). They are plain files,
never compiled into the program. A table resampled or integrated in memory is not written anywhere.

HEAD
    echo "Fetched: $TODAY, from https://files.cie.co.at/Publications-datasets/ (the file links on each dataset page)."
    echo
    echo "Licence as the source states it: each dataset page on cie.co.at links a metadata file (\`<name>.csv_metadata.json\`) whose"
    echo "\`rightsList\` says \"Creative Commons Attribution-ShareAlike 4.0 International\" (\`CC BY-SA 4.0\`, https://creativecommons.org/licenses/by-sa/4.0/)."
    echo "The text of the dataset pages themselves names no licence. The metadata files are kept here as published."
    echo
    echo "| File | Table (the metadata file's description) | Dataset page | Cite as |"
    echo "|---|---|---|---|"
    for row in "${TABLES[@]}"; do
      IFS='|' read -r file page cite <<<"$row"
      desc="$(python3 -I -c 'import json,sys; m=json.load(open(sys.argv[1])); print(m["descriptions"][0]["description"])' "$DEST/$file"*_metadata.json)"
      echo "| \`$file\` | $desc | $page | $cite |"
    done
    echo
    echo "Modified by: nobody. If a table is ever modified, the change is described in \`CHANGELOG.txt\` beside this file, with who made it."
    echo "Written by \`tools/film/fetch-cie.sh\`; do not edit by hand."
  } > "$DEST/ATTRIBUTION.md"
fi

(cd "$DEST" && ls | grep -v -x -e ATTRIBUTION.md -e SHA256SUMS | LC_ALL=C sort | xargs sha256sum > SHA256SUMS)
echo "wrote $DEST (SHA256SUMS: $(wc -l < "$DEST/SHA256SUMS") files)"
