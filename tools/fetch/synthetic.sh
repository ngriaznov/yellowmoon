#!/usr/bin/env bash
# Generate the synthetic ARW set into testdata/synthetic/ (git-ignored): one command, two steps.
#
#   tools/fetch/synthetic.sh                 the default set: about 37 mosaics and 61 ARW files, under 300 MB
#   tools/fetch/synthetic.sh --full          also the 6240 x 4160 full-size scene, every light under every body and the
#                                            evaluation scene (88 mosaics and 114 ARW files in all, about 740 MB, 6 minutes)
#   tools/fetch/synthetic.sh --force         regenerate everything (otherwise up-to-date files are kept)
#   tools/fetch/synthetic.sh --oracle        afterwards compare every file with LibRaw (needs rawpy) and ExifTool (if installed)
#   tools/fetch/synthetic.sh --only TEXT     only scenes whose name contains TEXT (repeatable)
#   tools/fetch/synthetic.sh --release       build the writer in release mode (faster for --full)
#   tools/fetch/synthetic.sh --no-verify     skip decoding each ARW again and comparing it with its mosaic
#   tools/fetch/synthetic.sh --help
#
# Step 1, `python3 -m training.synth.make_scenes`: physically based scenes (rawtoaces reflectances under blackbody, CIE D,
#   FL11 and LED lights, seen through the published Sony curves, Poisson-Gaussian noise) as raw mosaics + the embedded
#   JPEG + a JSON sidecar with the ground truth, into testdata/synthetic/scenes/.
# Step 2, `golden synth`: writes the ARW files (uncompressed 14, packed 12, cRAW, lossless L, lossless M/S) with
#   ym_raw::testgen into testdata/synthetic/arw/ and checks that each decodes back to its mosaic.
#
# Idempotent: both steps skip what is up to date. Nothing here is a photograph or a measurement; the files are ours,
# generated, and need no download. The --full evaluation scene contains CIE data (CC BY-SA 4.0): it stays in testdata/.
# Needs python3 with colour-science 0.4.7, numpy, scipy and pillow (pip install -r training/synth/requirements.txt),
# cargo, and the spectral sources (fetched here through tools/fetch/spectral.sh when missing).
# `YM_TESTDATA=/path` moves testdata/. Exit code 0 on success, 1 when a step or a round-trip check fails, 2 on usage.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
testdata="${YM_TESTDATA:-$repo_root/testdata}"
scenes="$testdata/synthetic/scenes"
arw="$testdata/synthetic/arw"

full=() force=() only_py=() only_rs=() release=() noverify=() oracle=0
while [ $# -gt 0 ]; do
  case "$1" in
    --full) full=(--full) ;;
    --force) force=(--force) ;;
    --only) shift; [ $# -gt 0 ] || { echo "synthetic.sh: --only needs a value" >&2; exit 2; }
            only_py+=(--only "$1"); only_rs+=(--only "$1") ;;
    --release) release=(--release) ;;
    --no-verify) noverify=(--no-verify) ;;
    --oracle) oracle=1 ;;
    -h|--help) sed -n '2,/^set -euo/p' "${BASH_SOURCE[0]}" | sed '$d' | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "synthetic.sh: unknown option $1 (see --help)" >&2; exit 2 ;;
  esac
  shift
done

for tool in python3 cargo; do
  command -v "$tool" >/dev/null 2>&1 || { echo "synthetic.sh: $tool is required" >&2; exit 127; }
done
if ! python3 -c "import colour, numpy, scipy, PIL" 2>/dev/null; then
  echo "synthetic.sh: missing Python packages; run: pip install -r training/synth/requirements.txt" >&2
  exit 1
fi

cd "$repo_root"
export CARGO_INCREMENTAL="${CARGO_INCREMENTAL:-0}"
export YM_TESTDATA="$testdata"
started=$(date +%s)

if [ ! -d "$testdata/sources/rawtoaces-data" ] || [ ! -d "$testdata/sources/aces-dev" ]; then
  echo "== spectral sources"
  tools/fetch/spectral.sh
fi

echo "== 1/2 scenes -> $scenes"
python3 -m training.synth.make_scenes --out "$scenes" ${full[@]+"${full[@]}"} ${force[@]+"${force[@]}"} ${only_py[@]+"${only_py[@]}"}

echo "== 2/2 ARW files -> $arw"
cargo run -q -p golden ${release[@]+"${release[@]}"} -- synth --scenes "$scenes" --out "$arw" \
  ${noverify[@]+"${noverify[@]}"} ${only_rs[@]+"${only_rs[@]}"}

if [ "$oracle" = 1 ]; then
  echo "== LibRaw oracle"
  python3 -I tools/golden/oracle_check.py --arw "$arw" --scenes "$scenes" ${only_rs[@]+"${only_rs[@]}"}
fi

python3 -I - "$arw" "$scenes" "$(( $(date +%s) - started ))" <<'PY'
import json, os, sys
arw, scenes, seconds = sys.argv[1], sys.argv[2], int(sys.argv[3])
files = json.load(open(os.path.join(arw, "index.json")))["files"]
by = {}
for f in files:
    n, b = by.get(f["variant"], (0, 0))
    by[f["variant"]] = (n + 1, b + f["bytes"])
mb = lambda b: f"{b / 1e6:8.1f} MB"
print()
print(f"{'variant':<16}{'files':>6}{'size':>13}")
for v in ("uncompressed14", "packed12", "craw", "lossless-l", "lossless-ms"):
    if v in by:
        print(f"{v:<16}{by[v][0]:>6}{mb(by[v][1]):>13}")
arw_total = sum(b for _, b in by.values())
print(f"{'ARW total':<16}{len(files):>6}{mb(arw_total):>13}")
def size(ext):
    return sum(os.path.getsize(os.path.join(scenes, n)) for n in os.listdir(scenes) if n.endswith(ext) and n != "index.json")
n_scenes = sum(1 for n in os.listdir(scenes) if n.endswith(".bin"))
print(f"{'scenes':<16}{n_scenes:>6}{mb(size('.bin') + size('.jpg') + size('.json')):>13}  (mosaics {mb(size('.bin')).strip()}, JPEGs {mb(size('.jpg')).strip()}, sidecars {mb(size('.json')).strip()})")
print(f"{'total':<16}{'':>6}{mb(arw_total + size('.bin') + size('.jpg') + size('.json')):>13}  in {seconds} s")
PY
