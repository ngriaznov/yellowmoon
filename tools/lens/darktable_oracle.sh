#!/usr/bin/env bash
# Lens-correction oracle: run ARW files through darktable-cli (GPL, only ever as a separate process)
# with its lens module on (embedded metadata) and save the module's own input and output, so that
# compare_with_darktable.py and record_oracle.py can derive darktable's vignetting gain and
# geometry and compare them with ours (docs/05-raw-processing.md section Verifying lens
# corrections; docs/13 "Lens corrections" gate).
#
# usage: tools/lens/darktable_oracle.sh [--out DIR] [--require-lens] [--lens-flags all|vignetting|distortion|ca]
#                                         [--lens-algorithm 1|2] [--lens-params HEXFILE [--lens-modversion N]] FILE.ARW ...
#
#   darktable 4.6.1 does not switch the lens module on by itself for Sony files (verified on synthetic
#   ARWs), so the script adds a lens history item: by default one it builds itself with
#   make_lens_params.py (module version 10, method "embedded metadata", the corrections of
#   --lens-flags, algorithm --lens-algorithm = 2, darktable's current one). --lens-params HEXFILE
#   overrides that with the hex of any lens-module op_params blob (e.g. captured from a GUI session:
#   sqlite3 library.db "select hex(op_params) from history where operation='lens'").
#
# For every input FILE it writes DIR/<stem>/{lens_in.pfm.gz,lens_out.pfm.gz,history.json,lens_params.hex}:
#   * lens_in / lens_out: the float RGB image (white-balanced camera RGB, linear, no colour matrix or
#     tone curve) that goes into and comes out of the lens module, dumped by `darktable-cli
#     --dump-pipe lens` (gzipped PFM, read by lens_io.py). Rendering to a 16-bit sRGB TIFF instead
#     mixes the planes through the camera matrix, which smears lateral CA (verified: 0.15 px against
#     0.003 px with the dump);
#   * history.json: the module history darktable used (its own defaults plus the lens row added here);
#   * lens_params.hex: the lens module op_params blob that was used.
# Exit codes: 0 ok; 2 usage / darktable missing; 3 (with --require-lens) the lens module changed
# nothing in a file (no embedded Sony tags darktable recognises, or an unknown camera); 4 render failed.
#
# Needs: darktable-cli >= 4.4 (Exif.SubImage1.{Vignetting,ChromaticAberration,Distortion}CorrParams
# support and --dump-pipe; 4.6.1 from Ubuntu 24.04 `apt-get install darktable` was used to write this),
# python3 with the standard sqlite3 module and gzip. Everything runs in an isolated config dir and
# temporary dir; nothing under $HOME is touched. Disk: about 220 MB of temporary files per 3000 x 2000
# render.
set -euo pipefail

OUT="build/lens-oracle"
REQUIRE=0
LENS_PARAMS=""
LENS_MODVERSION=10
LENS_FLAGS="all"
LENS_ALGORITHM=2
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FILES=()
while [ $# -gt 0 ]; do
  case "$1" in
    --out) OUT="$2"; shift 2 ;;
    --require-lens) REQUIRE=1; shift ;;
    --lens-params) LENS_PARAMS="$2"; shift 2 ;;
    --lens-modversion) LENS_MODVERSION="$2"; shift 2 ;;
    --lens-flags) LENS_FLAGS="$2"; shift 2 ;;
    --lens-algorithm) LENS_ALGORITHM="$2"; shift 2 ;;
    -h|--help) sed -n '2,32p' "$0"; exit 0 ;;
    -*) echo "unknown option $1" >&2; exit 2 ;;
    *) FILES+=("$1"); shift ;;
  esac
done
[ ${#FILES[@]} -gt 0 ] || { echo "no input files (see --help)" >&2; exit 2; }
command -v darktable-cli >/dev/null || { echo "darktable-cli not found; on Ubuntu: sudo apt-get install -y darktable" >&2; exit 2; }

render() { # render <input> <output.tif> <cfgdir> <librarydb> [extra darktable-cli options]
  local in="$1" out="$2" cfg="$3" lib="$4"
  shift 4
  rm -f "$out" # darktable-cli would otherwise write name_01.tif next to an existing file
  # --conf: 16-bit TIFF, no scene-referred defaults (no exposure/filmic), never write sidecar files
  # next to the input, no database updates beyond the isolated library.
  darktable-cli "$in" "$out" --core --configdir "$cfg" --library "$lib" \
    --conf plugins/imageio/format/tiff/bpp=16 \
    --conf plugins/darkroom/workflow=none \
    --conf write_sidecar_files=never \
    --conf ui_last/import_last_directory="$(dirname "$in")" "$@" >"$out.log" 2>&1 \
    || { echo "darktable-cli failed for $in (see $out.log)" >&2; return 4; }
}

add_lens_row() { # add_lens_row <librarydb> <hexfile> <modversion> : append an enabled lens history item
  python3 -I - "$1" "$2" "$3" <<'PY'
import sqlite3, sys
db = sqlite3.connect(sys.argv[1])
blob = bytes.fromhex(open(sys.argv[2]).read().strip())
imgid, num = db.execute("SELECT imgid, MAX(num) + 1 FROM history").fetchone()
tpl = db.execute("SELECT blendop_params, blendop_version FROM history ORDER BY num LIMIT 1").fetchone()
db.execute("INSERT INTO history (imgid, num, module, operation, op_params, enabled, blendop_params, blendop_version, multi_priority, multi_name, multi_name_hand_edited) VALUES (?,?,?,?,?,?,?,?,0,'',0)",
           (imgid, num, int(sys.argv[3]), "lens", blob, 1, tpl[0], tpl[1]))
db.execute("UPDATE images SET history_end=? WHERE id=?", (num + 1, imgid))
db.commit()
PY
}

dump_history() { # dump_history <librarydb> <history.json> <lens_params.hex>
  python3 -I - "$1" "$2" "$3" <<'PY'
import json, sqlite3, sys
db = sqlite3.connect(sys.argv[1])
rows = db.execute("SELECT num, operation, enabled, module, length(op_params), hex(op_params) FROM history ORDER BY num").fetchall()
hist = [dict(num=r[0], operation=r[1], enabled=r[2], modversion=r[3], params_len=r[4]) for r in rows]
json.dump(hist, open(sys.argv[2], "w"), indent=1)
lens = [r for r in rows if r[1] == "lens"]
open(sys.argv[3], "w").write(lens[-1][5] if lens else "")
PY
}

status=0
for f in "${FILES[@]}"; do
  stem="$(basename "${f%.*}")"
  dir="$OUT/$stem"
  rm -rf "$dir"; mkdir -p "$dir/tmp"
  cfg="$dir/cfg"; lib="$dir/library.db"

  # 1. First render: darktable imports the file and builds the default history (auto-applied modules).
  render "$f" "$dir/first.tif" "$cfg" "$lib" || { status=4; continue; }
  params="$LENS_PARAMS"
  if [ -z "$params" ]; then
    params="$dir/lens_params_in.hex"
    python3 -I "$HERE/make_lens_params.py" --flags "$LENS_FLAGS" --algorithm "$LENS_ALGORITHM" "$params"
  fi
  # 2. Add the lens module to the history and render again, dumping the module's input and output.
  add_lens_row "$lib" "$params" "$LENS_MODVERSION"
  TMPDIR="$dir/tmp" render "$f" "$dir/lens.tif" "$cfg" "$lib" --dump-pipe lens || { status=4; continue; }
  dump_history "$lib" "$dir/history.json" "$dir/lens_params.hex"
  in_pfm="$(find "$dir/tmp" -name '*lens_cpu_in_C.pfm' | head -n 1)"
  out_pfm="$(find "$dir/tmp" -name '*lens_cpu_out_C.pfm' | head -n 1)"
  if [ -z "$in_pfm" ] || [ -z "$out_pfm" ]; then
    echo "darktable did not dump the lens module for $f (see $dir/lens.tif.log)" >&2
    status=4; continue
  fi
  if cmp -s "$in_pfm" "$out_pfm"; then
    echo "NO EFFECT: darktable's lens module changed nothing in $f." >&2
    echo "  Check that the file carries Exif.SubImage1.{Vignetting,ChromaticAberration,Distortion}CorrParams" >&2
    echo "  (exiftool -a -G1 -s '$f' | grep -i Corr), that the CA tag starts with 2 * n (darktable rejects" >&2
    echo "  the data otherwise) and that darktable recognises the camera model." >&2
    [ "$REQUIRE" = 1 ] && status=3
  fi
  gzip -3 -c "$in_pfm" >"$dir/lens_in.pfm.gz"
  gzip -3 -c "$out_pfm" >"$dir/lens_out.pfm.gz"
  rm -rf "$dir/tmp" "$dir/first.tif" "$dir/first.tif.log" "$dir/lens.tif" "$dir/lens.tif.log" "$cfg" "$dir"/library.db*
  echo "$f -> $dir"
done
exit $status
