#!/usr/bin/env python3
"""Compares LibRaw (through rawpy) with the synthetic ARW files `golden synth` wrote.

    python3 -I tools/golden/oracle_check.py [--arw testdata/synthetic/arw] [--scenes testdata/synthetic/scenes]
                                            [--only TEXT] [--json OUT.json]

LibRaw is an independent decoder (LGPL/CDDL, test-only: Yellowmoon never links it). For every file listed in
`<arw>/index.json` it reads LibRaw's `raw_image` and checks two things:

* LibRaw == the mosaic the file was written from (sidecar `mosaic.sha256`): for the exact variants (uncompressed 14,
  packed 12, lossless L). That tests the writer and ym-raw's decoder against a third party at once.
* LibRaw == what ym-raw decodes (the index's `decoded_sha256`): for every variant LibRaw reads. cRAW is lossy, so the
  mosaic is not expected back; both decoders must agree on the quantised values (neither dithers). Lossless M and S
  are compared in LibRaw's view (replicated chroma, white balance left in), through the helper of `tools/oracle`
  that retargets the compression tag LibRaw's loader keys on.

One LibRaw rule is not a disagreement: for TIFF compression 7 with one sample per pixel (Lossless L) LibRaw 0.22 only
uses its Sony loader for the ILCE-1, ILCE-7x and ILCE-9x models (`src/metadata/tiff.cpp`: "Allow A1/Compression=7"); for
any other model it falls back to a generic lossless-JPEG loader that does not understand Sony's layout and leaves the
image empty (all zero, or whatever the buffer held before). The synthetic set writes lossless L for every body so that
ym-raw is exercised on all of them (of these bodies only the A1 writes such files in reality, research/notes/sony-arw-formats.md), so a lossless L file of a
model outside LibRaw's list that LibRaw does not reproduce is reported as "LibRaw model gate" and counted separately.
One that LibRaw does reproduce counts as an ordinary match.

ExifTool (apt `libimage-exiftool-perl`), when installed, is a third reader: it must report the metadata the sidecar says
was written (make, model, model ID, ISO, exposure, f-number, focal length, lens, capture time, SR2 black, white and
white balance) and extract the embedded preview, which must be the scene's JPEG (ignoring the COM padding).

Exit code 0 when every file agrees (gated files aside), 1 otherwise. Needs `pip install rawpy numpy`.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

try:
    import numpy as np
    import rawpy
except ImportError as exc:  # pragma: no cover
    sys.exit(f"oracle_check.py needs rawpy and numpy: {exc}")

REPO = Path(__file__).resolve().parents[2]
EXACT = {"uncompressed14", "packed12", "lossless-l"}


def load_helper():
    """The LibRaw dump helper of tools/oracle (handles the Lossless M/S pseudo-raw), or None."""
    path = REPO / "tools" / "oracle" / "rawpy_compare.py"
    if not path.exists():
        return None
    spec = importlib.util.spec_from_file_location("rawpy_compare", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def libraw_digest(path: Path, helper) -> tuple[str | None, str]:
    """(SHA-256 of LibRaw's little-endian u16 samples, kind)."""
    if helper is not None:
        with tempfile.NamedTemporaryFile(suffix=".bin") as tmp:
            meta = helper.describe(str(path), tmp.name)
        return meta.get("raw_image_sha256"), meta.get("kind", "")
    with rawpy.imread(str(path)) as raw:
        image = np.asarray(raw.raw_image)
        if image.ndim != 2:
            return None, "rgb (needs tools/oracle/rawpy_compare.py)"
        data = np.ascontiguousarray(image, dtype="<u2")
        return hashlib.sha256(data.tobytes()).hexdigest(), "bayer"


def lossless_l_gate(model: str) -> bool:
    """True for the models whose Lossless L files LibRaw 0.22 reads with its Sony loader."""
    return model == "ILCE-1" or model.startswith(("ILCE-7", "ILCE-9"))


def strip_com_padding(jpeg: bytes) -> bytes:
    """The JPEG without the blank COM segments (all spaces) that `golden synth` inserts after the first segment."""
    first = 4 + int.from_bytes(jpeg[4:6], "big")
    pos = first
    while jpeg[pos : pos + 2] == b"\xff\xfe":
        end = pos + 2 + int.from_bytes(jpeg[pos + 2 : pos + 4], "big")
        if jpeg[pos + 4 : end].strip(b" "):  # a real comment: the scene's own
            break
        pos = end
    return jpeg[:first] + jpeg[pos:]


# Tags by group (`-G1`): ExifTool also decodes a Sony maker-note table that says exposure undef and focal length 0 in
# these files (the test writer leaves that enciphered table empty), so the EXIF values are asked for by group.
EXIFTOOL_TAGS = ["-IFD0:Make", "-IFD0:Model", "-Sony:SonyModelID", "-ExifIFD:ISO", "-ExifIFD:ExposureTime",
                 "-ExifIFD:FNumber", "-ExifIFD:FocalLength", "-ExifIFD:LensModel", "-ExifIFD:DateTimeOriginal",
                 "-SR2SubIFD:BlackLevel", "-SR2SubIFD:WhiteLevel", "-SR2SubIFD:WB_RGGBLevels"]


def exiftool_check(index, args, sidecars) -> list[dict]:
    """Compares ExifTool's reading of every file with the sidecar; returns one row per file."""
    rows = []
    files = [args.arw / e["file"] for e in index]
    out = subprocess.run(["exiftool", "-j", "-n", "-G1", *EXIFTOOL_TAGS, *map(str, files)], capture_output=True, text=True, check=True)
    meta = {Path(m["SourceFile"]).name: m for m in json.loads(out.stdout)}
    for entry in index:
        w = sidecars[entry["scene"]]["writer"]
        m = meta[entry["file"]]
        problems = []

        def expect(name, got, want, tol=0.0):
            ok = (abs(float(got) - float(want)) <= tol) if tol and got is not None else got == want
            if not ok:
                problems.append(f"{name}: {got!r} != {want!r}")

        expect("Make", m.get("IFD0:Make"), w["make"])
        expect("Model", m.get("IFD0:Model"), w["model"])
        expect("SonyModelID", m.get("Sony:SonyModelID"), w["model_id"])
        expect("ISO", m.get("ExifIFD:ISO"), w["iso"])
        expect("ExposureTime", m.get("ExifIFD:ExposureTime"), w["exposure_time"][0] / w["exposure_time"][1], 1e-9)
        expect("FNumber", m.get("ExifIFD:FNumber"), w["f_number"][0] / w["f_number"][1], 1e-9)
        expect("FocalLength", m.get("ExifIFD:FocalLength"), w["focal_length_mm"], 1e-9)
        expect("LensModel", m.get("ExifIFD:LensModel"), w["lens_model"])
        expect("DateTimeOriginal", m.get("ExifIFD:DateTimeOriginal"), w["capture_time"])
        expect("BlackLevel", str(m.get("SR2SubIFD:BlackLevel")), " ".join(map(str, w["sr2"]["black"])))
        expect("WhiteLevel", str(m.get("SR2SubIFD:WhiteLevel")), " ".join(map(str, w["sr2"]["white"])))
        expect("WB_RGGBLevels", str(m.get("SR2SubIFD:WB_RGGBLevels")), " ".join(map(str, w["sr2"]["wb_rggb"])))
        preview = subprocess.run(["exiftool", "-b", "-PreviewImage", str(args.arw / entry["file"])], capture_output=True).stdout
        want = (args.scenes / w["preview"]["file"]).read_bytes()
        if not preview or strip_com_padding(preview) != want:
            problems.append("PreviewImage is not the scene's JPEG")
        rows.append({"file": entry["file"], "ok": not problems, "problems": problems})
    return rows


def judge(file: str, variant: str, model: str, digest: str | None, kind: str, ours: str | None, mosaic_sha: str) -> dict:
    """One file's verdict. ``digest`` is LibRaw's hash, ``ours`` ym-raw's (None: the index has none, because the file was
    written with --no-verify and never verified; there is nothing to compare with, which is not a disagreement) and
    ``mosaic_sha`` the hash of the mosaic the file was written from (compared for the exact variants only)."""
    row = {
        "file": file, "variant": variant, "libraw_kind": kind, "model": model,
        "libraw_equals_ours": (digest is not None and digest == ours) if ours is not None else None,
        "libraw_equals_mosaic": (digest == mosaic_sha) if variant in EXACT else None,
    }
    # LibRaw's model gate: a Lossless L file of a body LibRaw does not read that way, which LibRaw leaves empty
    row["libraw_model_gate"] = (
        variant == "lossless-l" and not lossless_l_gate(model)
        and not row["libraw_equals_ours"] and row["libraw_equals_mosaic"] is not True
    )
    row["ok"] = row["libraw_model_gate"] or (
        row["libraw_equals_ours"] is not False and row["libraw_equals_mosaic"] is not False
    )
    return row


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--arw", type=Path, default=REPO / "testdata" / "synthetic" / "arw")
    ap.add_argument("--scenes", type=Path, default=REPO / "testdata" / "synthetic" / "scenes")
    ap.add_argument("--only", action="append", default=[])
    ap.add_argument("--json", type=Path)
    args = ap.parse_args(argv)

    index = json.loads((args.arw / "index.json").read_text())["files"]
    helper = load_helper()
    rows = []
    sidecars: dict[str, dict] = {}
    for entry in index:
        if args.only and not any(t in entry["file"] for t in args.only):
            continue
        scene = entry["scene"]
        if scene not in sidecars:
            doc = json.loads((args.scenes / f"{scene}.json").read_text())
            sidecars[scene] = {"sha256": doc["mosaic"]["sha256"], "model": doc["writer"]["model"], "writer": doc["writer"]}
        digest, kind = libraw_digest(args.arw / entry["file"], helper)
        row = judge(entry["file"], entry["variant"], sidecars[scene]["model"], digest, kind,
                    entry.get("decoded_sha256"), sidecars[scene]["sha256"])
        rows.append(row)

    by_variant: dict[str, list] = defaultdict(list)
    for r in rows:
        by_variant[r["variant"]].append(r)
    print(f"LibRaw {'.'.join(map(str, rawpy.libraw_version))} against ym-raw and the mosaics, {len(rows)} files")
    print(f"{'variant':<16}{'files':>6}{'== mosaic':>12}{'== ym-raw':>12}{'model gate':>12}")
    for variant in ("uncompressed14", "packed12", "craw", "lossless-l", "lossless-ms"):
        rs = by_variant.get(variant, [])
        if not rs:
            continue
        compared = [r for r in rs if not r["libraw_model_gate"]]
        mosaic = "n/a (lossy)" if variant not in EXACT else f"{sum(bool(r['libraw_equals_mosaic']) for r in compared)}/{len(compared)}"
        if variant == "lossless-ms":
            mosaic = "n/a (RGB)"
        unverified = sum(r["libraw_equals_ours"] is None for r in compared)
        ours = f"{sum(r['libraw_equals_ours'] is True for r in compared)}/{len(compared) - unverified}"
        gate = str(len(rs) - len(compared)) if len(rs) != len(compared) else "-"
        print(f"{variant:<16}{len(rs):>6}{mosaic:>12}{ours:>12}{gate:>12}")
    bad = [r for r in rows if not r["ok"]]
    for r in bad:
        print("MISMATCH", r)
    no_hash = [r["file"] for r in rows if r["libraw_equals_ours"] is None]
    if no_hash:
        print(f"note: {len(no_hash)} files have no ym-raw hash in the index (written with --no-verify): "
              "compared with the mosaic only; run `golden synth` without --no-verify to compare with ym-raw too")
    exif_rows = []
    selected = [e for e in index if not args.only or any(t in e["file"] for t in args.only)]
    if shutil.which("exiftool") and selected:
        exif_rows = exiftool_check(selected, args, sidecars)
        ver = subprocess.run(["exiftool", "-ver"], capture_output=True, text=True).stdout.strip()
        good = sum(r["ok"] for r in exif_rows)
        print(f"ExifTool {ver}: metadata and embedded preview as written for {good}/{len(exif_rows)} files")
        for r in exif_rows:
            if not r["ok"]:
                print("EXIFTOOL MISMATCH", r["file"], r["problems"])
                bad.append(r)
    else:
        print("ExifTool: not installed, skipped")
    if args.json:
        args.json.write_text(json.dumps({"libraw": rows, "exiftool": exif_rows}, indent=1) + "\n")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
