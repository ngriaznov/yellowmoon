#!/usr/bin/env python3
"""Compare `ym info --json` with LibRaw (rawpy) and ExifTool on a set of ARW files.

This is the M0-04 corpus gate: "metadata fields (make, model, model ID, ISO, exposure, dimensions,
compression, black, white, WB) equal LibRaw/ExifTool values". It runs

    ym info --json FILE                    (our reader)
    tools/oracle/rawpy_meta.py FILE        (LibRaw 0.22 through rawpy)
    exiftool -j -n -G1 -a -u FILE          (ExifTool, optional)

and reports every field that both sides define and that differs. Exit status 0 when there are no
differences, 1 otherwise, 2 for usage errors. LibRaw and ExifTool are test oracles only: they run
as separate processes and are never linked into anything we ship (docs/14-licensing.md).

Usage:
    python3 tools/oracle/compare_metadata.py [--ym PATH] [--exiftool PATH] \
        [--report report.json] FILE_OR_DIR...

`--ym` is the command-line tool (default: target/debug/ym, what `cargo build -p ym-cli` writes; `--yellowmoon`
is the old name of the option and still works). `--exiftool` is the path of the `exiftool` script (run with perl) or the name of an installed
binary; without it ExifTool is used when `exiftool` is on PATH, and skipped otherwise. Folders are
searched recursively for *.ARW.

Documented differences that are not reported as mismatches:
  * LibRaw's `white_level` is its own `maximum` (2^bits - 1); ours is the SR2 white level (0x787F),
    which LibRaw calls linear_max. The latter is compared.
  * LibRaw applies per-model margins to width/height. Only the raw sensor size is compared.
  * ISO: ours prefers the Sony maker-note ISO (the sensitivity the camera measured). When that is
    the source it is compared with ExifTool's SonyISO exactly; when ours fell back to the EXIF tags
    (older bodies) it is compared with SonyISO within a third of a stop. It is compared with
    LibRaw's EXIF ISO within a third of a stop
    (ISO 12232's tolerance), except when the EXIF value saturated at 65535. The two differ on real
    files: the ILCE-9 sample of github.com/lclevy/sony_raw has EXIF ISO 2500 and SonyISO 2619
    (4.8 %, 0.07 stop).
"""

import argparse
import json
import math
import os
import shutil
import subprocess
import sys

# The measured SonyISO and the nominal EXIF ISO may differ by up to a third of a stop.
ISO_TOLERANCE_STOPS = 1 / 3

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))


def find_files(paths):
    out = []
    for path in paths:
        if os.path.isdir(path):
            for root, _dirs, files in os.walk(path):
                out.extend(
                    os.path.join(root, f) for f in files if f.lower().endswith(".arw")
                )
        else:
            out.append(path)
    return sorted(out)


def ours(ym, path):
    """Our reader's view of a file: the first line `ym info --json` prints, parsed (None when it prints nothing)."""
    # Apple Auto off: only the metadata is compared, and a Mac with Apple Intelligence would ask its model too.
    proc = subprocess.run([ym, "info", "--json", path, "--apple-auto", "off"], capture_output=True, text=True)
    lines = proc.stdout.strip().splitlines()
    if not lines:
        return None
    return json.loads(lines[0])


def libraw(path):
    proc = subprocess.run(
        [sys.executable, os.path.join(HERE, "rawpy_meta.py"), path],
        capture_output=True,
        text=True,
    )
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError:
        return {"error": proc.stdout or proc.stderr}


def exiftool_command(spec):
    if spec:
        if os.path.isfile(spec) and not os.access(spec, os.X_OK):
            return ["perl", spec]
        return [spec]
    found = shutil.which("exiftool")
    return [found] if found else None


def exiftool(command, path):
    proc = subprocess.run(
        command + ["-j", "-n", "-G1", "-a", "-u", path], capture_output=True, text=True
    )
    try:
        return json.loads(proc.stdout)[0]
    except (json.JSONDecodeError, IndexError):
        return None


def numbers(value):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return [float(value)]
    return [float(x) for x in str(value).split()]


def compare(path, mine, lr, et):
    """Returns a list of (field, ours, theirs, oracle) mismatches."""
    diffs = []

    def check(field, a, b, oracle, tolerance=0.0):
        if a is None or b is None:
            return
        if isinstance(a, list) or isinstance(b, list):
            ok = len(a) == len(b) and all(abs(x - y) <= tolerance for x, y in zip(a, b))
        elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
            ok = abs(a - b) <= tolerance
        else:
            ok = a == b
        if not ok:
            diffs.append((field, a, b, oracle))

    cam, exp, dims, raw = mine["camera"], mine["exposure"], mine["dimensions"], mine["raw"]
    if lr and "error" not in lr:
        check("make", cam["make"], lr.get("make"), "libraw")
        check("model", cam["model"], lr.get("model"), "libraw")
        check("raw_width", dims["width"], lr.get("raw_width"), "libraw")
        check("raw_height", dims["height"], lr.get("raw_height"), "libraw")
        check("cfa", raw["cfa"], lr.get("cfa"), "libraw")
        check("black", mine["black"], lr.get("black_level_rggb"), "libraw")
        check("white", mine["white"], lr.get("camera_white_level_rggb"), "libraw")
        check("wb", mine["wb_as_shot"], lr.get("wb_rggb"), "libraw")
        if exp["shutter_s"] is not None and lr.get("shutter_s"):
            check("shutter_s", exp["shutter_s"], lr["shutter_s"], "libraw", 1e-6 * lr["shutter_s"])
        if exp["aperture"] is not None:
            check("aperture", exp["aperture"], lr.get("aperture"), "libraw", 1e-4)
        if exp["focal_length_mm"] is not None:
            check("focal_length_mm", exp["focal_length_mm"], lr.get("focal_length_mm"), "libraw", 1e-3)
        iso_lr = lr.get("iso")
        if exp["iso"] is not None and iso_lr and iso_lr < 65535:
            if abs(math.log2(exp["iso"] / iso_lr)) > ISO_TOLERANCE_STOPS:
                diffs.append(("iso", exp["iso"], iso_lr, "libraw"))
    elif lr:
        diffs.append(("libraw", "readable", lr["error"], "libraw"))

    if et:
        check("make", cam["make"], et.get("IFD0:Make"), "exiftool")
        check("model", cam["model"], et.get("IFD0:Model"), "exiftool")
        check("model_id", cam["model_id"], et.get("Sony:SonyModelID"), "exiftool")
        check("raw_width", dims["width"], et.get("SubIFD:ImageWidth"), "exiftool")
        check("raw_height", dims["height"], et.get("SubIFD:ImageHeight"), "exiftool")
        check("bits_per_sample", raw["bits_per_sample"], et.get("SubIFD:BitsPerSample"), "exiftool")
        check("compression", raw["compression"], et.get("SubIFD:Compression"), "exiftool")
        sony_iso = et.get("Sony:SonyISO")
        if exp["iso"] is not None and sony_iso is not None:
            if exp.get("iso_source") == "sony_maker_note":
                check("sony_iso", exp["iso"], round(float(sony_iso)), "exiftool", 1)
            elif abs(math.log2(exp["iso"] / float(sony_iso))) > ISO_TOLERANCE_STOPS:
                # Older bodies keep SonyISO in tables we do not read (docs/04: the 0x2010
                # variants); the EXIF value we fell back to must still be about the same.
                diffs.append(("iso", exp["iso"], round(float(sony_iso)), "exiftool"))
        if exp["shutter_s"] is not None and et.get("ExifIFD:ExposureTime"):
            check("shutter_s", exp["shutter_s"], float(et["ExifIFD:ExposureTime"]), "exiftool", 1e-9)
        if exp["aperture"] is not None and et.get("ExifIFD:FNumber"):
            check("aperture", exp["aperture"], float(et["ExifIFD:FNumber"]), "exiftool", 1e-3)
        check("black", mine["black"], numbers(et.get("SR2SubIFD:BlackLevel")), "exiftool")
        check("wb", mine["wb_as_shot"], numbers(et.get("SR2SubIFD:WB_RGGBLevels")), "exiftool")
        white = numbers(et.get("SR2SubIFD:WhiteLevel"))
        if white and len(white) == 1:  # the A500 writes one value for all channels
            white = white * 4
        elif white and len(white) == 3:
            white = [white[0], white[1], white[1], white[2]]
        check("white", mine["white"], white, "exiftool")
        lens = mine.get("lens_corrections")
        for key, name in (
            ("vignetting", "VignettingCorrParams"),
            ("chromatic_aberration", "ChromaticAberrationCorrParams"),
            ("distortion", "DistortionCorrParams"),
        ):
            if lens is not None and f"SubIFD:{name}" in et and not lens.get(key):
                diffs.append((f"lens.{key}", "absent", "present", "exiftool"))
    return diffs


def build_parser():
    """The command line. The binary is `ym` (decision 2026-10-12); `--yellowmoon` is kept as an alias of `--ym`."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--ym", "--yellowmoon", dest="ym", default=os.path.join(REPO, "target", "debug", "ym"))
    parser.add_argument("--exiftool", default=None)
    parser.add_argument("--report", default=None)
    return parser


def main(argv):
    args = build_parser().parse_args(argv)

    if not os.path.exists(args.ym):
        print(f"ym binary not found at {args.ym} (build it with `cargo build -p ym-cli` or pass --ym)")
        return 2
    et_cmd = exiftool_command(args.exiftool)
    files = find_files(args.paths)
    if not files:
        print("no ARW files found")
        return 2

    report = {"files": [], "exiftool": bool(et_cmd)}
    failed = 0
    for path in files:
        mine = ours(args.ym, path)
        lr = libraw(path)
        et = exiftool(et_cmd, path) if et_cmd else None
        entry = {"path": path}
        if mine is None or mine.get("status") != "ok":
            entry["ours"] = mine
            entry["result"] = "unreadable-by-us"
            # LibRaw reading what we refuse is worth knowing; both failing is consistent.
            if lr and "error" not in lr:
                failed += 1
                entry["note"] = "LibRaw reads this file but ym info does not"
        else:
            diffs = compare(path, mine, lr, et)
            entry["differences"] = [
                {"field": f, "ours": a, "theirs": b, "oracle": o} for f, a, b, o in diffs
            ]
            entry["result"] = "ok" if not diffs else "differs"
            failed += bool(diffs)
        report["files"].append(entry)
        status = entry["result"]
        print(f"{status:>18}  {path}")
        for d in entry.get("differences", []):
            print(f"{'':>18}    {d['field']}: ours {d['ours']} vs {d['oracle']} {d['theirs']}")

    report["summary"] = {"total": len(files), "with_differences": failed}
    if args.report:
        with open(args.report, "w") as f:
            json.dump(report, f, indent=2)
            f.write("\n")
    print(f"{len(files)} files, {failed} with differences")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
