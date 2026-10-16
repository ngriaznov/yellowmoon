#!/usr/bin/env python3
"""Fingerprint ARRI look packages into data/looks/arri-packages.json (HUMAN M3-H2).

Usage:
    python3 -I tools/looks/fingerprint.py <zip>... [--package ID] [--packages FILE] [--dry-run]
    python3 -I tools/looks/fingerprint.py --reformat [--packages FILE]

For every zip it lists the look files (.cube LUTs, or ARRI Look Files: .alf4 or .aml; a package is
one kind) and records, per file, the path inside the zip, the size in bytes, the LUT size (the
LUT_3D_SIZE header of a .cube; the LUT size field of a look file's header) and the SHA-256. Nothing
else leaves the zip: **no LUT data is ever read into the output, written, or kept**. The result goes
into the matching package's "fingerprints" list in arri-packages.json, next to a small "observed"
summary (zip name, zip size, counts, folder names) that helps refine the name patterns by hand.

The package is recognised from the zip's name and contents using the patterns already in the file
(zip_name patterns, expected file count, LUT sizes, encoding words). If that is not unique, say
which one with --package.

Only the Python standard library is used. Run it with `-I` (isolated mode): it reads files from
the internet.

The ignored entries match the app's importer: __MACOSX folders, "._" files, .DS_Store, .xml,
PDFs and everything else that is not a look file. .alf4c is never part of a package, so the tool
skips it (the importer reads it as one of the person's own look files). A zip that holds more than
one kind of look file (.cube, .alf4, .aml) is refused. Entries whose paths would leave the folder
are skipped with a warning.

Exit codes: 0 done, 1 something went wrong (message on stderr), 2 usage error.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
import zipfile
from pathlib import Path, PurePosixPath

DEFAULT_PACKAGES = Path(__file__).resolve().parents[2] / "data" / "looks" / "arri-packages.json"
HEADER_BYTES = 64 * 1024
HEADER_LINES = 400
CHUNK = 1 << 20
INLINE_WIDTH = 140


class FingerprintError(Exception):
    """A problem the person running the tool can fix."""


# ---- Name folding, as in the app (crates/ym-looks/src/recognize/names.rs) -----------------------


def fold(text: str) -> str:
    """Lower case, letters and digits only."""
    return "".join(c for c in text.lower() if c.isalnum())


def pattern_matches(pattern: dict, folded: str) -> bool:
    """all_of words all present, one of any_of present (when given), none_of absent."""
    has = lambda word: fold(word) in folded  # noqa: E731
    return (
        all(has(w) for w in pattern.get("all_of", []))
        and (not pattern.get("any_of") or any(has(w) for w in pattern["any_of"]))
        and not any(has(w) for w in pattern.get("none_of", []))
    )


# ---- Reading a zip --------------------------------------------------------------------------------


def safe_path(name: str) -> str | None:
    """The entry name as a safe relative path with '/' separators, or None."""
    if not name or "\0" in name or "\\" in name:
        return None
    path = PurePosixPath(name)
    if path.is_absolute() or any(part in ("", "..", ".") for part in path.parts):
        return None
    return str(path)


def classify(path: str) -> str:
    """cube, alf4, aml, alf4c, xml, pdf, macos or other."""
    parts = path.split("/")
    name = parts[-1]
    if "__MACOSX" in parts or name.startswith("._") or name == ".DS_Store":
        return "macos"
    ext = name.rsplit(".", 1)[-1].lower() if "." in name else ""
    kinds = {"cube": "cube", "alf4": "alf4", "aml": "aml", "alf4c": "alf4c", "xml": "xml", "pdf": "pdf"}
    return kinds.get(ext, "other")


# The kinds a package is made of; the others are ignored, as in the app.
PACKAGE_KINDS = ("cube", "alf4", "aml")


def alf_lut_size(head: bytes, kind: str) -> int:
    """The LUT size in the header of a .alf4 or .aml file; 0 when the bytes are not that layout.

    Reads two header fields (where the LUT block is, and the size in it), never the LUT itself.
    """
    try:
        if kind == "alf4" and head[:4] == b"3FLA":
            block = struct.unpack_from("<I", head, 0x68)[0]
            return struct.unpack_from("<H", head, block + 0x42)[0]
        if kind == "aml" and head[:4] == b"KOLA":
            return struct.unpack_from("<I", head, 0x18)[0]
    except struct.error:
        pass
    return 0


def lut_size_of_header(head: bytes) -> int:
    """LUT_3D_SIZE from the first lines of a .cube file; 0 when there is none."""
    text = head.decode("utf-8", errors="replace").lstrip("\ufeff")
    for count, raw in enumerate(text.splitlines()):
        if count >= HEADER_LINES:
            break
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        words = line.split("#", 1)[0].split()
        if not words:
            continue
        first = words[0]
        if first[0] in "+-.0123456789":
            break  # the table starts; the header is over
        if first.upper() == "LUT_3D_SIZE" and len(words) == 2 and words[1].isdigit():
            return int(words[1])
    return 0


def read_zip(zip_path: Path) -> dict:
    """Names, sizes, LUT sizes and hashes of the look files in a zip, plus what was ignored."""
    found: dict[str, dict[str, dict]] = {kind: {} for kind in PACKAGE_KINDS}
    ignored = {"alf4c": 0, "xml": 0, "pdf": 0, "macos": 0, "other": 0, "unsafe": 0}
    warnings: list[str] = []
    try:
        archive = zipfile.ZipFile(zip_path)
    except (zipfile.BadZipFile, OSError) as error:
        raise FingerprintError(f"{zip_path.name}: cannot open as a zip ({error})") from error
    with archive:
        for info in archive.infolist():
            if info.is_dir():
                continue
            path = safe_path(info.filename)
            if path is None:
                ignored["unsafe"] += 1
                warnings.append(f"{zip_path.name}: skipped unsafe entry name {info.filename!r}")
                continue
            kind = classify(path)
            if kind not in PACKAGE_KINDS:
                ignored[kind] += 1
                continue
            if path in found[kind]:
                raise FingerprintError(f"{zip_path.name}: {path} appears twice")
            digest = hashlib.sha256()
            head = b""
            size = 0
            try:
                with archive.open(info) as stream:
                    while chunk := stream.read(CHUNK):
                        if len(head) < HEADER_BYTES:
                            head += chunk[: HEADER_BYTES - len(head)]
                        digest.update(chunk)
                        size += len(chunk)
            except (zipfile.BadZipFile, OSError, RuntimeError) as error:
                raise FingerprintError(f"{zip_path.name}: cannot read {path} ({error})") from error
            lut_size = lut_size_of_header(head) if kind == "cube" else alf_lut_size(head, kind)
            if lut_size == 0:
                what = "LUT_3D_SIZE" if kind == "cube" else "a LUT size in its header"
                warnings.append(f"{zip_path.name}: {path} has no {what} (recorded as 0)")
            found[kind][path] = {"path": path, "size": size, "lut_size": lut_size, "sha256": digest.hexdigest()}
    kinds = [k for k in PACKAGE_KINDS if found[k]]
    if len(kinds) > 1:
        raise FingerprintError(
            f"{zip_path.name}: holds both {' and '.join('.' + k for k in kinds)} files; "
            "fingerprint one package at a time"
        )
    kind = kinds[0] if kinds else "cube"
    return {
        "zip_name": zip_path.name,
        "zip_bytes": zip_path.stat().st_size,
        "format": kind,
        "files": [found[kind][p] for p in sorted(found[kind])],
        "ignored": ignored,
        "warnings": warnings,
    }


# ---- Choosing the package -------------------------------------------------------------------------


def score(package: dict, scan: dict) -> int:
    """How well a scanned zip fits a package's patterns (higher is better)."""
    points = 0
    stem = scan["zip_name"].rsplit(".", 1)[0]
    folded_zip = fold(stem)
    if any(pattern_matches(p, folded_zip) for p in package.get("zip_name", [])):
        points += 3
    if package.get("format", "cube") != scan["format"]:
        return -1  # a package of .cube files is not a zip of .alf4 files
    expected = package.get("expected", {})
    if expected.get("cube_files") == len(scan["files"]):
        points += 2
    sizes = {c["lut_size"] for c in scan["files"]}
    wanted = set(expected.get("lut_sizes", []))
    if wanted and sizes <= wanted:
        points += 1
    evidence = fold(stem + " " + " ".join(c["path"] for c in scan["files"]))
    if pattern_matches(package.get("markers", {}).get("encoding", {}), evidence):
        points += 1
    return points


def choose_package(db: dict, scan: dict, forced: str | None) -> dict:
    packages = db.get("packages", [])
    if forced is not None:
        for package in packages:
            if package["id"] == forced:
                if package.get("format", "cube") != scan["format"]:
                    raise FingerprintError(
                        f"{scan['zip_name']}: holds .{scan['format']} files, but {forced} is a package of "
                        f".{package.get('format', 'cube')} files"
                    )
                return package
        known = ", ".join(p["id"] for p in packages)
        raise FingerprintError(f"no package {forced!r} in the list (known: {known})")
    ranked = sorted(((score(p, scan), p["id"]) for p in packages), reverse=True)
    if not ranked or ranked[0][0] < 3 or (len(ranked) > 1 and ranked[0][0] == ranked[1][0]):
        table = ", ".join(f"{pid}={pts}" for pts, pid in ranked)
        raise FingerprintError(
            f"{scan['zip_name']}: cannot tell which package this is ({table}). Use --package ID."
        )
    return next(p for p in packages if p["id"] == ranked[0][1])


# ---- Writing the list -----------------------------------------------------------------------------


def apply_scan(package: dict, scan: dict) -> list[str]:
    """Writes the scan into the package. Returns notes about differences from the patterns."""
    notes: list[str] = []
    files = scan["files"]
    sizes = sorted({c["lut_size"] for c in files})
    folders = sorted({c["path"].split("/")[0] for c in files if "/" in c["path"]})
    package["fingerprints"] = [dict(c) for c in files]
    package["observed"] = {
        "zip_name": scan["zip_name"],
        "zip_bytes": scan["zip_bytes"],
        # The key is "cube_files" for the 3D-LUT packages (as before) and "files" for look files.
        ("cube_files" if scan["format"] == "cube" else "files"): len(files),
        "lut_sizes": sizes,
        "top_level_folders": folders,
        "ignored": {k: v for k, v in scan["ignored"].items() if v},
    }
    expected = package.setdefault("expected", {})
    if expected.get("cube_files") not in (None, len(files)):
        notes.append(
            f"{package['id']}: the pattern expected {expected['cube_files']} files, the zip has {len(files)}"
        )
    expected["cube_files"] = len(files)
    if expected.get("lut_sizes") and set(sizes) != set(expected["lut_sizes"]):
        notes.append(f"{package['id']}: the pattern expected LUT sizes {expected['lut_sizes']}, the zip has {sizes}")
    expected["lut_sizes"] = sizes
    return notes


def is_scalar(value) -> bool:
    return not isinstance(value, (dict, list))


def dump(value, level: int = 0) -> str:
    """JSON with two-space indent that keeps short lists and small objects on one line."""
    pad = "  " * level
    inner = "  " * (level + 1)
    if is_scalar(value):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, list):
        if not value:
            return "[]"
        if all(is_scalar(v) for v in value):
            one = "[" + ", ".join(json.dumps(v, ensure_ascii=False) for v in value) + "]"
            if len(one) + len(pad) <= INLINE_WIDTH:
                return one
        items = [inner + dump(v, level + 1) for v in value]
        return "[\n" + ",\n".join(items) + "\n" + pad + "]"
    if not value:
        return "{}"
    if all(is_scalar(v) or (isinstance(v, list) and all(is_scalar(x) for x in v)) for v in value.values()):
        one = (
            "{ "
            + ", ".join(f"{json.dumps(k, ensure_ascii=False)}: {dump(v)}" for k, v in value.items())
            + " }"
        )
        if len(one) + len(pad) <= INLINE_WIDTH:
            return one
    items = [f"{inner}{json.dumps(k, ensure_ascii=False)}: {dump(v, level + 1)}" for k, v in value.items()]
    return "{\n" + ",\n".join(items) + "\n" + pad + "}"


def load_db(path: Path) -> dict:
    try:
        db = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise FingerprintError(f"cannot read {path}: {error}") from error
    if not isinstance(db, dict) or db.get("version") != 1 or not isinstance(db.get("packages"), list):
        raise FingerprintError(f"{path} is not a version 1 arri-packages.json")
    return db


def save_db(path: Path, db: dict) -> None:
    text = dump(db) + "\n"
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(text, encoding="utf-8")
    temp.replace(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Write names, sizes, LUT sizes and SHA-256 of ARRI look packages into arri-packages.json."
    )
    parser.add_argument("zips", nargs="*", type=Path, metavar="ZIP", help="the downloaded package zips")
    parser.add_argument("--packages", type=Path, default=DEFAULT_PACKAGES, help="the list to update")
    parser.add_argument("--package", metavar="ID", help="the package id, when it cannot be told from the zip")
    parser.add_argument("--dry-run", action="store_true", help="show what would be written, change nothing")
    parser.add_argument("--reformat", action="store_true", help="rewrite the list in the tool's layout and stop")
    args = parser.parse_args(argv)
    if not args.zips and not args.reformat:
        parser.error("give at least one zip (or --reformat)")
    if args.package and len(args.zips) != 1:
        parser.error("--package needs exactly one zip")

    try:
        db = load_db(args.packages)
        seen: dict[str, str] = {}
        for zip_path in args.zips:
            scan = read_zip(zip_path)
            if not scan["files"]:
                left = ", ".join(f"{v} {k}" for k, v in scan["ignored"].items() if v)
                detail = f" (ignored: {left})" if left else ""
                raise FingerprintError(f"{zip_path.name}: no .cube, .alf4 or .aml files{detail}")
            package = choose_package(db, scan, args.package)
            if package["id"] in seen:
                raise FingerprintError(f"{zip_path.name} and {seen[package['id']]} are both {package['id']}")
            seen[package["id"]] = zip_path.name
            for warning in scan["warnings"]:
                print("warning:", warning, file=sys.stderr)
            for note in apply_scan(package, scan):
                print("note:", note, file=sys.stderr)
            print(
                f"{zip_path.name}: {package['id']}, {len(scan['files'])} files, "
                f"LUT sizes {sorted({c['lut_size'] for c in scan['files']})}"
            )
        if args.dry_run:
            print(dump({p["id"]: p.get("fingerprints", []) for p in db["packages"] if p["id"] in seen}))
            return 0
        save_db(args.packages, db)
        print(f"wrote {args.packages}")
    except FingerprintError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
