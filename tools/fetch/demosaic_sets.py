#!/usr/bin/env python3
"""The demosaic evaluation sets: Kodak (24 images) and McMaster (18 images), pinned by commit and SHA-256 (Python standard library only).

Use the wrapper ``demosaic-sets.sh``; it checks for python3 and runs this file in isolated mode.

    tools/fetch/demosaic-sets.sh              download (or verify) the files into testdata/demosaic/ and list them
    tools/fetch/demosaic-sets.sh --list       verify what is in testdata/demosaic/ against the table; no network
    tools/fetch/demosaic-sets.sh --force      also replace a file whose content differs from the pinned one
    tools/fetch/demosaic-sets.sh --help

These are the sets behind the demosaic PSNR of docs/05 (M1-12: Kodak 29.82 / 38.88 / 39.16 dB, McMaster 32.31 / 34.21 / 35.09 dB for
bilinear / Menon / LMMSE) and the demosaic gate of docs/13. The ignored test ``psnr_on_kodak_and_mcmaster`` in
crates/ym-gpu/tests/cpu_demosaic.rs reads ``testdata/demosaic/{kodak,mcmaster}``; run it with

    cargo test -p ym-gpu --test cpu_demosaic --release -- --ignored --nocapture

Where the files come from (``demosaic-sets.json`` is the table; the table, not the network, says what a file must be):

    kodak       cszn/FFDNet           testsets/Kodak24    24 PNG, 500 x 500    commit e787f46df2f3
    mcmaster    JingyunLiang/SwinIR   testsets/McMaster   18 TIFF, 500 x 500   commit 6545850fbf8d   (byte-identical to the copy in cszn/FFDNet)

Each file is downloaded from raw.githubusercontent.com at the pinned commit, and its size and SHA-256 are checked against the table
before it is written (to a ``.part`` file first, then renamed). A file that is already there with the pinned SHA-256 needs no network.
A file that differs (your own, or a different version) is never replaced without --force. Other files in the folder are never touched.

EVALUATION ONLY. Kodak's images are free to use (Eastman Kodak released them without restriction); the McMaster set is for academic
research. Neither repository records a licence for its copy. So the files are never committed (testdata/ is git-ignored), never shipped,
and no picture made from them goes into the repository (docs/validation-data.md, row 13).

Environment: YM_TESTDATA=/path moves testdata/. YM_DEMOSAIC_SETS=/path/to/table.json replaces the table (the unit test uses a folder read
through file://; same schema). Exit codes: 0 ok, 1 a file does not match the pin (or, with --list, is missing), 2 usage error or invalid
table, 3 nothing found (--list), 4 a host cannot be reached.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Tuple

EXIT_OK, EXIT_MISMATCH, EXIT_USAGE, EXIT_NOTHING, EXIT_NETWORK = 0, 1, 2, 3, 4

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DEFAULT_RAW_BASE = "https://raw.githubusercontent.com"


class TableError(Exception):
    """The table is not valid."""


class FetchError(Exception):
    """A download failed (the host, the network or a proxy refused it)."""


def load_table() -> Dict[str, Any]:
    """Reads and validates the table."""
    path = Path(os.environ.get("YM_DEMOSAIC_SETS") or HERE / "demosaic-sets.json")
    try:
        table = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise TableError(f"cannot read the table {path}: {error}") from error
    sources = table.get("sources")
    if not isinstance(sources, list) or not sources:
        raise TableError("the table has no sources")
    seen_ids = set()
    for source in sources:
        label = source.get("id", "?")
        for key in ("id", "repo", "commit", "dir", "dest", "licence_note"):
            if not isinstance(source.get(key), str) or not source[key]:
                raise TableError(f"source {label}: {key} is missing")
        if source["id"] in seen_ids:
            raise TableError(f"source {label}: the id is used twice")
        seen_ids.add(source["id"])
        if not re.fullmatch(r"[0-9a-f]{40}", source["commit"]):
            raise TableError(f"source {label}: commit must be a full 40 character hash")
        if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", source["repo"]):
            raise TableError(f"source {label}: repo must look like owner/name")
        if not re.fullmatch(r"[a-z0-9_-]+", source["dest"]):
            raise TableError(f"source {label}: dest must be a plain folder name")
        files = source.get("files")
        if not isinstance(files, list) or not files:
            raise TableError(f"source {label}: no files")
        names = set()
        for entry in files:
            name = entry.get("name")
            if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_. -]*", name) or ".." in name:
                raise TableError(f"source {label}: the file name {name!r} is not a plain name")
            if name in names:
                raise TableError(f"source {label}: {name} is listed twice")
            names.add(name)
            if not re.fullmatch(r"[0-9a-f]{64}", str(entry.get("sha256", ""))):
                raise TableError(f"source {label}: {name}: sha256 must be 64 hex digits")
            if not isinstance(entry.get("size"), int) or entry["size"] <= 0:
                raise TableError(f"source {label}: {name}: size must be a positive integer")
    return table


def sha256_of(data: bytes) -> str:
    """SHA-256 of ``data`` as hex."""
    return hashlib.sha256(data).hexdigest()


def download(url: str, timeout: float = 60.0, retries: int = 3) -> bytes:
    """Fetches ``url``. Network errors and 5xx answers are retried a few times; a 4xx answer or a missing file is not."""
    last: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as response:  # noqa: S310 (https, or file:// in the unit test)
                return response.read()
        except urllib.error.HTTPError as error:
            last = error
            if error.code < 500:
                break
        except (urllib.error.URLError, OSError, TimeoutError) as error:
            last = error
            if url.startswith("file:"):
                break
        time.sleep(0.5 * attempt)
    raise FetchError(f"{url}: {last}")


def file_url(raw_base: str, source: Dict[str, Any], name: str) -> str:
    """The raw URL of one file at the pinned commit (spaces and other odd characters percent-encoded)."""
    path = f"{source['repo']}/{source['commit']}/{source['dir']}/{name}"
    return raw_base.rstrip("/") + "/" + urllib.parse.quote(path)


def testdata_root() -> Path:
    """``testdata/`` of the checkout, or YM_TESTDATA."""
    return Path(os.environ.get("YM_TESTDATA") or REPO / "testdata")


def status_of(path: Path, entry: Dict[str, Any]) -> str:
    """``missing``, ``ok`` (the pinned content) or ``differs``."""
    if not path.is_file():
        return "missing"
    data = path.read_bytes()
    return "ok" if len(data) == entry["size"] and sha256_of(data) == entry["sha256"] else "differs"


def write_atomically(path: Path, data: bytes) -> None:
    """Writes via a ``.part`` file next to the target, so an interrupted run leaves no half file under the real name."""
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    try:
        part.write_bytes(data)
        os.replace(part, path)
    finally:
        if part.exists():
            part.unlink()


def notes(table: Dict[str, Any]) -> None:
    """Prints the licence note of every source: this tier is for evaluation only."""
    print("Evaluation only: never committed, never shipped (docs/validation-data.md, row 13).")
    for source in table["sources"]:
        print(f"  {source['id']}: {source['licence_note']}")


def list_sets(table: Dict[str, Any]) -> int:
    """``--list``: verify what is on disk against the table. No network."""
    root = testdata_root() / "demosaic"
    total_ok = 0
    problems = 0
    for source in table["sources"]:
        folder = root / source["dest"]
        ok = 0
        bad: List[str] = []
        for entry in source["files"]:
            state = status_of(folder / entry["name"], entry)
            if state == "ok":
                ok += 1
            else:
                bad.append(f"{entry['name']} ({'missing' if state == 'missing' else 'differs from the pinned file'})")
        total_ok += ok
        problems += len(bad)
        print(f"{source['id']}: {ok} of {len(source['files'])} files verified in {folder}")
        if bad:
            print("  not verified: " + ", ".join(bad))
    notes(table)
    if total_ok == 0:
        print("Nothing found. Run tools/fetch/demosaic-sets.sh to download the sets.")
        return EXIT_NOTHING
    return EXIT_MISMATCH if problems else EXIT_OK


def fetch_sets(table: Dict[str, Any], force: bool) -> int:
    """The default action: download what is missing and verify everything."""
    root = testdata_root() / "demosaic"
    raw_base = table.get("raw_base") or DEFAULT_RAW_BASE
    mismatches = 0
    network = 0
    for source in table["sources"]:
        folder = root / source["dest"]
        fresh = present = 0
        for entry in source["files"]:
            target = folder / entry["name"]
            state = status_of(target, entry)
            if state == "ok":
                present += 1
                continue
            if state == "differs" and not force:
                mismatches += 1
                print(
                    f"error: {target} is not the pinned file (size or SHA-256 differs); it is left alone. Use --force to replace it.",
                    file=sys.stderr,
                )
                continue
            url = file_url(raw_base, source, entry["name"])
            try:
                data = download(url)
            except FetchError as error:
                network += 1
                print(f"error: {source['id']}: cannot fetch {entry['name']}: {error}", file=sys.stderr)
                break  # the host (or the route to it) is not there: do not try the other files of this set
            if len(data) != entry["size"] or sha256_of(data) != entry["sha256"]:
                mismatches += 1
                print(
                    f"error: {source['id']}: {entry['name']} from {url} is not the pinned file "
                    f"(got {len(data)} bytes, SHA-256 {sha256_of(data)[:12]}...); nothing was written for it.",
                    file=sys.stderr,
                )
                continue
            write_atomically(target, data)
            fresh += 1
        have = sum(1 for e in source["files"] if status_of(folder / e["name"], e) == "ok")
        print(f"{source['id']}: {have} of {len(source['files'])} files verified in {folder} ({fresh} downloaded, {present} already there)")
    notes(table)
    if mismatches:
        return EXIT_MISMATCH
    return EXIT_NETWORK if network else EXIT_OK


def main(argv: List[str] | None = None) -> int:
    """Entry point; returns the exit code."""
    args = list(sys.argv[1:] if argv is None else argv)
    if any(a in ("-h", "--help") for a in args):
        print(__doc__)
        return EXIT_OK
    unknown = [a for a in args if a not in ("--list", "--force")]
    if unknown:
        print(f"error: unknown argument {unknown[0]} (see --help)", file=sys.stderr)
        return EXIT_USAGE
    try:
        table = load_table()
    except TableError as error:
        print(f"error: {error}", file=sys.stderr)
        return EXIT_USAGE
    if "--list" in args:
        return list_sets(table)
    return fetch_sets(table, force="--force" in args)


if __name__ == "__main__":
    sys.exit(main())
