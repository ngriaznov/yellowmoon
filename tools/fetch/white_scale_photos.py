#!/usr/bin/env python3
"""The Play Raw files behind the white scales of docs/03, pinned by SHA-256 (Python standard library only).

Use the wrapper ``white-scale-photos.sh``; it checks for python3 and runs this file in isolated mode.

    tools/fetch/white-scale-photos.sh              download (or verify) the files into testdata/white-scale/play-raw/ and list them
    tools/fetch/white-scale-photos.sh --list       verify what is in testdata/white-scale/play-raw/ against the table; no network
    tools/fetch/white-scale-photos.sh --force      also replace a file whose content differs from the pinned one
    tools/fetch/white-scale-photos.sh --help

A stand-in body (a Sony body without curves of its own) renders through another body's profile, and its white scale is fitted on the
white-balance presets its own files carry (``docs/03``, "White scale of stand-in bodies"; ``training/white_scale/fit.py``). The owner's
A1 photos (``owner-photos.sh``) and the local-eval files (``real-eval.sh``) cover some bodies. This set covers the others: 17 ARW files
that photographers shared in the Play Raw category of discuss.pixls.us, one to four per body (A7, A7R, A7S, A6700, NEX-5R, ZV-E10, RX100,
RX100 II, RX100 VII, RX10 IV, and the A1 file that the A1's consistency check leaves out). ``white-scale-photos.json`` is the table: the
name, body, size, SHA-256, download address, post, poster and licence of every file. The table, not the network, says what a file must be.

Each file is downloaded from the address the post's attachment link leads to on the forum's file store (the last part of that address is
the SHA-1 of the file), and its size and SHA-256 are checked against the table before it is written (to a ``.part`` file first, then
renamed). A file that is already there with the pinned SHA-256 needs no network. A file that differs (your own, or a different version)
is never replaced without --force. Other files in the folder are never touched.

EVALUATION ONLY. The posts state "Creative Commons, By-Attribution, Share-Alike" (the Play Raw template; no version number) or, for one
file, "Public Domain". The files are fetched on demand for the white-scale fit, never committed (testdata/ is git-ignored), never shipped
and never redistributed; the scales take only numbers from them, the presets the camera wrote (docs/14 rule 8).

Environment: YM_TESTDATA=/path moves testdata/. YM_WHITE_SCALE_PHOTOS=/path/to/table.json replaces the table (the unit test uses a folder
read through file://; same schema). Exit codes: 0 ok, 1 a file does not match the pin (or, with --list, is missing), 2 usage error or
invalid table, 3 nothing found (--list), 4 a host cannot be reached.
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
from typing import Any, Dict, List

EXIT_OK, EXIT_MISMATCH, EXIT_USAGE, EXIT_NOTHING, EXIT_NETWORK = 0, 1, 2, 3, 4

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
USER_AGENT = "yellowmoon-fetch/0.1.0 (+https://github.com/ngriaznov/yellowmoon; test-data download, one connection at a time)"
NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+-]*\.ARW")
#: The environment variable that swaps the table also lets its files be read from a folder (``file://``): only the unit test sets it.
TABLE_ENV = "YM_WHITE_SCALE_PHOTOS"


class TableError(Exception):
    """The table is not valid."""


class FetchError(Exception):
    """A download failed (the host, the network or a proxy refused it)."""


def table_path() -> Path:
    """The table: ``white-scale-photos.json`` next to this file, or ``$YM_WHITE_SCALE_PHOTOS``."""
    return Path(os.environ.get(TABLE_ENV) or HERE / "white-scale-photos.json")


def load_table() -> Dict[str, Any]:
    """Reads and validates the table."""
    path = table_path()
    try:
        table = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise TableError(f"cannot read the table {path}: {error}") from error
    hosts = table.get("hosts")
    if not isinstance(hosts, list) or not hosts or not all(isinstance(h, str) and h for h in hosts):
        raise TableError("the table has no hosts")
    files = table.get("files")
    if not isinstance(files, list) or not files:
        raise TableError("the table has no files")
    seen = set()
    for entry in files:
        name = entry.get("name")
        if not isinstance(name, str) or not NAME_RE.fullmatch(name):
            raise TableError(f"the file name {name!r} is not a plain .ARW name")
        if name in seen:
            raise TableError(f"{name} is listed twice")
        seen.add(name)
        for key in ("model", "url", "topic_url", "author", "licence"):
            if not isinstance(entry.get(key), str) or not entry[key]:
                raise TableError(f"{name}: {key} is missing")
        if not re.fullmatch(r"[0-9a-f]{64}", str(entry.get("sha256", ""))):
            raise TableError(f"{name}: sha256 must be 64 hex digits")
        if not isinstance(entry.get("size"), int) or entry["size"] <= 0:
            raise TableError(f"{name}: size must be a positive integer")
        parts = urllib.parse.urlsplit(entry["url"])
        if parts.scheme == "file" and TABLE_ENV in os.environ:
            continue
        if parts.scheme != "https" or (parts.hostname or "").lower() not in {h.lower() for h in hosts}:
            raise TableError(f"{name}: the address must be https on one of the table's hosts ({', '.join(hosts)})")
    return table


def sha256_of(data: bytes) -> str:
    """SHA-256 of ``data`` as hex."""
    return hashlib.sha256(data).hexdigest()


def download(url: str, timeout: float = 120.0, retries: int = 3) -> bytes:
    """Fetches ``url``. Network errors and 5xx answers are retried a few times; a 4xx answer or a missing file is not."""
    last: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 (https on the table's hosts, or file:// in the unit test)
                return response.read()
        except urllib.error.HTTPError as error:
            last = error
            if error.code < 500:
                break
        except (urllib.error.URLError, OSError, TimeoutError) as error:
            last = error
            if url.startswith("file:"):
                break
        time.sleep(1.0 * attempt)
    raise FetchError(f"{url}: {last}")


def testdata_root() -> Path:
    """``testdata/`` of the checkout, or YM_TESTDATA."""
    return Path(os.environ.get("YM_TESTDATA") or REPO / "testdata")


def folder() -> Path:
    """Where the files go: ``testdata/white-scale/play-raw``."""
    return testdata_root() / "white-scale" / "play-raw"


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


def note() -> None:
    """Prints the licence note: this set is for evaluation only."""
    print("Evaluation only: never committed, never shipped or redistributed. The posts state CC BY-SA (no version) or Public Domain; see white-scale-photos.json.")


def list_files(table: Dict[str, Any]) -> int:
    """``--list``: verify what is on disk against the table. No network."""
    root = folder()
    bad: List[str] = []
    ok = 0
    for entry in table["files"]:
        state = status_of(root / entry["name"], entry)
        if state == "ok":
            ok += 1
        else:
            bad.append(f"{entry['name']} ({'missing' if state == 'missing' else 'differs from the pinned file'})")
    print(f"{ok} of {len(table['files'])} files verified in {root}")
    if bad:
        print("  not verified: " + ", ".join(bad))
    note()
    if ok == 0:
        print("Nothing found. Run tools/fetch/white-scale-photos.sh to download the files.")
        return EXIT_NOTHING
    return EXIT_MISMATCH if bad else EXIT_OK


def fetch_files(table: Dict[str, Any], force: bool) -> int:
    """The default action: download what is missing and verify everything."""
    root = folder()
    mismatches = network = fresh = present = 0
    for entry in table["files"]:
        target = root / entry["name"]
        state = status_of(target, entry)
        if state == "ok":
            present += 1
            continue
        if state == "differs" and not force:
            mismatches += 1
            print(f"error: {target} is not the pinned file (size or SHA-256 differs); it is left alone. Use --force to replace it.", file=sys.stderr)
            continue
        try:
            data = download(entry["url"])
        except FetchError as error:
            network += 1
            print(f"error: cannot fetch {entry['name']}: {error}", file=sys.stderr)
            break  # the host (or the route to it) is not there: do not try the other files
        if len(data) != entry["size"] or sha256_of(data) != entry["sha256"]:
            mismatches += 1
            print(
                f"error: {entry['name']} from {entry['url']} is not the pinned file "
                f"(got {len(data)} bytes, SHA-256 {sha256_of(data)[:12]}...); nothing was written for it.",
                file=sys.stderr,
            )
            continue
        write_atomically(target, data)
        fresh += 1
    have = sum(1 for e in table["files"] if status_of(root / e["name"], e) == "ok")
    print(f"{have} of {len(table['files'])} files verified in {root} ({fresh} downloaded, {present} already there)")
    note()
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
        return list_files(table)
    return fetch_files(table, force="--force" in args)


if __name__ == "__main__":
    sys.exit(main())
