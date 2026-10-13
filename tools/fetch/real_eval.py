#!/usr/bin/env python3
"""The "local-eval" tier: real Sony ARW files for evaluation on THIS machine only (Python standard library, git 2.19+).

Use the wrapper ``real-eval.sh``; it checks for python3 and runs this file in isolated mode.

    tools/fetch/real-eval.sh              fetch (or verify) the files into testdata/real/ and list them
    tools/fetch/real-eval.sh --list       list what is in testdata/real/files (model, size, source); no network
    tools/fetch/real-eval.sh --force      also replace a file in testdata/real/files whose content differs from the pinned one
    tools/fetch/real-eval.sh --help

Nine files from three public GitHub repositories, each pinned to one commit and each file to one git blob id (``SOURCES``
below; the table, not the network, says what a file must be):

    metadata-extractor-images   DSLR-A500, ILCE-5000, ILCE-7M3, ILCE-7M4, NEX-5N, NEX-7          (arw/, about 140 MB)
    OpenImageIO-images          DSLR-A300                                                         (raw/, 10 MB)
    immich-test-assets          ILCE-6300 (cRAW) and ILCE-7M2 (14-bit uncompressed)               (formats/raw/Sony/, 75 MB)

For each source the script makes a shallow, blob-less fetch of the pinned commit (the commit and its trees, about a megabyte),
reads the blob ids of the wanted paths from the tree, checks them against the table, and downloads exactly those blobs. No
working tree is checked out and the git folder is deleted afterwards. Layout, all git-ignored:

    testdata/real/<source>/<path>        the pinned files, at their path in the repository (and a PINNED.txt that says where from)
    testdata/real/files/<Model>.ARW      a link to each, named as the local-only tests expect (crates/ym-core/tests/real_files.rs,
                                         render_real.rs, export_real.rs; tools/dngfit/tests/real_files.rs; tools/eval; training/illum/
                                         smoke_real.py); ILCE-6300_craw.ARW and ILCE-7M2_unc14.ARW carry the variant in the name

LICENCES ARE UNCLEAR OR RESTRICTED. None of the three repositories states a licence for these particular files that this project can
rely on (the reasons are printed with every run and recorded in docs/validation-data.md). So this tier is for local evaluation
only: never committed (testdata/ is git-ignored), never used in CI (the script refuses to run there), never uploaded, cached or
redistributed, and no picture made from them goes into the repository (golden references, screenshots). They are not part of the CC0
corpus (tools/fetch/samples.sh) or the golden tiers 1 to 3 (tools/fetch/golden.sh), and no gate in docs/13 depends on them.

Idempotent: a file that is already there with the pinned blob id needs no network; a link that already points at the pinned file is
left alone; a file in testdata/real/files that is not the pinned one (your own, or a different version) is never replaced without
--force; a pinned file under testdata/real/<source>/ that was edited is reported and left alone (delete the folder to fetch it again).

Environment: YM_TESTDATA=/path moves testdata/. YM_REAL_EVAL_SOURCES=/path/to/sources.json replaces the table (the unit test uses
local repositories; same schema as SOURCES). Exit codes: 0 ok, 1 a file does not match the pin (or a link was left alone), 2 usage
error, the table is invalid, or running in CI, 3 nothing found (--list), 4 a source cannot be fetched (the network, or a proxy in
front of it, refuses it, or the pinned commit is gone). Needs git and python3.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import struct
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

EXIT_OK, EXIT_MISMATCH, EXIT_USAGE, EXIT_NOTHING, EXIT_NETWORK = 0, 1, 2, 3, 4

#: The pinned sources. ``files[].blob`` is the git blob id (SHA-1 of ``"blob <size>\0" + content``) of the file at ``path`` in
#: ``commit``; ``name`` is the link in testdata/real/files. ``licence_note`` is what the repository says, printed with every run.
SOURCES: List[Dict[str, Any]] = [
    {
        "id": "metadata-extractor-images",
        "repo": "https://github.com/drewnoakes/metadata-extractor-images.git",
        "commit": "651ad0e67aa8d43d358ad05f9bc07b52d8b9ac6e",  # 2026-02-02, "Add JPEG with extended XMP that triggers bug"
        "licence_note": (
            "its README says 'You are free to use these media files however you wish', but it records no author and no licence "
            "for the individual files, which third parties contributed"
        ),
        "files": [
            {"path": "arw/Sony DSLR-A500.arw", "name": "DSLR-A500.ARW", "blob": "71ffa9e514f863f604d755456852acfdca5669ea"},
            {"path": "arw/Sony ILCE-5000 (A5000).arw", "name": "ILCE-5000.ARW", "blob": "8babe211d667574f3716095413840d3ef5a455c7"},
            {"path": "arw/Sony ILCE-7M3 (A7M3).arw", "name": "ILCE-7M3.ARW", "blob": "b5ceb57c6394cb1fc9e8af2886c52150d3aff835"},
            {"path": "arw/Sony ILCE-7M4 (A7M4).ARW", "name": "ILCE-7M4.ARW", "blob": "74411b2131c5351413079a29d89d204d1bbda5c0"},
            {"path": "arw/Sony NEX-5N.arw", "name": "NEX-5N.ARW", "blob": "7a3426e846b822363336fbf748aebf531d4acca5"},
            {"path": "arw/Sony NEX-7.arw", "name": "NEX-7.ARW", "blob": "694f73bb8fe560328815d7a1dc38b2d7389f0101"},
        ],
    },
    {
        "id": "OpenImageIO-images",
        "repo": "https://github.com/AcademySoftwareFoundation/OpenImageIO-images.git",
        "commit": "d13e98f92b4e75ae7cd949cc5aa1df219560fb93",  # 2026-08-25, "ktx2: add initial test files (#14)"
        "licence_note": (
            "THIRD-PARTY.md records CC BY-NC-SA 4.0 for raw/ (from rawsamples.ch, whose own text says only 'Creative Common "
            "License'); NonCommercial and ShareAlike terms, so not for redistribution here"
        ),
        "files": [
            {"path": "raw/RAW_SONY_A300.ARW", "name": "DSLR-A300.ARW", "blob": "423d5d6898043fcbb3b30472178c737c4f8cad91"},
        ],
    },
    {
        "id": "immich-test-assets",
        "repo": "https://github.com/immich-app/test-assets.git",
        "commit": "ec56cc6c1d76c45bc9615dc6ea232470a2b0a3a1",  # 2026-06-03, "chore: modify .github/FUNDING.yml"
        "licence_note": (
            "its README says 'All assets are public domain', but the repository's LICENSE is AGPL-3.0 and no author or origin is "
            "recorded for these two files (one commit, 'add formats'): a claim nobody has verified"
        ),
        "files": [
            {
                "path": "formats/raw/Sony/ILCE-6300/12bit-compressed-(3_2).arw",
                "name": "ILCE-6300_craw.ARW",
                "blob": "d5735732a20ae7eb32aac35a5c6ee62aa00de4cd",
            },
            {
                "path": "formats/raw/Sony/ILCE-7M2/14bit-uncompressed-(3_2).arw",
                "name": "ILCE-7M2_unc14.ARW",
                "blob": "130eebcdbdeca44417137ec4ea3e286b3bc5514e",
            },
        ],
    },
]

_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]*\.ARW$")
_SOURCE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_HEX40_RE = re.compile(r"^[0-9a-f]{40}$")


class UsageError(Exception):
    """A bad option or an invalid source table (exit code 2)."""


def say(msg: str) -> None:
    print(msg, flush=True)


def err(msg: str) -> None:
    print(f"real-eval.sh: {msg}", file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------------------------------------------------
# The table
# ---------------------------------------------------------------------------------------------------------------------


def load_sources() -> List[Dict[str, Any]]:
    """The source table: ``SOURCES``, or the JSON file named by ``YM_REAL_EVAL_SOURCES`` (checked either way)."""
    override = os.environ.get("YM_REAL_EVAL_SOURCES")
    if override:
        try:
            sources = json.loads(Path(override).read_text())
        except (OSError, ValueError) as e:
            raise UsageError(f"cannot read the source table {override}: {e}") from e
    else:
        sources = SOURCES
    validate_sources(sources)
    return sources


def validate_sources(sources: Any) -> None:
    """Refuses a table that could write outside testdata/ or name two files alike."""
    if not isinstance(sources, list) or not sources:
        raise UsageError("the source table must be a non-empty list")
    ids, names, blobs = set(), set(), set()
    for s in sources:
        if not isinstance(s, dict):
            raise UsageError("a source must be an object")
        sid = s.get("id")
        if not isinstance(sid, str) or not _SOURCE_ID_RE.match(sid) or sid in ids or sid == "files":
            raise UsageError(f"bad or repeated source id {sid!r}")
        ids.add(sid)
        for key in ("repo", "commit", "licence_note"):
            if not isinstance(s.get(key), str) or not s[key]:
                raise UsageError(f"source {sid}: {key} is missing")
        if not _HEX40_RE.match(s["commit"]):
            raise UsageError(f"source {sid}: commit must be a full 40-digit SHA-1")
        files = s.get("files")
        if not isinstance(files, list) or not files:
            raise UsageError(f"source {sid}: files must be a non-empty list")
        for f in files:
            if not isinstance(f, dict) or not all(isinstance(f.get(k), str) for k in ("path", "name", "blob")):
                raise UsageError(f"source {sid}: a file needs path, name and blob")
            parts = f["path"].split("/")
            if f["path"].startswith("/") or any(p in ("", ".", "..") for p in parts) or "\0" in f["path"] or f["path"].startswith("-"):
                raise UsageError(f"source {sid}: unsafe path {f['path']!r}")
            if not _NAME_RE.match(f["name"]) or f["name"] in names:
                raise UsageError(f"source {sid}: bad or repeated link name {f['name']!r}")
            if not _HEX40_RE.match(f["blob"]) or f["blob"] in blobs:
                raise UsageError(f"source {sid}: bad or repeated blob id for {f['path']}")
            names.add(f["name"])
            blobs.add(f["blob"])


# ---------------------------------------------------------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------------------------------------------------------


def blob_id(path: Path) -> str:
    """The git blob id (SHA-1 of "blob <size>\\0" + content) of a file: what the pinned commit lists for it."""
    h = hashlib.sha1()  # noqa: S324 - not security: git's own object id
    h.update(b"blob %d\0" % path.stat().st_size)
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def tiff_model(path: Path) -> Optional[str]:
    """The Model tag (0x0110) of IFD0 of a TIFF-based file (ARW is one), or None."""
    try:
        with path.open("rb") as f:
            head = f.read(8)
            if len(head) < 8 or head[:2] not in (b"II", b"MM"):
                return None
            e = "<" if head[:2] == b"II" else ">"
            if struct.unpack(e + "H", head[2:4])[0] != 42:
                return None
            f.seek(struct.unpack(e + "I", head[4:8])[0])
            (n,) = struct.unpack(e + "H", f.read(2))
            for _ in range(min(n, 1000)):
                ent = f.read(12)
                if len(ent) < 12:
                    return None
                tag, typ, cnt = struct.unpack(e + "HHI", ent[:8])
                if tag == 0x0110 and typ == 2 and 0 < cnt < 256:
                    if cnt <= 4:
                        raw = ent[8 : 8 + cnt]
                    else:
                        here = f.tell()
                        f.seek(struct.unpack(e + "I", ent[8:12])[0])
                        raw = f.read(cnt)
                        f.seek(here)
                    return raw.split(b"\0")[0].decode("ascii", "replace").strip() or None
    except (OSError, struct.error):
        return None
    return None


def is_arw(path: Path) -> bool:
    return path.name.lower().endswith(".arw")


# ---------------------------------------------------------------------------------------------------------------------
# Git
# ---------------------------------------------------------------------------------------------------------------------


def git(co: Path, *args: str, check: bool = True, timeout: int = 1800, binary: bool = False) -> subprocess.CompletedProcess:
    env = dict(os.environ, GIT_TERMINAL_PROMPT="0")
    return subprocess.run(
        ["git", "-C", str(co), *args],
        env=env,
        check=check,
        capture_output=True,
        text=not binary,
        timeout=timeout,
    )


class Unreachable(Exception):
    """git could not get the commit or a blob from the source (exit code 4)."""


class Mismatch(Exception):
    """What the source holds is not what the table pins (exit code 1)."""


def fetch_files(src: Dict[str, Any], co: Path, todo: List[Dict[str, str]]) -> None:
    """Fetches the pinned commit without blobs, then each wanted blob; writes the files under ``co`` (never half written)."""
    repo, commit = src["repo"], src["commit"]
    shutil.rmtree(co / ".git", ignore_errors=True)
    co.mkdir(parents=True, exist_ok=True)
    try:
        git(co, "init", "-q")
        git(co, "remote", "add", "origin", repo)
        git(co, "config", "remote.origin.promisor", "true")
        git(co, "config", "remote.origin.partialclonefilter", "blob:none")
        try:
            git(co, "fetch", "-q", "--depth", "1", "--filter=blob:none", "origin", commit)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as e:
            raise Unreachable(f"cannot fetch {repo} @ {commit[:12]}: the network (or a proxy in front of it) refuses it, or the pinned commit is gone") from e
        got = git(co, "rev-parse", "FETCH_HEAD").stdout.strip()
        if got != commit:
            raise Mismatch(f"unexpected commit {got} from {repo} (pinned: {commit})")
        for f in todo:
            listed = git(co, "rev-parse", "--verify", "-q", f"FETCH_HEAD:{f['path']}", check=False).stdout.strip()
            if listed != f["blob"]:
                raise Mismatch(
                    f"{f['path']} in {repo} @ {commit[:12]} is blob {listed or '(missing)'}, the table pins {f['blob']}"
                )
            target = co / f["path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            part = target.with_name("." + target.name + ".part")
            try:
                with part.open("wb") as out:
                    p = subprocess.run(
                        ["git", "-C", str(co), "cat-file", "blob", f["blob"]],
                        env=dict(os.environ, GIT_TERMINAL_PROMPT="0"),
                        stdout=out,
                        stderr=subprocess.PIPE,
                        timeout=1800,
                    )
                if p.returncode != 0:
                    why = (p.stderr.decode("utf-8", "replace").strip().splitlines() or ["git failed"])[-1]
                    raise Unreachable(f"cannot download {f['path']} from {repo}: {why}")
                if blob_id(part) != f["blob"]:
                    raise Mismatch(f"the download of {f['path']} is not blob {f['blob']}")
                os.replace(part, target)
            except subprocess.TimeoutExpired as e:
                raise Unreachable(f"the download of {f['path']} from {repo} timed out") from e
            finally:
                if part.exists():
                    part.unlink()
    finally:
        shutil.rmtree(co / ".git", ignore_errors=True)


def write_pinned_note(src: Dict[str, Any], co: Path) -> None:
    (co / "PINNED.txt").write_text(
        f"Local evaluation only. Fetched by tools/fetch/real-eval.sh; never commit, upload, cache or redistribute.\n"
        f"repository: {src['repo']}\ncommit:     {src['commit']}\nlicence:    {src['licence_note']}\n"
        + "".join(f"file:       {f['path']}  (blob {f['blob']}, linked as {f['name']})\n" for f in src["files"])
    )


def ensure_source(src: Dict[str, Any], real: Path) -> int:
    """Makes sure every file of the source is under ``real/<id>/`` with its pinned blob id. Returns an exit code."""
    co = real / src["id"]
    todo: List[Dict[str, str]] = []
    code = EXIT_OK
    for f in src["files"]:
        p = co / f["path"]
        if p.is_file():
            if blob_id(p) != f["blob"]:
                err(f"{p} is not the pinned file (blob {f['blob']}): it has local changes; delete {co} and run again")
                code = EXIT_MISMATCH
        else:
            todo.append(f)
    if code != EXIT_OK:
        return code
    if not todo:
        n = len(src["files"])
        say(f"real-eval.sh: {src['id']}: already fetched ({n} file{'s' if n != 1 else ''} at {src['commit'][:12]})")
        return EXIT_OK
    say(f"real-eval.sh: fetching {len(todo)} file(s) of {src['repo']} @ {src['commit'][:12]} (shallow, blob-less, only the pinned blobs)")
    existed = co.exists()
    try:
        fetch_files(src, co, todo)
    except Unreachable as e:
        err(str(e))
        _tidy(co, existed)
        return EXIT_NETWORK
    except Mismatch as e:
        err(str(e))
        _tidy(co, existed)
        return EXIT_MISMATCH
    except (OSError, subprocess.SubprocessError) as e:
        err(f"{src['id']}: {e}")
        _tidy(co, existed)
        return EXIT_NETWORK
    write_pinned_note(src, co)
    return EXIT_OK


def _tidy(co: Path, existed: bool) -> None:
    """After a failure: no git folder, and no folder at all when this run created it and nothing is in it."""
    shutil.rmtree(co / ".git", ignore_errors=True)
    if not existed:
        try:
            for p in sorted(co.rglob("*"), reverse=True):
                if p.is_dir():
                    p.rmdir()
            co.rmdir()
        except OSError:
            pass


# ---------------------------------------------------------------------------------------------------------------------
# Links and the listing
# ---------------------------------------------------------------------------------------------------------------------


def link_file(source: Path, target: Path, blob: str, force: bool) -> bool:
    """Links ``target`` to ``source``. Returns True for a problem (a file left alone)."""
    if os.path.lexists(target):
        if target.is_file() and blob_id(target) == blob:
            return False  # already the pinned file (a link to it, or an identical copy someone made earlier)
        if target.is_symlink() and not target.exists():
            target.unlink()  # a link whose file is gone (the checkout was deleted): ours, make it again
        elif target.is_dir() and not target.is_symlink():
            err(f"{target} is a folder; left alone")
            return True
        elif not force:
            err(f"{target} exists and is not the pinned file; left alone (--force replaces it)")
            return True
        else:
            target.unlink()
    os.symlink(os.path.relpath(source, target.parent), target)
    return False


def list_files(files_dir: Path, sources: List[Dict[str, Any]]) -> List[Tuple[str, int, str, str]]:
    known = {f["blob"]: s["id"] for s in sources for f in s["files"]}
    rows = []
    if files_dir.is_dir():
        for p in sorted(files_dir.iterdir()):
            if is_arw(p) and p.is_file():
                bid = blob_id(p)
                where = f"{known[bid]} (pinned)" if bid in known else "not from this script"
                rows.append((tiff_model(p) or "?", p.stat().st_size, p.name, where))
    return rows


def print_listing(files_dir: Path, rows: List[Tuple[str, int, str, str]]) -> None:
    say(f"{files_dir}  (local evaluation only)")
    if not rows:
        say("  no ARW files")
        return
    say(f"  {'model':<14}{'size':>10}  {'file':<22}{'source'}")
    for model, size, name, where in rows:
        say(f"  {model:<14}{size / 1e6:>7.1f} MB  {name:<22}{where}")
    say(f"  {len(rows)} files, {sum(r[1] for r in rows) / 1e6:.1f} MB")


def warn_licence(sources: List[Dict[str, Any]]) -> None:
    lines = [
        "",
        "  WARNING: LOCAL EVALUATION ONLY. THE LICENCE OF THESE FILES IS UNCLEAR OR RESTRICTED.",
        "  Use them on this machine to look at real Sony files, and nowhere else:",
        "    - never commit them or copy them into the repository (testdata/ is git-ignored);",
        "    - never use them in CI, and never upload or cache them there;",
        "    - never redistribute them, or pictures made from them (golden references, screenshots, bug-report attachments).",
        "  Gates and golden tests must not depend on them. Where each set comes from:",
    ]
    for s in sources:
        n = len(s["files"])
        lines.append(f"    - {s['id']} ({n} file{'s' if n != 1 else ''}): {s['licence_note']}.")
    lines.append("")
    print("\n".join(lines), file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------------------------------------------------


def main(argv: List[str]) -> int:
    listing, force = False, False
    for arg in argv:
        if arg == "--list":
            listing = True
        elif arg == "--force":
            force = True
        elif arg in ("-h", "--help"):
            print((__doc__ or "").strip("\n"))
            return EXIT_OK
        else:
            err(f"unknown option {arg} (see --help)")
            return EXIT_USAGE

    # Never in CI: a fork or a cache would republish files whose licence nobody has checked.
    if os.environ.get("CI") == "true" or os.environ.get("GITHUB_ACTIONS") == "true":
        err("refusing to run in CI. These files have an unclear licence and are for local evaluation only.")
        return EXIT_USAGE

    try:
        sources = load_sources()
    except UsageError as e:
        err(str(e))
        return EXIT_USAGE

    repo_root = Path(__file__).resolve().parent.parent.parent
    testdata = Path(os.environ.get("YM_TESTDATA") or repo_root / "testdata")
    real = testdata / "real"
    files_dir = real / "files"

    if listing:
        warn_licence(sources)
        rows = list_files(files_dir, sources)
        print_listing(files_dir, rows)
        if not rows:
            err("nothing found; run tools/fetch/real-eval.sh to fetch the files")
            return EXIT_NOTHING
        return EXIT_OK

    if shutil.which("git") is None:
        err("git is required")
        return 127
    warn_licence(sources)

    worst = EXIT_OK
    ready: List[Dict[str, Any]] = []
    for s in sources:
        code = ensure_source(s, real)
        if code == EXIT_OK:
            ready.append(s)
        elif code == EXIT_MISMATCH:
            worst = EXIT_MISMATCH  # a file that is not the pinned one outranks a source that could not be reached
        elif worst == EXIT_OK:
            worst = code

    problems = 0
    if ready:
        files_dir.mkdir(parents=True, exist_ok=True)
        for s in ready:
            for f in s["files"]:
                if link_file(real / s["id"] / f["path"], files_dir / f["name"], f["blob"], force):
                    problems += 1

    print_listing(files_dir, list_files(files_dir, sources))
    err("reminder: local evaluation only, the licences of these files are unclear or restricted; never commit, upload or use them in CI")
    if worst != EXIT_OK:
        return worst
    return EXIT_MISMATCH if problems else EXIT_OK


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
