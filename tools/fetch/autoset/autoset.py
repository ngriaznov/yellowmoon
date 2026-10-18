#!/usr/bin/env python3
"""Yellowmoon autoset dataset builder: CC0 Sony ARW files, streamed (Python standard library only).

Do not run this file directly; use the wrapper ``tools/fetch/autoset.sh``, which runs it with
``python3 -I``. The spec is ``docs/18-auto-settings.md`` section 6 (data and the row schema); the
usage is in ``tools/fetch/autoset/README.md``.

What it does
------------
* ``refresh``: query raw.pixls.us (one JSON listing) and RawDB (the paged set index, then one request
  per Sony set) and rewrite ``manifest.json``: one entry per distinct CC0 Sony ARW file, with its
  SHA-256, licence and where it came from. A file that both sources list is one entry; the other
  source's id is kept under ``also``.
* ``build``: for each manifest entry, download the file to ``<dest>/autoset/tmp/<sha256>.ARW.part``,
  check its SHA-256 (and its exact size, where the manifest has one), run the dump program on it,
  check the rows it prints, append them to the rows file, and delete the file. Resumable: an entry
  whose ``aug_k = 0`` row is already in the rows file is skipped, and an entry that failed is skipped
  until ``--retry-failed``.
* ``list``: one line per manifest entry.
* ``verify-rows``: check a rows file against the manifest (no network).

Rules the code enforces
-----------------------
* Licence CC0-1.0 only (checked per file, for both sources), extension ARW only.
* Raw files are never committed. They live in ``<dest>/autoset/tmp`` (under testdata/, git-ignored)
  and are deleted after use, also after a failure, unless ``--keep``.
* Downloads and redirects only over https (plain http only to loopback, for the tests), and only to
  the two source hosts or the storage domain RawDB redirects its downloads to.
* Polite: one request at a time, a pause between requests, an identified User-Agent, three tries with
  backoff on HTTP 429/5xx and on dropped or timed-out transfers.
* Rows are appended in one write, and only after every row of the file has passed validation.

Exit codes: 0 ok, 1 failures (or a manifest that breaks the rules), 2 usage error or missing tool,
3 nothing to do (empty manifest, or nothing matched), 4 network unreachable or refused, 5 not enough
disk, 130 interrupted.
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import errno
import hashlib
import html
import http.client
import json
import math
import os
import re
import shlex
import shutil
import socket
import ssl
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

VERSION = "0.1.0"
SCHEMA_VERSION = 1
USER_AGENT = "yellowmoon-autoset-fetch (+https://github.com/ngriaznov/yellowmoon)"

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_USAGE = 2
EXIT_EMPTY = 3
EXIT_NETWORK = 4
EXIT_DISK = 5
EXIT_INTERRUPTED = 130

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent.parent
DEFAULT_MANIFEST = HERE / "manifest.json"
DEFAULT_DUMP = REPO_ROOT / "target" / "release" / "examples" / "autoset_dump"
DUMP_BUILD_COMMAND = ["cargo", "build", "-p", "ym-core", "--release", "--example", "autoset_dump"]

#: The licence written into the manifest and the rows (docs/18 §6).
LICENCE = "CC0-1.0"
#: How RawDB writes the same licence.
RAWDB_LICENCE = "CC0 1.0"
PIXLS = "raw.pixls.us"
RAWDB = "rawdb.dnglab.org"
#: Source host name -> manifest id prefix.
SOURCE_KEY = {PIXLS: "pixls", RAWDB: "rawdb"}
SOURCE_HOSTS = frozenset({PIXLS, RAWDB})
#: RawDB answers a download with a redirect to signed object storage on this domain.
STORAGE_DOMAIN = "your-objectstorage.com"
LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})
DEFAULT_SOURCES = {
    "pixls": {"listing": "https://raw.pixls.us/json/getrepository.php?set=all"},
    "rawdb": {"api": "https://rawdb.dnglab.org/api/sets", "download": "https://rawdb.dnglab.org/api/download"},
}
DESCRIPTION = "CC0 Sony ARW files for the autoset training set (docs/18 §6). Raw files are never committed."
ENTRY_KEYS = ("id", "source", "body", "name", "mode", "url", "sha256", "size", "size_approx", "licence", "also")
FEATURES = 39
CHUNK = 1 << 20
#: Attempts per request (the first try included) and the first backoff in seconds (then doubled).
TRIES = 3
BACKOFF_S = 2.0
MAX_JSON_BYTES = 64 << 20
#: Free space that must remain after the download: 1 GiB, plus twice the file size (docs: brief AS-06).
DISK_MARGIN = 1 << 30
#: Assumed size for an entry with neither an exact nor an approximate size (never the case in the shipped
#: manifest; the largest CC0 Sony ARW listed is about 135 MB).
UNKNOWN_SIZE = 256 << 20
DUMP_TIMEOUT_S = 3600
HEX64 = re.compile(r"^[0-9a-f]{64}$")
TRANSIENT_CODES = frozenset({429, 500, 502, 503, 504})
TRANSIENT_ERRORS = (
    socket.timeout,
    TimeoutError,
    ConnectionResetError,
    ConnectionAbortedError,
    BrokenPipeError,
    http.client.HTTPException,
)
PIXLS_LICENCE_HREF = re.compile(r"""href=['"]([^'"]+)['"]""")
PIXLS_FILE_HTML = re.compile(
    r"""^<a href=['"]([^'"]+)['"]>([^<]*)</a><div class=['"]checksumdata['"]>"""
    r"""<span title=['"]sha256 Checksum['"]>([0-9A-Fa-f]{64})</span>&nbsp;\(([^)]*)\)</div>$"""
)
SIZE_APPROX = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([KMG]i?B)\s*$", re.IGNORECASE)
SIZE_UNITS = {"KB": 1 << 10, "MB": 1 << 20, "GB": 1 << 30, "KIB": 1 << 10, "MIB": 1 << 20, "GIB": 1 << 30}


# --------------------------------------------------------------------------------------------------
# Errors and messages


class AutosetError(Exception):
    """A stop with a message for the user; ``exit_code`` is the process status."""

    exit_code = EXIT_FAILED


class UsageError(AutosetError):
    """A bad command line or a bad environment setting."""

    exit_code = EXIT_USAGE


class SetupError(AutosetError):
    """A tool the build needs (the dump program, cargo) is missing or broken."""

    exit_code = EXIT_USAGE


class EmptyError(AutosetError):
    """The manifest is missing or empty, or the filters match nothing."""

    exit_code = EXIT_EMPTY


class ManifestError(AutosetError):
    """The manifest, or a listing answer, is malformed or breaks the data policy."""


class PolicyError(AutosetError):
    """A URL or a redirect points somewhere this tool may not go."""


class NetworkError(AutosetError):
    """A host cannot be reached (DNS, refused, proxy refusal, TLS). The run stops; nothing is recorded."""

    exit_code = EXIT_NETWORK


class DiskError(AutosetError):
    """Not enough free disk space for the next download."""

    exit_code = EXIT_DISK


class HttpStatusError(AutosetError):
    """The server answered with an error status that is not worth retrying."""

    def __init__(self, url: str, status: int, reason: str = "") -> None:
        super().__init__(f"{url}: HTTP {status} {reason}".strip())
        self.status = status


class TransientError(AutosetError):
    """A retryable failure: HTTP 429 or 5xx, or a transfer that dropped or timed out."""

    def __init__(self, message: str, *, status: Optional[int] = None, retry_after: Optional[float] = None) -> None:
        super().__init__(message)
        self.status = status
        self.retry_after = retry_after


class FileFailure(Exception):
    """One file failed at ``stage`` (download, verify, dump or rows). It is recorded and the run goes on."""

    def __init__(self, stage: str, message: str) -> None:
        super().__init__(message)
        self.stage = stage


def log(message: str = "") -> None:
    """Progress and diagnostics go to stderr; ``list`` and ``verify-rows`` print their reports to stdout."""
    print(message, file=sys.stderr, flush=True)


def network_help(err: NetworkError) -> str:
    return (
        f"error: {err}\n"
        "  The network, or a proxy in front of it, refuses this host. raw.pixls.us, rawdb.dnglab.org and the\n"
        "  RawDB storage host are blocked in some sandboxes and CI egress policies.\n"
        "  What to do: run this on a machine with open internet access."
    )


# --------------------------------------------------------------------------------------------------
# Small helpers


def key_of_id(identifier: str) -> str:
    """The source key of a manifest id: ``pixls`` or ``rawdb``."""
    return identifier.split(":", 1)[0]


def is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def is_finite_number(value: Any) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool)) and math.isfinite(value)


def is_zero(value: Any) -> bool:
    return is_finite_number(value) and value == 0


def features_ok(value: Any) -> bool:
    return isinstance(value, list) and len(value) == FEATURES and all(is_finite_number(x) for x in value)


def parse_size_approx(text: Any) -> Optional[int]:
    """Bytes from a listing's size text such as ``10.55MB``. Units are binary (MB = 1024 * 1024 bytes),
    which is the larger reading and so the safe one for the disk check."""
    if not isinstance(text, str):
        return None
    m = SIZE_APPROX.match(text)
    if not m:
        return None
    return int(float(m.group(1)) * SIZE_UNITS[m.group(2).upper()])


def fmt_gib(n: float) -> str:
    return f"{n / (1 << 30):.2f}"


def fmt_size(entry: Dict[str, Any]) -> str:
    if entry.get("size") is not None:
        return f"{entry['size'] / (1 << 20):.2f}MB"
    return entry.get("size_approx") or "?"


def effective_size(entry: Dict[str, Any]) -> int:
    return entry["size"] if entry.get("size") is not None else (parse_size_approx(entry.get("size_approx")) or 0)


def quote_segment(text: str) -> str:
    """Percent-encode one URL path segment the way JavaScript's encodeURIComponent does."""
    return urllib.parse.quote(text, safe="!*'()")


def quote_url_path(url: str) -> str:
    """Percent-encode the path of a URL (its spaces and other unsafe characters); keeps the host."""
    parts = urllib.parse.urlsplit(url)
    path = urllib.parse.quote(urllib.parse.unquote(parts.path), safe="/!$&'()*+,;=:@")
    return urllib.parse.urlunsplit((parts.scheme, parts.netloc, path, parts.query, ""))


def url_problem(url: str) -> Optional[str]:
    """Why ``url`` may not be fetched, or None. https only, to the source hosts and the storage domain;
    plain http is allowed to loopback only (the tests)."""
    parts = urllib.parse.urlsplit(url)
    host = (parts.hostname or "").lower()
    if parts.scheme == "http" and host in LOOPBACK_HOSTS:
        return None
    if parts.scheme != "https":
        return f"not an https URL: {url}"
    if host in SOURCE_HOSTS or host == STORAGE_DOMAIN or host.endswith("." + STORAGE_DOMAIN):
        return None
    return f"host {host!r} is not a source host or the RawDB storage domain: {url}"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def disk_free_bytes(path: Path) -> int:
    """Free bytes on the filesystem holding ``path`` (or its nearest existing parent)."""
    probe = path
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    return shutil.disk_usage(str(probe)).free


# --------------------------------------------------------------------------------------------------
# Manifest


def _check_manifest_url(url: str, where: str) -> None:
    problem = url_problem(url)
    if problem:
        raise ManifestError(f"{where}: {problem}")


def validate_entry(e: Any, where: str) -> None:
    """Raise ManifestError unless ``e`` is one well-formed manifest entry."""
    if not isinstance(e, dict):
        raise ManifestError(f"{where} is not an object")
    missing = [k for k in ENTRY_KEYS if k not in e]
    if missing:
        raise ManifestError(f"{where} lacks {', '.join(missing)}")
    source = e["source"]
    if source not in SOURCE_KEY:
        raise ManifestError(f"{where}: unknown source {source!r}")
    prefix = SOURCE_KEY[source] + ":"
    if not isinstance(e["id"], str) or not e["id"].startswith(prefix):
        raise ManifestError(f"{where}: id must start with {prefix!r}")
    for key in ("body", "name", "url", "sha256", "licence"):
        if not isinstance(e[key], str) or not e[key]:
            raise ManifestError(f"{where}: {key} must be a non-empty string")
    if "/" in e["name"]:
        raise ManifestError(f"{where}: name must be a file name, not a path")
    if e["mode"] is not None and not isinstance(e["mode"], str):
        raise ManifestError(f"{where}: mode must be a string or null")
    if not HEX64.match(e["sha256"]):
        raise ManifestError(f"{where}: sha256 must be 64 lower-case hex digits")
    if e["size"] is not None and (not is_int(e["size"]) or e["size"] < 0):
        raise ManifestError(f"{where}: size must be a byte count or null")
    if e["size_approx"] is not None and not isinstance(e["size_approx"], str):
        raise ManifestError(f"{where}: size_approx must be a string or null")
    if e["licence"] != LICENCE:
        raise ManifestError(f"{where}: licence must be {LICENCE}; the policy allows CC0 only")
    _check_manifest_url(e["url"], f"{where}.url")
    if not isinstance(e["also"], list) or not all(isinstance(a, str) and key_of_id(a) in SOURCE_KEY.values() for a in e["also"]):
        raise ManifestError(f"{where}: also must be a list of ids")


def validate_manifest(m: Any) -> None:
    """Raise ManifestError unless ``m`` is a well-formed manifest: unique ids, one entry per SHA-256."""
    if not isinstance(m, dict):
        raise ManifestError("the manifest must be a JSON object")
    if m.get("schema_version") != SCHEMA_VERSION:
        raise ManifestError(f"schema_version must be {SCHEMA_VERSION}")
    sources = m.get("sources")
    if not isinstance(sources, dict):
        raise ManifestError("sources is missing")
    for key, field in (("pixls", "listing"), ("rawdb", "api"), ("rawdb", "download")):
        block = sources.get(key)
        url = block.get(field) if isinstance(block, dict) else None
        if not isinstance(url, str):
            raise ManifestError(f"sources.{key}.{field} is missing")
        _check_manifest_url(url, f"sources.{key}.{field}")
    files = m.get("files")
    if not isinstance(files, list):
        raise ManifestError("files must be a list")
    ids: Set[str] = set()
    shas: Set[str] = set()
    for i, e in enumerate(files):
        where = f"files[{i}]"
        validate_entry(e, where)
        for name in [e["id"], *e["also"]]:
            if name in ids:
                raise ManifestError(f"{where}: id {name!r} appears twice")
            ids.add(name)
        if e["sha256"] in shas:
            raise ManifestError(f"{where}: sha256 {e['sha256']} is in two entries (a duplicate belongs in also)")
        shas.add(e["sha256"])


def load_manifest(path: Path) -> Dict[str, Any]:
    try:
        with open(path, encoding="utf-8") as f:
            m = json.load(f)
    except FileNotFoundError:
        raise EmptyError(f"no manifest at {path}: run tools/fetch/autoset.sh refresh first") from None
    except ValueError as err:
        raise ManifestError(f"the manifest is not valid JSON ({path}): {err}") from None
    validate_manifest(m)
    return m


def format_manifest(m: Dict[str, Any]) -> str:
    """The manifest text: one file entry per line, so a refresh shows as a small diff."""
    files = m["files"]
    lines = ['{"schema_version": ' + str(m["schema_version"]) + ",",
             " \"description\": " + json.dumps(m["description"], ensure_ascii=False) + ",",
             " \"sources\": " + json.dumps(m["sources"], ensure_ascii=False) + ",",
             ' "files": [']
    for i, e in enumerate(files):
        sep = "," if i < len(files) - 1 else ""
        lines.append("   " + json.dumps(e, ensure_ascii=False) + sep)
    lines.append(" ]}")
    return "\n".join(lines) + "\n"


def save_manifest(path: Path, m: Dict[str, Any]) -> None:
    validate_manifest(m)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write(format_manifest(m))
    os.replace(tmp, path)


def select_entries(m: Dict[str, Any], bodies: Optional[Sequence[str]] = None, ids: Optional[Sequence[str]] = None) -> List[Dict[str, Any]]:
    """Manifest entries (in manifest order) that match the filters; EmptyError if none does."""
    if not m["files"]:
        raise EmptyError("the manifest lists no files: run tools/fetch/autoset.sh refresh first")
    entries = m["files"]
    if ids:
        known = {e["id"] for e in entries}
        missing = [i for i in ids if i not in known]
        if missing:
            raise EmptyError("no manifest entry with id " + ", ".join(repr(i) for i in missing))
        wanted_ids = set(ids)
        entries = [e for e in entries if e["id"] in wanted_ids]
    if bodies:
        wanted = {b.lower() for b in bodies}
        entries = [e for e in entries if e["body"].lower() in wanted]
    if not entries:
        raise EmptyError("no manifest entry matches the filter")
    return entries


# --------------------------------------------------------------------------------------------------
# Listings (refresh)


@dataclasses.dataclass
class Candidate:
    """One file as a source lists it, before deduplication."""

    key: str  # "pixls" or "rawdb"
    id: str
    body: str
    name: str
    mode: Optional[str]
    url: str
    sha256: str
    size: Optional[int] = None
    size_approx: Optional[str] = None

    @property
    def source(self) -> str:
        return PIXLS if self.key == "pixls" else RAWDB


def parse_pixls(data: Any, stats: "collections.Counter[str]") -> List[Candidate]:
    """CC0 Sony ARW rows of the raw.pixls.us listing. Each row is
    ``[make, model, mode, megapixels, "", licence_html, date, file_html, exif_html]``."""
    rows = data.get("data") if isinstance(data, dict) else None
    if not isinstance(rows, list):
        raise ManifestError("the raw.pixls.us listing has no 'data' list")
    out: List[Candidate] = []
    for row in rows:
        if not isinstance(row, list) or len(row) < 8:
            stats["pixls rows not understood"] += 1
            continue
        make, model, mode, licence_html, file_html = row[0], row[1], row[2], row[5], row[7]
        if not isinstance(make, str) or make.strip() != "Sony":
            continue
        lic = PIXLS_LICENCE_HREF.search(licence_html) if isinstance(licence_html, str) else None
        if not lic or "publicdomain/zero/1.0" not in lic.group(1):
            stats["pixls not CC0 1.0"] += 1
            continue
        f = PIXLS_FILE_HTML.match(file_html) if isinstance(file_html, str) else None
        if not f:
            stats["pixls file markup not understood"] += 1
            continue
        href, text, sha, size_text = f.groups()
        name = html.unescape(text).strip()
        if not name.lower().endswith(".arw"):
            stats["pixls not ARW (SR2, SRF, ARQ)"] += 1
            continue
        body = str(model).strip() if model is not None else ""
        if not body:
            stats["pixls rows not understood"] += 1
            continue
        stats["pixls CC0 ARW rows"] += 1
        url = quote_url_path(html.unescape(href))
        out.append(
            Candidate(
                key="pixls",
                id=f"pixls:{body}/{name}",
                body=body,
                name=name,
                mode=str(mode) if mode is not None else None,
                url=url,
                sha256=sha.lower(),
                size_approx=size_text.strip() or None,
            )
        )
    return out


def rawdb_candidates(detail: Any, model: str, download: str, stats: "collections.Counter[str]") -> List[Candidate]:
    """The CC0 ARW files of one RawDB set (``GET /api/sets/Sony/<model>``)."""
    if not isinstance(detail, dict):
        raise ManifestError(f"RawDB set {model}: the answer is not an object")
    maker = detail.get("maker") if isinstance(detail.get("maker"), str) else "Sony"
    categories = detail.get("categories")
    if not isinstance(categories, dict):
        stats["RawDB sets without categories"] += 1
        return []
    out: List[Candidate] = []
    for files in categories.values():
        if not isinstance(files, list):
            continue
        for f in files:
            if not isinstance(f, dict):
                stats["RawDB files not understood"] += 1
                continue
            if str(f.get("extension", "")).lower() != "arw":
                stats["RawDB not ARW"] += 1
                continue
            if f.get("license") != RAWDB_LICENCE:
                stats["RawDB not CC0 1.0"] += 1
                continue
            sha = f.get("sha256")
            if not isinstance(sha, str) or not HEX64.match(sha.lower()):
                stats["RawDB ARW without sha256"] += 1
                continue
            path = f.get("path")
            rel = path.strip("/") if isinstance(path, str) else ""
            if not rel:
                stats["RawDB files not understood"] += 1
                continue
            stats["RawDB CC0 1.0 ARW files"] += 1
            tags = f.get("tags") if isinstance(f.get("tags"), list) else []
            segments = [quote_segment(p) for p in [maker, model, *rel.split("/")] if p]
            out.append(
                Candidate(
                    key="rawdb",
                    id=f"rawdb:{model}/{rel}",
                    body=model,
                    name=rel.rsplit("/", 1)[-1],
                    mode=", ".join(str(t) for t in tags) or None,
                    url=download.rstrip("/") + "/" + "/".join(segments),
                    sha256=sha.lower(),
                    size=f.get("size") if is_int(f.get("size")) else None,
                )
            )
    return out


def disambiguate(cands: List[Candidate]) -> List[Candidate]:
    """Different files that share an id (same body and name) each get the first 12 hex digits of their
    SHA-256 after a ``#``; the same file listed twice is kept once."""
    by_id: Dict[str, List[Candidate]] = {}
    for c in cands:
        by_id.setdefault(c.id, []).append(c)
    out: List[Candidate] = []
    for group in by_id.values():
        if len({c.sha256 for c in group}) > 1:
            for c in group:
                c.id = f"{c.id} #{c.sha256[:12]}"
        out.extend(group)
    seen: Set[Tuple[str, str]] = set()
    unique: List[Candidate] = []
    for c in out:
        if (c.id, c.sha256) in seen:
            continue
        seen.add((c.id, c.sha256))
        unique.append(c)
    return unique


def candidate_entry(c: Candidate) -> Dict[str, Any]:
    return {
        "id": c.id,
        "source": c.source,
        "body": c.body,
        "name": c.name,
        "mode": c.mode,
        "url": c.url,
        "sha256": c.sha256,
        "size": c.size,
        "size_approx": c.size_approx,
        "licence": LICENCE,
        "also": [],
    }


def merge(
    existing: List[Dict[str, Any]],
    fresh: Dict[str, List[Candidate]],
    refreshed: Set[str],
    accept_changed: bool,
) -> List[Dict[str, Any]]:
    """The new file list. Entries of the sources not refreshed are kept as they are; the refreshed
    sources' entries are rebuilt from ``fresh``. One entry per SHA-256: the pixls listing is preferred
    when both sources have a file, and the other source's ids go under ``also``.

    Raises ManifestError if an id already in the manifest would get another SHA-256, unless
    ``accept_changed``."""
    old_sha: Dict[str, str] = {}
    for e in existing:
        old_sha[e["id"]] = e["sha256"]
        for a in e["also"]:
            old_sha[a] = e["sha256"]
    changed = sorted(c.id for cands in fresh.values() for c in cands if c.id in old_sha and old_sha[c.id] != c.sha256)
    if changed and not accept_changed:
        shown = ", ".join(changed[:5]) + (" and more" if len(changed) > 5 else "")
        raise ManifestError(
            f"{len(changed)} id(s) in the manifest would get another SHA-256 ({shown}): the upstream files changed. "
            "Pass --accept-changed to accept the new hashes."
        )
    kept: Dict[str, Dict[str, Any]] = {}
    for e in existing:
        if key_of_id(e["id"]) in refreshed:
            continue
        copy = dict(e)
        copy["also"] = [a for a in e["also"] if key_of_id(a) not in refreshed]
        kept[e["sha256"]] = copy
    groups: Dict[str, List[Candidate]] = {}
    for key in ("pixls", "rawdb"):  # pixls first: its entry is the one kept when both sources have a file
        for c in fresh.get(key, []):
            groups.setdefault(c.sha256, []).append(c)
    for sha, cands in groups.items():
        entry = kept.get(sha)
        if entry is None:
            entry = candidate_entry(cands[0])
            kept[sha] = entry
            cands = cands[1:]
        for c in cands:
            if c.id != entry["id"] and c.id not in entry["also"]:
                entry["also"].append(c.id)
        if entry["size"] is None:
            entry["size"] = next((c.size for c in cands if c.size is not None), None)
        if entry["size_approx"] is None:
            entry["size_approx"] = next((c.size_approx for c in cands if c.size_approx), None)
    files = list(kept.values())
    for e in files:
        e["also"] = sorted(set(e["also"]) - {e["id"]})
    files.sort(key=lambda e: (e["body"], e["source"], e["name"], e["id"]))
    return files


def fetch_rawdb(http: "Http", src: Dict[str, Any], stats: "collections.Counter[str]") -> List[Candidate]:
    """Every Sony set in the paged RawDB index, then the detail of each set."""
    api, download = src["api"], src["download"]
    sets: List[Dict[str, Any]] = []
    offset = 0
    while True:
        query = urllib.parse.urlencode({"maker": "Sony", "offset": offset})
        page = http.get_json(f"{api}?{query}", "RawDB set index")
        batch = page.get("sets") if isinstance(page, dict) else None
        if not isinstance(batch, list):
            raise ManifestError("the RawDB set index has no 'sets' list")
        sets.extend(s for s in batch if isinstance(s, dict) and s.get("maker") == "Sony")
        offset += len(batch)
        total = page.get("total") if isinstance(page, dict) else None
        if not batch or not is_int(total) or offset >= total:
            break
    stats["RawDB Sony sets"] = len(sets)
    cands: List[Candidate] = []
    seen: Set[str] = set()
    for s in sets:
        model = s.get("model")
        if not isinstance(model, str) or not model or model in seen:
            continue
        seen.add(model)
        detail = http.get_json(f"{api}/{quote_segment('Sony')}/{quote_segment(model)}", f"RawDB set {model}")
        cands.extend(rawdb_candidates(detail, model, download, stats))
    return cands


# --------------------------------------------------------------------------------------------------
# HTTP: one request at a time, a pause between requests, retries on transient failures


class _AllowlistRedirect(urllib.request.HTTPRedirectHandler):
    """Follow a redirect only where ``url_problem`` allows it (RawDB sends downloads to object storage)."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[no-untyped-def]
        target = urllib.parse.urljoin(req.full_url, newurl)
        problem = url_problem(target)
        if problem:
            raise PolicyError(f"redirect refused: {problem}")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _tls_context() -> ssl.SSLContext:
    """The system CA bundle, or the file named by SSL_CERT_FILE. Verification is never switched off."""
    cafile = os.environ.get("SSL_CERT_FILE")
    try:
        return ssl.create_default_context(cafile=cafile) if cafile else ssl.create_default_context()
    except (OSError, ssl.SSLError) as err:
        raise UsageError(f"cannot read SSL_CERT_FILE ({cafile}): {err}") from None


def _retry_after(headers: Any) -> Optional[float]:
    value = headers.get("Retry-After") if headers is not None else None
    if value and value.strip().isdigit():
        return min(float(value), 300.0)
    return None


class Http:
    """Sequential, throttled HTTP client. ``delay`` seconds pass between the end of one request and the
    start of the next; a request that fails transiently is tried again, ``tries`` times in all."""

    def __init__(self, *, delay: float, timeout: float, tries: int, backoff: float) -> None:
        self.delay = max(0.0, delay)
        self.timeout = timeout
        self.tries = max(1, tries)
        self.backoff = max(0.0, backoff)
        self.opener = urllib.request.build_opener(_AllowlistRedirect(), urllib.request.HTTPSHandler(context=_tls_context()))
        self._last_end: Optional[float] = None

    def _pause(self) -> None:
        if self._last_end is not None and self.delay > 0:
            wait = self._last_end + self.delay - time.monotonic()
            if wait > 0:
                time.sleep(wait)

    def _request(self, url: str, consume: Callable[[Any], Any]) -> Any:
        problem = url_problem(url)
        if problem:
            raise PolicyError(problem)
        host = urllib.parse.urlsplit(url).hostname or url
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "*/*"})
        try:
            resp = self.opener.open(req, timeout=self.timeout)
        except urllib.error.HTTPError as err:
            err.close()
            if err.code in TRANSIENT_CODES:
                raise TransientError(
                    f"HTTP {err.code} from {host}", status=err.code, retry_after=_retry_after(err.headers)
                ) from None
            raise HttpStatusError(url, err.code, str(err.reason)) from None
        except urllib.error.URLError as err:
            if isinstance(err.reason, TRANSIENT_ERRORS):
                raise TransientError(f"{host}: {err.reason}") from None
            raise NetworkError(f"cannot reach {host}: {err.reason}") from None
        except TRANSIENT_ERRORS as err:
            raise TransientError(f"{host}: {err}") from None
        try:
            with resp:
                return consume(resp)
        except TRANSIENT_ERRORS as err:
            raise TransientError(f"{host}: the transfer was interrupted ({err})") from None

    def call(self, url: str, consume: Callable[[Any], Any], what: str) -> Any:
        """Run one request, retrying transient failures with backoff. ``consume`` reads the response."""
        for attempt in range(1, self.tries + 1):
            self._pause()
            try:
                return self._request(url, consume)
            except TransientError as err:
                if attempt >= self.tries:
                    raise
                wait = err.retry_after if err.retry_after is not None else self.backoff * (2 ** (attempt - 1))
                log(f"  {what}: {err}; trying again in {wait:.0f} s ({attempt + 1} of {self.tries})")
                time.sleep(wait)
            finally:
                self._last_end = time.monotonic()
        raise AssertionError("unreachable: the loop returns or raises")

    def get_bytes(self, url: str, what: str) -> bytes:
        """A whole small answer (listings and the RawDB index). Transfer trouble is a network failure here."""

        def consume(resp: Any) -> bytes:
            data = bytearray()
            while True:
                chunk = resp.read(1 << 16)
                if not chunk:
                    return bytes(data)
                data += chunk
                if len(data) > MAX_JSON_BYTES:
                    raise ManifestError(f"{what}: the answer is larger than {MAX_JSON_BYTES} bytes; refusing")

        try:
            return self.call(url, consume, what)
        except TransientError as err:
            if err.status is not None:
                raise HttpStatusError(url, err.status, str(err)) from None
            raise NetworkError(f"{what}: {err}") from None

    def get_json(self, url: str, what: str) -> Any:
        data = self.get_bytes(url, what)
        try:
            return json.loads(data.decode("utf-8"))
        except ValueError as err:
            raise ManifestError(f"{what}: not valid JSON ({err})") from None


def make_http(args: argparse.Namespace) -> Http:
    return Http(delay=args.delay, timeout=args.timeout, tries=TRIES, backoff=BACKOFF_S)


# --------------------------------------------------------------------------------------------------
# Download, dump, rows


def stream_to(resp: Any, part: Path) -> Tuple[str, int]:
    """Write the response to ``part`` in 1 MiB chunks while hashing it. Returns (sha256 hex, bytes)."""
    digest = hashlib.sha256()
    total = 0
    value = resp.headers.get("Content-Length")
    expected = int(value) if value and value.strip().isdigit() else None
    with open(part, "wb") as f:
        while True:
            chunk = resp.read(CHUNK)
            if not chunk:
                break
            digest.update(chunk)
            total += len(chunk)
            f.write(chunk)
    if expected is not None and total != expected:
        raise TransientError(f"the connection ended after {total} of {expected} bytes")
    return digest.hexdigest(), total


def download_verified(http: Http, e: Dict[str, Any], part: Path, raw: Path) -> None:
    """Download the entry to ``part``; rename it to ``raw`` only if its size (when exact) and SHA-256 match."""

    def consume(resp: Any) -> Tuple[str, int]:
        return stream_to(resp, part)

    try:
        digest, size = http.call(e["url"], consume, what=e["id"])
    except (HttpStatusError, TransientError, PolicyError) as err:
        raise FileFailure("download", str(err)) from None
    except OSError as err:  # the local write failed
        if err.errno == errno.ENOSPC:
            raise DiskError(f"the disk filled up while downloading {e['id']}: {err}") from None
        raise FileFailure("download", f"cannot write {part}: {err}") from None
    if e["size"] is not None and size != e["size"]:
        raise FileFailure("verify", f"the file has {size} bytes; the manifest says {e['size']}")
    if digest != e["sha256"]:
        raise FileFailure("verify", f"sha256 is {digest}; the manifest says {e['sha256']}")
    os.replace(part, raw)


def verify_file(path: Path, e: Dict[str, Any]) -> bool:
    if e["size"] is not None and path.stat().st_size != e["size"]:
        return False
    return sha256_file(path) == e["sha256"]


def check_disk(e: Dict[str, Any], workdir: Path, free_space: Callable[[Path], int]) -> None:
    """Refuse a download unless 2 x the file size + 1 GiB is free (DiskError, exit 5)."""
    size = e["size"] if e["size"] is not None else (parse_size_approx(e["size_approx"]) or UNKNOWN_SIZE)
    need = 2 * size + DISK_MARGIN
    free = free_space(workdir)
    if free < need:
        raise DiskError(
            f"not enough free disk space to download {e['id']}: about {fmt_gib(need)} GiB is needed in "
            f"{workdir}, {fmt_gib(free)} GiB is free. Free some space, or point --dest at a bigger disk, "
            "then run build again: it carries on where it stopped."
        )


def resolve_dump(cmd: Optional[str]) -> List[str]:
    """The dump program as argv: ``--dump``, else $YM_AUTOSET_DUMP, else the built example (built here
    with cargo if it does not exist yet)."""
    text = cmd if cmd is not None else os.environ.get("YM_AUTOSET_DUMP")
    if text is None or not text.strip():
        if not DEFAULT_DUMP.exists():
            log("the dump program is not built yet: " + " ".join(DUMP_BUILD_COMMAND))
            try:
                proc = subprocess.run(DUMP_BUILD_COMMAND, cwd=str(REPO_ROOT), check=False)
            except FileNotFoundError:
                raise SetupError("cargo is not installed; build the dump program yourself or pass --dump CMD") from None
            if proc.returncode != 0:
                raise SetupError(
                    f"cargo build of the dump program failed (exit {proc.returncode}); "
                    "pass --dump CMD to use another program"
                )
        if not DEFAULT_DUMP.exists():
            raise SetupError(f"no dump program at {DEFAULT_DUMP}; pass --dump CMD")
        return [str(DEFAULT_DUMP)]
    argv = shlex.split(text)
    if not argv:
        raise UsageError("the dump command is empty")
    if shutil.which(argv[0]) is None:
        raise SetupError(f"dump program not found: {argv[0]}")
    return argv


def run_dump(dump: List[str], e: Dict[str, Any], raw: Path) -> str:
    """Run ``<dump> --source <source> --licence CC0-1.0 <raw>`` and return its stdout."""
    command = [*dump, "--source", e["source"], "--licence", LICENCE, str(raw)]
    try:
        proc = subprocess.run(command, capture_output=True, timeout=DUMP_TIMEOUT_S, check=False)
    except subprocess.TimeoutExpired:
        raise FileFailure("dump", f"the dump program ran for more than {DUMP_TIMEOUT_S} s") from None
    except OSError as err:
        raise FileFailure("dump", f"cannot run {dump[0]}: {err}") from None
    stderr = proc.stderr.decode("utf-8", errors="replace").rstrip()
    if stderr:
        for line in stderr.splitlines():
            log(f"    dump: {line}")
    if proc.returncode != 0:
        last = stderr.splitlines()[-1] if stderr else "no message"
        raise FileFailure("dump", f"exit status {proc.returncode}: {last}")
    try:
        return proc.stdout.decode("utf-8")
    except UnicodeDecodeError as err:
        raise FileFailure("rows", f"the dump output is not UTF-8 ({err})") from None


def validate_rows(text: str, e: Dict[str, Any]) -> List[str]:
    """Check every row the dump printed for ``e`` and return them as JSON lines with the manifest's
    ``file``, ``source``, ``licence``, ``body`` (when null) and ``id``. Raises FileFailure("rows") on any
    problem: nothing is appended then."""
    lines: List[str] = []
    has_base = False
    for lineno, raw_line in enumerate(text.splitlines(), 1):
        if not raw_line.strip():
            continue
        try:
            row = json.loads(raw_line)
        except ValueError as err:
            raise FileFailure("rows", f"line {lineno} is not JSON ({err})") from None
        if not isinstance(row, dict):
            raise FileFailure("rows", f"line {lineno} is not a JSON object")
        if row.get("sha256") != e["sha256"]:
            raise FileFailure("rows", f"line {lineno}: sha256 {row.get('sha256')!r} is not the manifest's")
        if not is_finite_number(row.get("aug_k")):
            raise FileFailure("rows", f"line {lineno}: aug_k is not a finite number")
        if not features_ok(row.get("features")):
            raise FileFailure("rows", f"line {lineno}: features must be {FEATURES} finite numbers")
        if row.get("licence") not in (None, LICENCE):
            raise FileFailure("rows", f"line {lineno}: licence {row.get('licence')!r} is not {LICENCE}")
        row["file"] = e["name"]
        row["source"] = e["source"]
        row["licence"] = LICENCE
        if row.get("body") is None:
            row["body"] = e["body"]
        row["id"] = e["id"]
        if is_zero(row["aug_k"]):
            has_base = True
        try:
            lines.append(json.dumps(row, ensure_ascii=False, allow_nan=False))
        except ValueError as err:
            raise FileFailure("rows", f"line {lineno}: {err}") from None
    if not lines:
        raise FileFailure("rows", "the dump wrote no rows")
    if not has_base:
        raise FileFailure("rows", "no row with aug_k = 0")
    return lines


def append_rows(out: Path, lines: List[str]) -> None:
    """Append the rows in one write, then flush and fsync."""
    out.parent.mkdir(parents=True, exist_ok=True)
    prefix = ""
    if out.exists() and out.stat().st_size > 0:
        with open(out, "rb") as f:
            f.seek(-1, os.SEEK_END)
            if f.read(1) != b"\n":
                prefix = "\n"
    with open(out, "a", encoding="utf-8") as f:
        f.write(prefix + "\n".join(lines) + "\n")
        f.flush()
        os.fsync(f.fileno())


def read_done(out: Path) -> Set[str]:
    """The SHA-256s that already have an ``aug_k = 0`` row in the rows file."""
    done: Set[str] = set()
    if not out.exists():
        return done
    with open(out, encoding="utf-8", errors="replace") as f:
        for line in f:
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if isinstance(row, dict) and is_zero(row.get("aug_k")) and isinstance(row.get("sha256"), str):
                done.add(row["sha256"])
    return done


def read_failures(path: Path) -> Dict[str, Dict[str, Any]]:
    """One record per id that is still failing (the last one written)."""
    failed: Dict[str, Dict[str, Any]] = {}
    if not path.exists():
        return failed
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            if isinstance(rec, dict) and isinstance(rec.get("id"), str):
                failed[rec["id"]] = rec
    return failed


def update_failures(path: Path, e: Dict[str, Any], stage: Optional[str], error: Optional[str]) -> None:
    """Replace the record of ``e`` in failures.jsonl with a new failure, or drop it when ``stage`` is None.
    The file is rewritten atomically, so a record is never half-written."""
    lines: List[str] = []
    if path.exists():
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                if isinstance(rec, dict) and rec.get("id") != e["id"]:
                    lines.append(line.rstrip("\n"))
    if stage is not None:
        lines.append(json.dumps({"id": e["id"], "sha256": e["sha256"], "stage": stage, "error": error}, ensure_ascii=False))
    if not lines and not path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write("".join(line + "\n" for line in lines))
    os.replace(tmp, path)


def process_entry(
    e: Dict[str, Any],
    *,
    http: Http,
    workdir: Path,
    out: Path,
    keep: bool,
    dump: List[str],
    free_space: Callable[[Path], int],
) -> int:
    """Download (or reuse), dump, validate and append one entry; then delete its raw file unless ``keep``.
    Returns the number of rows appended. Raises FileFailure for a file-level failure."""
    workdir.mkdir(parents=True, exist_ok=True)
    raw = workdir / f"{e['sha256']}.ARW"
    part = workdir / f"{e['sha256']}.ARW.part"
    try:
        if raw.exists() and verify_file(raw, e):
            log("  using the raw file already in the work folder")
        else:
            raw.unlink(missing_ok=True)
            check_disk(e, workdir, free_space)
            log(f"  downloading {fmt_size(e)}")
            download_verified(http, e, part, raw)
        lines = validate_rows(run_dump(dump, e, raw), e)
        append_rows(out, lines)
        return len(lines)
    finally:
        part.unlink(missing_ok=True)
        if not keep:
            raw.unlink(missing_ok=True)


# --------------------------------------------------------------------------------------------------
# Commands


def cmd_refresh(args: argparse.Namespace, m: Dict[str, Any], path: Path) -> int:
    wanted = {"pixls", "rawdb"} if args.source == "all" else {args.source}
    http = make_http(args)
    stats: collections.Counter[str] = collections.Counter()
    fresh: Dict[str, List[Candidate]] = {}
    if "pixls" in wanted:
        log("listing raw.pixls.us ...")
        listing = http.get_json(m["sources"]["pixls"]["listing"], "raw.pixls.us listing")
        fresh["pixls"] = disambiguate(parse_pixls(listing, stats))
    if "rawdb" in wanted:
        log("listing the RawDB Sony sets (one request per set) ...")
        fresh["rawdb"] = disambiguate(fetch_rawdb(http, m["sources"]["rawdb"], stats))
    files = merge(m["files"], fresh, wanted, args.accept_changed)
    new_m = {
        "schema_version": SCHEMA_VERSION,
        "description": m.get("description", DESCRIPTION),
        "sources": m["sources"],
        "files": files,
    }
    validate_manifest(new_m)
    report_refresh(m["files"], files, stats)
    if args.dry_run:
        log("dry run: the manifest was not written")
        return EXIT_OK
    save_manifest(path, new_m)
    log(f"wrote {path}")
    return EXIT_OK


def report_refresh(old: List[Dict[str, Any]], files: List[Dict[str, Any]], stats: "collections.Counter[str]") -> None:
    old_ids = {e["id"] for e in old}
    new_ids = {e["id"] for e in files}
    total_bytes = sum(effective_size(e) for e in files)
    folded = sum(len(e["also"]) for e in files)
    log(f"pixls: {stats['pixls CC0 ARW rows']} CC0 Sony ARW rows")
    log(f"rawdb: {stats['RawDB CC0 1.0 ARW files']} CC0 1.0 ARW files in {stats['RawDB Sony sets']} Sony sets")
    excluded = {k: v for k, v in sorted(stats.items()) if "not" in k or "without" in k or "understood" in k}
    if excluded:
        log("excluded: " + ", ".join(f"{k} {v}" for k, v in excluded.items()))
    log(f"duplicates: {folded} listed files are the same file as an entry (kept under also)")
    log(
        f"manifest: {len(files)} files, {len({e['body'] for e in files})} bodies, "
        f"{total_bytes / (1 << 30):.2f} GiB (RawDB sizes are exact, pixls sizes approximate)"
    )
    log(f"changes against the old manifest: +{len(new_ids - old_ids)} -{len(old_ids - new_ids)}")


def cmd_list(args: argparse.Namespace, m: Dict[str, Any]) -> int:
    for e in select_entries(m, bodies=args.body):
        print("\t".join([e["id"], fmt_size(e), e["sha256"][:12], e["licence"], str(len(e["also"]))]))
    return EXIT_OK


def cmd_build(args: argparse.Namespace, m: Dict[str, Any]) -> int:
    if args.limit is not None and args.limit < 1:
        raise UsageError("--limit must be at least 1")
    entries = select_entries(m, bodies=args.body, ids=args.id)
    dest = Path(args.dest).resolve()
    workdir = dest / "autoset" / "tmp"
    out = Path(args.out).resolve() if args.out else dest / "autoset" / "rows.jsonl"
    failures_path = out.parent / "failures.jsonl"
    done = read_done(out)
    failed = read_failures(failures_path)
    http = make_http(args)
    counts: collections.Counter[str] = collections.Counter()
    todo: List[Dict[str, Any]] = []
    for e in entries:
        if e["sha256"] in done:
            counts["skipped (already in the rows file)"] += 1
        elif e["id"] in failed and not args.retry_failed:
            counts["skipped (failed earlier; --retry-failed tries again)"] += 1
        else:
            todo.append(e)
    log(f"{len(entries)} entries match; {len(todo)} to do")
    dump: Optional[List[str]] = None
    attempted = 0
    try:
        for n, e in enumerate(todo, 1):
            if args.limit is not None and attempted >= args.limit:
                break
            attempted += 1
            if dump is None:
                dump = resolve_dump(args.dump)
            log(f"[{n}/{len(todo)}] {e['id']}")
            try:
                written = process_entry(
                    e,
                    http=http,
                    workdir=workdir,
                    out=out,
                    keep=args.keep,
                    dump=dump,
                    free_space=disk_free_bytes,
                )
            except FileFailure as err:
                counts["failed"] += 1
                log(f"  failed ({err.stage}): {err}")
                update_failures(failures_path, e, err.stage, str(err))
                continue
            counts["done"] += 1
            counts["rows written"] += written
            log(f"  {written} rows appended")
            if e["id"] in failed:
                update_failures(failures_path, e, None, None)
    finally:
        log("summary: " + (", ".join(f"{k} {v}" for k, v in counts.items() if v) or "nothing to do"))
    return EXIT_FAILED if counts["failed"] else EXIT_OK


def cmd_verify_rows(args: argparse.Namespace, m: Dict[str, Any]) -> int:
    path = Path(args.rows)
    if not path.exists():
        raise UsageError(f"rows file not found: {path}")
    by_sha = {e["sha256"]: e for e in m["files"]}
    rows_per_sha: collections.Counter[str] = collections.Counter()
    base_sha: Set[str] = set()
    errors: List[str] = []
    total = 0
    with open(path, encoding="utf-8", errors="replace") as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            total += 1
            try:
                row = json.loads(line)
            except ValueError:
                errors.append(f"line {lineno}: not JSON")
                continue
            if not isinstance(row, dict):
                errors.append(f"line {lineno}: not a JSON object")
                continue
            sha = row.get("sha256")
            if sha not in by_sha:
                errors.append(f"line {lineno}: sha256 {sha!r} is not in the manifest")
                continue
            if not features_ok(row.get("features")):
                errors.append(f"line {lineno}: features are not {FEATURES} finite numbers")
                continue
            if not is_finite_number(row.get("aug_k")):
                errors.append(f"line {lineno}: aug_k is not a finite number")
                continue
            rows_per_sha[sha] += 1
            if is_zero(row["aug_k"]):
                base_sha.add(sha)
    per_key: collections.Counter[Tuple[str, str]] = collections.Counter()
    files_key: collections.Counter[Tuple[str, str]] = collections.Counter()
    entries_key: collections.Counter[Tuple[str, str]] = collections.Counter()
    for e in m["files"]:
        key = (e["source"], e["body"])
        entries_key[key] += 1
        if e["sha256"] in rows_per_sha:
            files_key[key] += 1
            per_key[key] += rows_per_sha[e["sha256"]]
    print(f"rows: {total} in {len(rows_per_sha)} files of the manifest's {len(m['files'])}")
    print(f"files with an aug_k = 0 row: {len(base_sha)}")
    print("source\tbody\tfiles with rows\trows")
    for key in sorted(entries_key):
        print(f"{key[0]}\t{key[1]}\t{files_key[key]} of {entries_key[key]}\t{per_key[key]}")
    if errors:
        for err in errors[:20]:
            print(f"error: {err}")
        if len(errors) > 20:
            print(f"error: and {len(errors) - 20} more")
        print(f"{len(errors)} error(s)")
        return EXIT_FAILED
    print("ok: every row's sha256 is in the manifest and its features are well formed")
    return EXIT_OK


def read_or_skeleton(path: Path) -> Dict[str, Any]:
    if path.exists():
        return load_manifest(path)
    return {"schema_version": SCHEMA_VERSION, "description": DESCRIPTION, "sources": DEFAULT_SOURCES, "files": []}


def run(args: argparse.Namespace) -> int:
    path = Path(args.manifest)
    if args.command == "refresh":
        return cmd_refresh(args, read_or_skeleton(path), path)
    m = load_manifest(path)
    if args.command == "list":
        return cmd_list(args, m)
    if args.command == "build":
        return cmd_build(args, m)
    if args.command == "verify-rows":
        return cmd_verify_rows(args, m)
    raise UsageError(f"unknown command {args.command!r}")


def add_manifest(p: argparse.ArgumentParser) -> None:
    p.add_argument("--manifest", default=str(DEFAULT_MANIFEST), help="manifest file (default: tools/fetch/autoset/manifest.json)")


def add_network(p: argparse.ArgumentParser) -> None:
    p.add_argument("--delay", type=float, default=1.0, help="seconds between requests (default 1; one at a time)")
    p.add_argument("--timeout", type=float, default=60.0, help="per-request timeout in seconds (default 60)")


def default_dest() -> str:
    return os.environ.get("YM_TESTDATA") or str(REPO_ROOT / "testdata")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="autoset.py",
        description="Yellowmoon autoset dataset: CC0 Sony ARW files, streamed. Use tools/fetch/autoset.sh.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    r = sub.add_parser("refresh", help="query raw.pixls.us and RawDB and rewrite the manifest")
    add_manifest(r)
    add_network(r)
    r.add_argument("--source", choices=["pixls", "rawdb", "all"], default="all", help="which source to query (default all)")
    r.add_argument("--dry-run", action="store_true", help="show the counts and changes; write nothing")
    r.add_argument("--accept-changed", action="store_true", help="accept new SHA-256 values for ids already in the manifest")

    ls = sub.add_parser("list", help="one line per manifest entry: id, size, SHA-256 prefix, licence, also-count")
    add_manifest(ls)
    ls.add_argument("--body", action="append", metavar="MODEL", help="only this camera model (repeatable)")

    b = sub.add_parser("build", help="stream the entries: download, verify, dump, check and append rows, delete")
    add_manifest(b)
    add_network(b)
    b.add_argument("--out", metavar="PATH", help="rows file to append to (default: <dest>/autoset/rows.jsonl)")
    b.add_argument("--dest", default=default_dest(), help="testdata folder (default testdata/, or $YM_TESTDATA); raw files go to <dest>/autoset/tmp")
    b.add_argument("--body", action="append", metavar="MODEL", help="only this camera model (repeatable)")
    b.add_argument("--id", action="append", metavar="ID", help="only this manifest id (repeatable)")
    b.add_argument("--limit", type=int, metavar="N", help="stop after N files (done or failed)")
    b.add_argument("--keep", action="store_true", help="keep the raw files instead of deleting them after their rows are written")
    b.add_argument("--dump", metavar="CMD", help="the dump program (default: $YM_AUTOSET_DUMP, else target/release/examples/autoset_dump)")
    b.add_argument("--retry-failed", action="store_true", help="try again the files listed in failures.jsonl")

    v = sub.add_parser("verify-rows", help="check a rows file against the manifest (no network)")
    add_manifest(v)
    v.add_argument("rows", metavar="ROWS", help="the rows file to check")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    if sys.version_info < (3, 8):
        print("error: Python 3.8 or newer is required", file=sys.stderr)
        return EXIT_USAGE
    try:
        args = build_parser().parse_args(argv)
    except SystemExit as e:  # argparse exits 2 on a usage error and 0 on --help
        return int(e.code or 0)
    try:
        return run(args)
    except NetworkError as err:
        log(network_help(err))
        return err.exit_code
    except AutosetError as err:
        log(f"error: {err}")
        return err.exit_code
    except KeyboardInterrupt:
        log("interrupted: the file in progress was removed; rows already written are kept")
        return EXIT_INTERRUPTED


if __name__ == "__main__":
    sys.exit(main())
