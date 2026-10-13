#!/usr/bin/env python3
"""Yellowmoon test-data fetcher (Python standard library only).

Do not run this file directly in scripts; use the wrappers ``samples.sh`` and ``golden.sh``, which
check for python3 and run it in isolated mode. Everything the wrappers accept is documented in
``tools/fetch/README.md`` (``docs/13-testing.md`` section "Test data" is the spec).

What it does
------------
* ``samples``: download the CC0 Sony ARW corpus listed in ``manifest.json`` into
  ``testdata/arw/<model>/`` and check every file's SHA-256 and size. ``--refresh`` queries the
  RawDB JSON API (and the raw.pixls.us directory listing) and rewrites the manifest.
* ``golden``: the same for the golden photo set: tier 1 (CC0) and tier 2 (freely licensed Play Raw
  files, with attribution). Tier 3 (copyrighted, evaluation only) is a documented local-only list and
  is never fetched.

Rules the code enforces (docs/14, docs/test-photo-sources.md)
-------------------------------------------------------------
* Corpus and tier 1: licence ``CC0-1.0`` only, checked per file. Tier 2: CC0, CC BY or CC BY-SA.
  Anything else (CC BY-NC-SA, unknown, unverified) is never downloaded.
* Polite: one connection at a time, a descriptive User-Agent, a pause between requests, back-off on
  HTTP 429/5xx, resumable downloads.
* Downloads and redirects may only go to the hosts named in the manifest, over https.
* Files are written to ``<dest>/arw/<model>/<name>`` or ``<dest>/golden/tier2/<model>/<name>``; names
  are sanitised so a manifest can never write outside ``<dest>``.
* When the network refuses, the scripts stop with a plain message and exit code 4.

Exit codes: 0 ok, 1 verification or policy failure, 2 usage error, 3 nothing to do (manifest empty or
filters match nothing), 4 network unreachable or refused.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import html.parser
import http.client
import json
import os
import re
import socket
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Set, Tuple

VERSION = "0.1.0"
SCHEMA_VERSION = 1
USER_AGENT = (
    f"yellowmoon-fetch/{VERSION} "
    "(+https://github.com/ngriaznov/yellowmoon; open-source test-data download, "
    "one connection at a time, CC0 and free-licence files only)"
)

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_USAGE = 2
EXIT_EMPTY = 3
EXIT_NETWORK = 4

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
DEFAULT_MANIFEST = HERE / "manifest.json"
DEFAULT_DEST = REPO_ROOT / "testdata"

#: Licences accepted for the decoder corpus and for golden tier 1.
CORPUS_LICENSES = frozenset({"CC0-1.0"})
#: Licences accepted for golden tier 2 (attribution is recorded and written to ATTRIBUTION.md).
TIER2_LICENSES = frozenset(
    {
        "CC0-1.0",
        "CC-BY-2.0",
        "CC-BY-3.0",
        "CC-BY-4.0",
        "CC-BY-SA-2.0",
        "CC-BY-SA-3.0",
        "CC-BY-SA-4.0",
    }
)
LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})
MAX_JSON_BYTES = 32 * 1024 * 1024
MAX_TEXT_BYTES = 8 * 1024 * 1024
HEX64 = re.compile(r"^[0-9a-f]{64}$")


# --------------------------------------------------------------------------------------------------
# Errors


class FetchError(Exception):
    """Base class; ``exit_code`` is the process exit status."""

    exit_code = EXIT_FAILED


class ManifestError(FetchError):
    """The manifest (or an API answer) is malformed or breaks the data policy."""


class PolicyError(FetchError):
    """A licence or host rule forbids the requested action."""


class VerifyError(FetchError):
    """A downloaded or existing file does not match the manifest."""


class EmptyError(FetchError):
    exit_code = EXIT_EMPTY


class NetworkError(FetchError):
    """The host could not be reached (DNS, refused, proxy tunnel refused, TLS, timeout)."""

    exit_code = EXIT_NETWORK

    def __init__(self, host: str, reason: object) -> None:
        super().__init__(f"cannot reach {host}: {reason}")
        self.host = host
        self.reason = str(reason)


class HttpStatusError(FetchError):
    """The server answered, but not with success."""

    def __init__(self, url: str, status: int, reason: str = "") -> None:
        super().__init__(f"{url}: HTTP {status} {reason}".strip())
        self.url = url
        self.status = status


def log(message: str = "") -> None:
    print(message, file=sys.stderr, flush=True)


# --------------------------------------------------------------------------------------------------
# Licences


def normalise_license(raw: object) -> str:
    """Map a free-form licence string to an SPDX-style id, or ``"unknown"``.

    Only identifiers we understand are returned; a Creative Commons string without a version is
    ``"unknown"`` on purpose (a human must pin the version).
    """
    if not isinstance(raw, str):
        return "unknown"
    s = raw.strip().lower()
    if not s:
        return "unknown"
    s = s.replace("creative commons", "cc")
    if re.search(r"\bcc[\s_-]*0\b|\bcc[\s_-]*zero\b|\bcc0", s):
        return "CC0-1.0"
    s = re.sub(r"[^a-z0-9.]+", " ", s)
    s = re.sub(r"\b(licen[cs]e|licen[cs]ed|international|unported|generic|spdx)\b", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    m = re.match(r"^cc by((?: nc)?(?: sa| nd)?) (?:v |version )?(\d\.\d)$", s)
    if not m:
        return "unknown"
    extras = m.group(1).strip().upper().replace(" ", "-")
    return "CC-BY" + ("-" + extras if extras else "") + "-" + m.group(2)


# --------------------------------------------------------------------------------------------------
# Manifest


def load_manifest(path: Path) -> Dict[str, Any]:
    try:
        with open(path, encoding="utf-8") as f:
            m = json.load(f)
    except FileNotFoundError:
        raise ManifestError(f"manifest not found: {path}") from None
    except json.JSONDecodeError as e:
        raise ManifestError(f"manifest is not valid JSON ({path}): {e}") from None
    validate_manifest(m)
    return m


def save_manifest(path: Path, m: Dict[str, Any]) -> None:
    validate_manifest(m)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(m, f, indent=2, ensure_ascii=False)
        f.write("\n")
    os.replace(tmp, path)


_SEGMENT_OK = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+()-]*$")


def sanitise_segment(part: str) -> str:
    """Make one path segment safe: only ``[A-Za-z0-9._+()-]``, no leading dot."""
    s = re.sub(r"[^A-Za-z0-9._+()-]", "_", part).lstrip(".")
    if not s:
        return "x_"
    return s if s[0].isalnum() else "x" + s


def flat_name(rel_path: str, make: str, model: str) -> str:
    """Turn a path inside a model's folder on the server into one flat, collision-free file name."""
    parts = [p for p in rel_path.replace("\\", "/").split("/") if p not in ("", ".", "..")]
    low = [p.lower() for p in parts]
    if len(parts) >= 3 and low[0] == make.lower() and low[1] == model.lower():
        parts = parts[2:]
    elif len(parts) >= 2 and low[0] == model.lower():
        parts = parts[1:]
    if not parts:
        raise ManifestError(f"empty file path {rel_path!r}")
    return "__".join(sanitise_segment(p) for p in parts)


def entry_dest_rel(e: Dict[str, Any]) -> str:
    if e["tier"] == 2:
        return f"golden/tier2/{e['body']}/{e['name']}"
    return f"arw/{e['body']}/{e['name']}"


def _require(cond: bool, message: str) -> None:
    if not cond:
        raise ManifestError(message)


def validate_entry(e: Any, where: str) -> None:
    _require(isinstance(e, dict), f"{where}: entry must be an object")
    for key in ("id", "source", "body", "name", "url", "size", "sha256", "license", "tier", "sets"):
        _require(key in e, f"{where}: missing field {key!r}")
    _require(isinstance(e["id"], str) and e["id"], f"{where}: bad id")
    where = f"{where} {e['id']}"
    _require(_SEGMENT_OK.match(str(e["body"])) is not None, f"{where}: unsafe body {e['body']!r}")
    _require(_SEGMENT_OK.match(str(e["name"])) is not None, f"{where}: unsafe name {e['name']!r}")
    _require(isinstance(e["size"], int) and e["size"] > 0, f"{where}: size must be a positive integer")
    _require(HEX64.match(str(e["sha256"])) is not None, f"{where}: sha256 must be 64 lower-case hex digits")
    _require(e["tier"] in (1, 2), f"{where}: tier must be 1 or 2")
    _require(
        isinstance(e["sets"], list) and e["sets"] and set(e["sets"]) <= {"corpus", "golden"},
        f"{where}: sets must be a non-empty subset of corpus, golden",
    )
    _require(isinstance(e["url"], str) and e["url"].startswith(("https://", "http://")), f"{where}: bad url")
    lic = e["license"]
    if e["tier"] == 1:
        _require(lic in CORPUS_LICENSES, f"{where}: tier 1 and the corpus take CC0-1.0 only, got {lic!r}")
    else:
        _require(lic in TIER2_LICENSES, f"{where}: licence {lic!r} is not allowed in golden tier 2")
        _require("corpus" not in e["sets"], f"{where}: tier 2 files are golden-only, never part of the corpus")
        att = e.get("attribution")
        _require(
            isinstance(att, dict) and att.get("author") and att.get("source_url"),
            f"{where}: tier 2 needs attribution.author and attribution.source_url",
        )


def validate_manifest(m: Any) -> None:
    _require(isinstance(m, dict), "manifest must be a JSON object")
    _require(m.get("schema_version") == SCHEMA_VERSION, f"manifest schema_version must be {SCHEMA_VERSION}")
    for key in ("sources", "bodies", "files", "pending", "excluded", "golden"):
        _require(key in m, f"manifest is missing {key!r}")
    _require(isinstance(m["files"], list), "manifest.files must be a list")
    seen_ids: Set[str] = set()
    seen_dest: Set[str] = set()
    for e in m["files"]:
        validate_entry(e, "files")
        _require(e["id"] not in seen_ids, f"duplicate id {e['id']}")
        seen_ids.add(e["id"])
        dest = entry_dest_rel(e).lower()
        _require(dest not in seen_dest, f"two entries write to the same path: {dest}")
        seen_dest.add(dest)
    for b in m["bodies"]:
        _require(isinstance(b, dict) and _SEGMENT_OK.match(str(b.get("model", ""))) is not None, "bad body entry")
    _require(isinstance(m["golden"].get("tier3", []), list), "manifest.golden.tier3 must be a list")


def allowed_hosts(m: Dict[str, Any]) -> Set[str]:
    hosts: Set[str] = set()
    for src in m["sources"].values():
        hosts.update(h.lower() for h in src.get("hosts", []))
        hosts.update(h.lower() for h in src.get("extra_hosts", []))
    return hosts


def select_entries(
    m: Dict[str, Any],
    *,
    sets: Set[str],
    tiers: Optional[Set[int]] = None,
    bodies: Optional[Sequence[str]] = None,
    per_body: Optional[int] = None,
    max_file_bytes: Optional[int] = None,
) -> List[Dict[str, Any]]:
    wanted = {b.lower() for b in bodies} if bodies else None
    out: List[Dict[str, Any]] = []
    counts: Dict[str, int] = {}
    for e in sorted(m["files"], key=lambda x: x["id"]):
        if not (set(e["sets"]) & sets):
            continue
        if tiers is not None and e["tier"] not in tiers:
            continue
        if wanted is not None and e["body"].lower() not in wanted:
            continue
        if max_file_bytes is not None and e["size"] > max_file_bytes:
            continue
        if per_body is not None and counts.get(e["body"], 0) >= per_body:
            continue
        counts[e["body"]] = counts.get(e["body"], 0) + 1
        out.append(e)
    return out


# --------------------------------------------------------------------------------------------------
# HTTP


def check_url(url: str, hosts: Set[str]) -> None:
    parts = urllib.parse.urlsplit(url)
    host = (parts.hostname or "").lower()
    if parts.scheme == "http" and host in LOOPBACK_HOSTS:
        return
    if parts.scheme != "https":
        raise PolicyError(f"refusing non-https URL: {url}")
    if host not in hosts and not any(host.endswith("." + h) for h in hosts):
        raise PolicyError(f"host {host!r} is not one of the manifest's source hosts ({', '.join(sorted(hosts))}): {url}")


class _AllowlistRedirect(urllib.request.HTTPRedirectHandler):
    def __init__(self, hosts: Set[str]) -> None:
        super().__init__()
        self.hosts = hosts

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[no-untyped-def]
        check_url(urllib.parse.urljoin(req.full_url, newurl), self.hosts)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _retry_after(headers: Any) -> Optional[float]:
    try:
        value = headers.get("Retry-After") if headers is not None else None
        return float(value) if value is not None and str(value).strip().isdigit() else None
    except (TypeError, ValueError):
        return None


class Http:
    """Sequential, throttled HTTP client. One request at a time, always."""

    def __init__(self, hosts: Set[str], *, delay: float = 1.0, timeout: float = 60.0, retries: int = 3) -> None:
        self.hosts = hosts
        self.delay = max(0.0, delay)
        self.timeout = timeout
        self.retries = max(0, retries)
        self.opener = urllib.request.build_opener(_AllowlistRedirect(hosts))
        self._last = 0.0
        self.requests = 0

    def _throttle(self) -> None:
        wait = self._last + self.delay - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        self._last = time.monotonic()

    def open(self, url: str, headers: Optional[Dict[str, str]] = None):  # type: ignore[no-untyped-def]
        check_url(url, self.hosts)
        host = urllib.parse.urlsplit(url).hostname or url
        attempt = 0
        while True:
            self._throttle()
            self.requests += 1
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, **(headers or {})})
            try:
                return self.opener.open(req, timeout=self.timeout)
            except urllib.error.HTTPError as e:
                if e.code in (429, 500, 502, 503, 504) and attempt < self.retries:
                    wait = _retry_after(e.headers) or float(2 ** (attempt + 1))
                    log(f"  HTTP {e.code} from {host}; waiting {wait:.0f} s before retry {attempt + 1}/{self.retries}")
                    e.close()
                    time.sleep(wait)
                    attempt += 1
                    continue
                e.close()
                raise HttpStatusError(url, e.code, str(e.reason)) from None
            except FetchError:
                raise
            except (urllib.error.URLError, OSError, http.client.HTTPException) as e:
                reason = e.reason if isinstance(e, urllib.error.URLError) else e
                transient = isinstance(reason, (socket.timeout, TimeoutError, ConnectionResetError)) or isinstance(
                    e, http.client.HTTPException
                )
                if transient and attempt < self.retries:
                    wait = float(2 ** (attempt + 1))
                    log(f"  {host}: {reason}; waiting {wait:.0f} s before retry {attempt + 1}/{self.retries}")
                    time.sleep(wait)
                    attempt += 1
                    continue
                raise NetworkError(host, reason) from None

    def get_bytes(self, url: str, *, limit: int, accept: str = "*/*") -> bytes:
        host = urllib.parse.urlsplit(url).hostname or url
        chunks: List[bytes] = []
        total = 0
        with self.open(url, {"Accept": accept}) as resp:
            try:
                while True:
                    chunk = resp.read(1 << 16)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > limit:
                        raise ManifestError(f"{url}: answer is larger than {limit} bytes; refusing")
                    chunks.append(chunk)
            except (OSError, http.client.HTTPException) as e:
                raise NetworkError(host, e) from None
        return b"".join(chunks)

    def get_json(self, url: str) -> Any:
        data = self.get_bytes(url, limit=MAX_JSON_BYTES, accept="application/json")
        try:
            return json.loads(data.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            raise ManifestError(f"{url}: not valid JSON ({e})") from None

    def get_text(self, url: str) -> str:
        data = self.get_bytes(url, limit=MAX_TEXT_BYTES, accept="text/html,*/*")
        return data.decode("utf-8", errors="replace")

    def download(self, url: str, part: Path, expected_size: Optional[int]) -> bool:
        """Download ``url`` into ``part`` (resuming if a partial file exists). Returns True if resumed."""
        host = urllib.parse.urlsplit(url).hostname or url
        resumed = False
        for attempt in range(self.retries + 1):
            start = part.stat().st_size if part.exists() else 0
            if expected_size is not None and start > expected_size:
                part.unlink()
                start = 0
            if expected_size is not None and start == expected_size:
                return resumed
            headers = {"Accept": "*/*"}
            if start:
                headers["Range"] = f"bytes={start}-"
            try:
                resp = self.open(url, headers)
            except HttpStatusError as e:
                if e.status == 416 and start:
                    part.unlink()
                    continue
                raise
            try:
                with resp:
                    if start and resp.status == 206:
                        mode, resumed = "ab", True
                    else:
                        mode, start = "wb", 0
                    length = resp.headers.get("Content-Length")
                    want = expected_size
                    if want is None and length is not None and length.strip().isdigit():
                        want = start + int(length)
                    # read1 keeps what arrived before a dropped connection, so the next attempt can resume from it
                    read = getattr(resp, "read1", resp.read)
                    with open(part, mode) as f:
                        while True:
                            chunk = read(1 << 20)
                            if not chunk:
                                break
                            f.write(chunk)
                    got = part.stat().st_size
                    if want is not None and got < want:
                        raise http.client.IncompleteRead(b"", want - got)
                return resumed
            except (OSError, http.client.HTTPException) as e:
                if attempt >= self.retries:
                    raise NetworkError(host, f"download interrupted: {e}") from None
                wait = float(2 ** (attempt + 1))
                log(f"  {host}: transfer interrupted ({e}); resuming in {wait:.0f} s")
                time.sleep(wait)
        raise NetworkError(host, "download did not complete")


def network_help(err: NetworkError) -> str:
    return (
        f"error: {err}\n"
        "  The network, or a proxy in front of it, refuses this host. rawdb.dnglab.org, raw.pixls.us and\n"
        "  discuss.pixls.us are blocked in some sandboxes and CI egress policies.\n"
        "  What to do: run this on a machine with open internet access (and commit the manifest, if you used\n"
        "  --refresh), or place the files by hand under testdata/arw/<model>/ and check them with --verify."
    )


# --------------------------------------------------------------------------------------------------
# Files


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1 << 20)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def safe_dest(root: Path, e: Dict[str, Any]) -> Path:
    root_r = root.resolve()
    dest = (root_r / entry_dest_rel(e)).resolve()
    if root_r != dest and root_r not in dest.parents:
        raise PolicyError(f"{e['id']}: destination escapes {root}")
    return dest


def file_status(root: Path, e: Dict[str, Any]) -> str:
    """Return ``ok``, ``missing``, ``size`` (wrong size) or ``hash`` (wrong content)."""
    dest = safe_dest(root, e)
    if not dest.is_file():
        return "missing"
    if dest.stat().st_size != e["size"]:
        return "size"
    return "ok" if sha256_file(dest) == e["sha256"] else "hash"


def fetch_entry(http: Http, root: Path, e: Dict[str, Any]) -> str:
    """Make sure ``e`` is on disk and verified. Returns ``cached`` or ``downloaded``."""
    dest = safe_dest(root, e)
    status = file_status(root, e)
    if status == "ok":
        return "cached"
    if status != "missing":
        log(f"  {dest.name}: on disk but {'wrong size' if status == 'size' else 'hash differs'}; fetching again")
        dest.unlink()
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    for attempt in (1, 2):
        resumed = http.download(e["url"], part, e["size"])
        size = part.stat().st_size
        digest = sha256_file(part)
        if size == e["size"] and digest == e["sha256"]:
            os.replace(part, dest)
            return "downloaded"
        part.unlink()
        if resumed and attempt == 1:
            log(f"  {dest.name}: resumed file failed verification; starting again from zero")
            continue
        raise VerifyError(
            f"{e['id']}: downloaded file does not match the manifest "
            f"(size {size} vs {e['size']}, sha256 {digest} vs {e['sha256']}). The file was deleted."
        )
    raise VerifyError(f"{e['id']}: could not download a verified copy")  # pragma: no cover


def fetch_unpinned(http: Http, root: Path, e: Dict[str, Any]) -> Tuple[int, str]:
    """Download a file that has no pinned hash yet (trust on first use). Returns ``(size, sha256)``."""
    dest = safe_dest(root, {**e, "tier": e.get("tier", 1)})
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    if part.exists():
        part.unlink()
    http.download(e["url"], part, None)
    size, digest = part.stat().st_size, sha256_file(part)
    os.replace(part, dest)
    return size, digest


def human_mb(n: int) -> str:
    return f"{n / (1024 * 1024):.1f} MB"


# --------------------------------------------------------------------------------------------------
# Link and HTML helpers


class LinkParser(html.parser.HTMLParser):
    """Collect ``(href, text)`` for every anchor."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.links: List[Tuple[str, str]] = []
        self._href: Optional[str] = None
        self._text: List[str] = []

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        if tag == "a":
            self._href = dict(attrs).get("href") or ""
            self._text = []

    def handle_data(self, data: str) -> None:
        if self._href is not None:
            self._text.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self._href is not None:
            self.links.append((self._href, "".join(self._text).strip()))
            self._href = None


def parse_links(markup: str) -> List[Tuple[str, str]]:
    p = LinkParser()
    p.feed(markup)
    p.close()
    return p.links


def is_arw(name: str) -> bool:
    return urllib.parse.urlsplit(name).path.lower().endswith(".arw")


_VARIANT_RULES: List[Tuple[str, str]] = [
    (r"uncompressed|\bunc\b|_unc_", "uncompressed"),
    (r"lossless[\s_.-]*m(edium)?\b|\bm[\s_.-]*lossless|yuv.*(m\b|medium)", "lossless-m"),
    (r"lossless[\s_.-]*s(mall)?\b|\bs[\s_.-]*lossless", "lossless-s"),
    (r"lossless|\bll\b", "lossless"),
    (r"craw|lossy|compressed", "craw-or-arw6"),
]


def guess_variant(text: str) -> str:
    """Best-effort variant hint from API text or file names. Authoritative variants come from ``ym info``."""
    t = text.lower()
    for pattern, label in _VARIANT_RULES:
        if re.search(pattern, t):
            return label
    return "unknown"


# --------------------------------------------------------------------------------------------------
# RawDB


_CTX_KEYS = ("license", "author", "compression", "notes", "description", "title")
_URL_KEYS = ("url", "download_url", "downloadUrl", "href", "link")


def iter_rawdb_records(node: Any, ctx: Optional[Dict[str, str]] = None) -> Iterator[Tuple[Dict[str, Any], Dict[str, str]]]:
    """Walk a RawDB answer and yield ``(file_record, inherited_context)`` for every ARW file.

    ``docs/test-photo-sources.md`` documents ``path``, ``size``, ``sha256`` and ``license`` per file.
    The exact nesting is not fixed, so any dict whose ``path`` ends in ``.arw`` is a file, and
    ``license`` / ``author`` / ``compression`` / ``notes`` are inherited from the enclosing dicts.
    """
    ctx = ctx or {}
    if isinstance(node, dict):
        local = dict(ctx)
        for key in _CTX_KEYS:
            value = node.get(key)
            if isinstance(value, str) and value:
                local[key] = value
        path = node.get("path")
        if isinstance(path, str) and is_arw(path):
            yield node, local
            return
        for value in node.values():
            if isinstance(value, (dict, list)):
                yield from iter_rawdb_records(value, local)
    elif isinstance(node, list):
        for item in node:
            yield from iter_rawdb_records(item, ctx)


def _as_int(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def rawdb_file_url(rec: Dict[str, Any], src: Dict[str, Any]) -> str:
    api = urllib.parse.urlsplit(src["api_base"])
    root = f"{api.scheme}://{api.netloc}"
    for key in _URL_KEYS:
        value = rec.get(key)
        if isinstance(value, str) and value:
            return urllib.parse.urljoin(root + "/", value)
    path = rec["path"]
    if path.startswith(("http://", "https://")):
        return path
    if path.startswith("/"):
        return root + urllib.parse.quote(path)
    template = src.get("file_url_template")
    if template:
        return template.format(root=root, path=urllib.parse.quote(path), make=src.get("make", "Sony"))
    raise ManifestError(
        "the RawDB answer has no download URL for "
        f"{path!r} (looked for {', '.join(_URL_KEYS)} and an absolute path) and "
        "manifest.sources.rawdb.file_url_template is not set. Look at the saved answer under "
        "testdata/.cache/api/ and set the template (placeholders: {root} {path} {make})."
    )


def build_rawdb_entries(
    doc: Any, src: Dict[str, Any], model: str, golden_bodies: Set[str]
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Return ``(accepted_entries, excluded, unhashed)`` for one model's RawDB answer."""
    make = src.get("make", "Sony")
    accepted: List[Dict[str, Any]] = []
    excluded: List[Dict[str, Any]] = []
    unhashed: List[Dict[str, Any]] = []
    seen: Set[str] = set()
    for rec, ctx in iter_rawdb_records(doc):
        name = flat_name(rec["path"], make, model)
        eid = f"rawdb:{model}/{name}"
        if eid in seen:
            continue
        seen.add(eid)
        raw_license = rec.get("license", ctx.get("license"))
        lic = normalise_license(raw_license)
        if lic not in CORPUS_LICENSES:
            excluded.append(
                {
                    "id": eid,
                    "source": "rawdb",
                    "body": model,
                    "license": lic,
                    "license_raw": raw_license if isinstance(raw_license, str) else None,
                    "reason": "licence is not CC0-1.0",
                }
            )
            continue
        sha = rec.get("sha256")
        sha = sha.lower() if isinstance(sha, str) and HEX64.match(sha.lower()) else None
        entry: Dict[str, Any] = {
            "id": eid,
            "source": "rawdb",
            "body": model,
            "name": name,
            "source_path": rec["path"],
            "url": rawdb_file_url(rec, src),
            "size": _as_int(rec.get("size")) or 0,
            "sha256": sha or "",
            "license": lic,
            "license_source": "rawdb-api",
            "tier": 1,
            "sets": ["corpus", "golden"] if model in golden_bodies else ["corpus"],
            "variant_hint": guess_variant(" ".join([rec["path"], ctx.get("compression", ""), ctx.get("notes", ""), ctx.get("title", "")])),
        }
        if isinstance(raw_license, str) and raw_license != lic:
            entry["license_raw"] = raw_license
        (accepted if sha and entry["size"] else unhashed).append(entry)
    return accepted, excluded, unhashed


# --------------------------------------------------------------------------------------------------
# Manifest refresh


def pick_bodies(m: Dict[str, Any], names: Optional[Sequence[str]], groups: Optional[Sequence[str]]) -> List[Dict[str, Any]]:
    bodies = m["bodies"]
    if names:
        wanted = {n.lower() for n in names}
        known = {b["model"].lower() for b in bodies}
        unknown = sorted(wanted - known)
        if unknown:
            raise ManifestError(f"unknown body {', '.join(unknown)}; add it to manifest.bodies first")
        bodies = [b for b in bodies if b["model"].lower() in wanted]
    if groups:
        bodies = [b for b in bodies if b.get("group") in set(groups)]
    return bodies


def probe_url(http: Http, entry: Dict[str, Any]) -> None:
    """Ask for one byte of the first file, to catch a wrong download-URL rule before it is written to the manifest."""
    try:
        with http.open(entry["url"], {"Range": "bytes=0-0"}):
            pass
    except HttpStatusError as e:
        raise ManifestError(
            f"the download URL derived for {entry['id']} answers HTTP {e.status}: {entry['url']}. "
            "Check how the RawDB answer names its files and set manifest.sources.rawdb.file_url_template. "
            "Manifest not written."
        ) from None


def cmd_refresh(args: argparse.Namespace, m: Dict[str, Any], http: Http, root: Path) -> int:
    sources = ["rawdb", "pixls"] if args.source == "all" else [args.source]
    bodies = pick_bodies(m, args.body, args.group)
    if not bodies:
        raise EmptyError("no bodies selected")
    golden_bodies = set(m["golden"].get("tier1_bodies", []))
    report: List[str] = []
    gaps: List[str] = []
    cache_dir = root / ".cache" / "api"

    if "rawdb" in sources:
        src = m["sources"]["rawdb"]
        models = [b["model"] for b in bodies if "rawdb" in b.get("sources", ["rawdb", "pixls"])]
        old = {e["id"]: e for e in m["files"] if e["source"] == "rawdb" and e["body"] in models}
        new_entries: List[Dict[str, Any]] = []
        excluded_new: List[Dict[str, Any]] = []
        missing: List[str] = []
        for model in models:
            url = f"{src['api_base'].rstrip('/')}/{urllib.parse.quote(src.get('make', 'Sony'))}/{urllib.parse.quote(model)}"
            log(f"RawDB {model}")
            try:
                doc = http.get_json(url)
            except HttpStatusError as e:
                if e.status == 404:
                    missing.append(model)
                    gaps.append(f"{model}: no RawDB set")
                    continue
                raise
            cache_dir.mkdir(parents=True, exist_ok=True)
            (cache_dir / f"rawdb-{model}.json").write_text(json.dumps(doc, indent=2), encoding="utf-8")
            accepted, excluded, unhashed = build_rawdb_entries(doc, src, model, golden_bodies)
            for entry in unhashed:
                if args.no_hash_download or args.dry_run:
                    report.append(f"skipped (API gives no size or sha256; --no-hash-download or --dry-run): {entry['id']}")
                    continue
                log(f"  no size or sha256 in the API for {entry['name']}; downloading once to pin it")
                api_sha = entry["sha256"]
                entry["size"], entry["sha256"] = fetch_unpinned(http, root, entry)
                if api_sha and api_sha != entry["sha256"]:
                    raise VerifyError(f"{entry['id']}: downloaded file hash {entry['sha256']} differs from the API's {api_sha}")
                accepted.append(entry)
            new_entries.extend(accepted)
            excluded_new.extend(excluded)
            if not accepted:
                gaps.append(f"{model}: no CC0 file in RawDB")
        if new_entries:
            probe_url(http, new_entries[0])
        new_by_id = {e["id"]: e for e in new_entries}
        changed = [i for i in new_by_id if i in old and old[i]["sha256"] != new_by_id[i]["sha256"]]
        if changed and not args.accept_changed:
            raise VerifyError(
                "upstream hashes changed for: " + ", ".join(sorted(changed)) +
                ". Files must not change silently; check why, then repeat with --accept-changed. Manifest not written."
            )
        added = sorted(set(new_by_id) - set(old))
        removed = sorted(i for i in old if i not in new_by_id and old[i]["body"] not in missing)
        report.append(f"rawdb: {len(added)} added, {len(removed)} removed, {len(changed)} changed, {len(excluded_new)} excluded (licence)")
        touched = set(models) - set(missing)
        m["files"] = [e for e in m["files"] if not (e["source"] == "rawdb" and e["body"] in touched)] + new_entries
        m["excluded"] = [x for x in m["excluded"] if not (x["source"] == "rawdb" and x["body"] in touched)] + excluded_new

    if "pixls" in sources:
        src = m["sources"]["pixls"]
        models = [b["model"] for b in bodies if "pixls" in b.get("sources", ["rawdb", "pixls"])]
        pinned = {e["id"] for e in m["files"] if e["source"] == "pixls"}
        pending: List[Dict[str, Any]] = [p for p in m["pending"] if p["source"] != "pixls" or p["body"] not in models]
        n_pending = 0
        seen_pending = {p["id"] for p in pending}
        for model in models:
            listing = f"{src['data_base'].rstrip('/')}/{urllib.parse.quote(src.get('make', 'Sony'))}/{urllib.parse.quote(model)}/"
            log(f"raw.pixls.us {model}")
            try:
                markup = http.get_text(listing)
            except HttpStatusError as e:
                if e.status == 404:
                    gaps.append(f"{model}: no raw.pixls.us folder")
                    continue
                raise
            for href, text in parse_links(markup):
                if not is_arw(href):
                    continue
                url = urllib.parse.urljoin(listing, href)
                name = flat_name(urllib.parse.unquote(urllib.parse.urlsplit(url).path.rsplit("/", 1)[-1]), src.get("make", "Sony"), model)
                eid = f"pixls:{model}/{name}"
                if eid in pinned or eid in seen_pending:
                    continue
                seen_pending.add(eid)
                pending.append(
                    {
                        "id": eid,
                        "source": "pixls",
                        "body": model,
                        "name": name,
                        "url": url,
                        "license": "unverified",
                        "note": "licence not stated by the listing; check the file on raw.pixls.us, then use --promote",
                    }
                )
                n_pending += 1
        m["pending"] = sorted(pending, key=lambda p: p["id"])
        report.append(f"pixls: {n_pending} candidate file(s) listed as pending (licence not checked, never downloaded until promoted)")

    m["files"] = sorted(m["files"], key=lambda e: e["id"])
    m["excluded"] = sorted(m["excluded"], key=lambda x: x["id"])
    for line in report:
        log(line)
    for g in gaps:
        log(f"gap: {g}")
    if args.dry_run:
        log("dry run: manifest not written")
        return EXIT_OK
    save_manifest(Path(args.manifest), m)
    log(f"manifest written: {args.manifest} ({len(m['files'])} files)")
    return EXIT_OK


def cmd_promote(args: argparse.Namespace, m: Dict[str, Any], http: Http, root: Path) -> int:
    pending = {p["id"]: p for p in m["pending"]}
    if args.promote not in pending:
        raise ManifestError(f"{args.promote} is not in manifest.pending (see `samples.sh --list --pending`)")
    p = pending[args.promote]
    lic = normalise_license(args.license)
    if lic not in CORPUS_LICENSES:
        raise PolicyError(f"only CC0-1.0 files can join the corpus; got {args.license!r}")
    if not args.checked_by:
        raise PolicyError("--checked-by is required: say who looked at the licence, and where (URL or name)")
    entry = {
        "id": p["id"],
        "source": p["source"],
        "body": p["body"],
        "name": p["name"],
        "source_path": p["name"],
        "url": p["url"],
        "size": 0,
        "sha256": "",
        "license": lic,
        "license_source": "manual",
        "license_checked_by": args.checked_by,
        "license_checked_on": datetime.date.today().isoformat(),
        "tier": 1,
        "sets": ["corpus", "golden"] if p["body"] in m["golden"].get("tier1_bodies", []) else ["corpus"],
        "variant_hint": guess_variant(p["name"]),
    }
    log(f"downloading {p['url']} to pin its hash")
    entry["size"], entry["sha256"] = fetch_unpinned(http, root, entry)
    m["files"].append(entry)
    m["pending"] = [x for x in m["pending"] if x["id"] != p["id"]]
    m["files"].sort(key=lambda e: e["id"])
    save_manifest(Path(args.manifest), m)
    log(f"promoted {entry['id']}: {human_mb(entry['size'])}, sha256 {entry['sha256']}")
    return EXIT_OK


# --------------------------------------------------------------------------------------------------
# Download, verify, list, coverage


def require_entries(args: argparse.Namespace, m: Dict[str, Any], entries: List[Dict[str, Any]], what: str) -> bool:
    """Handle the empty cases. Returns True if there is work to do."""
    if entries:
        return True
    if not m["files"]:
        message = (
            f"the manifest has no files yet ({what}). It ships empty because the corpus hosts "
            "(rawdb.dnglab.org, raw.pixls.us) were unreachable when it was written. On a machine that can reach "
            "them run `tools/fetch/samples.sh --refresh`, review the diff and commit tools/fetch/manifest.json."
        )
    else:
        message = f"no manifest entries match the given filters ({what})."
    if args.allow_empty:
        log(f"warning: {message}")
        return False
    raise EmptyError(message)


def run_downloads(args: argparse.Namespace, m: Dict[str, Any], entries: List[Dict[str, Any]], root: Path) -> int:
    total = sum(e["size"] for e in entries)
    if args.dry_run:
        for e in entries:
            log(f"would fetch {e['id']}  {human_mb(e['size'])}  {e['license']}  {e['url']}")
        log(f"{len(entries)} file(s), {human_mb(total)} in total")
        return EXIT_OK
    http = Http(allowed_hosts(m), delay=args.delay, timeout=args.timeout, retries=args.retries)
    cached = downloaded = 0
    for i, e in enumerate(entries, 1):
        log(f"[{i}/{len(entries)}] {e['id']}  {human_mb(e['size'])}")
        state = fetch_entry(http, root, e)
        cached += state == "cached"
        downloaded += state == "downloaded"
    log(f"done: {downloaded} downloaded, {cached} already present and verified, {len(entries)} total ({human_mb(total)})")
    return EXIT_OK


def run_verify(args: argparse.Namespace, entries: List[Dict[str, Any]], root: Path) -> int:
    bad = 0
    for e in entries:
        status = file_status(root, e)
        if status != "ok":
            bad += 1
            log(f"{status.upper():8} {e['id']}  -> {entry_dest_rel(e)}")
    log(f"verified {len(entries) - bad} of {len(entries)} file(s)")
    if bad and not args.allow_missing:
        return EXIT_FAILED
    return EXIT_OK


def print_list(entries: List[Dict[str, Any]]) -> None:
    for e in entries:
        print(f"{e['id']}\t{human_mb(e['size'])}\t{e['license']}\ttier{e['tier']}\t{e.get('variant_hint', '')}")


VARIANT_COLUMNS = ["uncompressed", "lossless", "lossless-m", "lossless-s", "craw-or-arw6", "unknown"]


def print_coverage(m: Dict[str, Any]) -> None:
    """Markdown table of variant hints per body, plus the list of bodies without any file."""
    print("| Body | Model | Group | " + " | ".join(VARIANT_COLUMNS) + " | Files |")
    print("|---|---|---|" + "---|" * (len(VARIANT_COLUMNS) + 1))
    gaps = []
    for b in m["bodies"]:
        files = [e for e in m["files"] if e["body"] == b["model"] and "corpus" in e["sets"]]
        counts = {v: 0 for v in VARIANT_COLUMNS}
        for e in files:
            counts[e.get("variant_hint", "unknown") if e.get("variant_hint") in counts else "unknown"] += 1
        cells = " | ".join(str(counts[v]) if counts[v] else "" for v in VARIANT_COLUMNS)
        print(f"| {b.get('name', '')} | {b['model']} | {b.get('group', '')} | {cells} | {len(files)} |")
        if not files:
            gaps.append(b["model"])
    print()
    print("Variant columns are hints guessed from the API text and file names (not authoritative; `ym info` is).")
    print(f"Bodies with no corpus file: {', '.join(gaps) if gaps else 'none'}")


# --------------------------------------------------------------------------------------------------
# Golden: tier 2 attribution, tier 3 list, candidate search


def write_attribution(root: Path, entries: List[Dict[str, Any]]) -> Optional[Path]:
    tier2 = [e for e in entries if e["tier"] == 2]
    if not tier2:
        return None
    lines = [
        "# Attribution for golden tier 2 files",
        "",
        "Generated by tools/fetch/golden.sh from tools/fetch/manifest.json. Do not edit.",
        "These files are freely licensed. They stay in the test bucket and in testdata/; they never ship in the app.",
        "",
        "| File | Camera | Author | Licence | Source |",
        "|---|---|---|---|---|",
    ]
    for e in sorted(tier2, key=lambda x: x["id"]):
        att = e["attribution"]
        title = att.get("title") or e["name"]
        lines.append(f"| {e['name']} | {e['body']} | {att['author']} | {e['license']} | [{title}]({att['source_url']}) |")
    out = root / "golden" / "ATTRIBUTION.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def print_tier3(m: Dict[str, Any], root: Path) -> None:
    print("Golden tier 3: copyrighted files for local evaluation only (docs/test-photo-sources.md).")
    print("They are NEVER fetched by these scripts, never committed, never uploaded, never cached in CI.")
    print("Download them by hand from the pages below, in a browser, and put them in the folder shown.\n")
    for item in m["golden"].get("tier3", []):
        print(f"* {item['provider']} ({item['body']}, about {item.get('files_expected', '?')} files)")
        print(f"    page:   {item['page_url']}")
        print(f"    terms:  {item['terms']}")
        print(f"    folder: {root / item['local_dir']}")
    print("\nAfter downloading, `golden.sh --index-tier3` records size and SHA-256 of what is there (local file only).")


def index_tier3(m: Dict[str, Any], root: Path) -> int:
    out: Dict[str, Any] = {"note": "Local only. Generated by golden.sh --index-tier3. Never commit or upload.", "files": []}
    for item in m["golden"].get("tier3", []):
        folder = root / item["local_dir"]
        if not folder.is_dir():
            continue
        for path in sorted(folder.rglob("*")):
            if path.is_file() and is_arw(path.name):
                out["files"].append(
                    {
                        "provider": item["provider"],
                        "body": item["body"],
                        "path": str(path.relative_to(root)),
                        "size": path.stat().st_size,
                        "sha256": sha256_file(path),
                    }
                )
    target = root / "golden" / "tier3" / "INDEX.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    log(f"indexed {len(out['files'])} local tier 3 file(s) -> {target}")
    return EXIT_OK


_LICENSE_HINT = re.compile(r"cc[\s-]*0|cc[\s-]*by(?:[\s-]*(?:nc|sa|nd))*(?:[\s-]*\d\.\d)?|creative\s+commons[^.<\n]{0,40}", re.I)


def cmd_candidates(args: argparse.Namespace, m: Dict[str, Any], http: Http, root: Path) -> int:
    """Search the Play Raw category for posts about the given models and save UNREVIEWED candidates.

    Nothing is written to the manifest: a person reads each post's licence and then uses --add-tier2.
    """
    src = m["sources"]["playraw"]
    base = src["api_base"].rstrip("/")
    models = args.candidates or src.get("search_models", ["ILCE-1", "ILCE-1M2", "ILCE-7RM4", "ILCE-7RM5", "ILCE-7CR"])
    results: List[Dict[str, Any]] = []
    for model in models:
        query = urllib.parse.quote(f"{model} category:play-raw")
        log(f"Play Raw search: {model}")
        found = http.get_json(f"{base}/search.json?q={query}")
        topics = found.get("topics", []) if isinstance(found, dict) else []
        for topic in topics[: args.max_topics]:
            tid = topic.get("id")
            if not isinstance(tid, int):
                continue
            doc = http.get_json(f"{base}/t/{tid}.json")
            posts = doc.get("post_stream", {}).get("posts", []) if isinstance(doc, dict) else []
            attachments: List[Dict[str, str]] = []
            hints: Set[str] = set()
            raw_hints: Set[str] = set()
            author = ""
            for post in posts:
                cooked = post.get("cooked", "") if isinstance(post, dict) else ""
                for href, text in parse_links(cooked):
                    if is_arw(href) or is_arw(text):
                        attachments.append({"url": urllib.parse.urljoin(base + "/", href), "name": text or href, "posted_by": str(post.get("username", ""))})
                        author = author or str(post.get("username", ""))
                text_only = re.sub(r"<[^>]+>", " ", cooked)
                for h in _LICENSE_HINT.finditer(text_only):
                    raw_hints.add(" ".join(h.group(0).split()))
                    hints.add(normalise_license(h.group(0)))
            if attachments:
                results.append(
                    {
                        "model": model,
                        "topic_id": tid,
                        "title": topic.get("title", ""),
                        "topic_url": f"{base}/t/{topic.get('slug', 'topic')}/{tid}",
                        "author": author,
                        "attachments": attachments,
                        "license_hints": sorted(h for h in hints if h != "unknown"),
                        "license_text": sorted(raw_hints)[:10],
                        "status": "unreviewed",
                    }
                )
    target = root / "golden" / "tier2-candidates.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({"note": "UNREVIEWED. Read each post's licence, then use golden.sh --add-tier2.", "candidates": results}, indent=2) + "\n", encoding="utf-8")
    log(f"{len(results)} candidate post(s) with ARW attachments -> {target}")
    return EXIT_OK


def cmd_add_tier2(args: argparse.Namespace, m: Dict[str, Any], http: Http, root: Path) -> int:
    lic = normalise_license(args.license)
    if lic not in TIER2_LICENSES:
        raise PolicyError(f"licence {args.license!r} ({lic}) is not allowed in golden tier 2: CC0, CC BY or CC BY-SA only")
    for need in ("author", "source_url"):
        if not getattr(args, need):
            raise PolicyError(f"--{need.replace('_', '-')} is required with --add-tier2")
    if not args.body or len(args.body) != 1:
        raise PolicyError("--add-tier2 needs exactly one --body MODEL")
    body = args.body[0]
    known = {b["model"] for b in m["bodies"]}
    if body not in known:
        raise ManifestError(f"unknown body {body!r}; add it to manifest.bodies first")
    url = args.add_tier2
    check_url(url, allowed_hosts(m))
    base_name = args.name or urllib.parse.unquote(urllib.parse.urlsplit(url).path.rsplit("/", 1)[-1])
    name = flat_name(base_name, "Sony", body)
    entry = {
        "id": f"playraw:{body}/{name}",
        "source": "playraw",
        "body": body,
        "name": name,
        "source_path": base_name,
        "url": url,
        "size": 0,
        "sha256": "",
        "license": lic,
        "license_source": "forum-post",
        "license_checked_on": datetime.date.today().isoformat(),
        "tier": 2,
        "sets": ["golden"],
        "variant_hint": guess_variant(name),
        "attribution": {"author": args.author, "source_url": args.source_url, "title": args.title or ""},
    }
    if any(e["id"] == entry["id"] for e in m["files"]):
        raise ManifestError(f"{entry['id']} is already in the manifest")
    log(f"downloading {url} to pin its hash")
    entry["size"], entry["sha256"] = fetch_unpinned(http, root, entry)
    m["files"].append(entry)
    m["files"].sort(key=lambda e: e["id"])
    save_manifest(Path(args.manifest), m)
    log(f"added {entry['id']}: {human_mb(entry['size'])}, sha256 {entry['sha256']}")
    return EXIT_OK


# --------------------------------------------------------------------------------------------------
# Command line


def add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--manifest", default=str(DEFAULT_MANIFEST), help="manifest file (default: tools/fetch/manifest.json)")
    p.add_argument("--dest", default=os.environ.get("YM_TESTDATA", str(DEFAULT_DEST)), help="test data folder (default: testdata/, or $YM_TESTDATA)")
    p.add_argument("--body", action="append", metavar="MODEL", help="only this camera model, e.g. ILCE-7RM4 (repeatable)")
    p.add_argument("--delay", type=float, default=1.0, help="seconds between requests (default 1; one connection at a time)")
    p.add_argument("--timeout", type=float, default=60.0, help="per-request timeout in seconds")
    p.add_argument("--retries", type=int, default=3, help="retries on HTTP 429/5xx and dropped transfers")
    p.add_argument("--allow-empty", action="store_true", help="do not fail when there is nothing to fetch")
    p.add_argument("--dry-run", action="store_true", help="show what would be done; fetch nothing, write nothing")
    p.add_argument("--verify", action="store_true", help="check files already on disk against the manifest; no network")
    p.add_argument("--allow-missing", action="store_true", help="with --verify: missing files are not an error")
    p.add_argument("--list", action="store_true", help="list the selected manifest entries and exit")
    p.add_argument("--per-body", type=int, metavar="N", help="at most N files per camera model (first by id)")
    p.add_argument("--max-file-mb", type=float, metavar="MB", help="skip files larger than this")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ymfetch.py",
        description="Yellowmoon test-data fetcher. Use tools/fetch/samples.sh and tools/fetch/golden.sh.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    s = sub.add_parser("samples", help="the CC0 ARW corpus (testdata/arw/<model>/)")
    add_common(s)
    s.add_argument("--refresh", action="store_true", help="query RawDB and raw.pixls.us and rewrite the manifest")
    s.add_argument("--source", choices=["rawdb", "pixls", "all"], default="all", help="with --refresh (default all)")
    s.add_argument("--group", action="append", choices=["required", "sibling", "recent", "legacy"], help="with --refresh: body groups")
    s.add_argument("--accept-changed", action="store_true", help="with --refresh: accept changed upstream hashes")
    s.add_argument("--no-hash-download", action="store_true", help="with --refresh: do not download files that lack a sha256")
    s.add_argument("--promote", metavar="ID", help="move a pending raw.pixls.us file into the corpus after a manual CC0 check")
    s.add_argument("--license", help="with --promote / --add-tier2: the licence you checked, e.g. CC0-1.0")
    s.add_argument("--checked-by", help="with --promote: who checked the licence, and where")
    s.add_argument("--pending", action="store_true", help="with --list: show raw.pixls.us candidates awaiting a licence check")
    s.add_argument("--coverage", action="store_true", help="print the body x variant coverage table (Markdown)")

    g = sub.add_parser("golden", help="the golden photo set: tier 1 (CC0) and tier 2 (free licence)")
    add_common(g)
    g.add_argument("--tier", type=int, action="append", choices=[1, 2], help="only this tier (repeatable; default both)")
    g.add_argument("--tier3", action="store_true", help="print the local-only tier 3 list and exit (never fetched)")
    g.add_argument("--index-tier3", action="store_true", help="record size and SHA-256 of the tier 3 files present locally")
    g.add_argument("--candidates", nargs="*", metavar="MODEL", help="search Play Raw for unreviewed tier 2 candidates (network)")
    g.add_argument("--max-topics", type=int, default=15, help="with --candidates: topics per model")
    g.add_argument("--add-tier2", metavar="URL", help="download one reviewed Play Raw file and add it to the manifest")
    g.add_argument("--license", help="with --add-tier2: the licence you checked, e.g. CC-BY-SA-4.0")
    g.add_argument("--author", help="with --add-tier2: the photographer's name")
    g.add_argument("--source-url", help="with --add-tier2: URL of the forum post")
    g.add_argument("--title", help="with --add-tier2: post title")
    g.add_argument("--name", help="with --add-tier2: file name to use")
    return parser


def run(args: argparse.Namespace) -> int:
    manifest_path = Path(args.manifest)
    m = load_manifest(manifest_path)
    root = Path(args.dest)
    max_bytes = int(args.max_file_mb * 1024 * 1024) if args.max_file_mb else None

    def new_http() -> Http:
        return Http(allowed_hosts(m), delay=args.delay, timeout=args.timeout, retries=args.retries)

    if args.command == "samples":
        if args.coverage:
            print_coverage(m)
            return EXIT_OK
        if args.list and args.pending:
            for p in m["pending"]:
                print(f"{p['id']}\t{p['license']}\t{p['url']}")
            return EXIT_OK
        if args.refresh:
            return cmd_refresh(args, m, new_http(), root)
        if args.promote:
            return cmd_promote(args, m, new_http(), root)
        entries = select_entries(m, sets={"corpus"}, bodies=args.body, per_body=args.per_body, max_file_bytes=max_bytes)
        if args.list:
            print_list(entries)
            return EXIT_OK
        if not require_entries(args, m, entries, "samples"):
            return EXIT_OK
        if args.verify:
            return run_verify(args, entries, root)
        return run_downloads(args, m, entries, root)

    # golden
    if args.tier3:
        print_tier3(m, root)
        return EXIT_OK
    if args.index_tier3:
        return index_tier3(m, root)
    if args.candidates is not None:
        return cmd_candidates(args, m, new_http(), root)
    if args.add_tier2:
        return cmd_add_tier2(args, m, new_http(), root)
    tiers = set(args.tier) if args.tier else {1, 2}
    entries = select_entries(m, sets={"golden"}, tiers=tiers, bodies=args.body, per_body=args.per_body, max_file_bytes=max_bytes)
    if args.list:
        print_list(entries)
        return EXIT_OK
    if not require_entries(args, m, entries, "golden set"):
        return EXIT_OK
    if args.verify:
        return run_verify(args, entries, root)
    code = run_downloads(args, m, entries, root)
    if code == EXIT_OK and not args.dry_run:
        attribution = write_attribution(root, entries)
        if attribution:
            log(f"attribution written: {attribution}")
    return code


def main(argv: Optional[Sequence[str]] = None) -> int:
    if sys.version_info < (3, 8):
        print("error: Python 3.8 or newer is required", file=sys.stderr)
        return EXIT_USAGE
    try:
        args = build_parser().parse_args(argv)
    except SystemExit as e:  # argparse exits with 2 on usage errors, 0 on --help
        return int(e.code or 0)
    try:
        return run(args)
    except NetworkError as e:
        log(network_help(e))
        return e.exit_code
    except HttpStatusError as e:
        hint = ""
        if e.status in (401, 403):
            hint = "\n  The site refuses automated access. Download the file in a browser once and pin it by hand."
        log(f"error: {e}{hint}")
        return e.exit_code
    except FetchError as e:
        log(f"error: {e}")
        return e.exit_code
    except KeyboardInterrupt:
        log("interrupted; partial downloads (*.part) are kept and will be resumed")
        return 130


if __name__ == "__main__":
    sys.exit(main())
