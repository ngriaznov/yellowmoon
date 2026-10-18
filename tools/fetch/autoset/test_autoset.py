"""Unit tests for autoset.py. Standard library only; no real network.

Run:  python3 -I -m unittest discover -s tools/fetch/autoset -p 'test_*.py' -v

A small HTTP server on 127.0.0.1 plays raw.pixls.us and RawDB: a listing, a paged set index that ignores
its maker filter, set details, a download that answers 303 to object storage, a file that fails once with
503, and a file that always fails. A fake dump program (written to the temp folder) stands in for the Rust
example and prints rows according to FAKE_DUMP_MODE. The answers follow docs/18 section 6 and the layouts
seen on the live sites when the manifest was written, so these tests prove the logic, not the live sites.
"""

from __future__ import annotations

import collections
import contextlib
import hashlib
import io
import json
import os
import shlex
import shutil
import socket
import socketserver
import sys
import tempfile
import threading
import unittest
import urllib.parse
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import autoset  # noqa: E402


def blob(seed: int, size: int) -> bytes:
    out = bytearray()
    counter = 0
    while len(out) < size:
        out += hashlib.sha256(f"{seed}:{counter}".encode()).digest()
        counter += 1
    return bytes(out[:size])


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


LIC_CC0 = "<a href='https://creativecommons.org/publicdomain/zero/1.0/' title='Creative Commons 0 - Public Domain' class='cc'>co</a>"
LIC_NC_SA = "<a href='http://creativecommons.org/licenses/by-nc-sa/4.0/' title='Creative Commons - Attribution, Non-Commercial, ShareAlike 4.0' class='cc'>ncsa</a>"

NAME_A = "Sony - TEST-1 - 14bit (3:2).ARW"
NAME_SHARED = "Sony - TEST-1 - shared.ARW"
NAME_RAWDB = "TEST-RAW-1_B.ARW"
NAME_FLAKY = "flaky.ARW"
NAME_ALWAYS = "always503.ARW"
NAME_GONE = "gone.ARW"
FILES: Dict[str, bytes] = {
    NAME_A: blob(1, 4000),
    NAME_SHARED: blob(2, 3500),
    NAME_RAWDB: blob(3, 5000),
    NAME_FLAKY: blob(4, 2500),
    NAME_ALWAYS: blob(5, 1000),
}
SHARED_RAW_SHA = sha(FILES[NAME_SHARED])
DNG_SHA = "1" * 64
NC_SHA = "2" * 64

RAWDB_SETS = [
    {"maker": "Sony", "model": "TEST-RAW-1", "license": "CC0 1.0"},
    {"maker": "Sony", "model": "TEST-RAW-2", "license": "CC BY-NC-SA 4.0"},
    # The server ignores maker= on purpose: the client must filter by maker itself.
    {"maker": "Canon", "model": "TEST-CANON", "license": "CC0 1.0"},
]
RAWDB_DETAILS: Dict[str, Dict[str, Any]] = {
    "TEST-RAW-1": {
        "maker": "Sony",
        "model": "TEST-RAW-1",
        "license": "CC0 1.0",
        "categories": {
            "raw_modes": [
                {"path": "raw_modes/TEST-RAW-1_B.ARW", "category": "raw_modes", "extension": "arw",
                 "size": len(FILES[NAME_RAWDB]), "license": "CC0 1.0", "sha256": sha(FILES[NAME_RAWDB]),
                 "tags": ["14bits", "uncompressed"]},
                {"path": "raw_modes/TEST-RAW-1_shared.ARW", "category": "raw_modes", "extension": "arw",
                 "size": len(FILES[NAME_SHARED]), "license": "CC0 1.0", "sha256": SHARED_RAW_SHA, "tags": []},
                {"path": "raw_modes/TEST-RAW-1_x.DNG", "category": "raw_modes", "extension": "dng",
                 "size": 10, "license": "CC0 1.0", "sha256": DNG_SHA, "tags": []},
                {"path": "raw_modes/TEST-RAW-1_nc.ARW", "category": "raw_modes", "extension": "arw",
                 "size": 10, "license": "CC BY-NC-SA 4.0", "sha256": NC_SHA, "tags": []},
            ]
        },
    },
    "TEST-RAW-2": {
        "maker": "Sony",
        "model": "TEST-RAW-2",
        "license": "CC BY-NC-SA 4.0",
        "categories": {"raw_modes": [{"path": "raw_modes/TEST-RAW-2_a.ARW", "extension": "arw", "size": 9,
                                      "license": "CC BY-NC-SA 4.0", "sha256": "3" * 64, "tags": []}]},
    },
}


def file_html(href: str, text: str, digest: str, size_text: str) -> str:
    return (f"<a href='{href}'>{text}</a><div class='checksumdata'>"
            f"<span title='sha256 Checksum'>{digest}</span>&nbsp;({size_text})</div>")


def listing(base: str) -> Dict[str, Any]:
    def row(model: str, mode: str, lic: str, name: str, digest: str) -> List[Any]:
        href = f"{base}/files/{name}"
        return ["Sony", model, mode, 1.0, "", lic, "2020-01-01", file_html(href, name, digest, "0.01MB"), "<a>exif</a>"]

    return {"data": [
        row("TEST-1", "14bit (3:2)", LIC_CC0, NAME_A, sha(FILES[NAME_A])),
        row("TEST-1", "14bit (3:2)", LIC_CC0, NAME_SHARED, SHARED_RAW_SHA),
        row("TEST-1", "nc", LIC_NC_SA, "Sony - TEST-1 - nc.ARW", "4" * 64),
        row("TEST-1", "arq", LIC_CC0, "Sony - TEST-1 - arq.ARQ", "5" * 64),
        ["Canon", "EOS 7D", "RAW", 1.0, "", LIC_CC0, "2020-01-01", file_html(f"{base}/files/Canon - x.CR2", "Canon - x.CR2", "6" * 64, "1.00MB"), "x"],
    ]}


class State:
    """What the fake server has been asked for, and the state of the flaky file."""

    def __init__(self) -> None:
        self.base = ""
        self.hits: List[str] = []
        self.flaky_left = 1


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    state = State()

    def log_message(self, *args: Any) -> None:
        pass

    def _send(self, code: int, body: bytes = b"", ctype: str = "application/octet-stream",
              extra: Optional[Dict[str, str]] = None) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        url = urllib.parse.urlsplit(self.path)
        path = urllib.parse.unquote(url.path)
        query = urllib.parse.parse_qs(url.query)
        st = self.state
        st.hits.append(path)
        name = path.rsplit("/", 1)[-1]
        if path == "/pixls/listing.json":
            self._send(200, json.dumps(listing(st.base)).encode(), "application/json")
        elif path == "/rawdb/api/sets":
            offset = int(query.get("offset", ["0"])[0])
            body = {"sets": RAWDB_SETS[offset:offset + 1], "total": len(RAWDB_SETS), "limit": 1, "offset": offset}
            self._send(200, json.dumps(body).encode(), "application/json")
        elif path.startswith("/rawdb/api/sets/Sony/"):
            detail = RAWDB_DETAILS.get(name)
            if detail is None:
                self._send(404, b"no such set")
            else:
                self._send(200, json.dumps(detail).encode(), "application/json")
        elif path.startswith("/rawdb/api/download/"):
            self._send(303, b"", "text/plain", {"Location": st.base + "/storage/" + urllib.parse.quote(name)})
        elif path.startswith("/storage/") or path.startswith("/files/"):
            if name in FILES:
                self._send(200, FILES[name])
            else:
                self._send(404, b"not here")
        elif path.startswith("/flaky/"):
            if st.flaky_left > 0:
                st.flaky_left -= 1
                self._send(503, b"busy")
            else:
                self._send(200, FILES[name])
        elif path.startswith("/always503/"):
            self._send(503, b"busy")
        elif path.startswith("/bad/"):
            self._send(200, blob(99, len(FILES[name])))
        elif path.startswith("/gone/"):
            self._send(404, b"gone")
        else:
            self._send(404, b"not found")


class QuietServer(socketserver.ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


FAKE_DUMP = r'''
"""Fake dump program for the autoset tests: prints rows for the file it is given (docs/18 section 6)."""
import hashlib, json, math, os, sys
args = sys.argv[1:]
mode = os.environ.get("FAKE_DUMP_MODE", "ok")
path = args[-1]
source = args[args.index("--source") + 1]
with open(path, "rb") as f:
    data = f.read()
digest = hashlib.sha256(data).hexdigest()
if mode == "exit1":
    sys.stderr.write("skip " + os.path.basename(path) + ": synthetic failure\n")
    sys.exit(1)
features = [0.01 * i for i in range(39)]
if mode == "short_features":
    features = features[:38]
if mode == "nan":
    features[3] = float("nan")
base = {"file": os.path.basename(path), "sha256": digest, "source": source, "licence": "CC0-1.0", "body": None,
        "capture_time": "2024-05-01T10:12:33", "aug_k": 0.0, "features": features, "key": -1.0,
        "meta": {"dro": "off", "iso": 100}, "label_ev": 0.5}
rows = [dict(base, aug_k=-1.0 if mode == "no_base" else 0.0)]
if mode != "no_base":
    rows.append(dict(base, aug_k=-1.0))
if mode == "sha_other":
    rows[0]["sha256"] = "0" * 64
if mode == "bad_licence":
    rows[0]["licence"] = "CC-BY-4.0"
if mode == "body_set":
    rows[0]["body"] = "FROM-DUMP"
if mode == "bad_json":
    print("{not json")
for r in rows:
    print(json.dumps(r, allow_nan=True))
'''


class Base(unittest.TestCase):
    """Starts the fake server once per test class and gives each test a temp folder."""

    server: HTTPServer
    base: str

    @classmethod
    def setUpClass(cls) -> None:
        cls.server = QuietServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()
        cls.base = f"http://127.0.0.1:{cls.server.server_address[1]}"
        Handler.state = State()
        Handler.state.base = cls.base

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()

    def setUp(self) -> None:
        Handler.state = State()
        Handler.state.base = self.base
        self.tmp = Path(tempfile.mkdtemp(prefix="autoset-test-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.dump_script = self.tmp / "fake_dump.py"
        self.dump_script.write_text(FAKE_DUMP, encoding="utf-8")
        env = mock.patch.dict(os.environ, {"NO_PROXY": "127.0.0.1,localhost", "no_proxy": "127.0.0.1,localhost",
                                           "FAKE_DUMP_MODE": "ok"})
        env.start()
        self.addCleanup(env.stop)
        backoff = mock.patch.object(autoset, "BACKOFF_S", 0.0)
        backoff.start()
        self.addCleanup(backoff.stop)

    @property
    def hits(self) -> List[str]:
        return Handler.state.hits

    def dump_cmd(self) -> str:
        return shlex.join([sys.executable, "-I", str(self.dump_script)])

    def manifest_path(self) -> Path:
        return self.tmp / "manifest.json"


def run_cli(argv: List[str]) -> Tuple[int, str]:
    err = io.StringIO()
    out = io.StringIO()
    with contextlib.redirect_stderr(err), contextlib.redirect_stdout(out):
        code = autoset.main(argv)
    return code, out.getvalue() + err.getvalue()


def make_entry(entry_id: str, source: str, body: str, name: str, url: str, digest: str,
               size: Optional[int], *, mode: Optional[str] = None, also: Optional[List[str]] = None) -> Dict[str, Any]:
    return {"id": entry_id, "source": source, "body": body, "name": name, "mode": mode, "url": url,
            "sha256": digest, "size": size, "size_approx": None if size is not None else "0.01MB",
            "licence": "CC0-1.0", "also": also or []}


def manifest_for(entries: List[Dict[str, Any]], base: str) -> Dict[str, Any]:
    return {"schema_version": 1, "description": "test", "files": entries,
            "sources": {"pixls": {"listing": base + "/pixls/listing.json"},
                        "rawdb": {"api": base + "/rawdb/api/sets", "download": base + "/rawdb/api/download"}}}


# --------------------------------------------------------------------------------------------------
# Parsing and policy (no network)


class ParsingTests(unittest.TestCase):
    def test_pixls_keeps_only_cc0_sony_arw(self) -> None:
        stats: collections.Counter[str] = collections.Counter()
        cands = autoset.parse_pixls(listing("https://example.invalid"), stats)
        self.assertEqual(len(cands), 2)
        a = next(c for c in cands if c.name == NAME_A)
        self.assertEqual(a.key, "pixls")
        self.assertEqual(a.id, "pixls:TEST-1/" + NAME_A)
        self.assertEqual(a.body, "TEST-1")
        self.assertEqual(a.mode, "14bit (3:2)")
        self.assertEqual(a.sha256, sha(FILES[NAME_A]))
        self.assertEqual(a.size_approx, "0.01MB")
        self.assertIsNone(a.size)
        self.assertEqual(a.source, autoset.PIXLS)
        self.assertEqual(stats["pixls not CC0 1.0"], 1)
        self.assertEqual(stats["pixls not ARW (SR2, SRF, ARQ)"], 1)
        self.assertEqual(stats["pixls CC0 ARW rows"], 2)

    def test_pixls_url_is_percent_encoded(self) -> None:
        cands = autoset.parse_pixls(listing("https://raw.pixls.us"), collections.Counter())
        a = next(c for c in cands if c.name == NAME_A)
        self.assertTrue(a.url.endswith("/files/Sony%20-%20TEST-1%20-%2014bit%20(3:2).ARW"), a.url)

    def test_rawdb_keeps_cc0_arw_only(self) -> None:
        stats: collections.Counter[str] = collections.Counter()
        cands = autoset.rawdb_candidates(RAWDB_DETAILS["TEST-RAW-1"], "TEST-RAW-1", "https://rawdb.dnglab.org/api/download", stats)
        self.assertEqual([c.name for c in cands], [NAME_RAWDB, "TEST-RAW-1_shared.ARW"])
        b = cands[0]
        self.assertEqual(b.id, "rawdb:TEST-RAW-1/raw_modes/TEST-RAW-1_B.ARW")
        self.assertEqual(b.size, len(FILES[NAME_RAWDB]))
        self.assertEqual(b.mode, "14bits, uncompressed")
        self.assertEqual(b.sha256, sha(FILES[NAME_RAWDB]))
        self.assertEqual(b.url, "https://rawdb.dnglab.org/api/download/Sony/TEST-RAW-1/raw_modes/TEST-RAW-1_B.ARW")
        self.assertEqual(stats["RawDB not ARW"], 1)
        self.assertEqual(stats["RawDB not CC0 1.0"], 1)

    def test_rawdb_download_url_encodes_each_segment(self) -> None:
        detail = {"maker": "Sony", "categories": {"raw_modes": [
            {"path": "raw_modes/DSC-HX95_ISO_100_12bits_Sony ARW Compressed.ARW", "extension": "ARW",
             "size": 5, "license": "CC0 1.0", "sha256": "a" * 64, "tags": ["12bits"]}]}}
        (c,) = autoset.rawdb_candidates(detail, "DSC-HX95", "https://rawdb.dnglab.org/api/download", collections.Counter())
        self.assertEqual(c.url, "https://rawdb.dnglab.org/api/download/Sony/DSC-HX95/raw_modes/DSC-HX95_ISO_100_12bits_Sony%20ARW%20Compressed.ARW")
        self.assertEqual(c.name, "DSC-HX95_ISO_100_12bits_Sony ARW Compressed.ARW")

    def test_disambiguate_suffixes_different_files_and_keeps_one_copy_of_the_same_file(self) -> None:
        def cand(digest: str) -> autoset.Candidate:
            return autoset.Candidate(key="pixls", id="pixls:M/x.ARW", body="M", name="x.ARW", mode=None,
                                     url="https://raw.pixls.us/x", sha256=digest)
        out = autoset.disambiguate([cand("a" * 64), cand("b" * 64), cand("a" * 64)])
        self.assertEqual(sorted(c.id for c in out), sorted([f"pixls:M/x.ARW #{'a' * 12}", f"pixls:M/x.ARW #{'b' * 12}"]))
        same = autoset.disambiguate([cand("c" * 64), cand("c" * 64)])
        self.assertEqual([c.id for c in same], ["pixls:M/x.ARW"])

    def test_merge_one_entry_per_sha_with_other_source_under_also(self) -> None:
        pixls = autoset.Candidate(key="pixls", id="pixls:TEST-1/" + NAME_SHARED, body="TEST-1", name=NAME_SHARED,
                                  mode="14bit", url="https://raw.pixls.us/a", sha256=SHARED_RAW_SHA, size_approx="3.40MB")
        rawdb = autoset.Candidate(key="rawdb", id="rawdb:TEST-RAW-1/raw_modes/TEST-RAW-1_shared.ARW", body="TEST-RAW-1",
                                  name="TEST-RAW-1_shared.ARW", mode=None, url="https://rawdb.dnglab.org/a",
                                  sha256=SHARED_RAW_SHA, size=len(FILES[NAME_SHARED]))
        files = autoset.merge([], {"pixls": [pixls], "rawdb": [rawdb]}, {"pixls", "rawdb"}, False)
        self.assertEqual(len(files), 1)
        e = files[0]
        self.assertEqual(e["id"], pixls.id)
        self.assertEqual(e["source"], autoset.PIXLS)
        self.assertEqual(e["also"], [rawdb.id])
        self.assertEqual(e["size"], len(FILES[NAME_SHARED]))  # exact size taken from RawDB
        self.assertEqual(e["size_approx"], "3.40MB")

    def test_merge_refuses_a_changed_sha_unless_accepted(self) -> None:
        old = [make_entry("pixls:M/x.ARW", autoset.PIXLS, "M", "x.ARW", "https://raw.pixls.us/x", "a" * 64, None)]
        new = {"pixls": [autoset.Candidate(key="pixls", id="pixls:M/x.ARW", body="M", name="x.ARW", mode=None,
                                           url="https://raw.pixls.us/x", sha256="b" * 64)]}
        with self.assertRaises(autoset.ManifestError) as ctx:
            autoset.merge(old, new, {"pixls"}, False)
        self.assertIn("--accept-changed", str(ctx.exception))
        accepted = autoset.merge(old, new, {"pixls"}, True)
        self.assertEqual(accepted[0]["sha256"], "b" * 64)

    def test_merge_keeps_entries_of_sources_not_refreshed(self) -> None:
        rawdb_entry = make_entry("rawdb:M/a.ARW", autoset.RAWDB, "M", "a.ARW", "https://rawdb.dnglab.org/a", "c" * 64, 7)
        pixls_entry = make_entry("pixls:M/old.ARW", autoset.PIXLS, "M", "old.ARW", "https://raw.pixls.us/old", "d" * 64, None)
        files = autoset.merge([rawdb_entry, pixls_entry], {"pixls": []}, {"pixls"}, False)
        self.assertEqual([e["id"] for e in files], ["rawdb:M/a.ARW"])

    def test_format_manifest_is_valid_json_with_one_entry_per_line(self) -> None:
        files = [make_entry("pixls:M/a.ARW", autoset.PIXLS, "M", "a.ARW", "https://raw.pixls.us/a", "e" * 64, None),
                 make_entry("rawdb:M/b.ARW", autoset.RAWDB, "M", "b.ARW", "https://rawdb.dnglab.org/b", "f" * 64, 9)]
        text = autoset.format_manifest({"schema_version": 1, "description": autoset.DESCRIPTION,
                                        "sources": autoset.DEFAULT_SOURCES, "files": files})
        self.assertEqual(json.loads(text)["files"][1]["size"], 9)
        entry_lines = [line for line in text.splitlines() if line.startswith("   {")]
        self.assertEqual(len(entry_lines), 2)
        self.assertNotIn("time", text.lower())  # no timestamps: the CI cache key hashes this file

    def test_validate_manifest_refuses_policy_breaches(self) -> None:
        good = make_entry("rawdb:M/a.ARW", autoset.RAWDB, "M", "a.ARW", "https://rawdb.dnglab.org/a", "a" * 64, 1)
        cases = {
            "licence": dict(good, licence="CC-BY-SA-4.0"),
            "sha": dict(good, sha256="ABC"),
            "host": dict(good, url="https://evil.example/a.ARW"),
            "plain http": dict(good, url="http://rawdb.dnglab.org/a.ARW"),
            "source": dict(good, source="example.org"),
            "name path": dict(good, name="x/y.ARW"),
        }
        for label, entry in cases.items():
            with self.subTest(label):
                with self.assertRaises(autoset.ManifestError):
                    autoset.validate_manifest({"schema_version": 1, "description": "d", "files": [entry],
                                               "sources": autoset.DEFAULT_SOURCES})

    def test_validate_manifest_refuses_one_sha_in_two_entries(self) -> None:
        a = make_entry("rawdb:M/a.ARW", autoset.RAWDB, "M", "a.ARW", "https://rawdb.dnglab.org/a", "a" * 64, 1)
        b = make_entry("rawdb:M/b.ARW", autoset.RAWDB, "M", "b.ARW", "https://rawdb.dnglab.org/b", "a" * 64, 1)
        with self.assertRaises(autoset.ManifestError):
            autoset.validate_manifest({"schema_version": 1, "description": "d", "files": [a, b], "sources": autoset.DEFAULT_SOURCES})

    def test_size_text_and_url_helpers(self) -> None:
        self.assertEqual(autoset.parse_size_approx("10.55MB"), int(10.55 * (1 << 20)))
        self.assertEqual(autoset.parse_size_approx("1GB"), 1 << 30)
        self.assertIsNone(autoset.parse_size_approx("10.55"))
        self.assertIsNone(autoset.parse_size_approx(""))
        self.assertEqual(autoset.quote_segment("a b(1)!*'.ARW"), "a%20b(1)!*'.ARW")
        self.assertEqual(autoset.quote_url_path("https://raw.pixls.us/getfile.php/1/nice/A B (3:2).ARW"),
                         "https://raw.pixls.us/getfile.php/1/nice/A%20B%20(3:2).ARW")
        self.assertIsNone(autoset.url_problem("https://raw.pixls.us/a"))
        self.assertIsNone(autoset.url_problem("https://nbg1.your-objectstorage.com/a"))
        self.assertIsNotNone(autoset.url_problem("https://example.org/a"))
        self.assertIsNone(autoset.url_problem("http://127.0.0.1:9/a"))


class ResolveDumpTests(unittest.TestCase):
    def test_explicit_command_is_split_with_shlex(self) -> None:
        argv = autoset.resolve_dump(f"{sys.executable} -I 'a path/dump.py'")
        self.assertEqual(argv, [sys.executable, "-I", "a path/dump.py"])

    def test_environment_overrides_the_default(self) -> None:
        with mock.patch.dict(os.environ, {"YM_AUTOSET_DUMP": sys.executable}):
            self.assertEqual(autoset.resolve_dump(None), [sys.executable])

    def test_missing_program_is_a_setup_error(self) -> None:
        with self.assertRaises(autoset.SetupError) as ctx:
            autoset.resolve_dump("/nonexistent/autoset-dump-xyz")
        self.assertEqual(ctx.exception.exit_code, autoset.EXIT_USAGE)

    def test_missing_default_builds_it_with_cargo_then_refuses_if_still_missing(self) -> None:
        with mock.patch.dict(os.environ), mock.patch.object(autoset, "DEFAULT_DUMP", Path("/nonexistent/autoset_dump")), \
                mock.patch.object(autoset.subprocess, "run", return_value=mock.Mock(returncode=1)) as run, \
                contextlib.redirect_stderr(io.StringIO()):
            os.environ.pop("YM_AUTOSET_DUMP", None)
            with self.assertRaises(autoset.SetupError):
                autoset.resolve_dump(None)
        self.assertEqual(run.call_args[0][0], autoset.DUMP_BUILD_COMMAND)


# --------------------------------------------------------------------------------------------------
# refresh against the local server


class RefreshTests(Base):
    def write_skeleton(self) -> None:
        """An empty manifest whose sources point at the local server (refresh reads them from here)."""
        self.manifest_path().write_text(json.dumps(manifest_for([], self.base)), encoding="utf-8")

    def test_refresh_writes_the_merged_manifest(self) -> None:
        path = self.manifest_path()
        self.write_skeleton()
        code, _ = run_cli(["refresh", "--manifest", str(path), "--delay", "0"])
        self.assertEqual(code, autoset.EXIT_OK)
        m = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual([e["id"] for e in m["files"]], [
            "pixls:TEST-1/" + NAME_A,
            "pixls:TEST-1/" + NAME_SHARED,
            "rawdb:TEST-RAW-1/raw_modes/TEST-RAW-1_B.ARW",
        ])
        shared = m["files"][1]
        self.assertEqual(shared["also"], ["rawdb:TEST-RAW-1/raw_modes/TEST-RAW-1_shared.ARW"])
        self.assertEqual(shared["size"], len(FILES[NAME_SHARED]))
        self.assertEqual(m["files"][2]["mode"], "14bits, uncompressed")
        self.assertNotIn("TEST-CANON", " ".join(self.hits))
        self.assertIn("/rawdb/api/sets/Sony/TEST-RAW-2", self.hits)
        self.assertEqual(sum(1 for h in self.hits if h == "/rawdb/api/sets"), 3)  # paged, one set per page

    def test_refresh_dry_run_writes_nothing(self) -> None:
        path = self.manifest_path()
        self.write_skeleton()
        before = path.read_text(encoding="utf-8")
        code, out = run_cli(["refresh", "--manifest", str(path), "--delay", "0", "--dry-run"])
        self.assertEqual(code, autoset.EXIT_OK)
        self.assertEqual(path.read_text(encoding="utf-8"), before)
        self.assertIn("dry run", out)
        self.assertIn("pixls: 2 CC0 Sony ARW rows", out)

    def test_refresh_refuses_a_changed_hash(self) -> None:
        path = self.manifest_path()
        old = manifest_for([make_entry("pixls:TEST-1/" + NAME_A, autoset.PIXLS, "TEST-1", NAME_A,
                                       autoset.quote_url_path(self.base + "/files/" + NAME_A), "9" * 64, None)], self.base)
        path.write_text(json.dumps(old), encoding="utf-8")
        before = path.read_text(encoding="utf-8")
        code, out = run_cli(["refresh", "--manifest", str(path), "--delay", "0"])
        self.assertEqual(code, autoset.EXIT_FAILED)
        self.assertIn("--accept-changed", out)
        self.assertEqual(path.read_text(encoding="utf-8"), before)

    def test_refresh_stops_with_network_exit_when_the_host_refuses(self) -> None:
        probe = socket.socket()
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
        probe.close()
        dead = f"http://127.0.0.1:{port}"
        path = self.manifest_path()
        path.write_text(json.dumps(manifest_for([], dead)), encoding="utf-8")
        code, out = run_cli(["refresh", "--manifest", str(path), "--delay", "0", "--source", "pixls"])
        self.assertEqual(code, autoset.EXIT_NETWORK)
        self.assertIn("refuses", out)


# --------------------------------------------------------------------------------------------------
# build against the local server with the fake dump


class BuildTests(Base):
    def setUp(self) -> None:
        super().setUp()
        self.dest = self.tmp / "testdata"
        self.out = self.tmp / "rows" / "rows.jsonl"
        self.workdir = self.dest / "autoset" / "tmp"
        self.failures = self.out.parent / "failures.jsonl"
        self.entry_ok = make_entry("pixls:TEST-1/" + NAME_A, autoset.PIXLS, "TEST-1", NAME_A,
                                   autoset.quote_url_path(f"{self.base}/files/{NAME_A}"), sha(FILES[NAME_A]), None)
        self.entry_raw = make_entry("rawdb:TEST-RAW-1/raw_modes/" + NAME_RAWDB, autoset.RAWDB, "TEST-RAW-1",
                                    NAME_RAWDB, f"{self.base}/rawdb/api/download/Sony/TEST-RAW-1/raw_modes/{NAME_RAWDB}",
                                    sha(FILES[NAME_RAWDB]), len(FILES[NAME_RAWDB]))

    def write_manifest(self, entries: List[Dict[str, Any]]) -> None:
        self.manifest_path().write_text(json.dumps(manifest_for(entries, self.base)), encoding="utf-8")

    def build(self, *extra: str) -> Tuple[int, str]:
        return run_cli(["build", "--manifest", str(self.manifest_path()), "--dest", str(self.dest),
                        "--out", str(self.out), "--dump", self.dump_cmd(), "--delay", "0", *extra])

    def rows(self) -> List[Dict[str, Any]]:
        if not self.out.exists():
            return []
        return [json.loads(line) for line in self.out.read_text(encoding="utf-8").splitlines() if line.strip()]

    def failure_records(self) -> List[Dict[str, Any]]:
        if not self.failures.exists():
            return []
        return [json.loads(line) for line in self.failures.read_text(encoding="utf-8").splitlines() if line.strip()]

    def test_downloads_verifies_dumps_appends_and_deletes(self) -> None:
        self.write_manifest([self.entry_ok, self.entry_raw])
        code, out = self.build()
        self.assertEqual(code, autoset.EXIT_OK, out)
        rows = self.rows()
        self.assertEqual(len(rows), 4)  # two files, an aug_k 0 row and an aug_k -1 row each
        first = rows[0]
        self.assertEqual(first["file"], NAME_A)
        self.assertEqual(first["source"], autoset.PIXLS)
        self.assertEqual(first["body"], "TEST-1")  # the dump wrote null; the manifest body is used
        self.assertEqual(first["id"], self.entry_ok["id"])
        self.assertEqual(first["licence"], "CC0-1.0")
        self.assertEqual(rows[2]["file"], NAME_RAWDB)
        self.assertEqual(rows[2]["source"], autoset.RAWDB)
        self.assertEqual(sorted(p.name for p in self.workdir.iterdir()), [])  # raw files deleted
        self.assertFalse(self.failures.exists())
        self.assertIn("/storage/" + NAME_RAWDB, self.hits)  # the 303 was followed

    def test_a_second_run_skips_what_is_done(self) -> None:
        self.write_manifest([self.entry_ok])
        self.assertEqual(self.build()[0], autoset.EXIT_OK)
        downloads = len([h for h in self.hits if h.startswith("/files/")])
        code, out = self.build()
        self.assertEqual(code, autoset.EXIT_OK)
        self.assertEqual(len([h for h in self.hits if h.startswith("/files/")]), downloads)
        self.assertIn("skipped (already in the rows file) 1", out)
        self.assertEqual(len(self.rows()), 2)

    def test_sha_mismatch_appends_nothing_records_it_and_deletes_the_file(self) -> None:
        bad = dict(self.entry_ok, url=autoset.quote_url_path(f"{self.base}/bad/{NAME_A}"), sha256=sha(FILES[NAME_A]))
        self.write_manifest([bad])
        code, out = self.build()
        self.assertEqual(code, autoset.EXIT_FAILED, out)
        self.assertEqual(self.rows(), [])
        self.assertEqual(self.failure_records()[0]["stage"], "verify")
        self.assertEqual(self.failure_records()[0]["id"], bad["id"])
        self.assertEqual(list(self.workdir.iterdir()), [])  # neither the .part nor the file remains

    def test_size_mismatch_is_a_verify_failure(self) -> None:
        wrong = dict(self.entry_raw, size=len(FILES[NAME_RAWDB]) + 1)
        self.write_manifest([wrong])
        code, _ = self.build()
        self.assertEqual(code, autoset.EXIT_FAILED)
        self.assertEqual(self.failure_records()[0]["stage"], "verify")

    def test_dump_exit_status_1_is_a_dump_failure_and_nothing_is_appended(self) -> None:
        self.write_manifest([self.entry_ok])
        with mock.patch.dict(os.environ, {"FAKE_DUMP_MODE": "exit1"}):
            code, out = self.build()
        self.assertEqual(code, autoset.EXIT_FAILED)
        self.assertEqual(self.rows(), [])
        record = self.failure_records()[0]
        self.assertEqual(record["stage"], "dump")
        self.assertIn("skip", out)  # the dump's own message is shown
        self.assertEqual(list(self.workdir.iterdir()), [])

    def test_malformed_rows_are_refused_as_a_whole(self) -> None:
        modes = ["short_features", "nan", "bad_json", "sha_other", "no_base", "bad_licence"]
        for mode in modes:
            with self.subTest(mode=mode):
                self.failures.unlink(missing_ok=True)
                self.out.unlink(missing_ok=True)
                self.write_manifest([self.entry_ok])
                with mock.patch.dict(os.environ, {"FAKE_DUMP_MODE": mode}):
                    code, _ = self.build()
                self.assertEqual(code, autoset.EXIT_FAILED)
                self.assertEqual(self.rows(), [])
                self.assertEqual(self.failure_records()[0]["stage"], "rows")
                self.assertEqual(list(self.workdir.iterdir()), [])

    def test_body_from_the_dump_is_kept_when_it_is_not_null(self) -> None:
        self.write_manifest([self.entry_ok])
        with mock.patch.dict(os.environ, {"FAKE_DUMP_MODE": "body_set"}):
            self.assertEqual(self.build()[0], autoset.EXIT_OK)
        self.assertEqual(self.rows()[0]["body"], "FROM-DUMP")

    def test_transient_503_is_retried_then_succeeds(self) -> None:
        flaky = make_entry("pixls:TEST-1/" + NAME_FLAKY, autoset.PIXLS, "TEST-1", NAME_FLAKY,
                           f"{self.base}/flaky/{NAME_FLAKY}", sha(FILES[NAME_FLAKY]), None)
        self.write_manifest([flaky])
        code, out = self.build()
        self.assertEqual(code, autoset.EXIT_OK, out)
        self.assertEqual(len([h for h in self.hits if h.startswith("/flaky/")]), 2)
        self.assertEqual(len(self.rows()), 2)

    def test_persistent_503_gives_up_after_three_tries(self) -> None:
        always = make_entry("pixls:TEST-1/" + NAME_ALWAYS, autoset.PIXLS, "TEST-1", NAME_ALWAYS,
                            f"{self.base}/always503/{NAME_ALWAYS}", sha(FILES[NAME_ALWAYS]), None)
        self.write_manifest([always])
        code, _ = self.build()
        self.assertEqual(code, autoset.EXIT_FAILED)
        self.assertEqual(len([h for h in self.hits if h.startswith("/always503/")]), autoset.TRIES)
        self.assertEqual(self.failure_records()[0]["stage"], "download")

    def test_missing_file_is_recorded_and_not_retried(self) -> None:
        gone = make_entry("pixls:TEST-1/gone.ARW", autoset.PIXLS, "TEST-1", NAME_GONE,
                          f"{self.base}/gone/{NAME_GONE}", "7" * 64, None)
        self.write_manifest([gone, self.entry_ok])
        code, _ = self.build()
        self.assertEqual(code, autoset.EXIT_FAILED)
        self.assertEqual(len([h for h in self.hits if h.startswith("/gone/")]), 1)
        self.assertEqual(len(self.rows()), 2)  # the next file still went through
        self.assertEqual(self.failure_records()[0]["id"], gone["id"])

    def test_disk_check_refuses_before_any_download(self) -> None:
        self.write_manifest([self.entry_ok])
        with mock.patch.object(autoset, "disk_free_bytes", lambda path: 0):
            code, out = self.build()
        self.assertEqual(code, autoset.EXIT_DISK)
        self.assertIn("not enough free disk space", out)
        self.assertEqual([h for h in self.hits if h.startswith("/files/")], [])
        self.assertEqual(self.rows(), [])

    def test_disk_check_needs_twice_the_size_plus_one_gib(self) -> None:
        # entry_ok has only an approximate size ("0.01MB"): the check uses that figure
        need = 2 * autoset.parse_size_approx("0.01MB") + autoset.DISK_MARGIN
        self.write_manifest([self.entry_ok])
        with mock.patch.object(autoset, "disk_free_bytes", lambda path: need - 1):
            code, _ = self.build()
        self.assertEqual(code, autoset.EXIT_DISK)
        with mock.patch.object(autoset, "disk_free_bytes", lambda path: need):
            code, _ = self.build()
        self.assertEqual(code, autoset.EXIT_OK)

    def test_keep_leaves_the_raw_file_in_the_work_folder(self) -> None:
        self.write_manifest([self.entry_ok])
        self.assertEqual(self.build("--keep")[0], autoset.EXIT_OK)
        self.assertTrue((self.workdir / f"{sha(FILES[NAME_A])}.ARW").exists())
        self.assertFalse((self.workdir / f"{sha(FILES[NAME_A])}.ARW.part").exists())

    def test_failed_file_is_skipped_until_retry_failed(self) -> None:
        self.write_manifest([self.entry_ok])
        with mock.patch.dict(os.environ, {"FAKE_DUMP_MODE": "exit1"}):
            self.assertEqual(self.build()[0], autoset.EXIT_FAILED)
        code, out = self.build()
        self.assertEqual(code, autoset.EXIT_OK)
        self.assertIn("failed earlier", out)
        self.assertEqual(self.rows(), [])
        code, _ = self.build("--retry-failed")
        self.assertEqual(code, autoset.EXIT_OK)
        self.assertEqual(len(self.rows()), 2)
        self.assertEqual(self.failure_records(), [])  # the record is cleared on success

    def test_limit_stops_after_n_files(self) -> None:
        self.write_manifest([self.entry_ok, self.entry_raw])
        self.assertEqual(self.build("--limit", "1")[0], autoset.EXIT_OK)
        self.assertEqual({r["id"] for r in self.rows()}, {self.entry_ok["id"]})

    def test_id_and_body_filters(self) -> None:
        self.write_manifest([self.entry_ok, self.entry_raw])
        self.assertEqual(self.build("--id", self.entry_raw["id"])[0], autoset.EXIT_OK)
        self.assertEqual({r["id"] for r in self.rows()}, {self.entry_raw["id"]})
        code, _ = self.build("--id", "pixls:none/none.ARW")
        self.assertEqual(code, autoset.EXIT_EMPTY)

    def test_interrupt_removes_the_file_in_progress_and_exits_130_without_a_traceback(self) -> None:
        self.write_manifest([self.entry_ok])

        def interrupted(*args: Any, **kwargs: Any) -> None:
            raise KeyboardInterrupt

        with mock.patch.object(autoset, "run_dump", interrupted):
            code, out = self.build()
        self.assertEqual(code, autoset.EXIT_INTERRUPTED)
        self.assertIn("interrupted", out)
        self.assertNotIn("Traceback", out)
        self.assertEqual(list(self.workdir.iterdir()), [])

    def test_missing_dump_program_is_refused_before_any_download(self) -> None:
        self.write_manifest([self.entry_ok])
        code, _ = run_cli(["build", "--manifest", str(self.manifest_path()), "--dest", str(self.dest),
                           "--out", str(self.out), "--dump", "/nonexistent/autoset-dump-xyz", "--delay", "0"])
        self.assertEqual(code, autoset.EXIT_USAGE)
        self.assertEqual([h for h in self.hits if h.startswith("/files/")], [])

    def test_empty_manifest_is_exit_3(self) -> None:
        self.write_manifest([])
        code, out = self.build()
        self.assertEqual(code, autoset.EXIT_EMPTY)
        self.assertIn("refresh", out)

    def test_refused_host_stops_the_run_and_records_nothing(self) -> None:
        probe = socket.socket()
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
        probe.close()
        dead = dict(self.entry_ok, url=autoset.quote_url_path(f"http://127.0.0.1:{port}/files/{NAME_A}"))
        self.write_manifest([dead])
        code, out = self.build()
        self.assertEqual(code, autoset.EXIT_NETWORK, out)
        self.assertEqual(self.failure_records(), [])

    def test_a_url_with_a_space_is_refused_by_policy_not_retried(self) -> None:
        raw_space = dict(self.entry_ok, url=f"{self.base}/files/{NAME_A}")
        self.write_manifest([raw_space])
        code, out = self.build()
        self.assertEqual(code, autoset.EXIT_FAILED)
        self.assertIn("percent-encoded", out)
        self.assertEqual([h for h in self.hits if h.startswith("/files/")], [])


    def test_verify_rows_accepts_what_build_wrote_and_lists_counts(self) -> None:
        self.write_manifest([self.entry_ok, self.entry_raw])
        self.assertEqual(self.build()[0], autoset.EXIT_OK)
        code, out = run_cli(["verify-rows", "--manifest", str(self.manifest_path()), str(self.out)])
        self.assertEqual(code, autoset.EXIT_OK, out)
        self.assertIn("files with an aug_k = 0 row: 2", out)
        self.assertIn("raw.pixls.us\tTEST-1\t1 of 1\t2", out)

    def test_verify_rows_flags_an_unknown_sha_and_bad_features(self) -> None:
        self.write_manifest([self.entry_ok])
        self.assertEqual(self.build()[0], autoset.EXIT_OK)
        bad_sha = {"sha256": "e" * 64, "features": [0.0] * 39, "aug_k": 0.0}
        bad_features = dict(self.rows()[0], features=[0.0] * 38)
        self.out.write_text(json.dumps(bad_sha) + "\n" + json.dumps(bad_features) + "\n", encoding="utf-8")
        code, out = run_cli(["verify-rows", "--manifest", str(self.manifest_path()), str(self.out)])
        self.assertEqual(code, autoset.EXIT_FAILED)
        self.assertIn("not in the manifest", out)
        self.assertIn("features are not 39", out)

    def test_list_prints_one_line_per_entry(self) -> None:
        self.write_manifest([self.entry_ok, self.entry_raw])
        code, out = run_cli(["list", "--manifest", str(self.manifest_path()), "--body", "TEST-RAW-1"])
        self.assertEqual(code, autoset.EXIT_OK)
        lines = [line for line in out.splitlines() if line]
        self.assertEqual(len(lines), 1)
        self.assertTrue(lines[0].startswith(self.entry_raw["id"] + "\t"))


if __name__ == "__main__":
    unittest.main()
