"""Unit tests for ymfetch.py. Standard library only; no real network.

Run:  python3 -I -m unittest discover -s tools/fetch -p 'test_*.py' -v

A small HTTP server on 127.0.0.1 plays RawDB, raw.pixls.us and a Discourse forum. The answers follow the
layout in docs/test-photo-sources.md, which is all we know of the real hosts, so these tests prove the
script logic (hashing, licence policy, politeness, failure messages), not conformance with the live sites.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
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
import ymfetch  # noqa: E402


def blob(seed: int, size: int = 3000) -> bytes:
    out = bytearray()
    counter = 0
    while len(out) < size:
        out += hashlib.sha256(f"{seed}:{counter}".encode()).digest()
        counter += 1
    return bytes(out[:size])


FILES: Dict[str, bytes] = {
    "/files/Sony/ILCE-TEST/Uncompressed/DSC00001.ARW": blob(1),
    "/Sony/ILCE-TEST/Lossless/DSC00002.ARW": blob(2, 5000),
    "/Sony/ILCE-TEST/Lossless/DSC00003.ARW": blob(3),
    "/data/Sony/ILCE-PIX/pix one.ARW": blob(4),
}


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class State:
    def __init__(self) -> None:
        self.hits: List[Tuple[str, Optional[str]]] = []
        self.rawdb_sha_override: Dict[str, str] = {}
        self.drop_after: Optional[int] = None  # first full download sends only this many bytes, then closes


class Handler(BaseHTTPRequestHandler):
    state: State = State()
    protocol_version = "HTTP/1.1"

    def log_message(self, *args: Any) -> None:  # silence
        pass

    def _send(self, code: int, body: bytes, ctype: str = "application/octet-stream", extra: Optional[Dict[str, str]] = None) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        path = urllib.parse.unquote(self.path.split("?")[0])
        self.state.hits.append((path, self.headers.get("Range")))
        if path == "/api/sets/Sony/ILCE-TEST":
            sets = {
                "name": "ILCE-TEST",
                "sets": [
                    {
                        "title": "Uncompressed",
                        "license": "CC0 1.0",
                        "files": [
                            {"path": "Sony/ILCE-TEST/Uncompressed/DSC00001.ARW", "size": len(FILES["/files/Sony/ILCE-TEST/Uncompressed/DSC00001.ARW"]),
                             "sha256": self.state.rawdb_sha_override.get("1", sha(FILES["/files/Sony/ILCE-TEST/Uncompressed/DSC00001.ARW"])),
                             "url": "/files/Sony/ILCE-TEST/Uncompressed/DSC00001.ARW"},
                            {"path": "Sony/ILCE-TEST/Uncompressed/readme.txt", "size": 3, "sha256": "0" * 64, "url": "/x"},
                        ],
                    },
                    {
                        "title": "Lossless",
                        "files": [
                            {"path": "/Sony/ILCE-TEST/Lossless/DSC00002.ARW", "size": 5000,
                             "sha256": sha(FILES["/Sony/ILCE-TEST/Lossless/DSC00002.ARW"]), "license": "CC0-1.0"},
                            {"path": "/Sony/ILCE-TEST/Lossless/DSC00003.ARW", "size": 3000,
                             "sha256": sha(FILES["/Sony/ILCE-TEST/Lossless/DSC00003.ARW"]), "license": "CC BY-NC-SA 4.0"},
                        ],
                    },
                ],
            }
            self._send(200, json.dumps(sets).encode(), "application/json")
        elif path == "/api/sets/Sony/ILCE-EMPTY":
            self._send(200, b'{"sets": []}', "application/json")
        elif path.startswith("/api/sets/Sony/"):
            self._send(404, b"not found", "text/plain")
        elif path == "/data/Sony/ILCE-PIX/":
            page = '<html><body><a href="?C=N;O=D">sort</a><a href="../">up</a><a href="pix%20one.ARW">pix one.ARW</a><a href="notes.txt">n</a></body></html>'
            self._send(200, page.encode(), "text/html")
        elif path in FILES:
            data = FILES[path]
            rng = self.headers.get("Range")
            if rng and rng.startswith("bytes="):
                start = int(rng[6:].split("-")[0])
                if start >= len(data):
                    self._send(416, b"", "text/plain")
                    return
                self._send(206, data[start:], extra={"Content-Range": f"bytes {start}-{len(data) - 1}/{len(data)}"})
            elif self.state.drop_after is not None:
                cut, self.state.drop_after = self.state.drop_after, None
                self.send_response(200)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data[:cut])
                self.wfile.flush()
                self.close_connection = True
            else:
                self._send(200, data)
        elif path == "/redirect":
            self._send(302, b"", "text/plain", {"Location": "https://evil.example/steal.ARW"})
        elif path == "/search.json":
            self._send(200, json.dumps({"topics": [{"id": 7, "title": "Test post", "slug": "test-post"}]}).encode(), "application/json")
        elif path == "/t/7.json":
            cooked = '<p>Shot on A1. License: CC BY-SA 4.0</p><a class="attachment" href="/uploads/short-url/abc.ARW">DSC_1.ARW</a>'
            self._send(200, json.dumps({"post_stream": {"posts": [{"username": "alice", "cooked": cooked}]}}).encode(), "application/json")
        else:
            self._send(404, b"not found", "text/plain")


class QuietServer(socketserver.ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


class ConnectRefuser(BaseHTTPRequestHandler):
    """Acts like the sandbox proxy: refuses every CONNECT tunnel with 403."""

    def log_message(self, *args: Any) -> None:
        pass

    def do_CONNECT(self) -> None:  # noqa: N802
        self.send_response(403, "Forbidden")
        self.send_header("Content-Length", "0")
        self.end_headers()


def serve(handler: type) -> Tuple[HTTPServer, str]:
    server = QuietServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_address[1]}"


def run_cli(argv: List[str], env: Optional[Dict[str, str]] = None) -> Tuple[int, str]:
    err = io.StringIO()
    with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()) as out:
        with mock.patch.dict(os.environ, env or {}):
            code = ymfetch.main(argv)
    return code, err.getvalue() + out.getvalue()


def entry_for(path: str, body: str = "ILCE-TEST", *, name: str, license: str = "CC0-1.0", tier: int = 1, base: str = "") -> Dict[str, Any]:
    data = FILES[path]
    e: Dict[str, Any] = {
        "id": f"rawdb:{body}/{name}", "source": "rawdb", "body": body, "name": name, "source_path": name,
        "url": base + path, "size": len(data), "sha256": sha(data), "license": license,
        "license_source": "test", "tier": tier, "sets": ["corpus"] if tier == 1 else ["golden"],
    }
    if tier == 2:
        e["attribution"] = {"author": "Alice", "source_url": "https://discuss.pixls.us/t/x/1", "title": "Test"}
    return e


class Base(unittest.TestCase):
    server: HTTPServer
    base: str

    @classmethod
    def setUpClass(cls) -> None:
        cls.server, cls.base = serve(Handler)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()

    def setUp(self) -> None:
        Handler.state = State()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / "testdata"
        self.manifest_path = Path(self.tmp.name) / "manifest.json"

    def manifest(self, files: Optional[List[Dict[str, Any]]] = None, bodies: Optional[List[str]] = None, **extra: Any) -> Dict[str, Any]:
        m: Dict[str, Any] = {
            "schema_version": 1,
            "sources": {
                "rawdb": {"api_base": self.base + "/api/sets", "make": "Sony", "hosts": ["rawdb.dnglab.org"], "extra_hosts": []},
                "pixls": {"data_base": self.base + "/data", "make": "Sony", "hosts": ["raw.pixls.us"], "extra_hosts": []},
                "playraw": {"api_base": self.base, "hosts": ["discuss.pixls.us"], "extra_hosts": []},
            },
            "bodies": [{"model": b, "name": b, "group": "required"} for b in (bodies or ["ILCE-TEST", "ILCE-EMPTY", "ILCE-NONE", "ILCE-PIX"])],
            "files": files or [], "pending": [], "excluded": [],
            "golden": {"tier1_bodies": ["ILCE-TEST"], "tier3": [
                {"id": "pl", "provider": "Photography Life", "body": "ILCE-1", "page_url": "https://photographylife.com/x",
                 "files_expected": 22, "terms": "personal use only", "local_dir": "golden/tier3/pl", "fetch": False}]},
        }
        m.update(extra)
        return m

    def write(self, m: Dict[str, Any]) -> None:
        self.manifest_path.write_text(json.dumps(m), encoding="utf-8")

    def cli(self, cmd: str, *args: str, env: Optional[Dict[str, str]] = None) -> Tuple[int, str]:
        return run_cli([cmd, "--manifest", str(self.manifest_path), "--dest", str(self.root), "--delay", "0", "--retries", "1", *args], env)

    def read_manifest(self) -> Dict[str, Any]:
        return json.loads(self.manifest_path.read_text(encoding="utf-8"))


class LicenceTests(unittest.TestCase):
    def test_normalise(self) -> None:
        cases = {
            "CC0 1.0": "CC0-1.0", "CC0-1.0": "CC0-1.0", "cc0": "CC0-1.0", "Creative Commons Zero v1.0 Universal": "CC0-1.0",
            "CC BY 4.0": "CC-BY-4.0", "CC-BY-SA-4.0": "CC-BY-SA-4.0", "Creative Commons Attribution-ShareAlike 4.0 International": "unknown",
            "CC BY-SA 4.0 International": "CC-BY-SA-4.0", "CC BY-NC-SA 4.0": "CC-BY-NC-SA-4.0", "CC BY-NC-SA": "unknown",
            "CC BY": "unknown", "GPL-3.0": "unknown", "": "unknown", None: "unknown", "all rights reserved": "unknown",
        }
        for raw, want in cases.items():
            with self.subTest(raw=raw):
                self.assertEqual(ymfetch.normalise_license(raw), want)

    def test_nc_is_never_accepted(self) -> None:
        for lic in ("CC-BY-NC-SA-4.0", "CC-BY-NC-4.0", "CC-BY-ND-4.0", "unknown"):
            self.assertNotIn(lic, ymfetch.CORPUS_LICENSES)
            self.assertNotIn(lic, ymfetch.TIER2_LICENSES)

    def test_flat_names(self) -> None:
        self.assertEqual(ymfetch.flat_name("Sony/ILCE-1/DSC1.ARW", "Sony", "ILCE-1"), "DSC1.ARW")
        self.assertEqual(ymfetch.flat_name("/Sony/ILCE-1/Lossless L/DSC1.ARW", "Sony", "ILCE-1"), "Lossless_L__DSC1.ARW")
        self.assertEqual(ymfetch.flat_name("../../etc/passwd.ARW", "Sony", "ILCE-1"), "etc__passwd.ARW")
        self.assertTrue(ymfetch.flat_name("(1).ARW", "Sony", "ILCE-1")[0].isalnum())


class ManifestTests(Base):
    def test_empty_manifest_fails_loudly_unless_allowed(self) -> None:
        self.write(self.manifest())
        code, out = self.cli("samples")
        self.assertEqual(code, ymfetch.EXIT_EMPTY)
        self.assertIn("no files yet", out)
        self.assertEqual(self.cli("samples", "--allow-empty")[0], 0)
        self.assertEqual(self.cli("golden")[0], ymfetch.EXIT_EMPTY)
        self.assertEqual(Handler.state.hits, [])

    def test_licence_policy_rejected_before_any_request(self) -> None:
        for lic in ("CC-BY-NC-SA-4.0", "unknown", "unverified", "CC-BY-4.0"):
            with self.subTest(lic=lic):
                self.write(self.manifest([entry_for("/Sony/ILCE-TEST/Lossless/DSC00003.ARW", name="x.ARW", license=lic, base=self.base)]))
                code, out = self.cli("samples")
                self.assertEqual(code, ymfetch.EXIT_FAILED, out)
                self.assertIn("CC0-1.0 only", out)
        self.assertEqual(Handler.state.hits, [])

    def test_tier2_needs_attribution_and_free_licence(self) -> None:
        e = entry_for("/Sony/ILCE-TEST/Lossless/DSC00003.ARW", name="p.ARW", license="CC-BY-SA-4.0", tier=2, base=self.base)
        del e["attribution"]
        self.write(self.manifest([e]))
        self.assertEqual(self.cli("golden")[0], ymfetch.EXIT_FAILED)
        e["attribution"] = {"author": "Alice", "source_url": "https://discuss.pixls.us/t/x/1"}
        e["license"] = "CC-BY-NC-SA-4.0"
        self.write(self.manifest([e]))
        self.assertEqual(self.cli("golden")[0], ymfetch.EXIT_FAILED)

    def test_unsafe_names_rejected(self) -> None:
        e = entry_for("/Sony/ILCE-TEST/Lossless/DSC00003.ARW", name="../../evil.ARW", base=self.base)
        self.write(self.manifest([e]))
        code, out = self.cli("samples")
        self.assertEqual(code, ymfetch.EXIT_FAILED)
        self.assertIn("unsafe name", out)
        self.assertFalse((Path(self.tmp.name) / "evil.ARW").exists())

    def test_duplicate_destinations_rejected(self) -> None:
        a = entry_for("/Sony/ILCE-TEST/Lossless/DSC00003.ARW", name="same.ARW", base=self.base)
        b = dict(a, id="pixls:ILCE-TEST/same.ARW", source="pixls")
        self.write(self.manifest([a, b]))
        self.assertEqual(self.cli("samples")[0], ymfetch.EXIT_FAILED)


class DownloadTests(Base):
    def files(self) -> List[Dict[str, Any]]:
        return [
            entry_for("/files/Sony/ILCE-TEST/Uncompressed/DSC00001.ARW", name="Uncompressed__DSC00001.ARW", base=self.base),
            entry_for("/Sony/ILCE-TEST/Lossless/DSC00002.ARW", name="Lossless__DSC00002.ARW", base=self.base),
        ]

    def test_download_verify_and_cache(self) -> None:
        self.write(self.manifest(self.files()))
        code, out = self.cli("samples")
        self.assertEqual(code, 0, out)
        for e in self.files():
            self.assertEqual((self.root / "arw" / "ILCE-TEST" / e["name"]).read_bytes(), FILES[e["url"].replace(self.base, "")])
        self.assertFalse(list(self.root.rglob("*.part")))
        hits_first = len(Handler.state.hits)
        code, out = self.cli("samples")
        self.assertEqual(code, 0)
        self.assertEqual(len(Handler.state.hits), hits_first, "a verified file must not be fetched again")
        self.assertIn("2 already present", out)
        self.assertEqual(self.cli("samples", "--verify")[0], 0)

    def test_verify_reports_missing_and_corrupt(self) -> None:
        self.write(self.manifest(self.files()))
        code, out = self.cli("samples", "--verify")
        self.assertEqual(code, ymfetch.EXIT_FAILED)
        self.assertIn("MISSING", out)
        self.assertEqual(self.cli("samples", "--verify", "--allow-missing")[0], 0)
        self.cli("samples")
        target = self.root / "arw" / "ILCE-TEST" / "Lossless__DSC00002.ARW"
        target.write_bytes(b"x" * 5000)
        code, out = self.cli("samples", "--verify")
        self.assertEqual(code, ymfetch.EXIT_FAILED)
        self.assertIn("HASH", out)
        code, _ = self.cli("samples")  # repairs it
        self.assertEqual(code, 0)
        self.assertEqual(target.read_bytes(), FILES["/Sony/ILCE-TEST/Lossless/DSC00002.ARW"])

    def test_hash_mismatch_deletes_the_download(self) -> None:
        bad = self.files()
        bad[0]["sha256"] = "0" * 64
        self.write(self.manifest(bad))
        code, out = self.cli("samples")
        self.assertEqual(code, ymfetch.EXIT_FAILED)
        self.assertIn("does not match the manifest", out)
        self.assertFalse(list(self.root.rglob("Uncompressed__DSC00001.ARW*")))

    def test_body_per_body_and_size_filters(self) -> None:
        self.write(self.manifest(self.files()))
        self.assertEqual(self.cli("samples", "--body", "ILCE-NONE")[0], ymfetch.EXIT_EMPTY)
        self.assertEqual(self.cli("samples", "--per-body", "1")[0], 0)
        self.assertEqual(len(list(self.root.rglob("*.ARW"))), 1)
        self.assertEqual(self.cli("samples", "--max-file-mb", "0.001")[0], ymfetch.EXIT_EMPTY)

    def test_dry_run_touches_nothing(self) -> None:
        self.write(self.manifest(self.files()))
        code, out = self.cli("samples", "--dry-run")
        self.assertEqual(code, 0)
        self.assertEqual(Handler.state.hits, [])
        self.assertFalse(self.root.exists())

    def test_resume_after_dropped_connection(self) -> None:
        self.write(self.manifest(self.files()[:1]))
        Handler.state.drop_after = 1000
        with mock.patch("ymfetch.time.sleep"):
            code, out = self.cli("samples")
        self.assertEqual(code, 0, out)
        ranges = [r for _, r in Handler.state.hits if r]
        self.assertTrue(ranges and ranges[0].startswith("bytes=1000-"), ranges)
        self.assertEqual((self.root / "arw/ILCE-TEST/Uncompressed__DSC00001.ARW").read_bytes(), FILES["/files/Sony/ILCE-TEST/Uncompressed/DSC00001.ARW"])

    def test_redirect_to_unlisted_host_is_refused(self) -> None:
        e = entry_for("/Sony/ILCE-TEST/Lossless/DSC00003.ARW", name="r.ARW", base=self.base)
        e["url"] = self.base + "/redirect"
        self.write(self.manifest([e]))
        code, out = self.cli("samples")
        self.assertEqual(code, ymfetch.EXIT_FAILED)
        self.assertIn("not one of the manifest's source hosts", out)

    def test_plain_http_to_a_real_host_is_refused(self) -> None:
        e = entry_for("/Sony/ILCE-TEST/Lossless/DSC00003.ARW", name="h.ARW", base="http://rawdb.dnglab.org")
        self.write(self.manifest([e]))
        code, out = self.cli("samples")
        self.assertEqual(code, ymfetch.EXIT_FAILED)
        self.assertIn("non-https", out)

    def test_golden_tier2_attribution_and_tier3_never_fetched(self) -> None:
        t2 = entry_for("/Sony/ILCE-TEST/Lossless/DSC00003.ARW", name="Play_One.ARW", license="CC-BY-SA-4.0", tier=2, base=self.base)
        t1 = entry_for("/files/Sony/ILCE-TEST/Uncompressed/DSC00001.ARW", name="Uncompressed__DSC00001.ARW", base=self.base)
        t1["sets"] = ["corpus", "golden"]
        self.write(self.manifest([t1, t2]))
        code, out = self.cli("golden")
        self.assertEqual(code, 0, out)
        self.assertTrue((self.root / "golden/tier2/ILCE-TEST/Play_One.ARW").is_file())
        self.assertTrue((self.root / "arw/ILCE-TEST/Uncompressed__DSC00001.ARW").is_file())
        attribution = (self.root / "golden/ATTRIBUTION.md").read_text(encoding="utf-8")
        self.assertIn("Alice", attribution)
        self.assertIn("CC-BY-SA-4.0", attribution)
        # the corpus run does not include tier 2
        self.assertEqual([e["id"] for e in ymfetch.select_entries(self.read_manifest(), sets={"corpus"})], [t1["id"]])
        # tier 3: listing only; asking for tier 3 as a fetch tier is a usage error
        hits = len(Handler.state.hits)
        code, out = self.cli("golden", "--tier3")
        self.assertEqual(code, 0)
        self.assertIn("NEVER fetched", out)
        self.assertEqual(self.cli("golden", "--tier", "3")[0], ymfetch.EXIT_USAGE)
        self.assertEqual(len(Handler.state.hits), hits)

    def test_index_tier3_is_local_only(self) -> None:
        self.write(self.manifest())
        folder = self.root / "golden/tier3/pl"
        folder.mkdir(parents=True)
        (folder / "A.ARW").write_bytes(b"abc")
        code, _ = self.cli("golden", "--index-tier3")
        self.assertEqual(code, 0)
        index = json.loads((self.root / "golden/tier3/INDEX.json").read_text(encoding="utf-8"))
        self.assertEqual(index["files"][0]["sha256"], sha(b"abc"))
        self.assertEqual(Handler.state.hits, [])


class NetworkFailureTests(Base):
    def test_connection_refused_has_a_clear_message(self) -> None:
        e = entry_for("/Sony/ILCE-TEST/Lossless/DSC00003.ARW", name="n.ARW", base="http://127.0.0.1:9")
        self.write(self.manifest([e]))
        code, out = self.cli("samples")
        self.assertEqual(code, ymfetch.EXIT_NETWORK, out)
        self.assertIn("cannot reach 127.0.0.1", out)
        self.assertIn("What to do", out)

    def test_proxy_refusing_connect_like_the_sandbox(self) -> None:
        proxy, url = serve(ConnectRefuser)
        self.addCleanup(proxy.server_close)
        self.addCleanup(proxy.shutdown)
        e = entry_for("/Sony/ILCE-TEST/Lossless/DSC00003.ARW", name="p.ARW", base="https://rawdb.dnglab.org")
        self.write(self.manifest([e]))
        env = {"https_proxy": url, "HTTPS_PROXY": url, "no_proxy": "", "NO_PROXY": ""}
        code, out = self.cli("samples", env=env)
        self.assertEqual(code, ymfetch.EXIT_NETWORK, out)
        self.assertIn("cannot reach rawdb.dnglab.org", out)
        self.assertIn("403", out)
        m = self.manifest([e])
        m["sources"]["rawdb"]["api_base"] = "https://rawdb.dnglab.org/api/sets"
        self.write(m)
        code, out = self.cli("samples", "--refresh", "--body", "ILCE-TEST", env=env)
        self.assertEqual(code, ymfetch.EXIT_NETWORK, out)
        self.assertEqual(self.read_manifest()["files"][0]["id"], e["id"], "a failed refresh must not touch the manifest")


class RefreshTests(Base):
    def refresh(self, *args: str) -> Tuple[int, str]:
        return self.cli("samples", "--refresh", "--source", "rawdb", "--body", "ILCE-TEST", "--body", "ILCE-NONE", "--body", "ILCE-EMPTY", *args)

    def test_refresh_rawdb(self) -> None:
        self.write(self.manifest())
        code, out = self.refresh()
        self.assertEqual(code, 0, out)
        m = self.read_manifest()
        ids = [e["id"] for e in m["files"]]
        self.assertEqual(ids, ["rawdb:ILCE-TEST/Lossless__DSC00002.ARW", "rawdb:ILCE-TEST/Uncompressed__DSC00001.ARW"])
        first = m["files"][1]
        self.assertEqual((first["license"], first["license_source"], first["tier"], first["sets"]), ("CC0-1.0", "rawdb-api", 1, ["corpus", "golden"]))
        self.assertEqual(first["sha256"], sha(FILES["/files/Sony/ILCE-TEST/Uncompressed/DSC00001.ARW"]))
        self.assertEqual(first["license_raw"], "CC0 1.0")
        self.assertEqual(first["variant_hint"], "uncompressed")
        self.assertEqual(m["files"][0]["variant_hint"], "lossless")
        self.assertEqual([x["id"] for x in m["excluded"]], ["rawdb:ILCE-TEST/Lossless__DSC00003.ARW"])
        self.assertEqual(m["excluded"][0]["license"], "CC-BY-NC-SA-4.0")
        self.assertIn("ILCE-NONE: no RawDB set", out)
        self.assertIn("ILCE-EMPTY: no CC0 file", out)
        self.assertTrue((self.root / ".cache/api/rawdb-ILCE-TEST.json").is_file())
        # the refreshed manifest is directly usable, and refresh is idempotent
        before = self.manifest_path.read_text(encoding="utf-8")
        self.assertEqual(self.cli("samples")[0], 0)
        self.assertEqual(self.refresh()[0], 0)
        self.assertEqual(self.manifest_path.read_text(encoding="utf-8"), before)

    def test_changed_upstream_hash_needs_explicit_acceptance(self) -> None:
        self.write(self.manifest())
        self.assertEqual(self.refresh()[0], 0)
        before = self.manifest_path.read_text(encoding="utf-8")
        Handler.state.rawdb_sha_override["1"] = "a" * 64
        code, out = self.refresh()
        self.assertEqual(code, ymfetch.EXIT_FAILED, out)
        self.assertIn("--accept-changed", out)
        self.assertEqual(self.manifest_path.read_text(encoding="utf-8"), before)
        self.assertEqual(self.refresh("--accept-changed")[0], 0)
        self.assertEqual([e["sha256"] for e in self.read_manifest()["files"] if e["name"].startswith("Uncompressed")], ["a" * 64])

    def test_dry_run_writes_nothing(self) -> None:
        self.write(self.manifest())
        before = self.manifest_path.read_text(encoding="utf-8")
        self.assertEqual(self.refresh("--dry-run")[0], 0)
        self.assertEqual(self.manifest_path.read_text(encoding="utf-8"), before)

    def test_unknown_body_and_missing_url_rule(self) -> None:
        self.write(self.manifest())
        self.assertEqual(self.cli("samples", "--refresh", "--body", "ILCE-9999")[0], ymfetch.EXIT_FAILED)
        src = {"api_base": self.base + "/api/sets", "make": "Sony"}
        with self.assertRaises(ymfetch.ManifestError):
            ymfetch.rawdb_file_url({"path": "relative/DSC.ARW"}, src)
        self.assertEqual(ymfetch.rawdb_file_url({"path": "relative/DSC.ARW"}, dict(src, file_url_template="{root}/dl/{path}")), self.base + "/dl/relative/DSC.ARW")

    def test_refresh_pixls_lists_pending_and_promote_pins_cc0(self) -> None:
        self.write(self.manifest())
        code, out = self.cli("samples", "--refresh", "--source", "pixls", "--body", "ILCE-PIX")
        self.assertEqual(code, 0, out)
        m = self.read_manifest()
        self.assertEqual(m["files"], [])
        self.assertEqual([p["id"] for p in m["pending"]], ["pixls:ILCE-PIX/pix_one.ARW"])
        self.assertEqual(m["pending"][0]["license"], "unverified")
        # nothing pending is ever downloaded by a plain run
        self.assertEqual(self.cli("samples")[0], ymfetch.EXIT_EMPTY)
        # promoting needs a CC0 statement and a name
        self.assertEqual(self.cli("samples", "--promote", "pixls:ILCE-PIX/pix_one.ARW", "--license", "CC-BY-NC-SA-4.0", "--checked-by", "me")[0], ymfetch.EXIT_FAILED)
        self.assertEqual(self.cli("samples", "--promote", "pixls:ILCE-PIX/pix_one.ARW", "--license", "CC0 1.0")[0], ymfetch.EXIT_FAILED)
        # the listing URL is percent-encoded by the server; serve it
        m = self.read_manifest()
        self.assertIn("pix%20one.ARW", m["pending"][0]["url"])
        code, out = self.cli("samples", "--promote", "pixls:ILCE-PIX/pix_one.ARW", "--license", "CC0 1.0", "--checked-by", "test: raw.pixls.us page, CC0")
        self.assertEqual(code, 0, out)
        m = self.read_manifest()
        self.assertEqual(m["pending"], [])
        promoted = m["files"][0]
        self.assertEqual(promoted["sha256"], sha(FILES["/data/Sony/ILCE-PIX/pix one.ARW"]))
        self.assertEqual((promoted["license"], promoted["license_source"]), ("CC0-1.0", "manual"))
        self.assertEqual(self.cli("samples", "--verify")[0], 0)
        # a later refresh keeps the promoted file and does not list it as pending again
        self.assertEqual(self.cli("samples", "--refresh", "--source", "pixls", "--body", "ILCE-PIX")[0], 0)
        m = self.read_manifest()
        self.assertEqual((len(m["files"]), m["pending"]), (1, []))

    def test_coverage_table(self) -> None:
        self.write(self.manifest())
        self.assertEqual(self.refresh()[0], 0)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            ymfetch.print_coverage(self.read_manifest())
        table = buf.getvalue()
        self.assertIn("| ILCE-TEST | ILCE-TEST | required | 1 | 1 |", table)
        self.assertIn("Bodies with no corpus file: ILCE-EMPTY, ILCE-NONE, ILCE-PIX", table)


class PlayRawTests(Base):
    def test_candidates_are_saved_unreviewed_and_manifest_untouched(self) -> None:
        self.write(self.manifest())
        before = self.manifest_path.read_text(encoding="utf-8")
        code, out = self.cli("golden", "--candidates", "ILCE-1")
        self.assertEqual(code, 0, out)
        found = json.loads((self.root / "golden/tier2-candidates.json").read_text(encoding="utf-8"))
        cand = found["candidates"][0]
        self.assertEqual((cand["status"], cand["author"], cand["license_hints"]), ("unreviewed", "alice", ["CC-BY-SA-4.0"]))
        self.assertTrue(cand["attachments"][0]["url"].endswith("/uploads/short-url/abc.ARW"))
        self.assertEqual(self.manifest_path.read_text(encoding="utf-8"), before)

    def test_add_tier2_checks_licence(self) -> None:
        self.write(self.manifest())
        url = self.base + "/Sony/ILCE-TEST/Lossless/DSC00003.ARW"
        common = ["golden", "--add-tier2", url, "--body", "ILCE-TEST", "--author", "Alice", "--source-url", "https://discuss.pixls.us/t/x/1"]
        self.assertEqual(self.cli(*common, "--license", "CC BY-NC-SA 4.0")[0], ymfetch.EXIT_FAILED)
        self.assertEqual(Handler.state.hits, [])
        code, out = self.cli(*common, "--license", "CC BY-SA 4.0", "--name", "Alice night.ARW")
        self.assertEqual(code, 0, out)
        e = self.read_manifest()["files"][0]
        self.assertEqual((e["tier"], e["license"], e["sets"], e["name"]), (2, "CC-BY-SA-4.0", ["golden"], "Alice_night.ARW"))
        self.assertEqual(self.cli("golden", "--verify")[0], 0)


REQUIRED_BODIES = [
    "ILCE-7RM2", "ILCE-7RM3", "ILCE-7RM4", "ILCE-7RM5", "ILCE-7M3", "ILCE-7M4", "ILCE-7M5", "ILCE-7RM6", "ILCE-7SM2",
    "ILCE-7SM3", "ILCE-9", "ILCE-9M3", "ILCE-1", "ILCE-1M2", "ILCE-6400", "ILCE-6700", "DSC-RX100M4", "ZV-E1", "ILME-FX3",
]  # docs/15 M0-09


class ShippedManifestTests(unittest.TestCase):
    def setUp(self) -> None:
        self.m = ymfetch.load_manifest(ymfetch.DEFAULT_MANIFEST)

    def test_covers_the_required_bodies(self) -> None:
        models = [b["model"] for b in self.m["bodies"]]
        self.assertEqual(len(models), len(set(models)), "duplicate body")
        for model in REQUIRED_BODIES:
            self.assertIn(model, models)
        required = [b["model"] for b in self.m["bodies"] if b["group"] == "required"]
        self.assertEqual(sorted(required), sorted(REQUIRED_BODIES))

    def test_every_entry_obeys_the_licence_policy(self) -> None:
        for e in self.m["files"]:
            self.assertIn(e["license"], ymfetch.CORPUS_LICENSES if e["tier"] == 1 else ymfetch.TIER2_LICENSES)
        self.assertEqual(sorted(self.m["policy"]["corpus_licenses"]), sorted(ymfetch.CORPUS_LICENSES))
        self.assertEqual(sorted(self.m["policy"]["tier2_licenses"]), sorted(ymfetch.TIER2_LICENSES))

    def test_tier3_is_listed_but_never_fetchable(self) -> None:
        tier3 = self.m["golden"]["tier3"]
        self.assertTrue(tier3)
        for item in tier3:
            self.assertIs(item["fetch"], False)
            self.assertTrue(item["page_url"].startswith("https://"))
        self.assertFalse([e for e in self.m["files"] if e["id"].startswith("tier3")])

    def test_hosts_are_pinned_to_the_documented_sources(self) -> None:
        self.assertEqual(ymfetch.allowed_hosts(self.m), {"rawdb.dnglab.org", "raw.pixls.us", "discuss.pixls.us"})


if __name__ == "__main__":
    unittest.main()
