"""Tests for real-eval.sh, the local-evaluation fetch script. Standard library and git only; no network.

Run:  python3 -I -m unittest discover -s tools/fetch -p 'test_*.py' -v

The script is run against two local git repositories shaped like the real sources (an arw/ folder with ARW files, text dumps and
other folders that must not be fetched; a raw/ and a formats/raw/Sony/ folder with a file name that has brackets in it), through
a source table of the same schema as the built-in one (YM_REAL_EVAL_SOURCES). That proves the script logic: the pin of commit
and blob ids, that only the listed blobs are downloaded, the links named by the table, idempotence, the protection of files it
did not write, one unreachable source not stopping the others, and the refusal to run in CI. The last test checks the built-in
table itself (names, hosts, pins) without any network. Nothing here touches github.com.
"""

from __future__ import annotations

import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List, Tuple

HERE = Path(__file__).resolve().parent
SCRIPT = HERE / "real-eval.sh"
sys.path.insert(0, str(HERE))


def tiff_with_model(model: str, byte_order: str = "<", padding: int = 4000) -> bytes:
    """A minimal TIFF (a header, IFD0 with one Model tag, some padding): all the script reads of an ARW file."""
    text = model.encode("ascii") + b"\0"
    e = byte_order
    head = (b"II" if e == "<" else b"MM") + struct.pack(e + "HI", 42, 8)
    ifd_size = 2 + 12 + 4
    value_offset = 8 + ifd_size
    if len(text) <= 4:
        entry = struct.pack(e + "HHI", 0x0110, 2, len(text)) + text.ljust(4, b"\0")
        tail = b""
    else:
        entry = struct.pack(e + "HHII", 0x0110, 2, len(text), value_offset)
        tail = text
    ifd = struct.pack(e + "H", 1) + entry + struct.pack(e + "I", 0)
    return head + ifd + tail + bytes((i * 7) % 251 for i in range(padding))


def git(cwd: Path, *args: str) -> str:
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.com",
        "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.com",
        "GIT_AUTHOR_DATE": "2026-01-01T00:00:00Z", "GIT_COMMITTER_DATE": "2026-01-01T00:00:00Z",
    }
    return subprocess.run(["git", *args], cwd=cwd, env=env, check=True, capture_output=True, text=True).stdout.strip()


def git_ok() -> bool:
    if shutil.which("git") is None:
        return False
    out = subprocess.run(["git", "--version"], capture_output=True, text=True).stdout.split()
    try:
        major, minor = (int(x) for x in out[2].split(".")[:2])
    except (IndexError, ValueError):
        return False
    return (major, minor) >= (2, 30)


def make_repo(root: Path, files: Dict[str, bytes]) -> Tuple[str, Dict[str, str]]:
    """A git repository at ``root`` that serves blobs by id; returns ``(commit, {path: blob id})``."""
    for rel, data in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
    git(root, "init", "-q", "-b", "main")
    git(root, "config", "uploadpack.allowFilter", "true")
    git(root, "config", "uploadpack.allowAnySHA1InWant", "true")
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", "fixture")
    return git(root, "rev-parse", "HEAD"), {rel: git(root, "rev-parse", f"HEAD:{rel}") for rel in files}


@unittest.skipUnless(git_ok() and shutil.which("bash") and shutil.which("python3"), "git 2.30+, bash and python3 are needed")
class RealEvalTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="ym-real-eval-"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.td = self.tmp / "testdata"
        # Source "alpha": names as in the real metadata-extractor-images: vendor prefix, a marketing name in brackets, mixed
        # case extensions, text dumps beside the files, other folders.
        self.alpha_dir = self.tmp / "alpha"
        alpha_files = {
            "arw/Sony ILCE-TEST1 (T1).arw": tiff_with_model("ILCE-TEST1"),
            "arw/Sony NEX-T2.ARW": tiff_with_model("NEX-T2", ">", 9000),
            "arw/Sony Short.arw": tiff_with_model("A1"),  # a Model value that fits in the entry
            "arw/Sony Unlisted.arw": tiff_with_model("UNLISTED"),
            "arw/metadata/java/Sony NEX-T2.ARW.txt": b"dump\n",
            "jpg/other.jpg": b"\xff\xd8not fetched",
        }
        self.alpha_commit, alpha_blobs = make_repo(self.alpha_dir, {k: v for k, v in alpha_files.items()})
        # Source "beta": one file by its plain path, one with brackets in the name under a deeper folder.
        self.beta_dir = self.tmp / "beta"
        beta_files = {
            "raw/RAW_BETA.ARW": tiff_with_model("DSLR-B1", ">", 6000),
            "raw/RAW_OTHER.ARW": tiff_with_model("OTHER"),
            "formats/raw/Sony/X1/12bit-compressed-(3_2).arw": tiff_with_model("ILCE-X1", "<", 7000),
        }
        self.beta_commit, beta_blobs = make_repo(self.beta_dir, beta_files)
        self.sources: List[Dict[str, Any]] = [
            {
                "id": "fixture-alpha",
                "repo": self.alpha_dir.as_uri(),
                "commit": self.alpha_commit,
                "licence_note": "alpha says nothing about the licence",
                "files": [
                    {"path": "arw/Sony ILCE-TEST1 (T1).arw", "name": "ILCE-TEST1.ARW", "blob": alpha_blobs["arw/Sony ILCE-TEST1 (T1).arw"]},
                    {"path": "arw/Sony NEX-T2.ARW", "name": "NEX-T2.ARW", "blob": alpha_blobs["arw/Sony NEX-T2.ARW"]},
                    {"path": "arw/Sony Short.arw", "name": "A1.ARW", "blob": alpha_blobs["arw/Sony Short.arw"]},
                ],
            },
            {
                "id": "fixture-beta",
                "repo": self.beta_dir.as_uri(),
                "commit": self.beta_commit,
                "licence_note": "beta is NonCommercial",
                "files": [
                    {"path": "raw/RAW_BETA.ARW", "name": "DSLR-B1.ARW", "blob": beta_blobs["raw/RAW_BETA.ARW"]},
                    {
                        "path": "formats/raw/Sony/X1/12bit-compressed-(3_2).arw",
                        "name": "ILCE-X1_craw.ARW",
                        "blob": beta_blobs["formats/raw/Sony/X1/12bit-compressed-(3_2).arw"],
                    },
                ],
            },
        ]
        self.all_names = ["A1.ARW", "DSLR-B1.ARW", "ILCE-TEST1.ARW", "ILCE-X1_craw.ARW", "NEX-T2.ARW"]

    def run_script(self, *args: str, extra: Dict[str, str] | None = None, unset: Tuple[str, ...] = ("CI", "GITHUB_ACTIONS")) -> Tuple[int, str, str]:
        table = self.tmp / "sources.json"
        table.write_text(json.dumps(self.sources))
        env = {k: v for k, v in os.environ.items() if k not in unset}
        env.update({
            "YM_TESTDATA": str(self.td),
            "YM_REAL_EVAL_SOURCES": str(table),
            "GIT_TERMINAL_PROMPT": "0",
        })
        env.update(extra or {})
        p = subprocess.run(["bash", str(SCRIPT), *args], env=env, capture_output=True, text=True, timeout=120)
        return p.returncode, p.stdout, p.stderr

    @property
    def files(self) -> Path:
        return self.td / "real" / "files"

    def names(self) -> List[str]:
        return sorted(p.name for p in self.files.iterdir()) if self.files.is_dir() else []

    # ---- the fetch ---------------------------------------------------------------------------------------------------
    def test_fetches_only_the_listed_blobs_and_links_by_table_name(self) -> None:
        code, out, err = self.run_script()
        self.assertEqual(code, 0, out + err)
        self.assertEqual(self.names(), self.all_names)
        for p in self.files.iterdir():
            self.assertTrue(p.is_symlink() and p.is_file(), p)
        alpha, beta = self.td / "real" / "fixture-alpha", self.td / "real" / "fixture-beta"
        self.assertTrue((alpha / "arw" / "Sony NEX-T2.ARW").is_file())
        self.assertTrue((beta / "formats" / "raw" / "Sony" / "X1" / "12bit-compressed-(3_2).arw").is_file())
        self.assertEqual((self.files / "ILCE-X1_craw.ARW").read_bytes(), (self.beta_dir / "formats/raw/Sony/X1/12bit-compressed-(3_2).arw").read_bytes())
        # nothing else of the repositories, no git folder, no half written file
        self.assertFalse((alpha / "arw" / "Sony Unlisted.arw").exists(), "a file next to the listed ones is not fetched")
        self.assertFalse((alpha / "arw" / "metadata").exists())
        self.assertFalse((alpha / "jpg").exists())
        self.assertFalse((beta / "raw" / "RAW_OTHER.ARW").exists())
        for src in (alpha, beta):
            self.assertFalse((src / ".git").exists(), "the git folder is deleted after the download")
            self.assertEqual([p for p in src.rglob("*") if p.name.endswith(".part")], [])
            self.assertIn("Local evaluation only", (src / "PINNED.txt").read_text())
        self.assertIn(self.alpha_commit, (alpha / "PINNED.txt").read_text())
        self.assertIn("beta is NonCommercial", (beta / "PINNED.txt").read_text())

    def test_lists_model_size_and_source_and_warns_about_every_licence(self) -> None:
        code, out, err = self.run_script()
        self.assertEqual(code, 0, out + err)
        for model in ("ILCE-TEST1", "NEX-T2", "A1", "DSLR-B1", "ILCE-X1"):
            self.assertIn(model, out)
        size = (self.alpha_dir / "arw" / "Sony ILCE-TEST1 (T1).arw").stat().st_size
        self.assertIn(f"{size / 1e6:.1f} MB", out)
        self.assertIn("5 files", out)
        self.assertIn("fixture-alpha (pinned)", out)
        self.assertIn("fixture-beta (pinned)", out)
        self.assertIn("LOCAL EVALUATION ONLY", err)
        self.assertIn("LICENCE OF THESE FILES IS UNCLEAR OR RESTRICTED", err)
        for word in ("never commit", "never use them in CI", "never redistribute"):
            self.assertIn(word, err)
        # what each repository says is printed with its name and file count
        self.assertIn("fixture-alpha (3 files): alpha says nothing about the licence", err)
        self.assertIn("fixture-beta (2 files): beta is NonCommercial", err)

    def test_second_run_needs_no_network(self) -> None:
        self.assertEqual(self.run_script()[0], 0)
        shutil.rmtree(self.alpha_dir)   # the sources are gone: a second run must not ask for them
        shutil.rmtree(self.beta_dir)
        code, out, err = self.run_script()
        self.assertEqual(code, 0, out + err)
        self.assertIn("fixture-alpha: already fetched (3 files at", out)
        self.assertIn("fixture-beta: already fetched (2 files at", out)
        self.assertEqual(self.names(), self.all_names)

    def test_a_deleted_file_is_fetched_again_and_the_others_are_kept(self) -> None:
        self.assertEqual(self.run_script()[0], 0)
        gone = self.td / "real" / "fixture-alpha" / "arw" / "Sony NEX-T2.ARW"
        gone.unlink()
        kept = self.td / "real" / "fixture-alpha" / "arw" / "Sony Short.arw"
        before = kept.stat().st_mtime_ns
        code, out, err = self.run_script()
        self.assertEqual(code, 0, out + err)
        self.assertIn("fetching 1 file(s)", out)
        self.assertTrue(gone.is_file())
        self.assertEqual(kept.stat().st_mtime_ns, before)
        self.assertTrue((self.files / "NEX-T2.ARW").is_file())

    def test_list_works_offline_and_reports_nothing_found_with_exit_3(self) -> None:
        code, out, err = self.run_script("--list")
        self.assertEqual(code, 3, out + err)
        self.assertIn("no ARW files", out)
        self.assertEqual(self.run_script()[0], 0)
        shutil.rmtree(self.alpha_dir)
        shutil.rmtree(self.beta_dir)
        code, out, err = self.run_script("--list")
        self.assertEqual(code, 0, out + err)
        self.assertIn("fixture-alpha (pinned)", out)
        self.assertIn("LOCAL EVALUATION ONLY", err)

    def test_list_recognises_a_pinned_copy_by_its_blob_id_without_the_source_folders(self) -> None:
        self.files.mkdir(parents=True)
        shutil.copyfile(self.beta_dir / "raw" / "RAW_BETA.ARW", self.files / "DSLR-B1.ARW")
        (self.files / "mine.ARW").write_bytes(tiff_with_model("MINE"))
        code, out, err = self.run_script("--list")
        self.assertEqual(code, 0, out + err)
        row = next(line for line in out.splitlines() if "DSLR-B1.ARW" in line)
        self.assertIn("fixture-beta (pinned)", row)
        row = next(line for line in out.splitlines() if "mine.ARW" in line)
        self.assertIn("not from this script", row)

    # ---- files the script did not write -------------------------------------------------------------------------------
    def test_a_different_file_is_left_alone_unless_forced(self) -> None:
        self.assertEqual(self.run_script()[0], 0)
        mine = self.files / "NEX-T2.ARW"
        mine.unlink()
        mine.write_bytes(b"my own file")
        code, out, err = self.run_script()
        self.assertEqual(code, 1, out + err)
        self.assertIn("left alone", err)
        self.assertEqual(mine.read_bytes(), b"my own file")
        self.assertIn("not from this script", out)
        code, out, err = self.run_script("--force")
        self.assertEqual(code, 0, out + err)
        self.assertTrue(mine.is_symlink())
        self.assertNotEqual(mine.read_bytes(), b"my own file")

    def test_an_identical_copy_is_accepted_and_not_replaced(self) -> None:
        self.files.mkdir(parents=True)
        copy = self.files / "NEX-T2.ARW"
        shutil.copyfile(self.alpha_dir / "arw" / "Sony NEX-T2.ARW", copy)
        code, out, err = self.run_script()
        self.assertEqual(code, 0, out + err)
        self.assertFalse(copy.is_symlink(), "an identical regular file stays as it is")
        self.assertEqual(self.names(), self.all_names)

    def test_a_link_to_a_file_that_is_gone_is_made_again(self) -> None:
        self.files.mkdir(parents=True)
        (self.files / "NEX-T2.ARW").symlink_to("../fixture-alpha/arw/not-there.arw")
        code, out, err = self.run_script()
        self.assertEqual(code, 0, out + err)
        self.assertTrue((self.files / "NEX-T2.ARW").is_file(), "the link points at the pinned file now")

    def test_other_files_in_the_folder_are_never_touched(self) -> None:
        self.files.mkdir(parents=True)
        other = self.files / "DSLR-A300.ARW"
        other.write_bytes(b"a file from somewhere else")
        self.assertEqual(self.run_script()[0], 0)
        self.assertEqual(other.read_bytes(), b"a file from somewhere else")
        code, out, _ = self.run_script("--force")
        self.assertEqual(code, 0)
        self.assertEqual(other.read_bytes(), b"a file from somewhere else")
        self.assertIn("DSLR-A300.ARW", out)

    def test_a_folder_in_the_way_is_never_removed(self) -> None:
        (self.files / "NEX-T2.ARW").mkdir(parents=True)
        code, out, err = self.run_script("--force")
        self.assertEqual(code, 1, out + err)
        self.assertIn("is a folder", err)
        self.assertTrue((self.files / "NEX-T2.ARW").is_dir())

    # ---- refusals and failures ----------------------------------------------------------------------------------------
    def test_refuses_to_run_in_ci(self) -> None:
        for var in ("CI", "GITHUB_ACTIONS"):
            for args in ((), ("--list",)):
                code, out, err = self.run_script(*args, extra={var: "true"})
                self.assertEqual(code, 2, (var, args, out, err))
                self.assertIn("refusing to run in CI", err)
        self.assertFalse(self.td.exists(), "nothing was fetched or created")

    def test_an_unreachable_source_exits_4_leaves_nothing_of_it_and_does_not_stop_the_others(self) -> None:
        self.sources[0]["repo"] = (self.tmp / "missing").as_uri()
        code, out, err = self.run_script()
        self.assertEqual(code, 4, out + err)
        self.assertIn("cannot fetch", err)
        self.assertFalse((self.td / "real" / "fixture-alpha").exists(), "nothing of the failed source is left")
        self.assertEqual(self.names(), ["DSLR-B1.ARW", "ILCE-X1_craw.ARW"])
        self.assertFalse((self.td / "real" / "fixture-beta" / ".git").exists())

    def test_a_commit_that_is_not_there_is_refused(self) -> None:
        self.sources[0]["commit"] = "0" * 40
        self.sources[1]["commit"] = "1" * 40
        code, out, err = self.run_script()
        self.assertEqual(code, 4, out + err)
        self.assertEqual(self.names(), [])

    def test_a_blob_id_that_is_not_the_one_in_the_commit_is_refused(self) -> None:
        self.sources[0]["files"][1]["blob"] = "2" * 40   # the table is wrong: the commit lists another blob for this path
        code, out, err = self.run_script()
        self.assertEqual(code, 1, out + err)
        self.assertIn("the table pins", err)
        self.assertIn("Sony NEX-T2.ARW", err)
        self.assertNotIn("NEX-T2.ARW", self.names())
        self.assertIn("DSLR-B1.ARW", self.names(), "the other source is still fetched")

    def test_a_modified_file_is_refused_and_kept(self) -> None:
        self.assertEqual(self.run_script()[0], 0)
        edited = self.td / "real" / "fixture-alpha" / "arw" / "Sony NEX-T2.ARW"
        edited.write_bytes(b"edited")
        code, out, err = self.run_script()
        self.assertEqual(code, 1, out + err)
        self.assertIn("local changes", err)
        self.assertEqual(edited.read_bytes(), b"edited")

    # ---- the table ---------------------------------------------------------------------------------------------------
    def test_an_invalid_table_exits_2_before_anything_is_fetched(self) -> None:
        good = json.loads(json.dumps(self.sources))
        bad_tables = {
            "a path that leaves the repository": ("files", 0, "path", "../outside.arw"),
            "an absolute path": ("files", 0, "path", "/etc/passwd"),
            "a link name with a slash": ("files", 0, "name", "../x.ARW"),
            "a link name that repeats": ("files", 1, "name", "ILCE-TEST1.ARW"),
            "a blob id that is not 40 hex digits": ("files", 0, "blob", "abc"),
            "a commit that is not a full SHA-1": (None, None, "commit", "main"),
            "the reserved id": (None, None, "id", "files"),
        }
        for what, (list_key, idx, key, value) in bad_tables.items():
            with self.subTest(what):
                self.sources = json.loads(json.dumps(good))
                target = self.sources[0] if list_key is None else self.sources[0][list_key][idx]
                target[key] = value
                code, out, err = self.run_script()
                self.assertEqual(code, 2, (out, err))
                self.assertFalse(self.td.exists(), "nothing was fetched or created")
        self.sources = json.loads(json.dumps(good))
        (self.tmp / "sources.json").write_text("not json")
        env = {**os.environ, "YM_TESTDATA": str(self.td), "YM_REAL_EVAL_SOURCES": str(self.tmp / "sources.json")}
        env.pop("CI", None)
        env.pop("GITHUB_ACTIONS", None)
        p = subprocess.run(["bash", str(SCRIPT)], env=env, capture_output=True, text=True, timeout=60)
        self.assertEqual(p.returncode, 2, p.stderr)
        self.assertIn("cannot read the source table", p.stderr)

    def test_usage(self) -> None:
        code, _, err = self.run_script("--bogus")
        self.assertEqual(code, 2)
        self.assertIn("unknown option", err)
        code, out, _ = self.run_script("--help")
        self.assertEqual(code, 0)
        self.assertIn("LICENCES ARE UNCLEAR OR RESTRICTED", out)

    def test_the_built_in_table_is_valid_and_pins_the_nine_files_the_local_tests_read(self) -> None:
        import real_eval  # the script's own module; standard library only

        real_eval.validate_sources(real_eval.SOURCES)
        by_source = {s["id"]: sorted(f["name"] for f in s["files"]) for s in real_eval.SOURCES}
        self.assertEqual(
            by_source,
            {
                "metadata-extractor-images": ["DSLR-A500.ARW", "ILCE-5000.ARW", "ILCE-7M3.ARW", "ILCE-7M4.ARW", "NEX-5N.ARW", "NEX-7.ARW"],
                "OpenImageIO-images": ["DSLR-A300.ARW"],
                "immich-test-assets": ["ILCE-6300_craw.ARW", "ILCE-7M2_unc14.ARW"],
            },
        )
        for s in real_eval.SOURCES:
            self.assertTrue(s["repo"].startswith("https://github.com/") and s["repo"].endswith(".git"), s["repo"])
            self.assertEqual(len(s["commit"]), 40)
            self.assertTrue(s["licence_note"])
        # the ones with a restriction say so in plain words
        notes = {s["id"]: s["licence_note"] for s in real_eval.SOURCES}
        self.assertIn("NonCommercial", notes["OpenImageIO-images"])
        self.assertIn("nobody has verified", notes["immich-test-assets"])


if __name__ == "__main__":
    unittest.main()
