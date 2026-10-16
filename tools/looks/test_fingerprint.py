"""Tests for fingerprint.py on synthetic zips. Run: python3 -I -m unittest tools/looks/test_fingerprint.py

Everything here is generated on the fly from our own formulas; no ARRI data is involved. Only
ARRI-like file names are used so the package patterns have something to match.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
TOOL = HERE / "fingerprint.py"
PACKAGES = HERE.parents[1] / "data" / "looks" / "arri-packages.json"


def cube_text(size: int, seed: int, title: str) -> bytes:
    """A small synthetic 3D LUT with a distinctive body (so tests can look for it)."""
    lines = [f'TITLE "{title}"', f"LUT_3D_SIZE {size}", "DOMAIN_MIN 0 0 0", "DOMAIN_MAX 1 1 1"]
    last = size - 1
    for b in range(size):
        for g in range(size):
            for r in range(size):
                lines.append(f"{r / last * 0.9 + seed * 0.001:.6f} {g / last:.6f} {b / last * 0.95:.6f}")
    return ("\n".join(lines) + "\n").encode()


def write_zip(path: Path, entries: dict[str, bytes]) -> Path:
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, data in entries.items():
            archive.writestr(name, data)
    return path


def alf4_bytes(size: int, seed: int) -> bytes:
    """A stand-in for an .alf4 file: the marker and the two header fields the tool reads (where the LUT
    block is, and the LUT size in it), then filler. Not a valid look file; the tool does not check one."""
    block = 0x558
    out = bytearray(0x658 + 6 * size**3)
    out[0:4] = b"3FLA"
    struct.pack_into("<I", out, 0x68, block)
    struct.pack_into("<H", out, block + 0x42, size)
    out[0x700:0x704] = struct.pack("<I", seed)
    return bytes(out)


def aml_bytes(size: int, seed: int) -> bytes:
    """A stand-in for an .aml file (see alf4_bytes)."""
    out = bytearray(0x668 + 6 * size**3)
    out[0:4] = b"KOLA"
    struct.pack_into("<I", out, 0x18, size)
    out[0x700:0x704] = struct.pack("<I", seed)
    return bytes(out)


def alf4_package(count: int, size: int) -> dict[str, bytes]:
    """Entries of a synthetic .alf4 package in ARRI's layout, junk included."""
    entries: dict[str, bytes] = {}
    groups = ["1000 Black-and-White", "3000 Film", "4000 Tinted"]
    root = "ARRI Look Library LogC4 (log-to-log) ALF4"
    for i in range(count):
        name = f"{root}/{groups[i % 3]}/LL-{1000 + 10 * (i + 1)}-Look-{i}.alf4"
        entries[name] = alf4_bytes(size, i + 1)
        entries[f"__MACOSX/{root}/{groups[i % 3]}/._LL-{1000 + 10 * (i + 1)}-Look-{i}.alf4"] = b"apple double"
    entries[f"{root}/.DS_Store"] = b"junk"
    return entries


def aml_package(count: int, size: int) -> dict[str, bytes]:
    """Entries of a synthetic .aml package: three strengths per look."""
    entries: dict[str, bytes] = {}
    root = "ARRI Look Library LogC3 to Rec709 ALF2"
    for i in range(count):
        for k, strength in enumerate(("Low", "Medium", "High")):
            entries[f"{root}/1000 Black-and-White/LL-{1110 + i}-Look-{i}-{strength}.aml"] = aml_bytes(size, 3 * i + k)
    entries[f"{root}/.DS_Store"] = b"junk"
    return entries


def look_package(count: int, size: int, enc: str, display: bool = False) -> dict[str, bytes]:
    """Entries of a synthetic package, junk included."""
    entries: dict[str, bytes] = {}
    groups = ["1000 Colour", "3000 Film", "4000 Mono"]
    for i in range(count):
        if display:
            name = f"{enc}_to_{['Gamma24_Rec709', 'P3_D65', 'Rec2020_PQ', 'HLG'][i % 4]}_{i}.cube"
        else:
            name = f"{groups[i % 3]}/Look_{i:02d}_{enc}.cube"
        entries[name] = cube_text(size, i + 1, f"{enc} {i}")
        entries[name.replace(".cube", ".xml")] = b"<adicam/>"
        entries["__MACOSX/" + name.replace("/", "/._")] = b"apple double"
    entries["Guide.pdf"] = b"%PDF-1.4"
    return entries


def run(*args: str | Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-I", str(TOOL), *map(str, args)], capture_output=True, text=True, check=False
    )


class FingerprintTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.db = self.dir / "arri-packages.json"
        shutil.copy(PACKAGES, self.db)
        # The shipped list carries the real fingerprints (names, sizes and hashes of ARRI's public
        # packages): start from a list without them, and shrink the expectations so the synthetic
        # packages are small.
        data = json.loads(self.db.read_text())
        for package in data["packages"]:
            package["fingerprints"] = []
            package.pop("observed", None)
            package["expected"]["cube_files"] = 4
            if package["expected"]["lut_sizes"]:
                package["expected"]["lut_sizes"] = [5]
            if package.get("format") == "aml":
                package["expected"]["cube_files"] = 12
        self.db.write_text(json.dumps(data, indent=2))

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def load(self) -> dict:
        return json.loads(self.db.read_text())

    def package(self, package_id: str) -> dict:
        return next(p for p in self.load()["packages"] if p["id"] == package_id)

    def test_fingerprints_a_look_library_and_recognises_it_from_the_zip_name(self) -> None:
        entries = look_package(4, 5, "LogC4")
        zip_path = write_zip(self.dir / "ARRI_Look_Library_LogC4.zip", entries)
        done = run(zip_path, "--packages", self.db)
        self.assertEqual(done.returncode, 0, done.stderr)
        package = self.package("look-library-logc4")
        prints = package["fingerprints"]
        self.assertEqual(len(prints), 4)
        by_path = {p["path"]: p for p in prints}
        self.assertEqual(sorted(by_path), sorted(n for n in entries if n.endswith(".cube") and "__MACOSX" not in n))
        for name, fp in by_path.items():
            data = entries[name]
            self.assertEqual(fp["size"], len(data))
            self.assertEqual(fp["lut_size"], 5)
            self.assertEqual(fp["sha256"], hashlib.sha256(data).hexdigest())
            self.assertEqual(set(fp), {"path", "size", "lut_size", "sha256"})
        observed = package["observed"]
        self.assertEqual(observed["zip_name"], "ARRI_Look_Library_LogC4.zip")
        self.assertEqual(observed["cube_files"], 4)
        self.assertEqual(observed["top_level_folders"], ["1000 Colour", "3000 Film", "4000 Mono"])
        self.assertEqual(observed["ignored"], {"xml": 4, "pdf": 1, "macos": 4})
        # The other packages are untouched.
        self.assertEqual(self.package("look-library-logc3")["fingerprints"], [])

    def test_never_writes_lut_data(self) -> None:
        entries = look_package(4, 5, "LogC4")
        zip_path = write_zip(self.dir / "ARRI_Look_Library_LogC4.zip", entries)
        self.assertEqual(run(zip_path, "--packages", self.db).returncode, 0)
        text = self.db.read_text()
        for name, data in entries.items():
            if name.endswith(".cube") and "__MACOSX" not in name:
                body = data.decode().splitlines()[5]  # a data row
                self.assertNotIn(body, text)
        # Nothing but short names and numbers: the list stays small whatever the LUT size.
        self.assertLess(len(text), 20_000)

    def test_keeps_the_note_and_the_other_top_level_fields(self) -> None:
        before = self.load()
        zip_path = write_zip(self.dir / "ARRI_Look_Library_LogC4.zip", look_package(4, 5, "LogC4"))
        self.assertEqual(run(zip_path, "--packages", self.db).returncode, 0)
        after = self.load()
        for key in ("version", "note", "patterns_note", "page_url", "display_names"):
            self.assertEqual(before[key], after[key], key)

    def test_second_run_changes_nothing(self) -> None:
        zip_path = write_zip(self.dir / "ARRI_Look_Library_LogC4.zip", look_package(4, 5, "LogC4"))
        self.assertEqual(run(zip_path, "--packages", self.db).returncode, 0)
        first = self.db.read_bytes()
        self.assertEqual(run(zip_path, "--packages", self.db).returncode, 0)
        self.assertEqual(self.db.read_bytes(), first)

    def test_several_zips_in_one_run(self) -> None:
        looks = write_zip(self.dir / "ARRI_Look_Library_LogC3.zip", look_package(4, 3, "LogC3"))
        displays = write_zip(self.dir / "ARRI_Display_LUTs_LogC4.zip", look_package(4, 5, "LogC4", display=True))
        done = run(looks, displays, "--packages", self.db)
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertEqual(len(self.package("look-library-logc3")["fingerprints"]), 4)
        self.assertEqual(len(self.package("display-luts-logc4")["fingerprints"]), 4)
        self.assertEqual(self.package("look-library-logc3")["expected"]["lut_sizes"], [3])

    def test_names_that_say_nothing_need_the_package_flag(self) -> None:
        zip_path = write_zip(self.dir / "download (1).zip", look_package(4, 3, "x"))
        done = run(zip_path, "--packages", self.db)
        self.assertEqual(done.returncode, 1)
        self.assertIn("--package", done.stderr)
        self.assertEqual(self.load()["packages"][2]["fingerprints"], [])
        done = run(zip_path, "--package", "look-library-logc3", "--packages", self.db)
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertEqual(len(self.package("look-library-logc3")["fingerprints"]), 4)

    def test_unknown_package_id_is_an_error(self) -> None:
        zip_path = write_zip(self.dir / "a.zip", look_package(4, 5, "LogC4"))
        done = run(zip_path, "--package", "nope", "--packages", self.db)
        self.assertEqual(done.returncode, 1)
        self.assertIn("nope", done.stderr)

    def test_a_zip_with_no_look_files_is_refused(self) -> None:
        zip_path = write_zip(self.dir / "ARRI_Look_Library_LogC4.zip", {"3000 Film/a.xml": b"x", "Guide.pdf": b"y"})
        done = run(zip_path, "--packages", self.db)
        self.assertEqual(done.returncode, 1)
        self.assertIn("no .cube, .alf4 or .aml files", done.stderr)

    def test_fingerprints_the_alf4_library(self) -> None:
        entries = alf4_package(4, 5)
        zip_path = write_zip(self.dir / "arri-look-library-logc4-log-to-log-alf4-data.zip", entries)
        done = run(zip_path, "--packages", self.db)
        self.assertEqual(done.returncode, 0, done.stderr)
        package = self.package("look-library-logc4-alf4")
        prints = package["fingerprints"]
        self.assertEqual(len(prints), 4)
        by_path = {p["path"]: p for p in prints}
        self.assertEqual(sorted(by_path), sorted(n for n in entries if n.endswith(".alf4") and "__MACOSX" not in n))
        for name, fp in by_path.items():
            self.assertEqual(fp["size"], len(entries[name]))
            self.assertEqual(fp["lut_size"], 5, "the LUT size comes from the file's header")
            self.assertEqual(fp["sha256"], hashlib.sha256(entries[name]).hexdigest())
            self.assertEqual(set(fp), {"path", "size", "lut_size", "sha256"})
        observed = package["observed"]
        self.assertEqual(observed["files"], 4)
        self.assertNotIn("cube_files", observed)
        self.assertEqual(observed["top_level_folders"], ["ARRI Look Library LogC4 (log-to-log) ALF4"])
        self.assertEqual(observed["ignored"], {"macos": 5})
        # The 3D-LUT package of the same encoding is untouched.
        self.assertEqual(self.package("look-library-logc4")["fingerprints"], [])

    def test_fingerprints_the_aml_library(self) -> None:
        entries = aml_package(4, 5)
        zip_path = write_zip(self.dir / "ARRI Look Library LogC3 (log-to-Rec709) ALF2_data.zip", entries)
        done = run(zip_path, "--packages", self.db)
        self.assertEqual(done.returncode, 0, done.stderr)
        package = self.package("look-library-logc3-rec709-alf2")
        self.assertEqual(len(package["fingerprints"]), 12)
        self.assertEqual({p["lut_size"] for p in package["fingerprints"]}, {5})
        self.assertEqual(package["observed"]["files"], 12)
        for other in ("look-library-logc3", "look-library-logc3-rec709"):
            self.assertEqual(self.package(other)["fingerprints"], [], other)

    def test_a_mixed_zip_and_a_wrong_forced_package_are_errors(self) -> None:
        mixed = {**look_package(2, 5, "LogC4"), **alf4_package(2, 5)}
        zip_path = write_zip(self.dir / "mixed.zip", mixed)
        done = run(zip_path, "--package", "look-library-logc4", "--packages", self.db)
        self.assertEqual(done.returncode, 1)
        self.assertIn("both", done.stderr)
        alf = write_zip(self.dir / "alf.zip", alf4_package(2, 5))
        done = run(alf, "--package", "look-library-logc4", "--packages", self.db)
        self.assertEqual(done.returncode, 1)
        self.assertIn("package of .cube files", done.stderr)

    def test_not_a_zip_is_an_error_not_a_crash(self) -> None:
        bad = self.dir / "broken.zip"
        bad.write_bytes(b"this is not a zip")
        done = run(bad, "--package", "look-library-logc4", "--packages", self.db)
        self.assertEqual(done.returncode, 1)
        self.assertIn("cannot open", done.stderr)
        self.assertNotIn("Traceback", done.stderr)

    def test_unsafe_entry_names_are_skipped_with_a_warning(self) -> None:
        entries = look_package(4, 5, "LogC4")
        zip_path = self.dir / "ARRI_Look_Library_LogC4.zip"
        write_zip(zip_path, entries)
        with zipfile.ZipFile(zip_path, "a") as archive:
            archive.writestr("../escape.cube", cube_text(5, 99, "x"))
        done = run(zip_path, "--packages", self.db)
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertIn("unsafe entry name", done.stderr)
        paths = [p["path"] for p in self.package("look-library-logc4")["fingerprints"]]
        self.assertFalse(any(".." in p for p in paths))

    def test_dry_run_prints_and_writes_nothing(self) -> None:
        before = self.db.read_bytes()
        zip_path = write_zip(self.dir / "ARRI_Look_Library_LogC4.zip", look_package(4, 5, "LogC4"))
        done = run(zip_path, "--packages", self.db, "--dry-run")
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertIn("sha256", done.stdout)
        self.assertEqual(self.db.read_bytes(), before)

    def test_differences_from_the_patterns_are_reported_as_notes(self) -> None:
        # The patterns say 4 files of size 5; this zip has 3 files of size 7.
        zip_path = write_zip(self.dir / "ARRI_Look_Library_LogC4.zip", look_package(3, 7, "LogC4"))
        done = run(zip_path, "--package", "look-library-logc4", "--packages", self.db)
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertIn("expected 4 files, the zip has 3", done.stderr)
        self.assertIn("expected LUT sizes [5]", done.stderr)
        package = self.package("look-library-logc4")
        self.assertEqual(package["expected"]["cube_files"], 3)
        self.assertEqual(package["expected"]["lut_sizes"], [7])

    def test_reformat_is_stable_and_keeps_meaning(self) -> None:
        before = self.load()
        self.assertEqual(run("--reformat", "--packages", self.db).returncode, 0)
        once = self.db.read_bytes()
        self.assertEqual(self.load(), before)
        self.assertEqual(run("--reformat", "--packages", self.db).returncode, 0)
        self.assertEqual(self.db.read_bytes(), once)

    def test_the_shipped_list_is_in_the_tools_layout_and_has_no_lut_data(self) -> None:
        shipped = PACKAGES.read_bytes()
        copy = self.dir / "copy.json"
        copy.write_bytes(shipped)
        self.assertEqual(run("--reformat", "--packages", copy).returncode, 0)
        self.assertEqual(copy.read_bytes(), shipped, "run `fingerprint.py --reformat` to restore the layout")
        self.assertLess(len(shipped), 200_000)
        # Fingerprints are names, sizes and hashes: no other key, no LUT rows anywhere.
        data = json.loads(shipped)
        for package in data["packages"]:
            for fp in package["fingerprints"]:
                self.assertEqual(set(fp), {"path", "size", "lut_size", "sha256"}, package["id"])
                self.assertRegex(fp["sha256"], r"^[0-9a-f]{64}$")
                self.assertTrue(fp["path"].endswith((".cube", ".alf4", ".aml")), fp["path"])
                self.assertLess(len(fp["path"]), 200)
        self.assertNotRegex(shipped.decode(), r"-?\d+\.\d+\s+-?\d+\.\d+\s+-?\d+\.\d+")

    def test_usage_errors_exit_2(self) -> None:
        self.assertEqual(run().returncode, 2)
        zips = [write_zip(self.dir / f"{i}.zip", {"a.cube": cube_text(3, i, "x")}) for i in range(2)]
        self.assertEqual(run(*zips, "--package", "look-library-logc4").returncode, 2)


if __name__ == "__main__":
    unittest.main()
