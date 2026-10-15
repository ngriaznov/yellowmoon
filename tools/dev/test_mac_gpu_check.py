"""The report of tools/dev/mac-gpu-check.sh against stand-ins for cargo, ym and golden (stdlib only).

The script builds and runs the real thing on a Mac; here `cargo` is a shell function-free shim that prints what cargo test
prints, and `ym` / `golden` are scripts that print what they print, so the shape of the one JSON file, the exit codes and the
refusal to write anywhere but the named file are checked in a second.
"""

import json
import os
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent / "mac-gpu-check.sh"

CARGO = r"""#!/bin/sh
# build: nothing to do; test: a plausible transcript (the first test binary fails when FAIL_ONE is set).
case "$1" in
  build) exit 0 ;;
  test)
    echo "   Running unittests src/lib.rs (target/release/deps/ym_gpu-0123456789abcdef)"
    echo "test a ... ok"
    echo "test result: ok. 12 passed; 0 failed; 1 ignored; 0 measured; 0 filtered out; finished in 0.1s"
    echo "     Running tests/gpu_chain.rs (target/release/deps/gpu_chain-fedcba9876543210)"
    if [ -n "$FAIL_ONE" ]; then
      echo "test the_bad_one ... FAILED"
      echo "test result: FAILED. 3 passed; 1 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.1s"
      exit 101
    fi
    echo "test result: ok. 4 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.1s"
    exit 0 ;;
esac
exit 0
"""

YM = r"""#!/bin/sh
if [ "$1" = doctor ] && [ "$2" = --json ]; then
  echo '{"version":"0.0.1","gpu":{"available":true,"name":"Stub \"GPU\""}}'
  exit 0
fi
if [ "$1" = doctor ] && [ "$2" = --bench ]; then
  if [ -n "$BENCH_DIFFERS" ]; then ident=false; else ident=true; fi
  echo '{"bench":1,"repeat":{"runs":2,"identical":'"$ident"'}}'
  exit 0
fi
exit 0
"""

GOLDEN = r"""#!/bin/sh
# golden device-check ... --report FILE
while [ $# -gt 0 ]; do
  if [ "$1" = --report ]; then
    printf '{\n  "metric": "gpu",\n  "outcomes": []\n}\n' > "$2"
  fi
  shift
done
exit 0
"""


def write_exec(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


class MacGpuCheck(unittest.TestCase):
    def setUp(self) -> None:
        self.dir = tempfile.TemporaryDirectory()
        root = Path(self.dir.name)
        self.shim = root / "shim"
        write_exec(self.shim / "cargo", CARGO)
        self.target = root / "target"
        write_exec(self.target / "release" / "ym", YM)
        write_exec(self.target / "release" / "golden", GOLDEN)
        self.photo = root / "photo.ARW"
        self.photo.write_bytes(b"x")
        self.out = root / "out" / "report.json"
        self.out.parent.mkdir()

    def tearDown(self) -> None:
        self.dir.cleanup()

    def run_script(self, *args: str, **env: str) -> subprocess.CompletedProcess:
        e = dict(os.environ)
        e["PATH"] = f"{self.shim}:{e['PATH']}"
        e["CARGO_TARGET_DIR"] = str(self.target)
        e.update(env)
        return subprocess.run(
            [str(SCRIPT), *args],
            capture_output=True,
            text=True,
            env=e,
            cwd=self.dir.name,
            timeout=120,
        )

    def test_the_report_has_the_documented_keys_and_nothing_else_is_written(self) -> None:
        before = {p for p in Path(self.dir.name).rglob("*")}
        r = self.run_script("--out", str(self.out), "--photo", str(self.photo), "--repeat", "2", "--with-cpu")
        self.assertEqual(r.returncode, 0, r.stderr)
        report = json.loads(self.out.read_text())
        self.assertEqual(
            set(report),
            {"schema", "generated", "machine", "git", "env", "doctor", "suites", "bench", "bench_cpu", "device_vs_cpu", "ok"},
        )
        self.assertEqual(report["schema"], 1)
        self.assertTrue(report["ok"])
        self.assertEqual(len(report["suites"]), 3)
        first = report["suites"][0]
        self.assertEqual((first["passed"], first["failed"], first["ignored"]), (16, 0, 1))
        self.assertEqual([b["name"] for b in first["binaries"]], ["ym_gpu", "gpu_chain"])
        self.assertEqual(first["failures"], [])
        self.assertIn("cargo test -p ym-gpu", first["command"])
        self.assertEqual(report["doctor"]["gpu"]["name"], 'Stub "GPU"')
        self.assertTrue(report["bench"]["repeat"]["identical"])
        self.assertIsNotNone(report["bench_cpu"])
        self.assertEqual(report["device_vs_cpu"]["metric"], "gpu")
        # The only new file is the report (the scratch folder is gone, no temporary is left beside it).
        after = {p for p in Path(self.dir.name).rglob("*")}
        self.assertEqual(after - before, {self.out})

    def test_a_failing_suite_and_differing_exports_fail_the_run_and_are_named(self) -> None:
        r = self.run_script("--out", str(self.out), "--photo", str(self.photo), FAIL_ONE="1", BENCH_DIFFERS="1")
        self.assertEqual(r.returncode, 1, r.stderr)
        report = json.loads(self.out.read_text())
        self.assertFalse(report["ok"])
        self.assertEqual(report["suites"][0]["exit"], 101)
        self.assertEqual(report["suites"][0]["failures"], ["the_bad_one"])
        self.assertFalse(report["bench"]["repeat"]["identical"])
        self.assertIsNone(report["bench_cpu"], "--with-cpu was not given")

    def test_without_a_photo_only_the_suites_run(self) -> None:
        r = self.run_script("--out", str(self.out), "--skip-tests")
        self.assertEqual(r.returncode, 0, r.stderr)
        report = json.loads(self.out.read_text())
        self.assertEqual(report["suites"], [])
        self.assertIsNone(report["bench"])
        self.assertIsNone(report["device_vs_cpu"])

    def test_usage_errors_exit_2_and_write_nothing(self) -> None:
        for args in (
            ["--out", str(self.dir.name)],  # a folder
            ["--out", str(self.out.with_suffix(".txt"))],  # not .json
            ["--out", str(self.out), "--photo", str(self.photo.with_name("nope.ARW"))],
            ["--out", str(self.out), "--repeat", "many"],
            ["--bogus"],
        ):
            r = self.run_script(*args)
            self.assertEqual(r.returncode, 2, (args, r.stderr))
        self.assertFalse(self.out.exists())

    def test_a_file_that_git_tracks_is_never_replaced(self) -> None:
        # The script's own folder is the repository: package.json is a tracked .json file there. Named by its
        # absolute path and, from the repository root, by its relative one, the script must refuse both (exit 2)
        # and leave the file as it is.
        root = SCRIPT.parent.parent.parent
        tracked = root / "package.json"
        tracking = subprocess.run(
            ["git", "ls-files", "--error-unmatch", "--", "package.json"], cwd=root, capture_output=True, text=True
        )
        if tracking.returncode != 0:
            self.skipTest("package.json is not tracked here (a source archive, not a clone)")
        before = tracked.read_bytes()
        for args, cwd in ((["--out", str(tracked), "--skip-tests"], None), (["--out", "package.json", "--skip-tests"], str(root))):
            e = dict(os.environ)
            e["PATH"] = f"{self.shim}:{e['PATH']}"
            e["CARGO_TARGET_DIR"] = str(self.target)
            r = subprocess.run(
                [str(SCRIPT), *args], capture_output=True, text=True, env=e, cwd=cwd or self.dir.name, timeout=120
            )
            self.assertEqual(r.returncode, 2, (args, r.stderr))
            self.assertIn("git tracks", r.stderr)
            self.assertEqual(tracked.read_bytes(), before)

    def test_a_relative_out_and_photo_are_read_from_the_folder_the_command_was_typed_in(self) -> None:
        # The script works from the repository root; the report and the photo must still be found where the user
        # said, not in the repository (`--out report.json` used to be written to the repository root).
        root = SCRIPT.parent.parent.parent
        stray = root / "rel-report.json"
        self.assertFalse(stray.exists(), "a file of an earlier run is in the way")
        try:
            r = self.run_script("--out", "rel-report.json", "--photo", "photo.ARW", "--skip-tests")
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertFalse(stray.exists(), "the report was written into the repository root")
            report = json.loads((Path(self.dir.name) / "rel-report.json").read_text())
            self.assertIsNotNone(report["bench"])
            # A relative folder that exists only here.
            r = self.run_script("--out", "out/nested.json", "--skip-tests")
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertTrue((self.out.parent / "nested.json").is_file())
            # A relative photo that does not exist here is refused, whatever the repository root holds.
            r = self.run_script("--out", "rel-report.json", "--photo", "README.md")
            self.assertEqual(r.returncode, 2, r.stderr)
        finally:
            stray.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
