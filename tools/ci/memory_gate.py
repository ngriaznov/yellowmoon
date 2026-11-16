#!/usr/bin/env python3
"""The `memory-8gb` job (docs/26-hardening.md P4-2, task H-12): the peak memory of a 50 MP and a 61 MP export as an 8 GB machine.

For each size it writes a synthetic ARW (the `gen_arw` example), then runs `ym render -v` to JPEG, TIFF 16-bit and DNG with
YELLOWMOON_RAM_GB=8, the CPU path, and reads two numbers for each run:

  * the peak resident memory of the process (`ru_maxrss` of the child, from `os.wait4`);
  * the peak of the memory ledger, which `ym render -v` prints ("Ledger peak: N bytes").

It fails when
  * the peak RSS of a run is above its bound in BOUNDS_MIB below; or
  * the ledger peak is not within a factor of two of the RSS (RSS / 2 <= ledger <= 2 x RSS): a large buffer that
    nobody counted, or a count with nothing behind it.

Python is test tooling only, never part of the app or the CLI. Usage:

    tools/ci/memory_gate.py --ym target/release/ym --gen target/release/examples/gen_arw --work /tmp/mem [--summary FILE]

The table of the numbers goes to stdout and, in Markdown, to --summary (CI passes $GITHUB_STEP_SUMMARY).
"""

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path

MIB = 1024 * 1024

# (label, camera model, width, height). 61 MP is the A7R IV's 9504 x 6336; 50 MP the A1's 8640 x 5760.
SIZES = [
    ("50MP", "ILCE-1", 8640, 5760),
    ("61MP", "ILCE-7RM4", 9504, 6336),
]

# (label, `ym render` arguments)
FORMATS = [
    ("JPEG", ["--format", "jpeg"]),
    ("TIFF16", ["--format", "tiff", "--bits", "16"]),
    ("DNG", ["--format", "dng"]),
]

# The bound on the peak RSS of each run, in MiB: (size, format) -> bound.
#
# FINAL, from CI run 43244397676 (job 131273066799, commit 9731571, release build, YELLOWMOON_RAM_GB=8, CPU path): each
# bound is that run's peak RSS x 1.10, rounded up to the next 10 MiB (the measured peak is in the comment of each line).
BOUNDS_MIB = {
    ("50MP", "JPEG"): 2310,  # CI 2093
    ("50MP", "TIFF16"): 1950,  # CI 1764
    ("50MP", "DNG"): 2380,  # CI 2157
    ("61MP", "JPEG"): 2770,  # CI 2516
    ("61MP", "TIFF16"): 2210,  # CI 2008
    ("61MP", "DNG"): 2740,  # CI 2483
}

# Within a factor of two (docs/26 H-13).
LEDGER_FACTOR = 2.0

LEDGER_LINE = re.compile(r"Ledger peak: (\d+) bytes")


def run_measured(cmd, env, cwd):
    """Runs `cmd`; returns (exit code, peak RSS in bytes of the child and its own children, stderr, seconds)."""
    started = time.monotonic()
    proc = subprocess.Popen(
        cmd, env=env, cwd=cwd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True
    )
    # wait4 gives the resource usage of this one child: ru_maxrss is in KiB on Linux (bytes on macOS).
    stderr = proc.stderr.read()
    _, status, usage = os.wait4(proc.pid, 0)
    proc.returncode = os.waitstatus_to_exitcode(status)
    scale = 1 if sys.platform == "darwin" else 1024
    return proc.returncode, usage.ru_maxrss * scale, stderr, time.monotonic() - started


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ym", required=True, type=Path)
    ap.add_argument("--gen", required=True, type=Path, help="the gen_arw example")
    ap.add_argument("--work", required=True, type=Path, help="a folder for the synthetic files and the exports")
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--ram-gb", default="8")
    ap.add_argument("--summary", type=Path)
    ap.add_argument("--only", help="a comma-separated list of size labels to run (50MP,61MP); default all")
    ap.add_argument("--formats", help="a comma-separated list of format labels to run (JPEG,TIFF16,DNG); default all")
    ap.add_argument("--no-bounds", action="store_true", help="record only (the first run of a new machine)")
    args = ap.parse_args()
    # The renders run in the work folder: paths given relative to here must not move with them.
    args.ym, args.gen, args.work = args.ym.resolve(), args.gen.resolve(), args.work.resolve()

    args.work.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.update(
        YELLOWMOON_RAM_GB=args.ram_gb,
        YELLOWMOON_GPU="off",
        YELLOWMOON_DATA=str(args.data.resolve()),
        YELLOWMOON_HOME=str((args.work / "home").resolve()),
        YELLOWMOON_TEST_APPLE_AUTO="none",
        YELLOWMOON_TEST_AUTO_PROVIDERS="none",
    )
    env.pop("RUST_LOG", None)

    rows = []
    failures = []
    only = set(args.only.split(",")) if args.only else None
    for size, model, width, height in SIZES:
        if only and size not in only:
            continue
        arw = args.work / f"{size}.ARW"
        if not arw.exists():
            subprocess.run([str(args.gen), model, str(width), str(height), str(arw)], check=True, stdout=subprocess.DEVNULL)
        for fmt, fmt_args in FORMATS:
            if args.formats and fmt not in args.formats.split(","):
                continue
            out = args.work / f"out-{size}-{fmt}"
            out.mkdir(exist_ok=True)
            cmd = [str(args.ym), "render", "-v", *fmt_args, "--overwrite", "--out", str(out), str(arw)]
            code, rss, stderr, seconds = run_measured(cmd, env, args.work)
            ledger_match = LEDGER_LINE.search(stderr)
            ledger = int(ledger_match.group(1)) if ledger_match else None
            bound = BOUNDS_MIB[(size, fmt)]
            problems = []
            if code != 0:
                problems.append(f"exit code {code}")
                sys.stderr.write(stderr[-2000:])
            if ledger is None:
                problems.append("no 'Ledger peak' line (ym render -v)")
            else:
                if ledger * LEDGER_FACTOR < rss:
                    problems.append(f"ledger {ledger / MIB:.0f} MiB is under half of the RSS {rss / MIB:.0f} MiB")
                if ledger > rss * LEDGER_FACTOR:
                    problems.append(f"ledger {ledger / MIB:.0f} MiB is over twice the RSS {rss / MIB:.0f} MiB")
            if not args.no_bounds and bound and rss > bound * MIB:
                problems.append(f"RSS {rss / MIB:.0f} MiB is over the bound of {bound} MiB")
            rows.append((size, fmt, rss, ledger, bound, seconds, problems))
            failures.extend(f"{size} {fmt}: {p}" for p in problems)
            # The export is not needed any more; keep the disk free.
            for f in out.iterdir():
                f.unlink()

    lines = [
        "| Size | Format | Peak RSS (MiB) | Ledger peak (MiB) | Ledger / RSS | Bound (MiB) | Time (s) | |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for size, fmt, rss, ledger, bound, seconds, problems in rows:
        ratio = f"{ledger / rss:.2f}" if ledger else "-"
        lines.append(
            f"| {size} | {fmt} | {rss / MIB:.0f} | {'-' if ledger is None else f'{ledger / MIB:.0f}'} | {ratio} | "
            f"{bound or '-'} | {seconds:.0f} | {'FAIL' if problems else 'ok'} |"
        )
    table = "\n".join(lines)
    print(f"YELLOWMOON_RAM_GB={args.ram_gb}, CPU path\n")
    print(table)
    if args.summary:
        with args.summary.open("a") as f:
            f.write(f"### memory-8gb (YELLOWMOON_RAM_GB={args.ram_gb}, CPU path)\n\n{table}\n\n")
            if failures:
                f.write("\n".join(f"- {x}" for x in failures) + "\n")
    if failures:
        print("\nFAILED:", *failures, sep="\n  ", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
