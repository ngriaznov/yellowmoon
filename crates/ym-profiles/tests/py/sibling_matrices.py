#!/usr/bin/env python3
"""Same-sensor pairing check of docs/03 (M1-04), with LibRaw as the independent tool.

docs/03: "Before a same-sensor pairing ships, check it: the sibling's ColorMatrix ... must match the
profile body's within 0.02 per coefficient after normalisation."  The spec names Adobe DNG Converter
samples, which V1 does not have (AGENTS.md V1 data rule: use public data or an independent tool).
LibRaw's per-model camera matrix is such a tool: it is the D65 ColorMatrix from Adobe's tables, found by
the file's Model string (the model ID does not choose it; checked with LibRaw 0.22.1).

The script takes a synthetic Sony ARW as a template, rewrites its Model string and model ID (tag 0xB001)
to each registry body, reads `rgb_xyz_matrix` back through rawpy, normalises every row to sum 1 and
prints the largest per-coefficient difference between each sibling and its profile body.

    python3 -I crates/ym-profiles/tests/py/sibling_matrices.py [--json] [--verify] [--matrices] [--template FILE]

LibRaw takes the first table entry whose name is a *prefix* of the Model string, so a body without an entry
of its own silently gets an older body's numbers (2026-10-19: the A1 II read as the A1, the A7 V as the
2013 A7, the A7R VI as the 2013 A7R). Every body read here is therefore traced to the entry it came from
(`entry_of`), and a body read from another body's entry fails the run unless `SHARED_ENTRIES` lists that
entry for it.

`--matrices` compares `training/white_scale/colour_matrices.json` with LibRaw for every registry body
instead: a body LibRaw reads from its own entry (or its listed shared one) must agree to 0.0001; a body it
reads from another body's entry is listed as not checkable with this build.

The pairings are the registry's `sensor_family` bodies plus the ones a profile's `provenance.checks`
records as considered and rejected ("... was considered for <profile>"): those have match kind `default`
now, because they failed (ILCE-7RM5 and ILCE-7CR against the A7R IV, 0.0934).

`--verify` compares the measured value of every pairing with the one recorded in the profile's
`provenance.checks` (to 0.0006, and the same pass or fail) and exits 1 on any difference: the recorded
numbers are a measurement, not an assumption.

Needs `rawpy` and the synthetic set (`tools/fetch/synthetic.sh`). Exit status 0 when every shipped pairing
is within the limit (and every rejected one is not; with `--verify`, when the records agree), 1 when
not, 2 when the tool cannot run. LibRaw is used here as a test oracle only; nothing of it ships.
"""

import json
import re
import struct
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
REGISTRY = ROOT / "data" / "cameras" / "registry.json"
PROFILES = ROOT / "data" / "cameras" / "profiles"
RECORD_TOLERANCE = 0.0006
MATRICES = ROOT / "training" / "white_scale" / "colour_matrices.json"
TEMPLATE = ROOT / "testdata" / "synthetic" / "arw" / "chart__D6500__ILCE-7M3__iso400__rggb__14b__uncompressed14.ARW"
LIMIT = 0.02
MATRIX_TOLERANCE = 0.0001
MODEL_ID_TAG = 0xB001
#: Not a Sony model ID: LibRaw does not know it, so a probe goes by its Model string alone.
UNKNOWN_MODEL_ID = 0xFFFF
#: Bodies LibRaw reads from the entry of a shorter Model string on purpose: the matrix is published once for the
#: bodies named (LibRaw's notes say so for the first five; the two A versions are listed under the base body's entry
#: in the published tables). Model string -> the entry's name.
SHARED_ENTRIES = {
    "DSC-RX100M4": "DSC-RX100M",
    "ILCE-7SM2": "ILCE-7S",
    "ILCE-7RM3A": "ILCE-7RM3",
    "ILCE-7RM4A": "ILCE-7RM4",
    "ILCE-9M2": "ILCE-9",
    "ILCE-6100A": "ILCE-6100",
    "ILCE-6400A": "ILCE-6400",
}


def patch(template: bytes, model: str, model_id: int) -> bytes:
    """The template with another IFD0 Model string and another SonyModelID."""
    out = bytearray(template)
    if out[:2] != b"II":
        raise SystemExit("the template must be a little-endian ARW")
    ifd0 = struct.unpack_from("<I", out, 4)[0]
    count = struct.unpack_from("<H", out, ifd0)[0]
    text = model.encode("ascii") + b"\0"
    for i in range(count):
        entry = ifd0 + 2 + 12 * i
        tag = struct.unpack_from("<H", out, entry)[0]
        if tag == 0x0110:  # Model
            if len(text) <= 4:  # TIFF keeps a value of up to four bytes in the entry itself
                struct.pack_into("<I4s", out, entry + 4, len(text), text)
            else:  # a longer one goes to the end of the file; the entry points at it
                struct.pack_into("<II", out, entry + 4, len(text), len(out))
                out += text
            break
    else:
        raise SystemExit("no Model tag in the template")
    needle = struct.pack("<HHI", MODEL_ID_TAG, 3, 1)  # SHORT, count 1
    at = bytes(out).find(needle)
    if at < 0:
        raise SystemExit("no SonyModelID in the template")
    struct.pack_into("<H", out, at + 8, model_id)
    return bytes(out)


def read_matrix(rawpy, np, template: bytes, model: str, model_id: int):
    """LibRaw's matrix (3 x 3, XYZ to camera RGB as published) for a Model string and model ID."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "probe.ARW"
        path.write_bytes(patch(template, model, model_id))
        with rawpy.imread(str(path)) as raw:
            return np.array(raw.rgb_xyz_matrix, dtype=float)[:3]


def color_matrix(rawpy, np, template: bytes, model: str, model_id: int):
    """LibRaw's matrix for a body, rows normalised to sum 1."""
    m = read_matrix(rawpy, np, template, model, model_id)
    return m / m.sum(axis=1, keepdims=True)


def entry_of(rawpy, np, template: bytes, model: str, model_id: int) -> str:
    """The name of the table entry LibRaw reads a body's matrix from.

    The entry is the first whose name is a prefix of the Model string, so every string from that name up to the
    full Model string finds the same entry, and a shorter one cannot. The string is shortened one character at a
    time while the numbers stay the same; the last string that gives them is the name. The full Model string
    means the body has an entry of its own. (A shorter entry with the very same numbers would make the name come
    out too short: a false alarm, never a missed one.) The shortened strings carry a model ID LibRaw does not know,
    so that it cannot find the body by its ID when the string means nothing to it.
    """
    full = read_matrix(rawpy, np, template, model, model_id)
    name = model
    while len(name) > 1 and np.array_equal(read_matrix(rawpy, np, template, name[:-1], UNKNOWN_MODEL_ID), full):
        name = name[:-1]
    return name


def entry_problem(model: str, entry: str) -> str | None:
    """Why the entry a body was read from is not acceptable, or ``None``."""
    want = SHARED_ENTRIES.get(model, model)
    if entry == want:
        return None
    if want == model:
        return f"{model} has no entry of its own in this LibRaw; it was read from {entry}'s"
    return f"{model} was read from {entry}'s entry, not from {want}'s"


def check_matrices(rawpy, np, template: bytes, bodies) -> int:
    """``--matrices``: the committed published matrices against LibRaw, body by body."""
    committed = json.loads(MATRICES.read_text())["bodies"]
    bad = 0
    for b in bodies:
        mine = np.array(committed[str(b["model_id"])]["xyz_to_camera"], dtype=float)
        entry = entry_of(rawpy, np, template, b["model"], b["model_id"])
        problem = entry_problem(b["model"], entry)
        label = f'{b["model"]} ({b["model_id"]})'
        if problem is not None:
            print(f"{label:22} not checkable: {problem}")
            continue
        diff = float(np.abs(read_matrix(rawpy, np, template, b["model"], b["model_id"]) - mine).max())
        ok = diff <= MATRIX_TOLERANCE
        bad += not ok
        print(f"{label:22} entry {entry:12} max |diff| {diff:.4f}  {'OK' if ok else 'DIFFERS'}")
    return 0 if bad == 0 else 1


def recorded_checks():
    """Every provenance check of every profile: {(model_id, profile id): (considered, result text)}."""
    out = {}
    for path in sorted(PROFILES.glob("*/profile.json")):
        profile = json.loads(path.read_text())
        for c in profile["provenance"]["checks"]:
            m = re.match(r"^(\S+) \((\d+)\) (uses|was considered for) (\S+)$", c["pairing"])
            if m:
                out[(int(m.group(2)), m.group(4))] = (m.group(3) == "was considered for", c["result"])
    return out


def main(argv):
    as_json = "--json" in argv
    verify = "--verify" in argv
    template_path = Path(argv[argv.index("--template") + 1]) if "--template" in argv else TEMPLATE
    try:
        import numpy as np
        import rawpy
    except ImportError as e:
        print(f"cannot run: {e} (pip3 install rawpy numpy)", file=sys.stderr)
        return 2
    if not template_path.is_file():
        print(f"cannot run: {template_path} is missing (tools/fetch/synthetic.sh)", file=sys.stderr)
        return 2
    template = template_path.read_bytes()
    bodies = json.loads(REGISTRY.read_text())["bodies"]
    if "--matrices" in argv:
        return check_matrices(rawpy, np, template, bodies)
    by_id = {b["model_id"]: b for b in bodies}
    exact = {b["profile"]: b for b in bodies if b["match"] == "exact"}
    recorded = recorded_checks()
    pairs = [(b, b["profile"], False) for b in bodies if b["match"] == "sensor_family"]
    pairs += [(by_id[mid], pid, True) for (mid, pid), (considered, _) in sorted(recorded.items()) if considered]
    rows = []
    for b, profile, rejected in pairs:
        base = exact[profile]
        a = color_matrix(rawpy, np, template, b["model"], b["model_id"])
        z = color_matrix(rawpy, np, template, base["model"], base["model_id"])
        diff = float(np.abs(a - z).max())
        entries = [entry_of(rawpy, np, template, x["model"], x["model_id"]) for x in (b, base)]
        problems = [p for x, e in zip((b, base), entries) if (p := entry_problem(x["model"], e)) is not None]
        row = {
            "pairing": f'{b["model"]} ({b["model_id"]}) -> {base["model"]}',
            "profile": profile,
            "max_abs_diff": round(diff, 4),
            "limit": LIMIT,
            "pass": diff <= LIMIT,
            "rejected": rejected,
            "entries": entries,
            "entry_problems": problems,
        }
        if verify:
            note = recorded.get((b["model_id"], profile))
            m = re.match(r"^(pass|fail): max \|diff\| ([0-9.]+) ", note[1]) if note else None
            row["recorded"] = None if m is None else {"verdict": m.group(1), "max_abs_diff": float(m.group(2))}
            row["agrees"] = (
                m is not None
                and abs(float(m.group(2)) - row["max_abs_diff"]) <= RECORD_TOLERANCE
                and (m.group(1) == "pass") == row["pass"]
            )
        rows.append(row)
    if as_json:
        print(json.dumps(rows, indent=2))
    else:
        for r in rows:
            verdict = "PASS" if r["pass"] else "FAIL"
            extra = " (rejected: not a sibling)" if r["rejected"] else ""
            if verify:
                extra += "  recorded " + (f'{r["recorded"]["verdict"]} {r["recorded"]["max_abs_diff"]:.4f}' if r["recorded"] else "nothing")
                extra += "  OK" if r["agrees"] else "  DIFFERS"
            print(f'{r["pairing"]:34} max |diff| {r["max_abs_diff"]:.4f}  limit {LIMIT}  {verdict}{extra}')
            for problem in r["entry_problems"]:
                print(f"  NOT ITS OWN MATRIX: {problem}")
    # A matrix read from another body's entry says nothing about the body: the comparison does not count.
    if any(r["entry_problems"] for r in rows):
        return 1
    if verify:
        return 0 if all(r["agrees"] for r in rows) else 1
    # A shipped pairing must pass; a rejected one is expected to fail (if it passes now, it can be reconsidered).
    return 0 if all(r["pass"] != r["rejected"] for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
