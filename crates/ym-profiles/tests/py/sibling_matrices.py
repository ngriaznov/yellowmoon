#!/usr/bin/env python3
"""Same-sensor pairing check of docs/03 (M1-04), with LibRaw as the independent tool.

docs/03: "Before a same-sensor pairing ships, check it: the sibling's ColorMatrix ... must match the
profile body's within 0.02 per coefficient after normalisation."  The spec names Adobe DNG Converter
samples, which V1 does not have (AGENTS.md V1 data rule: use public data or an independent tool).
LibRaw's per-model camera matrix is such a tool: it is the D65 ColorMatrix from Adobe's tables and it
is keyed by the Sony model ID in the file.

The script takes a synthetic Sony ARW as a template, rewrites its Model string and model ID (tag 0xB001)
to each registry body, reads `rgb_xyz_matrix` back through rawpy, normalises every row to sum 1 and
prints the largest per-coefficient difference between each sibling and its profile body.

    python3 -I crates/ym-profiles/tests/py/sibling_matrices.py [--json] [--verify] [--template FILE]

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
TEMPLATE = ROOT / "testdata" / "synthetic" / "uncompressed14__ILCE-7M3__noise.ARW"
LIMIT = 0.02
MODEL_ID_TAG = 0xB001


def patch(template: bytes, model: str, model_id: int) -> bytes:
    """The template with another IFD0 Model string and another SonyModelID."""
    out = bytearray(template)
    if out[:2] != b"II":
        raise SystemExit("the template must be a little-endian ARW")
    ifd0 = struct.unpack_from("<I", out, 4)[0]
    count = struct.unpack_from("<H", out, ifd0)[0]
    text = model.encode("ascii") + b"\0"
    offset = len(out)
    out += text  # the new string goes to the end of the file; the entry points at it
    for i in range(count):
        entry = ifd0 + 2 + 12 * i
        tag = struct.unpack_from("<H", out, entry)[0]
        if tag == 0x0110:  # Model
            struct.pack_into("<II", out, entry + 4, len(text), offset)
            break
    else:
        raise SystemExit("no Model tag in the template")
    needle = struct.pack("<HHI", MODEL_ID_TAG, 3, 1)  # SHORT, count 1
    at = bytes(out).find(needle)
    if at < 0:
        raise SystemExit("no SonyModelID in the template")
    struct.pack_into("<H", out, at + 8, model_id)
    return bytes(out)


def color_matrix(rawpy, np, template: bytes, model: str, model_id: int):
    """LibRaw's camera-to-XYZ matrix (3 x 3) for a body, rows normalised to sum 1."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "probe.ARW"
        path.write_bytes(patch(template, model, model_id))
        with rawpy.imread(str(path)) as raw:
            m = np.array(raw.rgb_xyz_matrix, dtype=float)[:3]
    return m / m.sum(axis=1, keepdims=True)


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
        row = {
            "pairing": f'{b["model"]} ({b["model_id"]}) -> {base["model"]}',
            "profile": profile,
            "max_abs_diff": round(diff, 4),
            "limit": LIMIT,
            "pass": diff <= LIMIT,
            "rejected": rejected,
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
    if verify:
        return 0 if all(r["agrees"] for r in rows) else 1
    # A shipped pairing must pass; a rejected one is expected to fail (if it passes now, it can be reconsidered).
    return 0 if all(r["pass"] != r["rejected"] for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
