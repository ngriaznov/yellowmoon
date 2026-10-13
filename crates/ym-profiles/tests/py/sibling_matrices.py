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

    python3 -I crates/ym-profiles/tests/py/sibling_matrices.py [--json] [--template FILE]

Needs `rawpy` and the synthetic set (`tools/fetch/synthetic.sh`). Exit status 0 when every pairing is
within the limit, 1 when one is not, 2 when the tool cannot run. LibRaw is used here as a test oracle
only; nothing of it ships.
"""

import json
import struct
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
REGISTRY = ROOT / "data" / "cameras" / "registry.json"
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


def main(argv):
    as_json = "--json" in argv
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
    exact = {b["profile"]: b for b in bodies if b["match"] == "exact"}
    rows = []
    for b in bodies:
        if b["match"] != "sensor_family":
            continue
        base = exact[b["profile"]]
        a = color_matrix(rawpy, np, template, b["model"], b["model_id"])
        z = color_matrix(rawpy, np, template, base["model"], base["model_id"])
        diff = float(np.abs(a - z).max())
        rows.append(
            {
                "pairing": f'{b["model"]} ({b["model_id"]}) -> {base["model"]}',
                "profile": b["profile"],
                "max_abs_diff": round(diff, 4),
                "limit": LIMIT,
                "pass": diff <= LIMIT,
            }
        )
    if as_json:
        print(json.dumps(rows, indent=2))
    else:
        for r in rows:
            verdict = "PASS" if r["pass"] else "FAIL"
            print(f'{r["pairing"]:34} max |diff| {r["max_abs_diff"]:.4f}  limit {LIMIT}  {verdict}')
    return 0 if all(r["pass"] for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
