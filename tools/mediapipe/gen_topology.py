#!/usr/bin/env python3
"""Writes `crates/ym-retouch/src/face_topology.rs`: the canonical face mesh topology of MediaPipe (docs/27 S-09).

Development only (AGENTS.md rule 6): run by hand when the sources change; the generated Rust file is committed.

    python tools/mediapipe/gen_topology.py --obj canonical_face_model.obj --connections face_mesh_connections.py

The two inputs are files of the MediaPipe repository (Apache-2.0, Copyright 2021 The MediaPipe Authors), refused unless their
SHA-256 is the pinned one:

  canonical_face_model.obj      mediapipe/modules/face_geometry/data/canonical_face_model.obj
                                the 468 vertices, the 898 triangles and the texture coordinates (the UV atlas) of the mesh
  face_mesh_connections.py      mediapipe/python/solutions/face_mesh_connections.py
                                the contour line sets (face oval, lips, eyes, eyebrows, irises)

The outputs are the triangles (vertex indices), the UV of every vertex, and the closed rings of landmark indices that the
retouch masks use. Only indices and UV numbers are written, rounded to 1e-6; the canonical 3D positions are not.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import sys
from pathlib import Path

PINS = {
    "canonical_face_model.obj": "8bac80443397e113f41a8b565ea72c59390bc031d9defab289dba7bc0c54e618",
    "face_mesh_connections.py": "cc7171e44d0612db38720cb2fd2516c395992cdd7457efe341543be6f3c22d87",
}

#: Rings that the connection sets do not give as one closed loop, written out (and checked against the sets below).
OUTER_LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
INNER_LIPS = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191]
RIGHT_EYEBROW = [46, 53, 52, 65, 55, 107, 66, 105, 63, 70]
LEFT_EYEBROW = [276, 283, 282, 295, 285, 336, 296, 334, 293, 300]


def check_pin(path: Path) -> bytes:
    data = path.read_bytes()
    got = hashlib.sha256(data).hexdigest()
    want = PINS[path.name]
    if got != want:
        raise SystemExit(f"{path}: SHA-256 {got}, the pin is {want}; nothing was written")
    return data


def ring(pairs) -> list[int]:
    """The closed loop of a connection set whose every vertex has two neighbours, as an index list."""
    adj: dict[int, list[int]] = {}
    for a, b in pairs:
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)
    if any(len(v) != 2 for v in adj.values()):
        raise SystemExit("a connection set is not a closed loop")
    start = min(adj)
    out, prev, cur = [start], None, start
    while True:
        nxt = [n for n in adj[cur] if n != prev]
        n = nxt[0]
        if n == start:
            break
        out.append(n)
        prev, cur = cur, n
    return out


def same_loop(a: list[int], b: list[int]) -> bool:
    return len(a) == len(b) and set(a) == set(b)


def array(name: str, doc: str, values: list[int]) -> str:
    body = ", ".join(str(v) for v in values)
    return f"/// {doc}\npub const {name}: [u16; {len(values)}] = [{body}];\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--obj", type=Path, required=True)
    parser.add_argument("--connections", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[2] / "crates/ym-retouch/src/face_topology.rs")
    args = parser.parse_args()

    obj = check_pin(args.obj).decode()
    check_pin(args.connections)
    spec = importlib.util.spec_from_file_location("face_mesh_connections", args.connections)
    c = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(c)

    vt: list[tuple[float, float]] = []
    triangles: list[tuple[int, int, int]] = []
    uv_of: dict[int, int] = {}
    n_vertices = 0
    for line in obj.splitlines():
        p = line.split()
        if not p:
            continue
        if p[0] == "v":
            n_vertices += 1
        elif p[0] == "vt":
            vt.append((float(p[1]), float(p[2])))
        elif p[0] == "f":
            idx = [tuple(int(x) for x in t.split("/")) for t in p[1:]]
            if len(idx) != 3:
                raise SystemExit("a face that is not a triangle")
            for v, t in idx:
                if uv_of.setdefault(v, t) != t:
                    raise SystemExit(f"vertex {v} has two texture coordinates")
            triangles.append(tuple(v - 1 for v, _ in idx))
    if n_vertices != 468 or len(vt) != 468 or len(triangles) != 898 or len(uv_of) != 468:
        raise SystemExit(f"unexpected model: {n_vertices} vertices, {len(vt)} uv, {len(triangles)} triangles")
    uv = [vt[uv_of[i + 1] - 1] for i in range(468)]

    oval = ring(c.FACEMESH_FACE_OVAL)
    left_eye = ring(c.FACEMESH_LEFT_EYE)
    right_eye = ring(c.FACEMESH_RIGHT_EYE)
    left_iris = ring(c.FACEMESH_LEFT_IRIS)
    right_iris = ring(c.FACEMESH_RIGHT_IRIS)
    lips = set(c.FACEMESH_LIPS)
    for name, r, want in (("outer lips", OUTER_LIPS, None), ("inner lips", INNER_LIPS, None)):
        pairs = {frozenset(p) for p in lips}
        for a, b in zip(r, r[1:] + r[:1]):
            if frozenset((a, b)) not in pairs:
                raise SystemExit(f"{name}: {a}-{b} is not a lip connection")
    for name, r, s in (
        ("right eyebrow", RIGHT_EYEBROW, c.FACEMESH_RIGHT_EYEBROW),
        ("left eyebrow", LEFT_EYEBROW, c.FACEMESH_LEFT_EYEBROW),
    ):
        used = {frozenset(p) for p in s}
        got = {frozenset((a, b)) for a, b in zip(r, r[1:] + r[:1])}
        if not used <= got:
            raise SystemExit(f"{name}: the written ring does not contain the eyebrow lines")

    out = [
        "//! The canonical face mesh of MediaPipe: triangles, texture coordinates and the landmark rings the masks use.",
        "//!",
        "//! GENERATED by `tools/mediapipe/gen_topology.py` from `canonical_face_model.obj` and `face_mesh_connections.py` of the",
        "//! MediaPipe repository (Apache-2.0, Copyright 2021 The MediaPipe Authors, https://github.com/google-ai-edge/mediapipe;",
        "//! `NOTICE` item 14). Do not edit by hand. Only vertex indices and texture coordinates are taken from the model.",
        "//!",
        "//! The landmark indices are those of Face Mesh V2's output: 0 to 467 are the mesh, 468 to 477 the irises.",
        "",
        "/// The number of vertices of the canonical mesh.",
        "pub const VERTICES: usize = 468;",
        "/// The number of triangles of the canonical mesh.",
        f"pub const TRIANGLE_COUNT: usize = {len(triangles)};",
        "",
        "/// The triangles as vertex indices. The winding is the canonical one: a face seen from the front, in picture",
        "/// coordinates (y down), has a negative signed area for every triangle that faces the camera.",
        f"pub const TRIANGLES: [[u16; 3]; {len(triangles)}] = [",
    ]
    for t in triangles:
        out.append(f"    [{t[0]}, {t[1]}, {t[2]}],")
    out += ["];", "", "/// The texture coordinate `(u, v)` of every vertex, `0..1` (v down, as the model's atlas image).", "pub const UV: [[f32; 2]; 468] = ["]
    for u, v in uv:
        out.append(f"    [{u:.6f}, {1.0 - v:.6f}],")
    out += ["];", ""]
    out.append(array("FACE_OVAL", "The face oval: 36 landmarks around the face, from the forehead (10) clockwise in the picture.", oval))
    out.append(array("OUTER_LIPS", "The outer contour of the lips.", OUTER_LIPS))
    out.append(array("INNER_LIPS", "The inner contour of the lips: the opening of the mouth.", INNER_LIPS))
    out.append(array("RIGHT_EYE", "The lid contour of the eye on the left of the picture (the person's right eye).", right_eye))
    out.append(array("LEFT_EYE", "The lid contour of the eye on the right of the picture (the person's left eye).", left_eye))
    out.append(array("RIGHT_EYEBROW", "The outline of the eyebrow on the left of the picture.", RIGHT_EYEBROW))
    out.append(array("LEFT_EYEBROW", "The outline of the eyebrow on the right of the picture.", LEFT_EYEBROW))
    out.append(array("RIGHT_IRIS", "The iris contour on the left of the picture (centre: 468).", right_iris))
    out.append(array("LEFT_IRIS", "The iris contour on the right of the picture (centre: 473).", left_iris))
    args.out.write_text("\n".join(out) + "\n")
    print(f"wrote {args.out} ({args.out.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
