#!/usr/bin/env python3
"""Converts Google's MediaPipe face networks from TFLite to safetensors (docs/27 S-08, docs/07).

Development and CI only (AGENTS.md rule 6): the app and `ym` never run Python; every network runs in Rust (`crates/ym-ml`,
module `tflite_graph`). Run by `.github/workflows/models.yml` (job `mediapipe`), which publishes the three outputs as assets of
the release `models-mediapipe-faces-1`.

    python tools/mediapipe/convert.py --out build/mediapipe [--download-dir build/mediapipe/cache] [--only blazeface]

For each model the script
  1. downloads the official file from Google's model bucket and checks its SHA-256 against the pin below (a file with another
     hash is refused, never converted);
  2. for Face Mesh V2, takes `face_landmarks_detector.tflite` out of `face_landmarker.task` (a zip) and checks that file too;
  3. reads the TFLite graph with the `tflite` FlatBuffers bindings and writes one safetensors file holding
       * every constant tensor the operators use, named `t<index>` (index = the TFLite tensor index). The f16 weights of the
         MediaPipe files sit behind DEQUANTIZE operators; the script folds those away and stores the f16 values unchanged, so the
         Rust runtime widens exactly the numbers TFLite does. The float32 weights of Selfie Multiclass are stored as f32;
       * in the header metadata, key `graph`: the operators in execution order with their tensor indices and all their builtin
         options, the integer constants (shapes, permutations, paddings, axes) and the graph's inputs and outputs, as JSON with
         sorted keys; and the keys `source`, `source_sha256`, `licence` and `converted_by`.
The output bytes depend on the input file only (the writer is our own, with sorted keys): the same input gives the same bytes.
  4. writes `SHA256SUMS` (the lines to copy into `data/models/<name>/SHA256SUMS`) and `conversion_report.json`.

The operators are not interpreted here. Rust implements the ones it needs and refuses a graph with another (`MlError::Unsupported`).
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import tflite

BUCKET = "https://storage.googleapis.com/mediapipe-models"
LICENCE = "Apache-2.0 (Google, MediaPipe model card)"

#: name -> what to download and what to write. The hashes are those of the files at the `/1/` URLs on 2026-11-11.
MODELS = {
    "blazeface": {
        "url": f"{BUCKET}/face_detector/blaze_face_full_range/float16/1/blaze_face_full_range.tflite",
        "sha256": "3698b18f063835bc609069ef052228fbe86d9c9a6dc8dcb7c7c2d69aed2b181b",
        "member": None,
        "member_sha256": None,
        "out": "blaze_face_full_range.f16.safetensors",
    },
    "face_mesh": {
        "url": f"{BUCKET}/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
        "sha256": "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff",
        "member": "face_landmarks_detector.tflite",
        "member_sha256": "c7d54204ce0448474c7f3fa9af494787c0965cbdd6f20fc72867e43046bd43d5",
        "out": "face_mesh_v2.f16.safetensors",
    },
    "selfie_multiclass": {
        "url": f"{BUCKET}/image_segmenter/selfie_multiclass_256x256/float32/1/selfie_multiclass_256x256.tflite",
        "sha256": "c6748b1253a99067ef71f7e26ca71096cd449baefa8f101900ea23016507e0e0",
        "member": None,
        "member_sha256": None,
        "out": "selfie_multiclass_256.f32.safetensors",
    },
}

OP_NAMES = {v: k for k, v in tflite.BuiltinOperator.__dict__.items() if not k.startswith("_") and isinstance(v, int)}
OPTION_TYPES = {v: k for k, v in tflite.BuiltinOptions.__dict__.items() if not k.startswith("_") and isinstance(v, int)}
TENSOR_DTYPES = {
    tflite.TensorType.FLOAT32: np.dtype("<f4"),
    tflite.TensorType.FLOAT16: np.dtype("<f2"),
    tflite.TensorType.INT32: np.dtype("<i4"),
}
#: Methods of a FlatBuffers table class that are not fields.
NOT_FIELDS = {"Init"}


SAFETENSORS_DTYPES = {np.dtype("<f2"): "F16", np.dtype("<f4"): "F32"}


def write_safetensors(tensors: dict[str, np.ndarray], metadata: dict[str, str]) -> bytes:
    """A safetensors file with the tensors in name order and the header keys sorted, so the bytes are always the same.

    The `safetensors` package keeps its metadata in a hash map, whose order changes from run to run; the format is simple enough
    to write directly (8 bytes of header length, the JSON header padded with spaces to a multiple of 8, the raw data).
    """
    header: dict = {"__metadata__": dict(sorted(metadata.items()))}
    offset = 0
    body = []
    for name in sorted(tensors):
        array = tensors[name]
        raw = np.ascontiguousarray(array).astype(array.dtype.newbyteorder("<"), copy=False).tobytes()
        header[name] = {
            "dtype": SAFETENSORS_DTYPES[array.dtype],
            "shape": [int(d) for d in array.shape],
            "data_offsets": [offset, offset + len(raw)],
        }
        offset += len(raw)
        body.append(raw)
    text = json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
    text += b" " * (-len(text) % 8)
    return len(text).to_bytes(8, "little") + text + b"".join(body)


def sha256_of(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def download(url: str, want: str, cache: Path | None) -> bytes:
    """The file at `url`, from `cache` when it is there with the right hash; refused unless its SHA-256 is `want`."""
    name = url.rsplit("/", 1)[-1]
    if cache is not None and (cache / name).is_file():
        data = (cache / name).read_bytes()
        if sha256_of(data) == want:
            return data
    request = urllib.request.Request(url, headers={"User-Agent": "yellowmoon-model-conversion"})
    with urllib.request.urlopen(request, timeout=300) as response:  # noqa: S310 (a fixed https URL)
        data = response.read()
    got = sha256_of(data)
    if got != want:
        raise SystemExit(f"{url}: SHA-256 {got}, the pin is {want}; nothing was converted")
    if cache is not None:
        cache.mkdir(parents=True, exist_ok=True)
        (cache / name).write_bytes(data)
    return data


def table_fields(table) -> dict:
    """All scalar fields and vectors of a FlatBuffers options table, found by reflection on the generated class."""
    out = {}
    cls = type(table)
    for name in sorted(dir(cls)):
        if name.startswith("_") or name in NOT_FIELDS or name.startswith("GetRootAs"):
            continue
        if name.endswith(("AsNumpy", "IsNone")):
            continue
        if name.endswith("Length") and name != "Length":
            base = name[: -len("Length")]
            if hasattr(cls, base):
                out[base] = [getattr(table, base)(j) for j in range(getattr(table, name)())]
            continue
        if name in out:
            continue
        value = getattr(table, name)
        if not callable(value):
            continue
        try:
            result = value()
        except TypeError:
            continue  # a vector accessor that needs an index
        if isinstance(result, (bool, int, float, np.integer, np.floating)):
            out[name] = result.item() if hasattr(result, "item") else result
    return out


def options_of(op) -> dict:
    kind = op.BuiltinOptionsType()
    if kind == 0:
        return {}
    cls = getattr(tflite, OPTION_TYPES[kind])
    table = cls()
    buf = op.BuiltinOptions()
    table.Init(buf.Bytes, buf.Pos)
    # Vector fields first (their `Length` twin names them), then scalars; vectors are lists of ints.
    fields = table_fields(table)
    return {k: v for k, v in sorted(fields.items())}


def constant(model, graph, index: int):
    """The numpy array of a constant tensor, or None when the tensor is computed."""
    tensor = graph.Tensors(index)
    buffer = model.Buffers(tensor.Buffer())
    if buffer is None or buffer.DataLength() == 0:
        return None
    dtype = TENSOR_DTYPES.get(tensor.Type())
    if dtype is None:
        raise SystemExit(f"tensor {index}: unsupported constant type {tensor.Type()}")
    raw = buffer.DataAsNumpy().tobytes()
    shape = [int(d) for d in tensor.ShapeAsNumpy()] if tensor.ShapeLength() else []
    array = np.frombuffer(raw, dtype=dtype)
    if int(np.prod(shape, dtype=np.int64)) != array.size:
        raise SystemExit(f"tensor {index}: {array.size} values for shape {shape}")
    return array.reshape(shape)


def convert(model_bytes: bytes, name: str, source: str, source_sha: str) -> tuple[bytes, dict]:
    model = tflite.Model.GetRootAsModel(model_bytes, 0)
    if model.SubgraphsLength() != 1:
        raise SystemExit(f"{name}: {model.SubgraphsLength()} subgraphs, one expected")
    graph = model.Subgraphs(0)

    weights: dict[str, np.ndarray] = {}  # t<index> -> array (f16 or f32)
    ints: dict[str, dict] = {}  # index -> {"shape", "data"}
    ops = []
    folded = 0
    for i in range(graph.OperatorsLength()):
        op = graph.Operators(i)
        code = model.OperatorCodes(op.OpcodeIndex())
        kind = OP_NAMES[max(code.BuiltinCode(), code.DeprecatedBuiltinCode())]
        ins = [int(op.Inputs(j)) for j in range(op.InputsLength())]
        outs = [int(op.Outputs(j)) for j in range(op.OutputsLength())]
        if kind == "DEQUANTIZE":
            # A constant f16 weight widened to f32: keep the f16 values under the f32 tensor's index.
            source_tensor = constant(model, graph, ins[0])
            if source_tensor is None or source_tensor.dtype != np.dtype("<f2"):
                raise SystemExit(f"{name}: DEQUANTIZE of something that is not a constant f16 tensor (operator {i})")
            weights[f"t{outs[0]}"] = source_tensor
            folded += 1
            continue
        for index in ins:
            if index < 0 or f"t{index}" in weights or str(index) in ints:
                continue
            array = constant(model, graph, index)
            if array is None:
                continue
            if array.dtype == np.dtype("<i4"):
                ints[str(index)] = {"shape": list(array.shape), "data": [int(v) for v in array.reshape(-1)]}
            else:
                weights[f"t{index}"] = array
        ops.append({"op": kind, "in": ins, "out": outs, "opt": options_of(op)})

    tensors = {}
    for tname, array in weights.items():
        if not np.all(np.isfinite(array.astype(np.float32))):
            raise SystemExit(f"{name}: {tname} holds a value that is not finite")
        tensors[tname] = np.ascontiguousarray(array)

    def io(indices):
        return [
            {
                "index": int(i),
                "name": graph.Tensors(i).Name().decode(),
                "shape": [int(d) for d in graph.Tensors(i).ShapeAsNumpy()],
            }
            for i in indices
        ]

    description = {
        "version": 1,
        "inputs": io([graph.Inputs(j) for j in range(graph.InputsLength())]),
        "outputs": io([graph.Outputs(j) for j in range(graph.OutputsLength())]),
        "ints": ints,
        "ops": ops,
    }
    metadata = {
        "format": "yellowmoon-tflite-graph-1",
        "model": name,
        "source": source,
        "source_sha256": source_sha,
        "licence": LICENCE,
        "converted_by": "tools/mediapipe/convert.py",
        "graph": json.dumps(description, sort_keys=True, separators=(",", ":")),
    }
    blob = write_safetensors(tensors, metadata)
    counts: dict[str, int] = {}
    for o in ops:
        counts[o["op"]] = counts.get(o["op"], 0) + 1
    report = {
        "operators": counts,
        "dequantize_folded": folded,
        "constant_tensors": len(tensors),
        "int_constants": len(ints),
        "parameters": int(sum(a.size for a in tensors.values())),
        "bytes": len(blob),
        "inputs": description["inputs"],
        "outputs": description["outputs"],
    }
    return blob, report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, required=True, help="directory for the .safetensors files, SHA256SUMS and the report")
    parser.add_argument("--download-dir", type=Path, default=None, help="cache for the downloaded originals")
    parser.add_argument("--only", choices=sorted(MODELS), action="append", help="convert only this model (repeatable)")
    args = parser.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    sums = []
    reports = {}
    for name in args.only or sorted(MODELS):
        spec = MODELS[name]
        data = download(spec["url"], spec["sha256"], args.download_dir)
        source_sha = spec["sha256"]
        if spec["member"]:
            with zipfile.ZipFile(io.BytesIO(data)) as z:
                data = z.read(spec["member"])
            source_sha = sha256_of(data)
            if source_sha != spec["member_sha256"]:
                raise SystemExit(f"{spec['member']}: SHA-256 {source_sha}, the pin is {spec['member_sha256']}")
        blob, report = convert(data, name, spec["url"] + (f"#{spec['member']}" if spec["member"] else ""), source_sha)
        (args.out / spec["out"]).write_bytes(blob)
        sums.append(f"{sha256_of(blob)}  {spec['out']}")
        report["source_sha256"] = source_sha
        reports[name] = report
        print(f"{name}: {spec['out']} {len(blob)} bytes, {report['parameters']} parameters, operators {report['operators']}")
    (args.out / "SHA256SUMS").write_text("\n".join(sums) + "\n")
    (args.out / "conversion_report.json").write_text(json.dumps(reports, indent=2, sort_keys=True) + "\n")
    print("\n".join(sums))
    return 0


if __name__ == "__main__":
    sys.exit(main())
