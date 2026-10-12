"""Reading and writing `jdd` weights as safetensors with an architecture header (docs/07 "Parity and versioning").

The safetensors file holds the state dict (float32) and, in ``__metadata__``, everything the Rust CPU
runtime (``crates/ym-ml``) needs to rebuild the network without a separate config file: the
architecture config JSON, the receptive field and the tile halo.
"""

from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path

import numpy as np

from safetensors import safe_open
from safetensors.torch import load_file

from .model import ARCH_NAME, ARCH_VERSION, IN_CHANNELS, JddConfig, JddNet, count_flops, count_parameters, named_weight_tensors
from .receptive_field import analyse


def sha256_file(path: str | Path) -> str:
    """Hex SHA-256 of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_metadata(model: JddNet, model_version: str = "0") -> dict[str, str]:
    """String-to-string header for the safetensors file."""
    rf = analyse(model)
    fl = count_flops(model, 64, 64)
    return {
        "format": ARCH_NAME,
        "arch_version": str(ARCH_VERSION),
        "model_version": model_version,
        "config": model.cfg.to_json(),
        "in_channels": str(IN_CHANNELS),
        "out_channels": "3",
        "rf_half_px": str(rf.rf_half_px),
        "rf_full_px": str(rf.rf_full_px),
        "halo_half_px": str(rf.halo_half_px),
        "halo_full_px": str(rf.halo_full_px),
        "params": str(count_parameters(model)),
        "flops_per_output_pixel": f"{fl['per_output_pixel']:.1f}",
        "tile_multiple_half_px": "4",
    }


def write_safetensors(tensors: dict, metadata: dict[str, str], path: str | Path) -> None:
    """Write a safetensors file with a *deterministic* byte layout (sorted names and metadata keys).

    ``safetensors.torch.save_file`` serialises the metadata from a hash map, so the same weights can give
    different bytes (and SHA-256) in two runs. Weight files carry a SHA-256 (docs/07), so we write the
    documented format ourselves: 8-byte little-endian header length, JSON header padded with spaces to a
    multiple of 8 bytes, then the raw little-endian float32 data in name order.
    """
    header: dict = {"__metadata__": dict(sorted(metadata.items()))}
    chunks: list[bytes] = []
    offset = 0
    for name in sorted(tensors):
        arr = np.ascontiguousarray(tensors[name].detach().cpu().numpy().astype("<f4"))
        raw = arr.tobytes()
        header[name] = {"dtype": "F32", "shape": list(arr.shape), "data_offsets": [offset, offset + len(raw)]}
        chunks.append(raw)
        offset += len(raw)
    blob = json.dumps(header, separators=(",", ":")).encode()
    blob += b" " * ((-len(blob)) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(blob)))
        f.write(blob)
        for c in chunks:
            f.write(c)


def save_weights(model: JddNet, path: str | Path, model_version: str = "0", extra: dict[str, str] | None = None) -> str:
    """Write ``path`` (safetensors) and return its SHA-256."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = build_metadata(model, model_version)
    if extra:
        meta.update({k: str(v) for k, v in extra.items()})
    write_safetensors(named_weight_tensors(model), meta, path)
    return sha256_file(path)


def read_metadata(path: str | Path) -> dict[str, str]:
    """The ``__metadata__`` header of a weights file."""
    with safe_open(str(path), framework="pt") as f:
        return dict(f.metadata() or {})


def load_weights(path: str | Path) -> JddNet:
    """Rebuild the network described by the file's header and load its tensors."""
    meta = read_metadata(path)
    if meta.get("format") != ARCH_NAME:
        raise ValueError(f"{path}: not a {ARCH_NAME} weights file (format={meta.get('format')!r})")
    model = JddNet(JddConfig.from_json(meta["config"]))
    model.load_state_dict(load_file(str(path)))
    return model.eval()


def write_json(path: str | Path, obj: object) -> None:
    """Deterministic JSON file (sorted keys, 2-space indent, trailing newline)."""
    Path(path).write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")

