"""Python port of the Sony cRAW block codec of ``docs/04-raw-decoding.md`` (encoder and decoder).

Training-time simulation only (docs/07, sample generation step 5): a 14-bit mosaic goes through the
*lossy* cRAW encoder and decoder, which produces the same stripe errors real cRAW files have
("blondie" artefacts) and the per-pixel quantisation step ``qstep`` the denoise network is given.
The Rust test encoder (M2-03) is an independent implementation of the same text.

Format (docs/04):

* A row of ``W`` pixels (``W`` a multiple of 32) is stored as ``W`` bytes. Every 32 pixels form two
  interleaved 16-pixel blocks: first the even columns, then the odd columns (same CFA colour).
* A block is 128 bits, least-significant bit first: 11-bit ``max``, 11-bit ``min``, 4-bit index of
  the max pixel, 4-bit index of the min pixel, then 14 x 7-bit deltas for the other pixels in index
  order. Decoded value: ``min(2047, (delta << shift) + min)`` with
  ``shift = max(0, bitlen(max - min) - 7)``.
* The 11-bit value goes through the camera curve (tag 0x7010, four knots, slope doubling 1, 2, 4, 8,
  16), indexed with ``value << 1``; the output is on the 14-bit scale.

Quantisation step (``qstep``, in output DN): the width of the interval that must contain the true
value. Exact pixels (max and min) are known to +-half a value step through the curve; delta pixels
to +-half of ``2**shift`` value steps (the top cell, ``delta == 127``, is wider and is handled
exactly). Every true value is within ``+-qstep/2`` of the decoded value (tested).

The encoder is not Sony's: it picks, for every delta, the representable value nearest to the true
14-bit value in the DN domain. Real Sony encoders may round differently; the error statistics are
what matters for training.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

BLOCK_PIXELS = 16
BLOCK_BITS = 128
GROUP_PIXELS = 32
VALUE_MAX = 2047
DEFAULT_KNOTS = (1500, 2400, 3100, 3800)
# Black level of the 14-bit scale used by the sampler; the curve is dense (slope 1) below knot 1.


@dataclass(frozen=True)
class CrawCurve:
    """The four-knot tone curve of tag 0x7010 and the table it expands to."""

    knots: tuple[int, int, int, int] = DEFAULT_KNOTS

    def __post_init__(self) -> None:
        k = self.knots
        if not (0 < k[0] < k[1] < k[2] < k[3] < 4095):
            raise ValueError(f"knots must be strictly increasing within (0, 4095): {k}")

    @property
    def table(self) -> np.ndarray:
        """``curve[0..4095]`` exactly as LibRaw builds it (``curve[j] = curve[j-1] + (1 << segment)``)."""
        pts = (0, *self.knots, 4095)
        curve = np.zeros(4096, dtype=np.int64)
        for seg in range(5):
            lo, hi = pts[seg], pts[seg + 1]
            curve[lo + 1 : hi + 1] = curve[lo] + (np.arange(1, hi - lo + 1) << seg)
        return curve

    @property
    def values(self) -> np.ndarray:
        """DN (14-bit scale) of every 11-bit value: ``curve[value << 1]``, 2048 entries."""
        return self.table[0:4096:2].copy()

    def tag_values(self) -> tuple[int, int, int, int]:
        """The four u16 values stored in tag 0x7010 (``(v >> 2) & 0xFFF`` is the knot)."""
        return tuple(int(k) << 2 for k in self.knots)  # type: ignore[return-value]

    @classmethod
    def from_tag_values(cls, values: tuple[int, int, int, int]) -> CrawCurve:
        """Build the curve from raw tag values."""
        return cls(tuple((int(v) >> 2) & 0xFFF for v in values))  # type: ignore[arg-type]


def _check_width(width: int) -> None:
    if width % GROUP_PIXELS:
        raise ValueError(f"cRAW width must be a multiple of {GROUP_PIXELS}, got {width}")


def _to_blocks(img: np.ndarray) -> np.ndarray:
    """``[H, W]`` -> ``[H, W/32, 2, 16]``: per row, per group, (even-column block, odd-column block)."""
    h, w = img.shape
    g = img.reshape(h, w // GROUP_PIXELS, BLOCK_PIXELS, 2)  # [..., pair index, parity]
    return np.ascontiguousarray(np.moveaxis(g, 3, 2))


def _from_blocks(blocks: np.ndarray) -> np.ndarray:
    h, groups = blocks.shape[0], blocks.shape[1]
    g = np.moveaxis(blocks, 2, 3)  # [H, G, 16, 2]
    return np.ascontiguousarray(g.reshape(h, groups * GROUP_PIXELS))


def _shift_for_range(rng_: np.ndarray) -> np.ndarray:
    """``shift = max(0, bitlen(max - min) - 7)``, equal to LibRaw's ``while (sh < 4 and 0x80 << sh <= max - min)``."""
    sh = np.zeros(rng_.shape, dtype=np.int64)
    for s in range(4):
        sh += (0x80 << s) <= rng_
    return sh


@dataclass
class CrawBlocks:
    """Parsed fields of all blocks: arrays of shape ``[H, G, 2]`` (+ ``[..., 16]`` for the pixels)."""

    vmax: np.ndarray
    vmin: np.ndarray
    imax: np.ndarray
    imin: np.ndarray
    delta: np.ndarray  # [H, G, 2, 16]; entries at imax / imin are unused (0)


def encode_blocks(dn: np.ndarray, curve: CrawCurve) -> CrawBlocks:
    """Quantise a 14-bit-scale ``[H, W]`` image into block fields (no bit packing)."""
    dn = np.asarray(dn)
    _check_width(dn.shape[1])
    cv = curve.values.astype(np.float64)  # monotone increasing, length 2048
    dn_f = np.clip(dn.astype(np.float64), cv[0], cv[-1])
    # Continuous position on the 11-bit axis (piecewise-linear inverse of the curve).
    pos = np.interp(dn_f, cv, np.arange(2048, dtype=np.float64))
    v = np.rint(pos).astype(np.int64)
    vb = _to_blocks(v)  # [H, G, 2, 16]
    pb = _to_blocks(pos)
    db = _to_blocks(dn_f)
    vmax = vb.max(axis=-1)
    vmin = vb.min(axis=-1)
    imax = vb.argmax(axis=-1)
    imin = vb.argmin(axis=-1)
    const = imax == imin  # constant block: use two distinct indices
    imin = np.where(const, (imax + 1) % BLOCK_PIXELS, imin)
    sh = _shift_for_range(vmax - vmin)
    step = (1 << sh)[..., None]  # [H, G, 2, 1]
    base = ((pb - vmin[..., None]) / step)
    d0 = np.clip(np.floor(base), 0, 127).astype(np.int64)
    d1 = np.clip(d0 + 1, 0, 127)

    def dn_of(d: np.ndarray) -> np.ndarray:
        val = np.minimum(VALUE_MAX, (d << sh[..., None]) + vmin[..., None])
        return cv[val]

    e0 = np.abs(dn_of(d0) - db)
    e1 = np.abs(dn_of(d1) - db)
    delta = np.where(e1 < e0, d1, d0)
    idx = np.arange(BLOCK_PIXELS)
    is_exact = (idx == imax[..., None]) | (idx == imin[..., None])
    delta = np.where(is_exact, 0, delta)
    return CrawBlocks(vmax, vmin, imax, imin, delta)


def decode_blocks(blocks: CrawBlocks, curve: CrawCurve) -> tuple[np.ndarray, np.ndarray]:
    """Block fields -> ``(dn, qstep)`` as ``[H, W]`` int64 / float64 arrays (14-bit scale)."""
    cv = curve.values.astype(np.float64)
    vmax, vmin = blocks.vmax, blocks.vmin
    sh = _shift_for_range(vmax - vmin)
    idx = np.arange(BLOCK_PIXELS)
    val = np.minimum(VALUE_MAX, (blocks.delta << sh[..., None]) + vmin[..., None])
    val = np.where(idx == blocks.imax[..., None], vmax[..., None], val)
    val = np.where(idx == blocks.imin[..., None], vmin[..., None], val)
    dn = cv[val]
    # qstep: width (in DN) of the cell the true value was rounded from: the larger of the DN gaps to the
    # two neighbouring representable values (the curve is convex, so the gap is not symmetric). Exact
    # pixels are known to the nearest 11-bit value; the top cell (delta 127) ends at ``max`` instead.
    axis = np.arange(2048, dtype=np.float64)

    def c_at(x: np.ndarray) -> np.ndarray:
        return np.interp(np.clip(x, 0.0, 2047.0), axis, cv)

    exact = (idx == blocks.imax[..., None]) | (idx == blocks.imin[..., None])
    step = np.where(exact, 1.0, (1 << sh)[..., None].astype(np.float64))
    up = c_at(val + step) - cv[val]
    down = cv[val] - c_at(val - step)
    top = (blocks.delta == 127) & ~exact
    top_w = ((vmax - vmin)[..., None] - (127 << sh[..., None])).astype(np.float64) + 0.5  # + inverse-curve rounding
    up = np.where(top, 2.0 * (c_at(val + top_w) - cv[val]), up)
    qstep = np.maximum(up, down)
    return _from_blocks(dn), _from_blocks(qstep)


def pack_blocks(blocks: CrawBlocks) -> bytes:
    """Serialise block fields to the on-disk byte layout (``H * W`` bytes, row after row)."""
    h, g = blocks.vmax.shape[:2]
    n = h * g * 2
    fields = np.zeros((n, BLOCK_BITS), dtype=np.uint8)

    def put(arr: np.ndarray, nbits: int, off: int) -> None:
        bits = (arr.reshape(n)[:, None] >> np.arange(nbits)) & 1
        fields[:, off : off + nbits] = bits

    put(blocks.vmax, 11, 0)
    put(blocks.vmin, 11, 11)
    put(blocks.imax, 4, 22)
    put(blocks.imin, 4, 26)
    delta = blocks.delta.reshape(n, BLOCK_PIXELS)
    imax = blocks.imax.reshape(n)
    imin = blocks.imin.reshape(n)
    pix = np.arange(BLOCK_PIXELS)
    keep = (pix[None, :] != imax[:, None]) & (pix[None, :] != imin[:, None])  # [n, 16], 14 True per row
    ordered = delta[keep].reshape(n, 14)
    bits = (ordered[:, :, None] >> np.arange(7)) & 1  # [n, 14, 7]
    fields[:, 30:128] = bits.reshape(n, 98)
    packed = np.packbits(fields, axis=1, bitorder="little")  # [n, 16]
    return packed.tobytes()


def unpack_blocks(data: bytes, height: int, width: int) -> CrawBlocks:
    """Inverse of :func:`pack_blocks`."""
    _check_width(width)
    g = width // GROUP_PIXELS
    n = height * g * 2
    raw = np.frombuffer(data, dtype=np.uint8)
    if raw.size != height * width:
        raise ValueError(f"expected {height * width} bytes, got {raw.size}")
    bits = np.unpackbits(raw.reshape(n, 16), axis=1, bitorder="little").astype(np.int64)

    def get(off: int, nbits: int) -> np.ndarray:
        return (bits[:, off : off + nbits] << np.arange(nbits)).sum(axis=1)

    vmax = get(0, 11).reshape(height, g, 2)
    vmin = get(11, 11).reshape(height, g, 2)
    imax = get(22, 4).reshape(height, g, 2)
    imin = get(26, 4).reshape(height, g, 2)
    d14 = (bits[:, 30:128].reshape(n, 14, 7) << np.arange(7)).sum(axis=2)  # [n, 14]
    pix = np.arange(BLOCK_PIXELS)
    keep = (pix[None, :] != imax.reshape(n)[:, None]) & (pix[None, :] != imin.reshape(n)[:, None])
    delta = np.zeros((n, BLOCK_PIXELS), dtype=np.int64)
    delta[keep] = d14.reshape(-1)
    return CrawBlocks(vmax, vmin, imax, imin, delta.reshape(height, g, 2, BLOCK_PIXELS))


def encode_image(dn: np.ndarray, curve: CrawCurve | None = None) -> bytes:
    """Encode a ``[H, W]`` 14-bit-scale image to cRAW bytes."""
    return pack_blocks(encode_blocks(dn, curve or CrawCurve()))


def decode_image(data: bytes, height: int, width: int, curve: CrawCurve | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Decode cRAW bytes to ``(dn, qstep)`` ``[H, W]`` arrays on the 14-bit scale."""
    return decode_blocks(unpack_blocks(data, height, width), curve or CrawCurve())


def roundtrip(dn: np.ndarray, curve: CrawCurve | None = None, *, through_bytes: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Encode then decode ``dn`` (any width; padded to a multiple of 32 by edge replication).

    Returns ``(decoded_dn, qstep)`` cropped to the input shape. ``through_bytes=False`` skips the bit
    packing (identical result, a little faster); the tests check both paths agree.
    """
    curve = curve or CrawCurve()
    h, w = dn.shape
    pad = (-w) % GROUP_PIXELS
    src = np.pad(dn, ((0, 0), (0, pad)), mode="edge") if pad else dn
    blocks = encode_blocks(src, curve)
    if through_bytes:
        blocks = unpack_blocks(pack_blocks(blocks), h, src.shape[1])
    out, q = decode_blocks(blocks, curve)
    return out[:, :w], q[:, :w]
