import numpy as np
import pytest

from training.jdd import craw
from training.jdd.craw import CrawCurve


def libraw_curve(knots):
    """The curve exactly as LibRaw's loop builds it (independent, naive implementation)."""
    pts = [0, *knots, 4095]
    curve = [0] * 4096
    for i in range(5):
        for j in range(pts[i] + 1, pts[i + 1] + 1):
            curve[j] = curve[j - 1] + (1 << i)
    return np.array(curve)


def test_curve_matches_naive_construction_and_tag_roundtrip():
    c = CrawCurve()
    assert (c.table == libraw_curve(c.knots)).all()
    assert (np.diff(c.table) > 0).all()
    assert c.values[-1] >= 16383  # reaches the 14-bit white
    assert CrawCurve.from_tag_values(c.tag_values()).knots == c.knots
    with pytest.raises(ValueError):
        CrawCurve(knots=(10, 5, 20, 30))


def reference_block_bytes(vmax, vmin, imax, imin, deltas14):
    """Pack one block with plain Python integers: the bit layout of docs/04 (LSB first)."""
    word = vmax | (vmin << 11) | (imax << 22) | (imin << 26)
    for k, d in enumerate(deltas14):
        word |= d << (30 + 7 * k)
    return word.to_bytes(16, "little")


def test_bit_layout_against_independent_packer():
    rng = np.random.default_rng(1)
    # One row of 32 pixels = even block + odd block; craft the blocks directly.
    vmax, vmin, imax, imin = 1500, 1400, 3, 11
    deltas = [int(v) for v in rng.integers(0, 100, size=14)]
    blocks = reference_block_bytes(vmax, vmin, imax, imin, deltas) + reference_block_bytes(1000, 900, 0, 15, deltas)
    parsed = craw.unpack_blocks(blocks, 1, 32)
    assert parsed.vmax.tolist() == [[[1500, 1000]]]
    assert parsed.vmin.tolist() == [[[1400, 900]]]
    assert parsed.imax.tolist() == [[[3, 0]]] and parsed.imin.tolist() == [[[11, 15]]]
    # deltas fill the 14 non-extreme slots in index order
    slots = [i for i in range(16) if i not in (3, 11)]
    assert [int(parsed.delta[0, 0, 0, s]) for s in slots] == deltas
    assert craw.pack_blocks(parsed) == blocks
    # decoding: shift 0 for range 100 (< 128), exact max/min
    dn, q = craw.decode_image(blocks, 1, 32)
    cv = CrawCurve().values
    assert dn[0, 2 * 3] == cv[1500] and dn[0, 2 * 11] == cv[1400]  # even columns hold block 0
    assert dn[0, 2 * slots[0]] == cv[min(2047, deltas[0] + 1400)]
    assert dn.shape == q.shape == (1, 32)


@pytest.mark.parametrize("kind", ["noise", "ramp", "constant", "extremes", "stripes"])
def test_true_value_within_half_qstep(kind):
    rng = np.random.default_rng(7)
    h, w = 48, 128
    if kind == "noise":
        img = np.clip(rng.normal(3000, 2500, (h, w)), 0, 16383)
    elif kind == "ramp":
        img = np.tile(np.linspace(0, 16383, w), (h, 1))
    elif kind == "constant":
        img = np.full((h, w), 777.0)
    elif kind == "extremes":
        img = rng.choice([0.0, 16383.0, 512.0], size=(h, w))
    else:  # the "blondie" case: high-contrast alternating columns
        img = np.where(np.arange(w) % 4 < 2, 600.0, 9000.0)[None, :] + rng.normal(0, 30, (h, w))
    img = np.rint(np.clip(img, 0, 16383)).astype(np.int64)
    dn, q = craw.roundtrip(img)
    err = np.abs(dn - img)
    assert (err <= q / 2 + 1e-9).all(), (err - q / 2).max()
    assert (q >= 1).all()


def test_lossy_and_bytes_path_equals_direct_path():
    rng = np.random.default_rng(2)
    img = np.clip(rng.normal(4000, 3000, (16, 64)), 0, 16383).astype(np.int64)
    a = craw.roundtrip(img, through_bytes=True)
    b = craw.roundtrip(img, through_bytes=False)
    assert (a[0] == b[0]).all() and (a[1] == b[1]).all()
    assert len(craw.encode_image(img)) == 16 * 64  # one byte per pixel
    assert (np.abs(a[0] - img) > 0).any()  # it is lossy


def test_roundtrip_pads_odd_widths():
    img = np.full((4, 40), 2000, dtype=np.int64)
    dn, q = craw.roundtrip(img)
    assert dn.shape == q.shape == (4, 40)
    assert (np.abs(dn - img) <= q / 2).all()


def test_qstep_grows_with_signal():
    """Quantisation is coarser in highlights (curve slope doubles per segment)."""
    rng = np.random.default_rng(3)
    lo = np.clip(rng.normal(700, 20, (32, 64)), 0, 16383).astype(np.int64)
    hi = np.clip(rng.normal(14000, 20, (32, 64)), 0, 16383).astype(np.int64)
    assert craw.roundtrip(hi)[1].mean() > 4 * craw.roundtrip(lo)[1].mean()
