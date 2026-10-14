//! Bit-level comparisons for the kernel tests: tile = whole frame, device = host wrapper, poisoned
//! run = plain run (`docs/gpu-plan.md` D5, D7 and X-10).
//!
//! A value is compared by its bits, so `NaN == NaN` when the bits agree and `-0.0 != 0.0`: the claim of
//! these tests is "the same bytes", not "close".

#![allow(dead_code)]

use std::sync::OnceLock;

use ym_gpu::cpu::tiling::{Rect, tile_grid};
use ym_gpu::device::{Gpu, StoragePreference};

/// The index and the two values of the first sample whose bits differ, if any. Different lengths
/// count as a difference at the end of the shorter one.
pub fn first_difference(a: &[f32], b: &[f32]) -> Option<(usize, f32, f32)> {
    if let Some(i) = (0..a.len().min(b.len())).find(|&i| a[i].to_bits() != b[i].to_bits()) {
        return Some((i, a[i], b[i]));
    }
    (a.len() != b.len()).then(|| (a.len().min(b.len()), f32::NAN, f32::NAN))
}

/// The largest absolute difference; infinity when either side has a NaN where the other has not.
pub fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "lengths differ");
    a.iter().zip(b).fold(0.0f32, |m, (&x, &y)| {
        let d = if x.is_nan() && y.is_nan() {
            0.0
        } else if x.is_nan() || y.is_nan() {
            f32::INFINITY
        } else {
            (x - y).abs()
        };
        m.max(d)
    })
}

/// Panics unless `a` and `b` are the same bits, naming the first sample that differs.
pub fn assert_bit_equal(what: &str, a: &[f32], b: &[f32]) {
    if let Some((i, x, y)) = first_difference(a, b) {
        let diffs = a.iter().zip(b).filter(|(x, y)| x.to_bits() != y.to_bits()).count();
        panic!(
            "{what}: not bit-identical; first difference at sample {i}: {x:e} ({:#010x}) against {y:e} ({:#010x}); {diffs} of {} samples differ, max |diff| {:e}",
            x.to_bits(),
            y.to_bits(),
            a.len().min(b.len()),
            max_abs_diff(&a[..a.len().min(b.len())], &b[..a.len().min(b.len())]),
        );
    }
}

/// Panics when any sample is NaN or infinite: what a stale read of a poisoned buffer looks like.
pub fn assert_all_finite(what: &str, data: &[f32]) {
    if let Some(i) = data.iter().position(|v| !v.is_finite()) {
        panic!(
            "{what}: sample {i} of {} is {:e} (a NaN here is a read of a buffer nobody wrote)",
            data.len(),
            data[i]
        );
    }
}

/// Pastes `tile` (planar, `planes` planes of `rect.w x rect.h`) into `frame` (planar, `planes`
/// planes of `width x height`) at `rect`.
pub fn paste(frame: &mut [f32], width: usize, height: usize, planes: usize, rect: Rect, tile: &[f32]) {
    assert_eq!(tile.len(), planes * rect.w * rect.h, "tile {rect:?} has the wrong length");
    for c in 0..planes {
        for y in 0..rect.h {
            let dst = c * width * height + (rect.y + y) * width + rect.x;
            let src = c * rect.w * rect.h + y * rect.w;
            frame[dst..dst + rect.w].copy_from_slice(&tile[src..src + rect.w]);
        }
    }
}

/// D5: renders the `width x height` frame whole, then as tiles of each size in `tile_sizes`
/// (cores on the even grid of `tile_grid`), and requires every tiled frame to equal the whole one
/// bit for bit. `render(rect)` returns the planar output (`planes` planes) of exactly `rect`; the
/// whole frame is `render(Rect::whole(width, height))`. Returns the whole frame so the caller can
/// compare it with a reference.
pub fn assert_tiles_equal_whole(
    what: &str,
    width: usize,
    height: usize,
    planes: usize,
    tile_sizes: &[usize],
    render: impl Fn(Rect) -> Vec<f32>,
) -> Vec<f32> {
    let whole = render(Rect::whole(width, height));
    assert_eq!(whole.len(), planes * width * height, "{what}: whole frame length");
    for &tile_px in tile_sizes {
        let mut tiled = vec![f32::NAN; whole.len()];
        for rect in tile_grid(width, height, tile_px, tile_px, 2) {
            let tile = render(rect);
            paste(&mut tiled, width, height, planes, rect, &tile);
        }
        assert_bit_equal(&format!("{what}: tiles of {tile_px} px against the whole frame"), &tiled, &whole);
    }
    whole
}

/// A device without poison mode, shared by the tests that compare a poisoned run with a plain one.
pub fn plain_gpu() -> Option<&'static Gpu> {
    static PLAIN: OnceLock<Option<Gpu>> = OnceLock::new();
    PLAIN
        .get_or_init(|| super::start_unpoisoned(StoragePreference::Auto))
        .as_ref()
}

/// X-10 / D7: runs `run` on the poisoned test device (every buffer starts as NaN) and on a plain one
/// (every new buffer starts as zeros), requires the same bits from both and no NaN or infinity in the
/// result. Returns the result. Skips (returning `None`) when there is no adapter.
pub fn assert_poison_neutral(what: &str, run: impl Fn(&Gpu) -> Vec<f32>) -> Option<Vec<f32>> {
    let poisoned = super::gpu()?;
    let plain = plain_gpu()?;
    let a = run(&poisoned.gpu);
    let b = run(plain);
    assert_bit_equal(&format!("{what}: poisoned device against plain device"), &a, &b);
    assert_all_finite(&format!("{what} (poisoned device)"), &a);
    Some(a)
}
