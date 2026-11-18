//! Motion's fast method (`docs/33-blur.md` 3.5, 11 item 50): the exact line of the oracle, gathered over only the pixels it
//! touches.
//!
//! The oracle's [`super::line_table`] is a square table of `(2 half + 1)^2` weights of which a line touches about
//! `l (|cos a| + |sin a|) + 2`: at most two pixels a column of the dominant axis. This method keeps those weights, **the same
//! numbers in the same order**, and adds them in the oracle's order, so a pixel's sums are the oracle's bit for bit: the fast
//! method is exact, and the bound of `docs/33` 8 (3e-3) is met with room. It costs `2 l` taps a pixel instead of `(l + 2)^2`.
//!
//! The spec's design of cells (the planes reduced to cells of `f` pixels, 17 taps along the line at spacing `f`, bilinear reads)
//! was measured and does not hold the bound: a line has no blur across it, so reducing to cells smears a light or a grain by
//! `f` pixels across the line, and bilinear reads at a fractional offset across the line do the same by a pixel (item 50, and
//! `tests/blur_motion.rs`). The cost is the price of that and is reported in `docs/33` 9.

use super::{Plane4, Table, line_table};

/// One tap of the line: the offset of the pixel it reads from the pixel it makes, and its weight (the exact length of the line
/// inside that pixel, normalised so the weights sum to 1).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Tap {
    /// Offset in columns.
    pub dx: i32,
    /// Offset in rows (down).
    pub dy: i32,
    /// The weight.
    pub w: f64,
}

/// The nonzero weights of the line of `length_px` pixels at `angle_deg`, in the oracle's order (rows from the top, columns from
/// the left): what [`super::line_table`] holds, without its zeros. A line of no length gives the one tap `(0, 0)` of weight 1.
#[must_use]
pub fn taps(length_px: f64, angle_deg: f64) -> Vec<Tap> {
    taps_of(&line_table(length_px, angle_deg))
}

/// The nonzero weights of a table, in its order.
#[must_use]
pub fn taps_of(table: &Table) -> Vec<Tap> {
    let side = (2 * table.half + 1) as usize;
    table
        .weights
        .iter()
        .enumerate()
        .filter(|(_, w)| **w != 0.0)
        .map(|(i, &w)| Tap {
            dx: (i % side) as i32 - table.half as i32,
            dy: (i / side) as i32 - table.half as i32,
            w,
        })
        .collect()
}

/// How far from a pixel the method reads, in pixels: the table's half width, `ceil(l / 2) + 1`. The scheduler's reach
/// ([`super::reach_px`]) is never less.
#[must_use]
pub fn reads_px(length_px: f64) -> usize {
    if length_px <= 1e-6 {
        return 0;
    }
    (length_px / 2.0).ceil() as usize + 1
}

/// The sums of the weighted planes `p` (a window of `w x h` pixels) along the line: for each pixel the taps are added in the
/// oracle's order, taps outside the window skipped. Written tap by tap over rows so the inner loop is a straight run.
#[must_use]
pub fn gather(p: &[Plane4], w: usize, h: usize, length_px: f64, angle_deg: f64) -> Vec<Plane4> {
    let line = taps(length_px, angle_deg);
    let mut out = vec![[0.0; 4]; w * h];
    for t in &line {
        let (dx, dy) = (i64::from(t.dx), i64::from(t.dy));
        let x_lo = (-dx).max(0) as usize;
        let x_hi = (w as i64 - dx).clamp(0, w as i64) as usize;
        if x_lo >= x_hi {
            continue;
        }
        for y in 0..h {
            let yy = y as i64 + dy;
            if yy < 0 || yy >= h as i64 {
                continue;
            }
            let src = &p[yy as usize * w..(yy as usize + 1) * w];
            let dst = &mut out[y * w..(y + 1) * w];
            for x in x_lo..x_hi {
                let q = &src[(x as i64 + dx) as usize];
                let o = &mut dst[x];
                for c in 0..4 {
                    o[c] += t.w * q[c];
                }
            }
        }
    }
    out
}
