//! The deterministic area average that reduces a reference to its scene and analysis pictures (`docs/34-reference-grade.md`
//! section 6.5, RG-05).
//!
//! An output pixel is the mean of the source over the rectangle it covers: a box filter whose weights are the exact fractions of
//! each source pixel that falls inside the rectangle. A picture is never enlarged. The weights are integers (the overlap of two
//! intervals whose ends are multiples of `1 / (source * output)`), so they carry no rounding; the sums are `f64`, taken in a fixed
//! order (source rows top to bottom, source columns left to right). Every output row is computed on its own from the source, so the
//! result is the same bits at every thread count and on every run.

/// The size of a picture of `width x height` reduced so that its long edge is at most `max_long_edge` px (never enlarged); the
/// short edge keeps the proportions, rounded, and is at least 1.
#[must_use]
pub fn reduced_size(width: usize, height: usize, max_long_edge: usize) -> (usize, usize) {
    let long = width.max(height);
    if long <= max_long_edge || long == 0 {
        return (width, height);
    }
    let scale = |side: usize| -> usize { ((side * max_long_edge + long / 2) / long).max(1) };
    if width >= height {
        (max_long_edge, scale(height))
    } else {
        (scale(width), max_long_edge)
    }
}

/// For each of `out` output cells over `src` source cells: the first source cell and the integer weights (the overlap of the source
/// cell with the output cell, in units of `1 / out` of a source cell). The weights of a cell sum to `src`.
fn spans(src: usize, out: usize) -> Vec<(usize, Vec<u64>)> {
    let (src_u, out_u) = (src as u64, out as u64);
    (0..out)
        .map(|i| {
            // The output cell is [i * src, (i + 1) * src) in units of 1 / out of a source cell.
            let lo = i as u64 * src_u;
            let hi = lo + src_u;
            let first = (lo / out_u) as usize;
            let mut weights = Vec::new();
            let mut j = first;
            loop {
                let a = (j as u64 * out_u).max(lo);
                let b = ((j as u64 + 1) * out_u).min(hi);
                if a >= b {
                    break;
                }
                weights.push(b - a);
                j += 1;
                if j >= src {
                    break;
                }
            }
            (first, weights)
        })
        .collect()
}

/// Reduces a picture of `src_w x src_h` to `out_w x out_h` (neither larger than the source) by the area average. `pixel(x, y)` gives
/// the source pixel (linear sRGB, three channels); it is called from several threads. The result is `out_w * out_h` pixels of three
/// `f64`, rows top to bottom.
///
/// When the sizes are equal the pixels are returned as they are.
///
/// # Panics
///
/// When an output side is 0 or larger than the source's.
#[must_use]
pub fn area_average<F>(
    src_w: usize,
    src_h: usize,
    out_w: usize,
    out_h: usize,
    pixel: F,
) -> Vec<[f64; 3]>
where
    F: Fn(usize, usize) -> [f64; 3] + Sync,
{
    assert!(out_w >= 1 && out_h >= 1 && out_w <= src_w && out_h <= src_h);
    let mut out = vec![[0.0f64; 3]; out_w * out_h];
    if (out_w, out_h) == (src_w, src_h) {
        for y in 0..src_h {
            for x in 0..src_w {
                out[y * out_w + x] = pixel(x, y);
            }
        }
        return out;
    }
    let cols = spans(src_w, out_w);
    let rows = spans(src_h, out_h);
    let (inv_w, inv_h) = (1.0 / src_w as f64, 1.0 / src_h as f64);
    // One output row: for each source row it covers, the horizontal mean of that row, weighted and added in order.
    let one_row = |oy: usize, dst: &mut [[f64; 3]]| {
        let (first_row, row_weights) = &rows[oy];
        let mut line = vec![[0.0f64; 3]; out_w];
        for (k, &wy) in row_weights.iter().enumerate() {
            let sy = first_row + k;
            for (ox, (first_col, col_weights)) in cols.iter().enumerate() {
                let mut acc = [0.0f64; 3];
                for (i, &wx) in col_weights.iter().enumerate() {
                    let p = pixel(first_col + i, sy);
                    let w = wx as f64;
                    acc[0] += w * p[0];
                    acc[1] += w * p[1];
                    acc[2] += w * p[2];
                }
                let w = wy as f64;
                for c in 0..3 {
                    line[ox][c] += w * (acc[c] * inv_w);
                }
            }
        }
        for (d, l) in dst.iter_mut().zip(&line) {
            *d = [l[0] * inv_h, l[1] * inv_h, l[2] * inv_h];
        }
    };
    let threads = std::thread::available_parallelism()
        .map_or(1, usize::from)
        .min(out_h);
    if threads <= 1 {
        for (oy, dst) in out.chunks_mut(out_w).enumerate() {
            one_row(oy, dst);
        }
    } else {
        let rows_per = out_h.div_ceil(threads);
        std::thread::scope(|scope| {
            for (t, block) in out.chunks_mut(rows_per * out_w).enumerate() {
                let one_row = &one_row;
                scope.spawn(move || {
                    for (i, dst) in block.chunks_mut(out_w).enumerate() {
                        one_row(t * rows_per + i, dst);
                    }
                });
            }
        });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sizes_keep_the_proportions_and_never_enlarge() {
        assert_eq!(reduced_size(4000, 3000, 1024), (1024, 768));
        assert_eq!(reduced_size(3000, 4000, 1024), (768, 1024));
        assert_eq!(reduced_size(1024, 1024, 1024), (1024, 1024));
        assert_eq!(reduced_size(400, 300, 1024), (400, 300));
        assert_eq!(reduced_size(10_000, 3, 512), (512, 1));
        assert_eq!(reduced_size(1920, 1080, 512), (512, 288));
    }

    #[test]
    fn the_weights_of_a_cell_sum_to_the_source_size_and_cover_it_exactly() {
        for (src, out) in [
            (10, 3),
            (7, 7),
            (1000, 512),
            (513, 512),
            (5, 1),
            (4096, 1024),
        ] {
            let s = spans(src, out);
            let mut covered = vec![0u64; src];
            for (first, weights) in &s {
                assert_eq!(weights.iter().sum::<u64>(), src as u64, "{src}->{out}");
                for (k, w) in weights.iter().enumerate() {
                    covered[first + k] += w;
                }
            }
            // Every source cell is given out in full, once, over all the output cells.
            assert!(covered.iter().all(|&c| c == out as u64), "{src}->{out}");
        }
    }

    #[test]
    fn a_ramp_averages_to_the_midpoints_and_a_constant_stays_constant() {
        // 10 -> 5: the mean of pairs.
        let out = area_average(10, 1, 5, 1, |x, _| [x as f64, 2.0 * x as f64, 7.0]);
        for (i, p) in out.iter().enumerate() {
            let mid = 2.0 * i as f64 + 0.5;
            assert!((p[0] - mid).abs() < 1e-12);
            assert!((p[1] - 2.0 * mid).abs() < 1e-12);
            assert!((p[2] - 7.0).abs() < 1e-12);
        }
        // 3 -> 2: fractions 2/3 and 1/3.
        let out = area_average(3, 1, 2, 1, |x, _| [[3.0, 6.0, 9.0][x], 0.0, 0.0]);
        // The first output covers 1.5 source cells: (3 + 0.5 * 6) / 1.5 = 4; the second (0.5 * 6 + 9) / 1.5 = 8.
        assert!((out[0][0] - 4.0).abs() < 1e-12, "{out:?}");
        assert!((out[1][0] - 8.0).abs() < 1e-12, "{out:?}");
        // A constant picture of any size is its own mean.
        let out = area_average(1013, 777, 333, 217, |_, _| [0.25, 0.5, 0.75]);
        for p in out {
            assert!(
                (p[0] - 0.25).abs() < 1e-12
                    && (p[1] - 0.5).abs() < 1e-12
                    && (p[2] - 0.75).abs() < 1e-12
            );
        }
    }

    #[test]
    fn the_sum_of_the_picture_is_kept() {
        // The mean over the whole picture is the same before and after (up to rounding).
        let f = |x: usize, y: usize| {
            [
                (x * 31 + y * 17) as f64 % 255.0 / 255.0,
                (x ^ y) as f64 % 97.0 / 97.0,
                0.5,
            ]
        };
        let (sw, sh, ow, oh) = (301, 211, 128, 91);
        let src_mean: f64 = (0..sh)
            .flat_map(|y| (0..sw).map(move |x| (x, y)))
            .map(|(x, y)| f(x, y)[0])
            .sum::<f64>()
            / (sw * sh) as f64;
        let out = area_average(sw, sh, ow, oh, f);
        let out_mean: f64 = out.iter().map(|p| p[0]).sum::<f64>() / (ow * oh) as f64;
        assert!((src_mean - out_mean).abs() < 1e-12, "{src_mean} {out_mean}");
    }

    #[test]
    fn equal_sizes_return_the_pixels() {
        let out = area_average(5, 4, 5, 4, |x, y| [x as f64, y as f64, 0.0]);
        assert_eq!(out[7], [2.0, 1.0, 0.0]);
    }

    /// The bits do not depend on the number of threads: a run on one thread of the source's rows, written the long way, equals the
    /// scoped threads' result.
    #[test]
    fn the_result_is_the_same_bits_on_a_second_run_and_row_by_row() {
        let f = |x: usize, y: usize| {
            let v = ((x * 2654435761 + y * 40503) % 1000) as f64 / 999.0;
            [v, 1.0 - v, v * v]
        };
        let a = area_average(257, 190, 100, 74, f);
        let b = area_average(257, 190, 100, 74, f);
        assert_eq!(a, b);
        // Row by row, a thread each, would be this; recompute each output row alone with the one-row path.
        let cols = spans(257, 100);
        let rows = spans(190, 74);
        for oy in [0usize, 1, 40, 73] {
            let (first_row, row_weights) = &rows[oy];
            let mut line = vec![[0.0f64; 3]; 100];
            for (k, &wy) in row_weights.iter().enumerate() {
                for (ox, (first_col, col_weights)) in cols.iter().enumerate() {
                    let mut acc = [0.0f64; 3];
                    for (i, &wx) in col_weights.iter().enumerate() {
                        let p = f(first_col + i, first_row + k);
                        for c in 0..3 {
                            acc[c] += wx as f64 * p[c];
                        }
                    }
                    for c in 0..3 {
                        line[ox][c] += wy as f64 * (acc[c] * (1.0 / 257.0));
                    }
                }
            }
            for (ox, l) in line.iter().enumerate() {
                let want = [
                    l[0] * (1.0 / 190.0),
                    l[1] * (1.0 / 190.0),
                    l[2] * (1.0 / 190.0),
                ];
                assert_eq!(a[oy * 100 + ox], want);
            }
        }
    }
}
