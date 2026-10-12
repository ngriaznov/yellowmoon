//! The 2 MP crop grid and the per-window statistics (`docs/06-noise.md` steps 1 to 2).
//!
//! The estimator reads 16 square tiles spread over the frame (a 4 x 4 grid, about 2 MP in total).
//! Each tile is split into its four CFA planes (half resolution) and each plane into **windows** of
//! 6 x 6 plane pixels, i.e. 3 x 3 non-overlapping 2 x 2 Haar blocks. For a block of plane pixels
//! `a b / c d`:
//!
//! ```text
//! m  = (a + b + c + d) / 4          block mean (the LL coefficient divided by 2)
//! dh = (a - b + c - d) / 2          horizontal detail
//! dv = (a + b - c - d) / 2          vertical detail
//! dd = (a - b - c + d) / 2          diagonal detail (HH)
//! ```
//!
//! For white noise of variance `s^2` the four are uncorrelated, and `dh`, `dv`, `dd` each have
//! variance `s^2`, so `dd` is the noise measurement. `dd` also cancels row offsets (`a - b` and
//! `c - d` are within-row differences) and any linear ramp. The window's **energy** measures
//! texture without using its own `dd`, so selecting windows by low energy does not bias the `dd`
//! variance (for Gaussian noise `dd` is independent of `dh`, `dv` and of other windows):
//!
//! ```text
//! energy = sum_blocks[(dh - mean dh)^2 + (dv - mean dv)^2] / 16  +  mean over neighbours of dd^2
//! ```
//!
//! The first term is the detrended horizontal and vertical texture of the window (16 degrees of
//! freedom); the second is the diagonal texture of the up to eight neighbouring windows of the same
//! plane, which the window's own `dd` cannot be used for but which catches a texture that only
//! lives on the diagonal. Both terms have expectation `s^2` for white noise, so the energy of a
//! flat window is `2 s^2`.
//!
//! Each window also stores two plane-fit residual sums for the filtered-file check.

use super::{EstimateError, EstimatorConfig, source::PlaneSource};

/// Plane pixels per window side.
pub const WIN: usize = 6;

/// Photosites per window side (`2 * WIN`).
pub const WIN_PX: usize = 2 * WIN;

/// Haar blocks per window side.
pub const BLOCKS: usize = WIN / 2;

/// Haar detail coefficients per window.
pub const COEFFS: usize = BLOCKS * BLOCKS;

/// Degrees of freedom of the plane-fit residual of the 36 pixels of a window.
pub const DOF_PIX: f64 = 33.0;

/// Degrees of freedom of the plane-fit residual of the 9 block means of a window.
pub const DOF_BIN: f64 = 6.0;

/// Statistics of one 6 x 6 window of one CFA plane.
#[derive(Clone, Copy, Debug)]
pub struct Window {
    /// CFA plane `2 * (y & 1) + (x & 1)`.
    pub plane: u8,
    /// Mean of the 36 values.
    pub mean: f32,
    /// Texture energy in variance units (`2 s^2` for flat white noise), see the module docs.
    pub energy: f32,
    /// The nine diagonal Haar detail coefficients.
    pub d: [f32; COEFFS],
    /// Mean squared quantisation step over the 36 pixels (normalised units squared).
    pub q2: f32,
    /// Number of pixels at the clip level.
    pub clipped: u8,
}

/// Where the 16 tiles sit.
#[derive(Clone, Debug)]
pub struct TilePlan {
    /// Tile side in photosites (a multiple of 12).
    pub size: usize,
    /// Top-left corners (even coordinates), row-major.
    pub origins: Vec<(usize, usize)>,
}

impl TilePlan {
    /// Windows per tile side.
    pub fn windows_per_axis(&self) -> usize {
        self.size / WIN_PX
    }

    /// Windows per tile over the four planes.
    pub fn windows_per_tile(&self) -> usize {
        let n = self.windows_per_axis();
        4 * n * n
    }

    /// Flat index of a window inside its tile.
    pub fn window_index(&self, plane: usize, wi: usize, wj: usize) -> usize {
        let n = self.windows_per_axis();
        (plane * n + wi) * n + wj
    }
}

/// Chooses the tile layout for a `width x height` frame.
///
/// # Errors
///
/// Returns [`EstimateError::FrameTooSmall`] when tiles would be smaller than 48 photosites.
pub fn plan(width: usize, height: usize, cfg: &EstimatorConfig) -> Result<TilePlan, EstimateError> {
    let per_axis = cfg.tiles_per_axis.max(1);
    let target = (cfg.grid_pixels as f64 / (per_axis * per_axis) as f64).sqrt();
    let mut size = (target / WIN_PX as f64).floor() as usize * WIN_PX;
    let limit = width.min(height) / per_axis / WIN_PX * WIN_PX;
    size = size.min(limit);
    if size < 4 * WIN_PX {
        return Err(EstimateError::FrameTooSmall { width, height });
    }
    let spread = |extent: usize, i: usize| -> usize {
        if per_axis == 1 {
            (extent - size) / 2 / 2 * 2
        } else {
            let pos = (extent - size) as f64 * i as f64 / (per_axis - 1) as f64;
            (pos.round() as usize) / 2 * 2
        }
    };
    let mut origins = Vec::with_capacity(per_axis * per_axis);
    for iy in 0..per_axis {
        for ix in 0..per_axis {
            origins.push((spread(width, ix), spread(height, iy)));
        }
    }
    Ok(TilePlan { size, origins })
}

/// The pixels of one tile.
pub struct TileData {
    /// Normalised values, `size * size`, row-major.
    pub pixels: Vec<f32>,
    /// Quantisation steps (normalised), if the decoder supplied a map.
    pub qstep: Option<Vec<f32>>,
}

/// Reads one tile from the sources.
pub fn load_tile(
    source: &dyn PlaneSource,
    qstep: Option<&dyn PlaneSource>,
    origin: (usize, usize),
    size: usize,
) -> TileData {
    let mut pixels = vec![0.0f32; size * size];
    source.read_rect(origin.0, origin.1, size, size, &mut pixels);
    let qstep = qstep.map(|q| {
        let mut v = vec![0.0f32; size * size];
        q.read_rect(origin.0, origin.1, size, size, &mut v);
        v
    });
    TileData { pixels, qstep }
}

/// Computes the windows of a tile in the order `[plane][wi][wj]`.
pub fn tile_windows(tile: &TileData, size: usize, clip_level: f32, q_norm: f32) -> Vec<Window> {
    let n = size / WIN_PX;
    let mut out = Vec::with_capacity(4 * n * n);
    for plane in 0..4 {
        for wi in 0..n {
            for wj in 0..n {
                out.push(window(tile, size, plane, wi, wj, clip_level, q_norm));
            }
        }
    }
    // Add the diagonal texture of the neighbouring windows to each window's energy.
    let dd2: Vec<f32> = out
        .iter()
        .map(|w| w.d.iter().map(|d| d * d).sum::<f32>() / COEFFS as f32)
        .collect();
    for plane in 0..4 {
        for wi in 0..n {
            for wj in 0..n {
                let (mut sum, mut count) = (0.0f32, 0u32);
                for di in -1i32..=1 {
                    for dj in -1i32..=1 {
                        if di == 0 && dj == 0 {
                            continue;
                        }
                        let (ii, jj) = (wi as i32 + di, wj as i32 + dj);
                        if ii < 0 || jj < 0 || ii >= n as i32 || jj >= n as i32 {
                            continue;
                        }
                        sum += dd2[(plane * n + ii as usize) * n + jj as usize];
                        count += 1;
                    }
                }
                let w = &mut out[(plane * n + wi) * n + wj];
                if count > 0 {
                    w.energy += sum / count as f32;
                }
            }
        }
    }
    out
}

/// Plane-fit residual sums of one window, for the filtered-file check: `(ss_pix, ss_bin)`.
///
/// `ss_pix` is the residual sum of squares of the 36 pixels after a plane fit (33 degrees of
/// freedom) and `ss_bin` the same for the 3 x 3 block means (6 degrees of freedom); both divided by
/// their degrees of freedom estimate the pixel variance and the variance of a 2 x 2 mean, 4 times
/// smaller for white noise. Computed on demand, for the few windows the check looks at.
pub fn residual_sums(
    tile: &TileData,
    size: usize,
    plane: usize,
    wi: usize,
    wj: usize,
) -> (f64, f64) {
    let (py, px) = (plane >> 1, plane & 1);
    let mut z = [[0.0f64; WIN]; WIN];
    for (r, row) in z.iter_mut().enumerate() {
        let base = (2 * (WIN * wi + r) + py) * size + 2 * WIN * wj + px;
        for (c, slot) in row.iter_mut().enumerate() {
            *slot = f64::from(tile.pixels[base + 2 * c]);
        }
    }
    // Pixel coordinates run -2.5..2.5 (sum of squares 17.5 per row, 105 over the window); block
    // coordinates run -1..1 (sum of squares 6 over the 9 blocks).
    let (mut s, mut szz, mut si, mut sj) = (0.0, 0.0, 0.0, 0.0);
    for (r, row) in z.iter().enumerate() {
        for (c, &v) in row.iter().enumerate() {
            s += v;
            szz += v * v;
            si += (r as f64 - 2.5) * v;
            sj += (c as f64 - 2.5) * v;
        }
    }
    let ss_pix = (szz - s * s / 36.0 - (si * si + sj * sj) / 105.0).max(0.0);
    let (mut s, mut smm, mut si, mut sj) = (0.0, 0.0, 0.0, 0.0);
    for bi in 0..BLOCKS {
        for bj in 0..BLOCKS {
            let v = 0.25
                * (z[2 * bi][2 * bj]
                    + z[2 * bi][2 * bj + 1]
                    + z[2 * bi + 1][2 * bj]
                    + z[2 * bi + 1][2 * bj + 1]);
            s += v;
            smm += v * v;
            si += (bi as f64 - 1.0) * v;
            sj += (bj as f64 - 1.0) * v;
        }
    }
    let ss_bin = (smm - s * s / 9.0 - (si * si + sj * sj) / 6.0).max(0.0);
    (ss_pix, ss_bin)
}

fn window(
    tile: &TileData,
    size: usize,
    plane: usize,
    wi: usize,
    wj: usize,
    clip_level: f32,
    q_norm: f32,
) -> Window {
    let (py, px) = (plane >> 1, plane & 1);
    let clip_threshold = clip_level - 1.0e-5;
    let mut z = [[0.0f64; WIN]; WIN];
    let mut clipped = 0u8;
    let mut q2_sum = 0.0f64;
    for (r, row) in z.iter_mut().enumerate() {
        let y = 2 * (WIN * wi + r) + py;
        let base = y * size + 2 * WIN * wj + px;
        for (c, slot) in row.iter_mut().enumerate() {
            let v = tile.pixels[base + 2 * c];
            clipped += u8::from(v >= clip_threshold);
            *slot = f64::from(v);
            if let Some(q) = &tile.qstep {
                let qv = f64::from(q[base + 2 * c]);
                q2_sum += qv * qv;
            }
        }
    }
    let q2 = if tile.qstep.is_some() {
        q2_sum / (WIN * WIN) as f64
    } else {
        f64::from(q_norm) * f64::from(q_norm)
    };

    let mut dh = [0.0f64; COEFFS];
    let mut dv = [0.0f64; COEFFS];
    let mut dd = [0.0f64; COEFFS];
    let mut m = [0.0f64; COEFFS];
    for bi in 0..BLOCKS {
        for bj in 0..BLOCKS {
            let (a, b) = (z[2 * bi][2 * bj], z[2 * bi][2 * bj + 1]);
            let (c, d) = (z[2 * bi + 1][2 * bj], z[2 * bi + 1][2 * bj + 1]);
            let k = bi * BLOCKS + bj;
            dh[k] = 0.5 * (a - b + c - d);
            dv[k] = 0.5 * (a + b - c - d);
            dd[k] = 0.5 * (a - b - c + d);
            m[k] = 0.25 * (a + b + c + d);
        }
    }
    let n = COEFFS as f64;
    let mean = m.iter().sum::<f64>() / n;
    let (hbar, vbar) = (dh.iter().sum::<f64>() / n, dv.iter().sum::<f64>() / n);
    let energy: f64 = (dh.iter().map(|v| (v - hbar).powi(2)).sum::<f64>()
        + dv.iter().map(|v| (v - vbar).powi(2)).sum::<f64>())
        / 16.0;

    Window {
        plane: plane as u8,
        mean: mean as f32,
        energy: energy as f32,
        d: dd.map(|v| v as f32),
        q2: q2 as f32,
        clipped,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::estimate::source::F32Plane;
    use crate::model::synth::SynthRng;

    #[test]
    fn plan_for_a_50_mp_frame_is_about_2_mp() {
        let cfg = EstimatorConfig::default();
        let p = plan(8192, 6144, &cfg).unwrap();
        assert_eq!(p.size, 348);
        assert_eq!(p.origins.len(), 16);
        let total: usize = p.origins.len() * p.size * p.size;
        assert!((1_900_000..=2_000_000).contains(&total), "{total}");
        for &(x, y) in &p.origins {
            assert!(x % 2 == 0 && y % 2 == 0);
            assert!(x + p.size <= 8192 && y + p.size <= 6144);
        }
        // Tiles do not overlap.
        let mut xs: Vec<usize> = p.origins.iter().map(|o| o.0).collect();
        xs.sort_unstable();
        xs.dedup();
        assert_eq!(xs.len(), 4);
        assert!(xs.windows(2).all(|w| w[1] - w[0] >= p.size));
        assert_eq!(p.windows_per_tile(), 4 * 29 * 29);
    }

    #[test]
    fn plan_shrinks_tiles_for_small_frames_and_rejects_tiny_ones() {
        let cfg = EstimatorConfig::default();
        let p = plan(1024, 768, &cfg).unwrap();
        assert_eq!(p.size, 192);
        assert!(plan(120, 120, &cfg).is_err());
    }

    fn white_tile(size: usize, sigma: f64, seed: u64) -> TileData {
        let mut rng = SynthRng::new(seed);
        TileData {
            pixels: (0..size * size)
                .map(|_| (0.3 + sigma * rng.normal()) as f32)
                .collect(),
            qstep: None,
        }
    }

    #[test]
    fn white_noise_windows_have_unit_coefficient_variance_and_the_white_ratio() {
        let size = 480;
        let sigma = 0.01;
        let tile = white_tile(size, sigma, 4);
        let wins = tile_windows(&tile, size, 1.0, 0.0);
        assert_eq!(wins.len(), 4 * 40 * 40);
        let n = (wins.len() * COEFFS) as f64;
        let var: f64 = wins
            .iter()
            .flat_map(|w| w.d.iter())
            .map(|&d| f64::from(d).powi(2))
            .sum::<f64>()
            / n;
        assert!(
            (var / (sigma * sigma) - 1.0).abs() < 0.02,
            "dd variance ratio {}",
            var / (sigma * sigma)
        );
        // Plane-fit residuals: SS / dof is unbiased, the binned one is a quarter of the pixel one.
        let n_w = size / WIN_PX;
        let (mut pix, mut bin) = (0.0, 0.0);
        for plane in 0..4 {
            for wi in 0..n_w {
                for wj in 0..n_w {
                    let (p, b) = residual_sums(&tile, size, plane, wi, wj);
                    pix += p / DOF_PIX;
                    bin += b / DOF_BIN;
                }
            }
        }
        assert!((pix / wins.len() as f64 / (sigma * sigma) - 1.0).abs() < 0.02);
        assert!(
            (bin / pix - 0.25).abs() < 0.01,
            "white-noise binned ratio {}",
            bin / pix
        );
        // Energy of flat white noise is 2 s^2 (one s^2 from each of its two terms).
        let e: f64 = wins.iter().map(|w| f64::from(w.energy)).sum::<f64>() / wins.len() as f64;
        assert!(
            (e / (2.0 * sigma * sigma) - 1.0).abs() < 0.03,
            "energy ratio {}",
            e / (2.0 * sigma * sigma)
        );
    }

    #[test]
    fn ramps_and_row_offsets_do_not_reach_the_diagonal_coefficient() {
        let size = 96;
        let mut px = vec![0.0f32; size * size];
        for y in 0..size {
            let row_offset = ((y * 7919) % 13) as f32 * 0.003;
            for x in 0..size {
                px[y * size + x] = 0.2 + 0.0004 * x as f32 + 0.0007 * y as f32 + row_offset;
            }
        }
        let tile = TileData {
            pixels: px,
            qstep: None,
        };
        let wins = tile_windows(&tile, size, 1.0, 0.0);
        for w in &wins {
            for &d in &w.d {
                assert!(d.abs() < 1e-6, "HH leaked: {d}");
            }
        }
        // A pure linear ramp in x has no energy after detrending (the row offsets do, through dv).
        // (Neighbouring windows have dd = 0 too, so the whole energy is zero.)
        let mut px = vec![0.0f32; size * size];
        for y in 0..size {
            for x in 0..size {
                px[y * size + x] = 0.2 + 0.001 * x as f32;
            }
        }
        let wins = tile_windows(
            &TileData {
                pixels: px,
                qstep: None,
            },
            size,
            1.0,
            0.0,
        );
        assert!(wins.iter().all(|w| w.energy < 1e-9));
    }

    #[test]
    fn texture_raises_energy_and_clipping_is_counted() {
        let size = 96;
        // Vertical stripes whose edges fall inside the Haar blocks: horizontal detail that
        // changes sign from block to block (a pure Nyquist stripe has a constant dh, which the
        // detrending treats as a ramp; it has no diagonal detail either, so it cannot leak into dd).
        let mut stripes = vec![0.5f32; size * size];
        // Horizontal stripes: vertical detail.
        let mut rows = vec![0.5f32; size * size];
        // A checkerboard on the plane grid: only the diagonal detail sees it, and only the
        // neighbouring windows' diagonal term can reject it.
        let mut checker = vec![0.5f32; size * size];
        for y in 0..size {
            for x in 0..size {
                if (x / 2).div_ceil(2) % 2 == 0 {
                    stripes[y * size + x] = 0.6;
                }
                if (y / 2).div_ceil(2) % 2 == 0 {
                    rows[y * size + x] = 0.6;
                }
                if (x / 2 + y / 2) % 2 == 0 {
                    checker[y * size + x] = 0.6;
                }
            }
        }
        for (name, px) in [("stripes", stripes), ("rows", rows), ("checker", checker)] {
            let wins = tile_windows(
                &TileData {
                    pixels: px,
                    qstep: None,
                },
                size,
                1.0,
                0.0,
            );
            // Windows with at least one neighbour see the texture.
            assert!(
                wins.iter().all(|w| w.energy > 1e-3),
                "{name}: texture not seen"
            );
        }
        let px = vec![1.0f32; size * size];
        let wins = tile_windows(
            &TileData {
                pixels: px,
                qstep: None,
            },
            size,
            1.0,
            0.0,
        );
        assert!(
            wins.iter()
                .all(|w| w.clipped == 36 && w.d.iter().all(|&d| d == 0.0))
        );
    }

    #[test]
    fn qstep_plane_gives_the_mean_square_step() {
        let size = 48;
        let tile = TileData {
            pixels: vec![0.3; size * size],
            qstep: Some(vec![0.002; size * size]),
        };
        let wins = tile_windows(&tile, size, 1.0, 1e-4);
        assert!((f64::from(wins[0].q2) - 4e-6).abs() < 1e-10);
        let tile = TileData {
            pixels: vec![0.3; size * size],
            qstep: None,
        };
        let wins = tile_windows(&tile, size, 1.0, 1e-3);
        assert!((f64::from(wins[0].q2) - 1e-6).abs() < 1e-10);
    }

    #[test]
    fn tiles_load_through_a_source() {
        let (w, h) = (96, 96);
        let data: Vec<f32> = (0..w * h).map(|i| i as f32).collect();
        let src = F32Plane::new(&data, w, h).unwrap();
        let t = load_tile(&src, None, (4, 2), 48);
        assert_eq!(t.pixels[0], (2 * w + 4) as f32);
        assert_eq!(t.pixels[48], (3 * w + 4) as f32);
    }
}
