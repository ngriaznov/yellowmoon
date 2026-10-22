//! The coarse colour pass after R6's single-frame denoise (`docs/algorithms/05-single-frame-denoise.md`,
//! *Coarse colour*).
//!
//! Method A filters 8 × 8 blocks with candidates within 8 px, so it cannot see colour noise that varies over
//! more than about 20 px: on a flat colour the fine grain goes and smooth colour patches of 16 to 64 px stay,
//! which the fine grain used to hide. This stage removes that coarse colour noise and leaves the luma alone:
//!
//! 1. The working encoding (`denoise_space`) and its Y/Cb/Cr (`ycbcr`), as method A uses them.
//! 2. Cells of [`CELL_PX`] × [`CELL_PX`] pixels anchored to the frame's origin (`RgbImage::place`, reading
//!    G4): the means of Y, Cb and Cr over each cell's pixels in the buffer. A cell clipped by the frame's edge
//!    is the mean of the pixels it has.
//! 3. The noise the coarse colour may have at each cell: `h = strength · σC(cell)`, with `σC` the chroma noise
//!    of the working encoding at the cell's colour from the photo's noise model and R4 gains ([`LevelNoise`],
//!    as method A's chroma threshold) and `strength = c · K_COARSE` (`c = Colour / 50`).
//! 4. A bilateral filter of the cells' colour: over the window of (2 [`WINDOW_RADIUS`] + 1)² cells around
//!    the cell (cut at the buffer's cells), each cell `j` weighs `exp(−d² / (2 h²))`, `d²` the squared
//!    distance of its Cb and Cr from the cell's own and `h` the smaller of the two cells' widths (cells are
//!    averaged only within the noise of both: a dark saturated colour, whose noise in the encoding is many
//!    times a light neighbour's, does not take on its colour); the correction is the weighted mean of those
//!    differences (the weighted mean of the colour minus the cell's own, exactly 0 where the window's colour
//!    is the cell's). Cells within the noise of each other are averaged, so a blotch goes; a cell across a
//!    colour edge, or a colour variation well above the noise, weighs nothing, so colour does not bleed.
//! 5. Every pixel takes the bilinear interpolation of the corrections of the four cells around it (cell
//!    centres at `CELL_PX k + (CELL_PX − 1) / 2` in frame coordinates, neighbours cut at the buffer's cells)
//!    as a change of Cb and Cr, which leaves Y as it is: `ΔR = CR_DIV ΔCr`, `ΔB = CB_DIV ΔCb`,
//!    `ΔG = −(KR ΔR + KB ΔB) / KG` in the encoding, then decoded. A pixel whose correction is 0 keeps its
//!    input bits.
//!
//! A pixel reads the cells of its own and the next window: [`HALO_PX`] of context on each side, so a tile
//! with that halo equals the whole frame bit for bit (the cells that a buffer cuts lie in the margin the
//! caller drops). The stage is continuous in its input (no decision jumps), so the device differs from it by
//! rounding alone.
//!
//! [`K_COARSE`], [`CELL_PX`] and [`WINDOW_RADIUS`] were measured with `examples/denoise_gate.rs` and real
//! photos (`docs/algorithms/05`, *Measurement*).

use super::RgbImage;
use super::bm_dct::LevelNoise;
use super::denoise_space::{WORKING_GAIN_EV, decode, encode};
use super::ycbcr::{CB_DIV, CR_DIV, KB, KG, KR};

/// The side of a cell, in pixels.
pub const CELL_PX: usize = 4;
/// The radius of the window, in cells: (2 · 8 + 1)² = 289 cells, 68 px.
pub const WINDOW_RADIUS: usize = 8;
/// The width `h` of the bilateral filter's colour weight in standard deviations of the photo's chroma noise
/// per pixel ([`LevelNoise::chroma_sigma`]), at Auto Colour.
pub const K_COARSE: f64 = 0.2;
/// The context a pixel reads on each side. A pixel `j` px into its cell interpolates between the cell before
/// and its own (`j < CELL_PX / 2`) or its own and the next; each of those reads its window, so the farthest
/// pixel read is `CELL_PX (WINDOW_RADIUS + 1) + j` to the left or `CELL_PX (WINDOW_RADIUS + 1) + CELL_PX − 1 − j`
/// to the right: at most `CELL_PX (WINDOW_RADIUS + 1) + CELL_PX / 2 − 1`.
pub const HALO_PX: usize = halo_for(WINDOW_RADIUS);

/// The context for a window of `radius` cells ([`HALO_PX`]).
const fn halo_for(radius: usize) -> usize {
    CELL_PX * (radius + 1) + CELL_PX / 2 - 1
}

/// The resolved stage: the photo's noise and the strength.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CoarseChroma {
    /// The photo's noise model and gains (its `chroma_scale` is method A's and is not used here).
    pub noise: LevelNoise,
    /// `h / σC`: [`K_COARSE`] times `c = Colour / 50`.
    pub strength: f64,
    /// The radius of the window in cells ([`WINDOW_RADIUS`]).
    pub radius: usize,
}

impl CoarseChroma {
    /// The stage for method A's noise and the Colour control (`color`, 0 … 100; Auto 50); `None` when it
    /// would leave every pixel as it is (Colour 0 or a photo without noise).
    #[must_use]
    pub fn resolve(noise: LevelNoise, color: f32) -> Option<CoarseChroma> {
        let strength = f64::from(color) / 50.0 * K_COARSE;
        let has_noise = noise.a > 0.0 || noise.b > 0.0;
        (strength > 0.0 && strength.is_finite() && has_noise).then_some(CoarseChroma {
            noise,
            strength,
            radius: WINDOW_RADIUS,
        })
    }

    /// The context a pixel reads on each side ([`HALO_PX`] at the default radius).
    #[must_use]
    pub const fn halo_px(&self) -> usize {
        halo_for(self.radius)
    }

    /// Applies the stage to `img` (balanced linear camera RGB; a window of the frame with [`HALO_PX`] of
    /// context, or the whole frame).
    #[must_use]
    pub fn apply(&self, img: &RgbImage) -> RgbImage {
        let (w, h) = (img.width, img.height);
        if w == 0 || h == 0 {
            return img.clone();
        }
        let n = w * h;
        let gain = WORKING_GAIN_EV;
        // The encoded channels and Y/Cb/Cr per pixel.
        let mut enc = [vec![0.0f64; n], vec![0.0f64; n], vec![0.0f64; n]];
        for (c, plane) in enc.iter_mut().enumerate() {
            for (i, v) in plane.iter_mut().enumerate() {
                *v = encode(f64::from(img.data[c * n + i]), gain);
            }
        }
        let grid = Grid::new(img);
        let cells = grid.cell_means(&enc);
        let inv = self.weights(&cells);
        let delta = self.corrections(&grid, &cells, &inv);
        let mut out = img.clone();
        for y in 0..h {
            for x in 0..w {
                let (db, dr) = grid.interpolate(&delta, x, y);
                if db == 0.0 && dr == 0.0 {
                    continue;
                }
                let i = y * w + x;
                let de_r = CR_DIV * dr;
                let de_b = CB_DIV * db;
                let de_g = -(KR * de_r + KB * de_b) / KG;
                for (c, de) in [de_r, de_g, de_b].into_iter().enumerate() {
                    out.data[c * n + i] = decode(enc[c][i] + de, gain) as f32;
                }
            }
        }
        out
    }

    /// `1 / (2 h²)` of every cell, `None` where `h` is not above 0 or the weight would not be finite.
    fn weights(&self, cells: &[[f64; 3]]) -> Vec<Option<f64>> {
        cells
            .iter()
            .map(|&c| {
                let h = self.strength * self.noise.chroma_sigma(c);
                let inv = 1.0 / (2.0 * h * h);
                (h > 0.0 && h.is_finite() && inv.is_finite()).then_some(inv)
            })
            .collect()
    }

    /// The correction `(ΔCb, ΔCr)` of every cell (module documentation, steps 3 and 4).
    fn corrections(&self, grid: &Grid, cells: &[[f64; 3]], inv: &[Option<f64>]) -> Vec<[f64; 2]> {
        let r = self.radius as isize;
        let (cw, ch) = (grid.cols as isize, grid.rows as isize);
        let mut out = vec![[0.0; 2]; cells.len()];
        for cy in 0..ch {
            for cx in 0..cw {
                let k = (cy * cw + cx) as usize;
                let (own, Some(own_inv)) = (cells[k], inv[k]) else {
                    continue;
                };
                let (mut sw, mut sb, mut sr) = (0.0, 0.0, 0.0);
                for wy in (cy - r).max(0)..=(cy + r).min(ch - 1) {
                    for wx in (cx - r).max(0)..=(cx + r).min(cw - 1) {
                        let j = (wy * cw + wx) as usize;
                        // Two cells are averaged only within the noise of both: the narrower weight.
                        let Some(j_inv) = inv[j] else { continue };
                        let v = cells[j];
                        let (db, dr) = (v[1] - own[1], v[2] - own[2]);
                        let wt = (-(db * db + dr * dr) * own_inv.max(j_inv)).exp();
                        sw += wt;
                        sb += wt * db;
                        sr += wt * dr;
                    }
                }
                // The weighted mean of the differences: exactly 0 where the window's colour is the cell's.
                // The cell itself weighs 1, so `sw >= 1`.
                out[k] = [sb / sw, sr / sw];
            }
        }
        out
    }
}

/// The buffer's cells: frame-anchored, cut at the buffer's edges.
struct Grid {
    width: usize,
    height: usize,
    /// Frame x and y of the buffer's first pixel.
    x0: usize,
    y0: usize,
    /// Frame index of the first cell column and row.
    cx0: usize,
    cy0: usize,
    cols: usize,
    rows: usize,
}

impl Grid {
    fn new(img: &RgbImage) -> Grid {
        let (x0, y0) = (img.place.x0, img.place.y0);
        let cx0 = x0 / CELL_PX;
        let cy0 = y0 / CELL_PX;
        let cols = (x0 + img.width - 1) / CELL_PX - cx0 + 1;
        let rows = (y0 + img.height - 1) / CELL_PX - cy0 + 1;
        Grid {
            width: img.width,
            height: img.height,
            x0,
            y0,
            cx0,
            cy0,
            cols,
            rows,
        }
    }

    /// The means of Y, Cb and Cr of the encoded channels over each cell's pixels in the buffer, in raster
    /// order of the cells (pixels added in raster order).
    fn cell_means(&self, enc: &[Vec<f64>; 3]) -> Vec<[f64; 3]> {
        let mut sums = vec![[0.0f64; 3]; self.cols * self.rows];
        let mut counts = vec![0usize; self.cols * self.rows];
        for y in 0..self.height {
            let cy = (self.y0 + y) / CELL_PX - self.cy0;
            for x in 0..self.width {
                let cx = (self.x0 + x) / CELL_PX - self.cx0;
                let i = y * self.width + x;
                let (r, g, b) = (enc[0][i], enc[1][i], enc[2][i]);
                let yy = KR * r + KG * g + KB * b;
                let k = cy * self.cols + cx;
                sums[k][0] += yy;
                sums[k][1] += (b - yy) / CB_DIV;
                sums[k][2] += (r - yy) / CR_DIV;
                counts[k] += 1;
            }
        }
        sums.iter()
            .zip(&counts)
            .map(|(s, &c)| s.map(|v| v / c as f64))
            .collect()
    }

    /// The bilinear interpolation of the cells' corrections at the buffer's pixel `(x, y)`.
    fn interpolate(&self, delta: &[[f64; 2]], x: usize, y: usize) -> (f64, f64) {
        let axis = |p: usize, first: usize, len: usize| -> (usize, usize, f64) {
            // The cell centres are at CELL_PX k + (CELL_PX - 1) / 2; t is the frame position in cells.
            let t = (p as f64 - (CELL_PX as f64 - 1.0) / 2.0) / CELL_PX as f64;
            let k0 = t.floor();
            let frac = t - k0;
            let clamp = |k: f64| -> usize {
                let local = k - first as f64;
                local.clamp(0.0, (len - 1) as f64) as usize
            };
            (clamp(k0), clamp(k0 + 1.0), frac)
        };
        let (xa, xb, fx) = axis(self.x0 + x, self.cx0, self.cols);
        let (ya, yb, fy) = axis(self.y0 + y, self.cy0, self.rows);
        let at = |cx: usize, cy: usize| delta[cy * self.cols + cx];
        let lerp = |a: f64, b: f64, t: f64| a + (b - a) * t;
        let mut out = [0.0; 2];
        for (k, o) in out.iter_mut().enumerate() {
            let top = lerp(at(xa, ya)[k], at(xb, ya)[k], fx);
            let bottom = lerp(at(xa, yb)[k], at(xb, yb)[k], fx);
            *o = lerp(top, bottom, fy);
        }
        (out[0], out[1])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu::Placement;
    use crate::cpu::synth;

    fn noise() -> LevelNoise {
        LevelNoise {
            a: 4e-4,
            b: 6e-7,
            gains: [2.0, 1.0, 1.5],
            encoding_gain: WORKING_GAIN_EV.exp2(),
            chroma_scale: 0.0,
        }
    }

    fn image(w: usize, h: usize, f: impl Fn(usize, usize, usize) -> f32) -> RgbImage {
        let mut data = Vec::with_capacity(3 * w * h);
        for c in 0..3 {
            for y in 0..h {
                for x in 0..w {
                    data.push(f(c, x, y));
                }
            }
        }
        RgbImage::from_planar(w, h, data).unwrap()
    }

    /// An image from Y/Cb/Cr of the working encoding at every pixel.
    fn from_ycc(w: usize, h: usize, f: impl Fn(usize, usize) -> [f64; 3]) -> RgbImage {
        let mut img = RgbImage::new(w, h);
        let n = w * h;
        for y in 0..h {
            for x in 0..w {
                let [yy, cb, cr] = f(x, y);
                let r = yy + CR_DIV * cr;
                let b = yy + CB_DIV * cb;
                let g = (yy - KR * r - KB * b) / KG;
                for (c, e) in [r, g, b].into_iter().enumerate() {
                    img.data[c * n + y * w + x] = decode(e, WORKING_GAIN_EV) as f32;
                }
            }
        }
        img
    }

    /// The width `h` of the colour weight at mid grey, Auto Colour.
    fn h_grey() -> f64 {
        K_COARSE * noise().chroma_sigma([0.46, 0.0, 0.0])
    }

    /// A flat grey with smooth colour blotches of about 20 to 40 px and amplitude `amp` (working encoding), as
    /// method A leaves them.
    fn blotchy(w: usize, h: usize, amp: f64) -> RgbImage {
        from_ycc(w, h, |x, y| {
            let (fx, fy) = (x as f64 / 12.0, y as f64 / 12.0);
            let t = (fx * 2.1 + fy * 0.7).sin() * (fy * 1.9 - fx * 0.4).cos();
            [0.46, amp * t, -0.7 * amp * t]
        })
    }

    fn chroma_spread(img: &RgbImage, inner: usize) -> f64 {
        let n = img.plane_len();
        let w = img.width;
        let mut v = Vec::new();
        for y in inner..img.height - inner {
            for x in inner..w - inner {
                let i = y * w + x;
                v.push(f64::from(img.data[i] - img.data[2 * n + i]));
            }
        }
        let mean = v.iter().sum::<f64>() / v.len() as f64;
        (v.iter().map(|d| (d - mean).powi(2)).sum::<f64>() / v.len() as f64).sqrt()
    }

    #[test]
    fn colour_zero_or_no_noise_leaves_no_stage() {
        assert_eq!(CoarseChroma::resolve(noise(), 0.0), None);
        let silent = LevelNoise {
            a: 0.0,
            b: 0.0,
            ..noise()
        };
        assert_eq!(CoarseChroma::resolve(silent, 50.0), None);
        let s = CoarseChroma::resolve(noise(), 50.0).expect("stage");
        assert!((s.strength - K_COARSE).abs() < 1e-12);
        assert_eq!(HALO_PX, 37);
    }

    #[test]
    fn coarse_colour_blotches_within_the_noise_are_removed_and_luma_is_kept() {
        let img = blotchy(96, 80, 0.7 * h_grey());
        let out = CoarseChroma::resolve(noise(), 50.0).unwrap().apply(&img);
        assert!(
            chroma_spread(&out, 24) < 0.3 * chroma_spread(&img, 24),
            "{} against {}",
            chroma_spread(&out, 24),
            chroma_spread(&img, 24)
        );
        // The luma of the working encoding does not move.
        let n = img.plane_len();
        for i in 0..n {
            let y = |im: &RgbImage| {
                let e = |c: usize| encode(f64::from(im.data[c * n + i]), WORKING_GAIN_EV);
                KR * e(0) + KG * e(1) + KB * e(2)
            };
            assert!((y(&out) - y(&img)).abs() < 1e-6, "pixel {i}");
        }
    }

    #[test]
    fn colour_variation_well_above_the_noise_is_kept() {
        let img = blotchy(96, 80, 12.0 * h_grey());
        let out = CoarseChroma::resolve(noise(), 50.0).unwrap().apply(&img);
        let (a, b) = (chroma_spread(&img, 24), chroma_spread(&out, 24));
        assert!(b > 0.9 * a, "{b} against {a}");
    }

    #[test]
    fn a_colour_edge_far_above_the_noise_keeps_its_colours() {
        // Red-ish on the left, blue-ish on the right, same luma: the cells away from the edge stay as they
        // were, and the edge is not smeared.
        let grey = 0.18 / WORKING_GAIN_EV.exp2() as f32;
        let img = image(96, 64, |c, x, _| {
            let k = if x < 48 { 0.3 } else { -0.3 };
            match c {
                0 => grey * (1.0 + k),
                2 => grey * (1.0 - k),
                _ => grey,
            }
        });
        let out = CoarseChroma::resolve(noise(), 100.0).unwrap().apply(&img);
        let n = img.plane_len();
        for y in 0..64 {
            for x in [0, 10, 40, 46, 49, 52, 80, 95] {
                let i = y * 96 + x;
                for c in [0, 2] {
                    let (a, b) = (img.data[c * n + i], out.data[c * n + i]);
                    assert!(
                        (a - b).abs() <= 1e-4 * a.abs().max(1e-3),
                        "({x}, {y}) channel {c}: {a} -> {b}"
                    );
                }
            }
        }
    }

    #[test]
    fn a_dark_saturated_colour_beside_a_light_grey_keeps_its_colour() {
        // At a high noise level a dark saturated colour's chroma noise in the encoding is many times a light
        // grey's; with the narrower of the two widths the dark side does not take on the grey's colour.
        let loud = LevelNoise {
            a: 4e-3,
            b: 4e-6,
            ..noise()
        };
        // A dark blue (h ≈ 0.013 at Auto Colour) beside a light grey (h ≈ 0.007), 0.023 apart in Cb and Cr: within
        // the dark side's width, well outside the grey's. With the dark cells' own width they moved by 16 %.
        let img = image(96, 64, |c, x, _| {
            if x < 48 {
                [0.002, 0.0015, 0.004][c]
            } else {
                0.2
            }
        });
        let out = CoarseChroma::resolve(loud, 50.0).unwrap().apply(&img);
        let n = img.plane_len();
        let chroma = |im: &RgbImage, i: usize| {
            let e = |c: usize| encode(f64::from(im.data[c * n + i]), WORKING_GAIN_EV);
            let y = KR * e(0) + KG * e(1) + KB * e(2);
            [(e(2) - y) / CB_DIV, (e(0) - y) / CR_DIV]
        };
        let mut worst = 0.0f64;
        for y in 0..64 {
            for x in 0..48 {
                let (a, b) = (chroma(&img, y * 96 + x), chroma(&out, y * 96 + x));
                worst = worst.max((a[0] - b[0]).hypot(a[1] - b[1]) / a[0].hypot(a[1]));
            }
        }
        assert!(worst < 0.05, "the dark side moved by {worst} of its chroma");
    }

    #[test]
    fn a_tile_with_the_halo_equals_the_whole_frame_bit_for_bit() {
        let (fw, fh) = (151, 117);
        let mut rng = synth::Rng::new(7);
        let mut img = blotchy(fw, fh, 2.0 * h_grey());
        img.data
            .iter_mut()
            .for_each(|v| *v *= 1.0 + 0.05 * rng.gaussian());
        let stage = CoarseChroma::resolve(noise(), 70.0).unwrap();
        let whole = stage.apply(&img);
        for (tx, ty, tw, th) in [
            (37usize, 29usize, 61usize, 47usize),
            (0, 0, 50, 50),
            (90, 70, 61, 47),
            (5, 3, 9, 7),
        ] {
            let (x0, y0) = (tx.saturating_sub(HALO_PX), ty.saturating_sub(HALO_PX));
            let (x1, y1) = ((tx + tw + HALO_PX).min(fw), (ty + th + HALO_PX).min(fh));
            let (bw, bh) = (x1 - x0, y1 - y0);
            let mut buf = image(bw, bh, |c, x, y| {
                img.data[c * fw * fh + (y0 + y) * fw + x0 + x]
            });
            buf.place = Placement {
                x0,
                y0,
                frame_width: fw,
                frame_height: fh,
            };
            let got = stage.apply(&buf);
            for c in 0..3 {
                for y in ty..(ty + th).min(fh) {
                    for x in tx..(tx + tw).min(fw) {
                        assert_eq!(
                            got.data[c * bw * bh + (y - y0) * bw + x - x0].to_bits(),
                            whole.data[c * fw * fh + y * fw + x].to_bits(),
                            "tile ({tx}, {ty}): ({x}, {y}) channel {c}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn a_smaller_halo_does_not_reproduce_the_frame() {
        // The halo is tight: one pixel less changes some pixel of the core.
        let (fw, fh) = (120, 100);
        let mut rng = synth::Rng::new(11);
        let mut img = blotchy(fw, fh, 2.0 * h_grey());
        img.data
            .iter_mut()
            .for_each(|v| *v *= 1.0 + 0.05 * rng.gaussian());
        let stage = CoarseChroma::resolve(noise(), 50.0).unwrap();
        let whole = stage.apply(&img);
        let halo = HALO_PX - 1;
        let mut any = false;
        for tx in 40..44 {
            let (x0, x1) = (tx - halo, tx + 20 + halo);
            let bw = x1 - x0;
            let mut buf = image(bw, fh, |c, x, y| img.data[c * fw * fh + y * fw + x0 + x]);
            buf.place = Placement {
                x0,
                y0: 0,
                frame_width: fw,
                frame_height: fh,
            };
            let got = stage.apply(&buf);
            for y in 0..fh {
                for x in tx..tx + 20 {
                    if got.data[y * bw + x - x0].to_bits() != whole.data[y * fw + x].to_bits() {
                        any = true;
                    }
                }
            }
        }
        assert!(any, "a halo of {halo} px already reproduces the frame");
    }

    #[test]
    fn outputs_are_finite_and_tiny_images_work() {
        let stage = CoarseChroma::resolve(noise(), 100.0).unwrap();
        for (w, h) in [(1, 1), (3, 2), (5, 9), (0, 0)] {
            let img = image(w, h, |c, x, y| {
                ((x * 7 + y * 3 + c) % 5) as f32 * 0.05 - 0.01
            });
            let out = stage.apply(&img);
            assert_eq!((out.width, out.height), (w, h));
            assert!(out.data.iter().all(|v| v.is_finite()));
        }
        // A flat image is a fixed point.
        let flat = image(40, 40, |c, _, _| [0.1, 0.12, 0.08][c]);
        assert_eq!(stage.apply(&flat), flat);
    }
}
