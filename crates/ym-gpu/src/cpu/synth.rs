//! Deterministic synthetic images for the CPU reference tests and benchmarks: zone plate, edges,
//! colour patches, fine text, a natural-looking random scene and line grids (`docs/13-testing.md`).
//!
//! Everything is seeded and uses only the small PRNG here, so a test (or the later GPU parity
//! tests) sees the same pixels on every machine. Images are linear `f32` in `[0, 1]` unless a
//! function says otherwise.

use super::{Cfa, Mosaic, RgbImage};

/// SplitMix64: a tiny, fast, well-mixed generator. Not for cryptography.
#[derive(Clone, Debug)]
pub struct Rng(u64);

impl Rng {
    /// A generator seeded with `seed`.
    pub fn new(seed: u64) -> Rng {
        Rng(seed)
    }

    /// Next 64 random bits.
    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `[0, 1)`.
    #[inline]
    pub fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }

    /// Standard normal (Box-Muller).
    pub fn gaussian(&mut self) -> f32 {
        let u1 = self.next_f32().max(1e-12);
        let u2 = self.next_f32();
        (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
    }

    /// Uniform in `[lo, hi)`.
    pub fn range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.next_f32()
    }
}

/// Samples a full-resolution RGB image through a Bayer filter (no optical low-pass).
pub fn mosaic_from_rgb(img: &RgbImage, cfa: Cfa) -> Mosaic {
    let (w, h) = (img.width, img.height);
    let mut m = Mosaic::new(w, h, cfa);
    for y in 0..h {
        for x in 0..w {
            m.data[y * w + x] = img.plane(cfa.color_at(x, y).plane())[y * w + x];
        }
    }
    m
}

/// A coloured zone plate: radial chirp `0.5 + 0.5 cos(k r^2)` with a different phase per channel,
/// reaching the Nyquist frequency at the image corner.
pub fn zone_plate(w: usize, h: usize) -> RgbImage {
    let mut img = RgbImage::new(w, h);
    let (cx, cy) = (w as f32 * 0.5, h as f32 * 0.5);
    let r_max2 = cx * cx + cy * cy;
    // Phase advance per pixel of d(k r^2)/dr = 2 k r reaches pi at r = r_max.
    let k = std::f32::consts::PI / (2.0 * r_max2.sqrt()) * 0.9;
    for y in 0..h {
        for x in 0..w {
            let dx = x as f32 + 0.5 - cx;
            let dy = y as f32 + 0.5 - cy;
            let r2 = dx * dx + dy * dy;
            let phase = k * r2;
            img.set_pixel(
                x,
                y,
                [
                    0.5 + 0.5 * phase.cos(),
                    0.5 + 0.5 * (phase + 0.7).cos(),
                    0.5 + 0.5 * (phase * 1.0 + 1.4).cos(),
                ],
            );
        }
    }
    img
}

/// Straight and slanted edges between saturated colour regions, with a small grey ramp: stresses
/// zipper and false-colour handling.
pub fn edges(w: usize, h: usize) -> RgbImage {
    let mut img = RgbImage::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let (fx, fy) = (x as f32 / w as f32, y as f32 / h as f32);
            let mut c = [0.85, 0.85, 0.85];
            // Vertical and horizontal edges.
            if fx > 0.15 && fx < 0.35 && fy > 0.15 && fy < 0.45 {
                c = [0.9, 0.1, 0.1];
            }
            if fx > 0.45 && fx < 0.80 && fy > 0.10 && fy < 0.30 {
                c = [0.1, 0.2, 0.9];
            }
            // Slanted edges: a wedge and a disc.
            if fy > 0.55
                && (fx - 0.25) * 2.0 < (fy - 0.55) * 3.0
                && fx > 0.05
                && fx < 0.40
                && fy < 0.95
            {
                c = [0.1, 0.8, 0.2];
            }
            let (dx, dy) = ((fx - 0.65) * w as f32, (fy - 0.72) * h as f32);
            if dx * dx + dy * dy < (0.18 * w.min(h) as f32).powi(2) {
                c = [0.95, 0.8, 0.1];
            }
            // Fine diagonal line pair on the right.
            let d = ((fx - 0.88) * w as f32 - (fy - 0.5) * h as f32 * 0.6).abs();
            if d < 1.0 && fy > 0.3 && fy < 0.9 {
                c = [0.05, 0.05, 0.05];
            }
            img.set_pixel(x, y, c);
        }
    }
    img
}

/// A grid of flat colour patches (neutrals, primaries, secondaries, skin-like tones).
pub fn color_patches(w: usize, h: usize) -> RgbImage {
    const PATCHES: [[f32; 3]; 24] = [
        [0.17, 0.10, 0.07],
        [0.58, 0.36, 0.28],
        [0.12, 0.20, 0.33],
        [0.10, 0.17, 0.07],
        [0.28, 0.24, 0.43],
        [0.14, 0.52, 0.42],
        [0.62, 0.26, 0.04],
        [0.06, 0.10, 0.34],
        [0.53, 0.09, 0.12],
        [0.10, 0.04, 0.13],
        [0.37, 0.52, 0.06],
        [0.65, 0.42, 0.03],
        [0.03, 0.05, 0.30],
        [0.06, 0.28, 0.06],
        [0.45, 0.04, 0.04],
        [0.78, 0.60, 0.01],
        [0.52, 0.10, 0.28],
        [0.00, 0.22, 0.36],
        [0.90, 0.90, 0.90],
        [0.59, 0.59, 0.59],
        [0.36, 0.36, 0.36],
        [0.20, 0.20, 0.20],
        [0.09, 0.09, 0.09],
        [0.02, 0.02, 0.02],
    ];
    let mut img = RgbImage::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let px = (x * 6 / w).min(5);
            let py = (y * 4 / h).min(3);
            img.set_pixel(x, y, PATCHES[py * 6 + px]);
        }
    }
    img
}

/// 5x7 glyphs (rows top to bottom, 5 bits each, MSB left) for the text test.
fn glyph(c: char) -> [u8; 7] {
    match c {
        '0' => [
            0b01110, 0b10001, 0b10011, 0b10101, 0b11001, 0b10001, 0b01110,
        ],
        '1' => [
            0b00100, 0b01100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110,
        ],
        '2' => [
            0b01110, 0b10001, 0b00001, 0b00010, 0b00100, 0b01000, 0b11111,
        ],
        '3' => [
            0b11110, 0b00001, 0b00001, 0b01110, 0b00001, 0b00001, 0b11110,
        ],
        '4' => [
            0b00010, 0b00110, 0b01010, 0b10010, 0b11111, 0b00010, 0b00010,
        ],
        '5' => [
            0b11111, 0b10000, 0b11110, 0b00001, 0b00001, 0b10001, 0b01110,
        ],
        '6' => [
            0b00110, 0b01000, 0b10000, 0b11110, 0b10001, 0b10001, 0b01110,
        ],
        '7' => [
            0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b01000, 0b01000,
        ],
        '8' => [
            0b01110, 0b10001, 0b10001, 0b01110, 0b10001, 0b10001, 0b01110,
        ],
        '9' => [
            0b01110, 0b10001, 0b10001, 0b01111, 0b00001, 0b00010, 0b01100,
        ],
        'Y' => [
            0b10001, 0b10001, 0b01010, 0b00100, 0b00100, 0b00100, 0b00100,
        ],
        'E' => [
            0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b11111,
        ],
        'L' => [
            0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b11111,
        ],
        'O' => [
            0b01110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110,
        ],
        'W' => [
            0b10001, 0b10001, 0b10001, 0b10101, 0b10101, 0b11011, 0b10001,
        ],
        'M' => [
            0b10001, 0b11011, 0b10101, 0b10101, 0b10001, 0b10001, 0b10001,
        ],
        'N' => [
            0b10001, 0b11001, 0b10101, 0b10011, 0b10001, 0b10001, 0b10001,
        ],
        _ => [0; 7],
    }
}

/// Fine text: lines of "YELLOWMOON 0123456789" in dark ink at several scales and colours on a
/// light background, anti-aliased by 3x3 supersampling. `scale` is the glyph pixel size of the
/// first line (the next lines use `scale + 1`, ...).
pub fn fine_text(w: usize, h: usize, scale: usize) -> RgbImage {
    let mut img = RgbImage::new(w, h);
    img.data.fill(0.9);
    let inks = [
        [0.05, 0.05, 0.05],
        [0.7, 0.05, 0.05],
        [0.05, 0.2, 0.7],
        [0.05, 0.5, 0.1],
    ];
    let text: Vec<char> = "YELLOWMOON 0123456789 ".chars().collect();
    let mut y0 = 3usize;
    let mut line = 0usize;
    while y0 + 7 * scale + 3 < h {
        let s = scale + line / 2;
        let ink = inks[line % inks.len()];
        let mut x0 = 3usize;
        let mut ci = line;
        while x0 + 5 * s < w {
            let g = glyph(text[ci % text.len()]);
            for (gy, row) in g.iter().enumerate() {
                for gx in 0..5 {
                    if row >> (4 - gx) & 1 == 1 {
                        for py in 0..s {
                            for px in 0..s {
                                let (x, y) = (x0 + gx * s + px, y0 + gy * s + py);
                                if x < w && y < h {
                                    img.set_pixel(x, y, ink);
                                }
                            }
                        }
                    }
                }
            }
            x0 += 6 * s;
            ci += 1;
        }
        y0 += 9 * s;
        line += 1;
        if line > 12 {
            break;
        }
    }
    // Anti-alias with a 3x3 box (equivalent to supersampling the hard glyph rasters at 1/3 px).
    box_blur3(&img)
}

fn box_blur3(img: &RgbImage) -> RgbImage {
    let (w, h) = (img.width, img.height);
    let mut out = RgbImage::new(w, h);
    for c in 0..3 {
        let src = img.plane(c);
        let dst = out.plane_mut(c);
        for y in 0..h {
            for x in 0..w {
                let mut s = 0.0;
                for dy in -1isize..=1 {
                    for dx in -1isize..=1 {
                        let xx = (x as isize + dx).clamp(0, w as isize - 1) as usize;
                        let yy = (y as isize + dy).clamp(0, h as isize - 1) as usize;
                        s += src[yy * w + xx];
                    }
                }
                dst[y * w + x] = s / 9.0;
            }
        }
    }
    out
}

/// A natural-looking scene: smooth colour fields, a few hundred soft-edged shapes and a trace of
/// noise (sigma 0.002), in `[0, 1]`. Cheap enough for tens of megapixels (`O(shapes + pixels)`).
pub fn natural(w: usize, h: usize, seed: u64) -> RgbImage {
    let mut rng = Rng::new(seed);
    let mut img = RgbImage::new(w, h);
    let grey0 = rng.range(0.25, 0.55);
    let base: [f32; 3] = std::array::from_fn(|_| grey0 + rng.range(-0.08, 0.08));
    let grad: [[f32; 2]; 3] = std::array::from_fn(|_| [rng.range(-0.3, 0.3), rng.range(-0.3, 0.3)]);
    for y in 0..h {
        for x in 0..w {
            let (fx, fy) = (x as f32 / w as f32 - 0.5, y as f32 / h as f32 - 0.5);
            let p: [f32; 3] = std::array::from_fn(|c| base[c] + grad[c][0] * fx + grad[c][1] * fy);
            img.set_pixel(x, y, p);
        }
    }
    let shapes = 30 + (w * h) / 40_000;
    for _ in 0..shapes.min(600) {
        let cx = rng.range(0.0, w as f32);
        let cy = rng.range(0.0, h as f32);
        let rx = rng.range(3.0, (w as f32 * 0.12).max(6.0));
        let ry = rx * rng.range(0.4, 1.6);
        // Photographic channel correlation: a grey level with a modest tint, not three free colours.
        let grey = rng.range(0.05, 0.95);
        let col: [f32; 3] =
            std::array::from_fn(|_| (grey + rng.range(-0.15, 0.15)).clamp(0.02, 0.98));
        let soft = rng.range(0.4, 2.5);
        let (x0, x1) = (
            (cx - rx * 2.0).max(0.0) as usize,
            ((cx + rx * 2.0) as usize + 1).min(w),
        );
        let (y0, y1) = (
            (cy - ry * 2.0).max(0.0) as usize,
            ((cy + ry * 2.0) as usize + 1).min(h),
        );
        for y in y0..y1 {
            for x in x0..x1 {
                let d = (((x as f32 - cx) / rx).powi(2) + ((y as f32 - cy) / ry).powi(2)).sqrt();
                let a = ((1.0 - d) * rx.min(ry) / soft + 0.5).clamp(0.0, 1.0);
                if a > 0.0 {
                    let px = img.pixel(x, y);
                    img.set_pixel(
                        x,
                        y,
                        std::array::from_fn(|c| px[c] * (1.0 - a) + col[c] * a),
                    );
                }
            }
        }
    }
    for v in img.data.iter_mut() {
        *v = (*v + 0.002 * rng.gaussian()).clamp(0.0, 1.0);
    }
    img
}

/// A cheap mosaic for performance runs: a smooth gradient plus uniform noise, in `[0, 0.9]`.
pub fn noise_mosaic(w: usize, h: usize, cfa: Cfa, seed: u64) -> Mosaic {
    let mut m = Mosaic::new(w, h, cfa);
    let mut rng = Rng::new(seed);
    for y in 0..h {
        let fy = y as f32 / h as f32;
        let row = &mut m.data[y * w..(y + 1) * w];
        for (x, v) in row.iter_mut().enumerate() {
            *v = 0.1 + 0.6 * (x as f32 / w as f32) * (1.0 - 0.5 * fy) + 0.2 * rng.next_f32();
        }
    }
    m
}

/// Adds Gaussian noise of standard deviation `sigma` to every sample.
pub fn add_noise(data: &mut [f32], sigma: f32, seed: u64) {
    let mut rng = Rng::new(seed);
    for v in data {
        *v += sigma * rng.gaussian();
    }
}

/// An anti-aliased line grid evaluated analytically at continuous position `(x, y)`: dark
/// Gaussian-profile lines of standard deviation `line_sigma_px` every `spacing_px`, offset by
/// `offset_px`, on a light background. Returns 1 (background) .. about 0.1 (line crossing).
pub fn grid_value(x: f64, y: f64, spacing_px: f64, offset_px: f64, line_sigma_px: f64) -> f64 {
    let dist = |t: f64| {
        let u = (t - offset_px).rem_euclid(spacing_px);
        u.min(spacing_px - u)
    };
    let g = |d: f64| (-0.5 * (d / line_sigma_px).powi(2)).exp();
    let line = g(dist(x)).max(g(dist(y)));
    1.0 - 0.9 * line
}

/// Like [`grid_value`] but only the lines along one axis: vertical lines (varying `x`) when
/// `vertical` is true, horizontal lines (varying `y`) otherwise. Used where line crossings would
/// disturb a centroid tracker.
pub fn lines_value(
    x: f64,
    y: f64,
    vertical: bool,
    spacing_px: f64,
    offset_px: f64,
    line_sigma_px: f64,
) -> f64 {
    let t = if vertical { x } else { y };
    let u = (t - offset_px).rem_euclid(spacing_px);
    let d = u.min(spacing_px - u);
    1.0 - 0.9 * (-0.5 * (d / line_sigma_px).powi(2)).exp()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rng_is_deterministic_and_in_range() {
        let mut a = Rng::new(7);
        let mut b = Rng::new(7);
        for _ in 0..100 {
            let (x, y) = (a.next_f32(), b.next_f32());
            assert_eq!(x, y);
            assert!((0.0..1.0).contains(&x));
        }
        let mut g = Rng::new(1);
        let n = 20000;
        let (mut s, mut s2) = (0.0f64, 0.0f64);
        for _ in 0..n {
            let v = f64::from(g.gaussian());
            s += v;
            s2 += v * v;
        }
        assert!((s / n as f64).abs() < 0.03 && (s2 / n as f64 - 1.0).abs() < 0.05);
    }

    #[test]
    fn generators_stay_in_range_and_are_deterministic() {
        for img in [
            zone_plate(64, 48),
            edges(64, 48),
            color_patches(64, 48),
            fine_text(96, 64, 1),
            natural(64, 48, 5),
        ] {
            assert!(img.data.iter().all(|v| (-1e-6..=1.0 + 1e-6).contains(v)));
        }
        assert_eq!(natural(40, 30, 9), natural(40, 30, 9));
        assert_ne!(natural(40, 30, 9), natural(40, 30, 10));
    }

    #[test]
    fn mosaic_samples_the_right_channel() {
        let img = color_patches(16, 16);
        let m = mosaic_from_rgb(&img, Cfa::GBRG);
        for y in 0..16 {
            for x in 0..16 {
                assert_eq!(m.get(x, y), img.pixel(x, y)[m.color_at(x, y).plane()]);
            }
        }
    }

    #[test]
    fn grid_lines_are_dark_on_the_line_and_light_between() {
        assert!(grid_value(50.0, 17.3, 50.0, 0.0, 1.0) < 0.2);
        assert!(grid_value(25.0, 25.0, 50.0, 0.0, 1.0) > 0.99);
    }
}
