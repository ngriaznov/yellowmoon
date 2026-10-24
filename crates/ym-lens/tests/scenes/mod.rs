//! Synthetic scenes with known blur, drawn analytically with a fixed seed, for the bokeh detector and L1.
//!
//! * [`subject_scene`]: a sharp textured subject with fine strands (hair) in front of a textured background
//!   made of three vertical bands blurred by discs of known radii.
//! * [`night_scene`]: bright flat discs of known radii on a dark noisy ground.
//! * [`sharp_scene`]: a textured scene with no blur at all.
//! * [`flat_scene`]: large flat patches, a blurred band and a sharp one.
//!
//! The textures are Voronoi mosaics (sharp straight edges, supersampled for anti-aliasing) plus very fine
//! noise, so the blurred layers have edges of every contrast and orientation to be read, like foliage or
//! buildings, and the disc blur is exact (area-weighted rows).
#![allow(dead_code, clippy::needless_range_loop)]

use std::f64::consts::PI;

/// A small deterministic generator (splitmix64).
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0xD1B5_4A32_D192_ED03)
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in [0, 1).
    pub fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    pub fn range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.unit()
    }

    /// Standard normal (Box-Muller).
    pub fn normal(&mut self) -> f64 {
        let u = self.unit().max(1e-12);
        let v = self.unit();
        (-2.0 * u.ln()).sqrt() * (2.0 * PI * v).cos()
    }
}

/// A position-hashed value in [0, 1).
fn hash01(x: i64, y: i64, salt: u64) -> f64 {
    let mut z = (x as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
        ^ (y as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9)
        ^ salt.wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 30;
    z = z.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z ^= z >> 27;
    z = z.wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    (z >> 11) as f64 / (1u64 << 53) as f64
}

/// The colour of a Voronoi mosaic with cells of about `cell` pixels at `(x, y)`.
fn mosaic(x: f64, y: f64, cell: f64, salt: u64) -> [f64; 3] {
    let (gx, gy) = ((x / cell).floor() as i64, (y / cell).floor() as i64);
    let mut best = (f64::INFINITY, 0i64, 0i64);
    for dy in -1..=1 {
        for dx in -1..=1 {
            let (cx, cy) = (gx + dx, gy + dy);
            let px = (cx as f64 + hash01(cx, cy, salt)) * cell;
            let py = (cy as f64 + hash01(cx, cy, salt ^ 0x55)) * cell;
            let d = (px - x).powi(2) + (py - y).powi(2);
            if d < best.0 {
                best = (d, cx, cy);
            }
        }
    }
    let (_, cx, cy) = best;
    let lum = 0.04 + 0.8 * hash01(cx, cy, salt ^ 0xA1).powf(1.5);
    let tint = |s: u64| 0.75 + 0.5 * hash01(cx, cy, salt ^ s);
    [lum * tint(0xB2), lum * tint(0xC3), lum * tint(0xD4)]
}

/// A planar RGB image.
#[derive(Clone)]
pub struct Rgb {
    pub w: usize,
    pub h: usize,
    pub data: Vec<f32>,
}

impl Rgb {
    pub fn new(w: usize, h: usize) -> Self {
        Rgb {
            w,
            h,
            data: vec![0.0; 3 * w * h],
        }
    }

    pub fn plane(&self, c: usize) -> &[f32] {
        &self.data[c * self.w * self.h..(c + 1) * self.w * self.h]
    }

    pub fn plane_mut(&mut self, c: usize) -> &mut [f32] {
        let n = self.w * self.h;
        &mut self.data[c * n..(c + 1) * n]
    }

    /// Luminance as the detector computes it.
    pub fn luminance(&self) -> Vec<f32> {
        let n = self.w * self.h;
        (0..n)
            .map(|i| 0.25 * self.data[i] + 0.5 * self.data[n + i] + 0.25 * self.data[2 * n + i])
            .collect()
    }
}

/// The texture of a band: a mosaic of cells of about `cell` px with 12 % pixel noise, supersampled 2 x 2.
/// Only columns `x0 .. x1` are drawn (the rest is 0).
pub fn texture(w: usize, h: usize, cell: f64, salt: u64, x0: usize, x1: usize) -> Rgb {
    use rayon::prelude::*;
    let mut img = Rgb::new(w, h);
    let n = w * h;
    let rows: Vec<Vec<[f32; 3]>> = (0..h)
        .into_par_iter()
        .map(|y| {
            (x0..x1.min(w))
                .map(|x| {
                    let mut acc = [0.0f64; 3];
                    for sy in 0..2 {
                        for sx in 0..2 {
                            let c = mosaic(
                                x as f64 + (sx as f64 + 0.5) / 2.0,
                                y as f64 + (sy as f64 + 0.5) / 2.0,
                                cell,
                                salt,
                            );
                            for k in 0..3 {
                                acc[k] += c[k] / 4.0;
                            }
                        }
                    }
                    let grain = 1.0 + 0.24 * (hash01(x as i64, y as i64, salt ^ 0x77) - 0.5);
                    [
                        (acc[0] * grain) as f32,
                        (acc[1] * grain) as f32,
                        (acc[2] * grain) as f32,
                    ]
                })
                .collect()
        })
        .collect();
    for (y, row) in rows.iter().enumerate() {
        for (i, px) in row.iter().enumerate() {
            for k in 0..3 {
                img.data[k * n + y * w + x0 + i] = px[k];
            }
        }
    }
    img
}

/// `img` blurred by a disc of radius `r` pixels, exactly (rows are area-weighted, the horizontal span of a row
/// has fractional ends). The border is replicated.
pub fn disc_blur(img: &[f32], w: usize, h: usize, r: f64) -> Vec<f32> {
    if r < 0.05 {
        return img.to_vec();
    }
    // Area of the disc in the row band [t0, t1] (the antiderivative of the chord).
    let g = |t: f64| -> f64 {
        let t = t.clamp(-r, r);
        t * (r * r - t * t).max(0.0).sqrt() + r * r * (t / r).asin()
    };
    let rows = r.ceil() as i64;
    let mut half_widths = Vec::new();
    let mut total = 0.0;
    for dy in -rows..=rows {
        let area = g(dy as f64 + 0.5) - g(dy as f64 - 0.5);
        half_widths.push(area / 2.0);
        total += area;
    }
    use rayon::prelude::*;
    // Prefix sums per row.
    let rows_prefix: Vec<Vec<f64>> = (0..h)
        .map(|y| {
            let mut p = vec![0.0f64; w + 1];
            for x in 0..w {
                p[x + 1] = p[x] + f64::from(img[y * w + x]);
            }
            p
        })
        .collect();
    let cumulative = |y: usize, t: f64| -> f64 {
        // Integral of row y from 0 to t (pixel j covers [j, j + 1)); beyond the ends the edge pixel repeats.
        let p = &rows_prefix[y];
        if t <= 0.0 {
            return t * f64::from(img[y * w]);
        }
        if t >= w as f64 {
            return p[w] + (t - w as f64) * f64::from(img[y * w + w - 1]);
        }
        let j = t.floor() as usize;
        p[j] + (t - j as f64) * f64::from(img[y * w + j])
    };
    let mut out = vec![0.0f32; w * h];
    out.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        for (x, o) in row.iter_mut().enumerate() {
            let mut acc = 0.0f64;
            for (k, hw) in half_widths.iter().enumerate() {
                let yy = (y as i64 + k as i64 - rows).clamp(0, h as i64 - 1) as usize;
                let c = x as f64 + 0.5;
                acc += cumulative(yy, c + hw) - cumulative(yy, c - hw);
            }
            *o = (acc / total) as f32;
        }
    });
    out
}

/// Gaussian-ish sensor noise at a level: `sigma(x) = sqrt(a x + b^2)`.
#[derive(Clone, Copy, Debug)]
pub struct Noise {
    pub shot: f64,
    pub read: f64,
}

impl Noise {
    pub const NONE: Noise = Noise {
        shot: 0.0,
        read: 0.0,
    };
    pub const LOW: Noise = Noise {
        shot: 2e-4,
        read: 0.002,
    };
    pub const MID: Noise = Noise {
        shot: 1.5e-3,
        read: 0.004,
    };
    pub const HIGH: Noise = Noise {
        shot: 8e-3,
        read: 0.008,
    };

    pub fn apply(&self, img: &mut Rgb, rng: &mut Rng) {
        if self.shot == 0.0 && self.read == 0.0 {
            return;
        }
        for v in &mut img.data {
            let x = f64::from(*v).max(0.0);
            let s = (self.shot * x + self.read * self.read).sqrt();
            *v = (x + s * rng.normal()).max(0.0) as f32;
        }
    }
}

/// A scene with the truth the detector is checked against.
#[derive(Clone)]
pub struct Scene {
    pub img: Rgb,
    /// The disc radius in pixels that blurred each pixel (0 for sharp pixels).
    pub truth_r: Vec<f32>,
    /// True where the truth is well defined (away from blur boundaries and the subject's outline).
    pub trusted: Vec<bool>,
    /// Sharp subject pixels, eroded by 2 px.
    pub subject: Vec<bool>,
    /// Pixels of a strand (coverage above one half).
    pub strand: Vec<bool>,
    /// Known discs: centre x, centre y, radius.
    pub discs: Vec<(f64, f64, f64)>,
    /// A focus point on the subject, as fractions.
    pub focus: Option<(f64, f64)>,
    /// The scene without its discs and noise, planar (night scenes only; empty otherwise).
    pub ground: Vec<f32>,
}

impl Scene {
    /// The same scene with sensor noise added.
    pub fn noisy(&self, noise: Noise, seed: u64) -> Scene {
        let mut s = self.clone();
        noise.apply(&mut s.img, &mut Rng::new(seed));
        s
    }
}

/// Polyline strands from the subject's top edge into the background: 1 to 1.6 px wide dark curves.
fn draw_strands(
    img: &mut Rgb,
    rng: &mut Rng,
    strand: &mut [bool],
    count: usize,
    from: &dyn Fn(f64) -> (f64, f64),
) {
    let (w, h) = (img.w, img.h);
    let n = w * h;
    for _ in 0..count {
        let t = rng.unit();
        let (x0, y0) = from(t);
        let angle = -PI / 2.0 + rng.range(-1.0, 1.0);
        let len = rng.range(40.0, 95.0);
        let bend = rng.range(-0.012, 0.012);
        let width = rng.range(1.0, 1.6);
        let tone = rng.range(0.01, 0.04);
        // Sample the curve densely and stamp anti-aliased dots.
        let steps = (len * 3.0) as usize;
        for s in 0..=steps {
            let u = s as f64 / 3.0;
            let a = angle + bend * u;
            let (px, py) = (x0 + u * a.cos(), y0 + u * a.sin());
            for yy in (py - 2.0).floor() as i64..=(py + 2.0).ceil() as i64 {
                for xx in (px - 2.0).floor() as i64..=(px + 2.0).ceil() as i64 {
                    if xx < 0 || yy < 0 || xx >= w as i64 || yy >= h as i64 {
                        continue;
                    }
                    let d =
                        ((xx as f64 + 0.5 - px).powi(2) + (yy as f64 + 0.5 - py).powi(2)).sqrt();
                    let cov = (0.5 + width / 2.0 - d).clamp(0.0, 1.0);
                    if cov > 0.0 {
                        let i = yy as usize * w + xx as usize;
                        for c in 0..3 {
                            let v = &mut img.data[c * n + i];
                            *v = (f64::from(*v) * (1.0 - cov) + tone * cov) as f32;
                        }
                        if cov > 0.5 {
                            strand[i] = true;
                        }
                    }
                }
            }
        }
    }
}

/// Erodes a mask by `r` pixels (square structuring element).
pub fn erode(mask: &[bool], w: usize, h: usize, r: i64) -> Vec<bool> {
    let mut out = vec![false; w * h];
    for y in 0..h as i64 {
        for x in 0..w as i64 {
            let mut all = true;
            'win: for dy in -r..=r {
                for dx in -r..=r {
                    let (nx, ny) = (x + dx, y + dy);
                    if nx < 0
                        || ny < 0
                        || nx >= w as i64
                        || ny >= h as i64
                        || !mask[(ny * w as i64 + nx) as usize]
                    {
                        all = false;
                        break 'win;
                    }
                }
            }
            out[(y * w as i64 + x) as usize] = all;
        }
    }
    out
}

/// Three background bands blurred by discs of `radii` pixels in front of nothing, a sharp textured subject with
/// strands in the middle. `w x h` is the frame (the lens plane at scale 1).
pub fn subject_scene(w: usize, h: usize, radii: [f64; 3], noise: Noise, seed: u64) -> Scene {
    let n = w * h;
    let mut rng = Rng::new(seed);
    let cuts = [w as f64 * 0.34, w as f64 * 0.68];
    let band_of = |x: f64| -> usize {
        if x < cuts[0] {
            0
        } else if x < cuts[1] {
            1
        } else {
            2
        }
    };
    let cells = [34.0, 28.0, 40.0];
    let mut bg = Rgb::new(w, h);
    let mut den = vec![0.0f32; n];
    for (i, &r) in radii.iter().enumerate() {
        let lo = if i == 0 { 0 } else { cuts[i - 1] as usize };
        let hi = if i == 2 { w } else { cuts[i] as usize + 1 };
        let mut tex = texture(w, h, cells[i], seed + 11 * i as u64, lo, hi);
        // Mask the texture to its band; the band's own mask is blurred with it so the layers normalise.
        let mask: Vec<f32> = (0..n)
            .map(|p| {
                if band_of((p % w) as f64 + 0.5) == i {
                    1.0
                } else {
                    0.0
                }
            })
            .collect();
        for c in 0..3 {
            let p = tex.plane_mut(c);
            for (v, m) in p.iter_mut().zip(&mask) {
                *v *= m;
            }
        }
        for c in 0..3 {
            let blurred = disc_blur(tex.plane(c), w, h, r);
            for (d, s) in bg.plane_mut(c).iter_mut().zip(&blurred) {
                *d += s;
            }
        }
        for (d, s) in den.iter_mut().zip(&disc_blur(&mask, w, h, r)) {
            *d += s;
        }
    }
    for c in 0..3 {
        for (v, d) in bg.plane_mut(c).iter_mut().zip(&den) {
            *v /= d.max(1e-6);
        }
    }
    // Truth radius and where it is trusted.
    let mut truth_r = vec![0.0f32; n];
    let mut trusted = vec![true; n];
    for y in 0..h {
        for x in 0..w {
            let b = band_of(x as f64 + 0.5);
            truth_r[y * w + x] = radii[b] as f32;
            for (k, &cut) in cuts.iter().enumerate() {
                let reach = radii[k].max(radii[k + 1]) * 2.0 + 6.0;
                if (x as f64 + 0.5 - cut).abs() < reach {
                    trusted[y * w + x] = false;
                }
            }
        }
    }
    // The subject: a sharp ellipse with its own finer mosaic and stripes.
    let (scx, scy, sax, say) = (
        w as f64 * 0.5,
        h as f64 * 0.72,
        w as f64 * 0.13,
        h as f64 * 0.42,
    );
    let subj_tex = texture(
        w,
        h,
        11.0,
        seed + 99,
        (scx - sax - 3.0).max(0.0) as usize,
        (scx + sax + 3.0) as usize,
    );
    let mut subject = vec![false; n];
    let mut alpha = vec![0.0f32; n];
    for y in 0..h {
        for x in 0..w {
            let (dx, dy) = ((x as f64 + 0.5 - scx) / sax, (y as f64 + 0.5 - scy) / say);
            let d = (dx * dx + dy * dy).sqrt();
            // Distance in pixels to the outline, approximately.
            let px = (d - 1.0) * (sax.min(say));
            let a = (0.5 - px).clamp(0.0, 1.0);
            alpha[y * w + x] = a as f32;
            if a >= 0.999 {
                subject[y * w + x] = true;
            }
        }
    }
    let mut img = bg.clone();
    for c in 0..3 {
        for i in 0..n {
            let (x, y) = (i % w, i / w);
            let stripe = 1.0 + 0.25 * ((x as f64 * 0.9 + y as f64 * 0.4).sin());
            let s = f64::from(subj_tex.data[c * n + i]) * stripe;
            let a = f64::from(alpha[i]);
            img.data[c * n + i] = (f64::from(bg.data[c * n + i]) * (1.0 - a) + s * a) as f32;
        }
    }
    // Pixels near the subject's outline are not trusted for the background's truth.
    let subj_dilated = {
        let inv: Vec<bool> = alpha.iter().map(|a| *a < 0.001).collect();
        erode(&inv, w, h, 10)
    };
    for i in 0..n {
        if !subj_dilated[i] {
            trusted[i] = false;
        }
    }
    let mut strand = vec![false; n];
    // Strands start on the upper half of the outline.
    let start = move |t: f64| -> (f64, f64) {
        let a = PI * (1.1 + 0.8 * t);
        (scx + sax * a.cos(), scy + say * a.sin())
    };
    draw_strands(&mut img, &mut rng, &mut strand, 70, &start);
    // Strand pixels and their neighbourhood are not trusted for the background's truth either.
    let strand_near = {
        let inv: Vec<bool> = strand.iter().map(|s| !*s).collect();
        erode(&inv, w, h, 4)
    };
    for i in 0..n {
        if !strand_near[i] {
            trusted[i] = false;
        }
    }
    for i in 0..n {
        if alpha[i] > 0.5 || strand[i] {
            truth_r[i] = 0.0;
        }
    }
    noise.apply(&mut img, &mut rng);
    Scene {
        img,
        truth_r,
        trusted,
        subject: erode(&subject, w, h, 2),
        strand,
        discs: Vec::new(),
        focus: Some((0.5, 0.75)),
        ground: Vec::new(),
    }
}

/// Bright flat discs of the given radii on a dark ground.
pub fn night_scene(w: usize, h: usize, radii: &[f64], noise: Noise, seed: u64) -> Scene {
    night_scene_spaced(w, h, radii, noise, seed, 2.0)
}

/// [`night_scene`] with the discs at least `spacing` times the sum of their radii apart, centre to centre.
pub fn night_scene_spaced(
    w: usize,
    h: usize,
    radii: &[f64],
    noise: Noise,
    seed: u64,
    spacing: f64,
) -> Scene {
    let n = w * h;
    let mut rng = Rng::new(seed);
    let mut img = Rgb::new(w, h);
    // Ground: dark with a slow gradient.
    for y in 0..h {
        for x in 0..w {
            let g = 0.004 + 0.004 * (y as f64 / h as f64);
            for c in 0..3 {
                img.data[c * n + y * w + x] = (g * (0.8 + 0.2 * c as f64)) as f32;
            }
        }
    }
    let ground = img.data.clone();
    let mut discs: Vec<(f64, f64, f64)> = Vec::new();
    let mut tries = 0;
    for &r in radii {
        loop {
            tries += 1;
            assert!(tries < 100_000, "cannot place the discs");
            let margin = 3.0 * r + 4.0;
            let (cx, cy) = (
                rng.range(margin, w as f64 - margin),
                rng.range(margin, h as f64 - margin),
            );
            if discs.iter().all(|&(ox, oy, or)| {
                ((ox - cx).powi(2) + (oy - cy).powi(2)).sqrt() > spacing * (or + r) + 6.0
            }) {
                discs.push((cx, cy, r));
                break;
            }
        }
    }
    for (k, &(cx, cy, r)) in discs.iter().enumerate() {
        let level = 0.4 * 2f64.powf(rng.range(0.0, 4.0));
        let tint = [
            1.0,
            0.7 + 0.3 * hash01(k as i64, 1, 5),
            0.4 + 0.5 * hash01(k as i64, 2, 5),
        ];
        let (x0, x1) = (
            (cx - r - 2.0).floor().max(0.0) as usize,
            ((cx + r + 2.0).ceil() as usize).min(w),
        );
        let (y0, y1) = (
            (cy - r - 2.0).floor().max(0.0) as usize,
            ((cy + r + 2.0).ceil() as usize).min(h),
        );
        for y in y0..y1 {
            for x in x0..x1 {
                let d = ((x as f64 + 0.5 - cx).powi(2) + (y as f64 + 0.5 - cy).powi(2)).sqrt();
                // Anti-aliased coverage of the disc (a one pixel ramp).
                let cov = (r + 0.5 - d).clamp(0.0, 1.0);
                if cov > 0.0 {
                    for c in 0..3 {
                        let v = &mut img.data[c * n + y * w + x];
                        *v += (level * tint[c] * cov) as f32;
                    }
                }
            }
        }
    }
    let mut truth_r = vec![0.0f32; n];
    for &(cx, cy, r) in &discs {
        for y in 0..h {
            for x in 0..w {
                let d = ((x as f64 + 0.5 - cx).powi(2) + (y as f64 + 0.5 - cy).powi(2)).sqrt();
                if d <= r {
                    truth_r[y * w + x] = r as f32;
                }
            }
        }
    }
    noise.apply(&mut img, &mut rng);
    Scene {
        img,
        truth_r,
        trusted: vec![true; n],
        subject: vec![false; n],
        strand: vec![false; n],
        discs,
        focus: None,
        ground,
    }
}

/// A textured scene with no blur beyond the sampling.
pub fn sharp_scene(w: usize, h: usize, noise: Noise, seed: u64) -> Scene {
    let n = w * h;
    let mut rng = Rng::new(seed);
    let mut img = texture(w, h, 14.0, seed, 0, w);
    noise.apply(&mut img, &mut rng);
    Scene {
        img,
        truth_r: vec![0.0; n],
        trusted: vec![true; n],
        subject: vec![true; n],
        strand: vec![false; n],
        discs: Vec::new(),
        focus: None,
        ground: Vec::new(),
    }
}

/// Large flat patches: a flat left half, a flat bright right third and a middle band of mosaic blurred by 8 px.
pub fn flat_scene(w: usize, h: usize, noise: Noise, seed: u64) -> Scene {
    let n = w * h;
    let mut rng = Rng::new(seed);
    let mut img = Rgb::new(w, h);
    let tex = texture(w, h, 30.0, seed, 0, w);
    let blurred: Vec<Vec<f32>> = (0..3).map(|c| disc_blur(tex.plane(c), w, h, 8.0)).collect();
    let mut truth_r = vec![0.0f32; n];
    for y in 0..h {
        for x in 0..w {
            let fx = x as f64 / w as f64;
            for c in 0..3 {
                img.data[c * n + y * w + x] = if fx < 0.4 {
                    [0.25, 0.3, 0.2][c]
                } else if fx < 0.7 {
                    blurred[c][y * w + x]
                } else {
                    [0.7, 0.65, 0.6][c]
                };
            }
            if (0.4..0.7).contains(&fx) {
                truth_r[y * w + x] = 8.0;
            }
        }
    }
    noise.apply(&mut img, &mut rng);
    Scene {
        img,
        truth_r,
        trusted: vec![true; n],
        subject: vec![false; n],
        strand: vec![false; n],
        discs: Vec::new(),
        focus: None,
        ground: Vec::new(),
    }
}

/// A smooth, soft-edged disc (the subject) in a textured background blurred by 9 px everywhere, with the focus
/// point on the disc: the subject is in focus but has nothing sharp in it (smooth skin, a soft silhouette), so
/// only the focus point says so.
pub fn anchor_scene(w: usize, h: usize, noise: Noise, seed: u64) -> Scene {
    let n = w * h;
    let mut rng = Rng::new(seed);
    let tex = texture(w, h, 32.0, seed, 0, w);
    let mut img = Rgb::new(w, h);
    for c in 0..3 {
        img.plane_mut(c)
            .copy_from_slice(&disc_blur(tex.plane(c), w, h, 9.0));
    }
    let (cx, cy, r) = (w as f64 * 0.5, h as f64 * 0.5, h as f64 * 0.2);
    let mut subject = vec![false; n];
    for y in 0..h {
        for x in 0..w {
            let d = ((x as f64 + 0.5 - cx).powi(2) + (y as f64 + 0.5 - cy).powi(2)).sqrt();
            // A soft edge 24 px wide.
            let t = ((r + 12.0 - d) / 24.0).clamp(0.0, 1.0);
            let a = t * t * (3.0 - 2.0 * t);
            if a > 0.0 {
                // Smooth shading across the subject.
                let shade = 1.0 - 0.35 * (d / r).min(1.0).powi(2);
                for (c, tone) in [0.32, 0.27, 0.23].into_iter().enumerate() {
                    let v = &mut img.data[c * n + y * w + x];
                    *v = (f64::from(*v) * (1.0 - a) + tone * shade * a) as f32;
                }
            }
            subject[y * w + x] = d < r - 14.0;
        }
    }
    noise.apply(&mut img, &mut rng);
    Scene {
        img,
        truth_r: vec![9.0; n],
        trusted: vec![true; n],
        subject: erode(&subject, w, h, 4),
        strand: vec![false; n],
        discs: Vec::new(),
        focus: Some((0.5, 0.5)),
        ground: Vec::new(),
    }
}

/// A textured background blurred by `bg_radius` px with bright flat discs of the given radii on it.
pub fn balls_over_texture_scene(
    w: usize,
    h: usize,
    bg_radius: f64,
    radii: &[f64],
    noise: Noise,
    seed: u64,
) -> Scene {
    let n = w * h;
    let mut rng = Rng::new(seed);
    let tex = texture(w, h, 30.0, seed, 0, w);
    let mut img = Rgb::new(w, h);
    for c in 0..3 {
        img.plane_mut(c)
            .copy_from_slice(&disc_blur(tex.plane(c), w, h, bg_radius));
    }
    let mut discs: Vec<(f64, f64, f64)> = Vec::new();
    for &r in radii {
        loop {
            let margin = 3.0 * r + 4.0;
            let (cx, cy) = (
                rng.range(margin, w as f64 - margin),
                rng.range(margin, h as f64 - margin),
            );
            if discs.iter().all(|&(ox, oy, or)| {
                ((ox - cx).powi(2) + (oy - cy).powi(2)).sqrt() > 2.4 * (or + r) + 6.0
            }) {
                discs.push((cx, cy, r));
                break;
            }
        }
    }
    for &(cx, cy, r) in &discs {
        let (x0, x1) = (
            (cx - r - 2.0).floor().max(0.0) as usize,
            ((cx + r + 2.0).ceil() as usize).min(w),
        );
        let (y0, y1) = (
            (cy - r - 2.0).floor().max(0.0) as usize,
            ((cy + r + 2.0).ceil() as usize).min(h),
        );
        for y in y0..y1 {
            for x in x0..x1 {
                let d = ((x as f64 + 0.5 - cx).powi(2) + (y as f64 + 0.5 - cy).powi(2)).sqrt();
                let a = (r + 0.5 - d).clamp(0.0, 1.0);
                if a > 0.0 {
                    for (c, tone) in [4.0, 3.4, 2.6].into_iter().enumerate() {
                        let v = &mut img.data[c * n + y * w + x];
                        *v = (f64::from(*v) * (1.0 - a) + tone * a) as f32;
                    }
                }
            }
        }
    }
    noise.apply(&mut img, &mut rng);
    Scene {
        img,
        truth_r: vec![bg_radius as f32; n],
        trusted: vec![true; n],
        subject: vec![false; n],
        strand: vec![false; n],
        discs,
        focus: None,
        ground: Vec::new(),
    }
}
