//! Deterministic synthetic skin for the tests of the classical models: a smooth skin tone in
//! linear light, texture from a seeded generator, pores, and lesions with their pigments.

use super::grey::Grey;
use crate::types::LinearAwg4Image;

/// SplitMix64, a tiny seeded generator.
pub struct Rng(pub u64);

impl Rng {
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// Uniform in `[0, 1)`.
    pub fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Uniform in `[lo, hi)`.
    pub fn range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.unit()
    }
}

/// What a lesion is made of.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    /// Haemoglobin: green and blue down, red nearly unchanged.
    Papule,
    /// Light melanin spot at blemish scale (a post-acne mark): all channels down, red least.
    Brown,
    /// Dark brown mole.
    Mole,
}

/// A round lesion with a soft edge.
#[derive(Debug, Clone, Copy)]
pub struct Lesion {
    pub kind: Kind,
    pub cx: f64,
    pub cy: f64,
    /// Radius of the full-strength core in pixels.
    pub radius: f64,
}

impl Lesion {
    pub fn new(kind: Kind, cx: f64, cy: f64, radius: f64) -> Self {
        Self {
            kind,
            cx,
            cy,
            radius,
        }
    }

    /// Linear-light factor per channel at full strength.
    pub fn factors(&self) -> [f64; 3] {
        match self.kind {
            Kind::Papule => [0.97, 0.72, 0.80],
            Kind::Brown => [0.82, 0.68, 0.60],
            Kind::Mole => [0.50, 0.38, 0.30],
        }
    }

    /// Weight 0..1 at a distance from the centre: 1 inside 0.75 R, 0 beyond 1.25 R.
    pub fn weight(&self, dist: f64) -> f64 {
        let t = ((1.25 * self.radius - dist) / (0.5 * self.radius)).clamp(0.0, 1.0);
        t * t * (3.0 - 2.0 * t)
    }
}

/// Skin without lesions: tone, gradient, texture (white and 2 px grain, about 1.5 % together) and
/// 40 pores. The same seed gives the same skin, so the version with lesions differs only at them.
pub fn clean_skin(w: usize, h: usize, seed: u64) -> Vec<[f64; 3]> {
    clean_skin_tone(w, h, seed, [0.42, 0.30, 0.22])
}

/// [`clean_skin`] with another skin tone (the linear RGB of the mean skin).
pub fn clean_skin_tone(w: usize, h: usize, seed: u64, tone: [f64; 3]) -> Vec<[f64; 3]> {
    clean_skin_rough(w, h, seed, tone, 1.0, 40, 0.12)
}

/// [`clean_skin_tone`] with a texture `gain` (1 is the 1.5 % of the other functions), `pores`
/// pores and a pore `depth` (the fraction of light a pore's centre takes away).
pub fn clean_skin_rough(
    w: usize,
    h: usize,
    seed: u64,
    tone: [f64; 3],
    gain: f64,
    pores: usize,
    depth: f64,
) -> Vec<[f64; 3]> {
    let mut rng = Rng(seed);
    let white: Vec<f64> = (0..w * h).map(|_| rng.range(-1.0, 1.0)).collect();
    let grain_src: Vec<f64> = (0..w * h).map(|_| rng.range(-1.0, 1.0)).collect();
    let grain = Grey::new(w, h, grain_src).gaussian(2.0);
    let mut px: Vec<[f64; 3]> = (0..w * h)
        .map(|i| {
            let (x, y) = ((i % w) as f64, (i / w) as f64);
            let g = 1.0 + 0.04 * (x / w as f64 - 0.5) + 0.03 * (y / h as f64 - 0.5);
            let t = 1.0 + gain * (0.008 * white[i] + 0.16 * grain.data[i]);
            let f = g * t;
            [tone[0] * f, tone[1] * f, tone[2] * f]
        })
        .collect();
    for _ in 0..pores {
        let (cx, cy) = (
            rng.range(5.0, w as f64 - 5.0),
            rng.range(5.0, h as f64 - 5.0),
        );
        for y in (cy as usize).saturating_sub(3)..((cy as usize) + 4).min(h) {
            for x in (cx as usize).saturating_sub(3)..((cx as usize) + 4).min(w) {
                let d2 = (x as f64 - cx).powi(2) + (y as f64 - cy).powi(2);
                let wt = (-d2 / (2.0 * 0.8 * 0.8)).exp();
                for (c, f) in [1.0 - 0.67 * depth, 1.0 - depth, 1.0 - depth]
                    .iter()
                    .enumerate()
                {
                    px[y * w + x][c] *= 1.0 - wt * (1.0 - f);
                }
            }
        }
    }
    px
}

/// Applies lesions to a clean patch.
pub fn with_lesions(w: usize, h: usize, clean: &[[f64; 3]], lesions: &[Lesion]) -> LinearAwg4Image {
    let mut px = clean.to_vec();
    for l in lesions {
        let r = (1.3 * l.radius).ceil() as i64 + 1;
        for y in (l.cy as i64 - r).max(0)..(l.cy as i64 + r + 1).min(h as i64) {
            for x in (l.cx as i64 - r).max(0)..(l.cx as i64 + r + 1).min(w as i64) {
                let d = ((x as f64 - l.cx).powi(2) + (y as f64 - l.cy).powi(2)).sqrt();
                let wt = l.weight(d);
                for (c, f) in l.factors().iter().enumerate() {
                    px[y as usize * w + x as usize][c] *= 1.0 - wt * (1.0 - f);
                }
            }
        }
    }
    LinearAwg4Image {
        width_px: w as u32,
        height_px: h as u32,
        data: px
            .iter()
            .map(|p| [p[0] as f32, p[1] as f32, p[2] as f32])
            .collect(),
    }
}

/// Clean skin as an image.
pub fn skin_image(w: usize, h: usize, seed: u64, lesions: &[Lesion]) -> LinearAwg4Image {
    with_lesions(w, h, &clean_skin(w, h, seed), lesions)
}

/// Random lesions at least `gap` pixels apart and `margin` from the borders: `counts` of
/// (papules, brown spots, moles) with radii in the given ranges (papule, brown, mole).
pub fn random_lesions(
    rng: &mut Rng,
    w: usize,
    h: usize,
    counts: (usize, usize, usize),
    radii: [(f64, f64); 3],
    margin: f64,
    gap: f64,
) -> Vec<Lesion> {
    let mut out: Vec<Lesion> = Vec::new();
    for (kind, n, (lo, hi)) in [
        (Kind::Mole, counts.2, radii[2]),
        (Kind::Papule, counts.0, radii[0]),
        (Kind::Brown, counts.1, radii[1]),
    ] {
        let mut placed = 0;
        let mut tries = 0;
        while placed < n && tries < 2000 {
            tries += 1;
            let l = Lesion::new(
                kind,
                rng.range(margin, w as f64 - margin),
                rng.range(margin, h as f64 - margin),
                rng.range(lo, hi),
            );
            if out
                .iter()
                .all(|o| ((o.cx - l.cx).powi(2) + (o.cy - l.cy).powi(2)).sqrt() >= gap)
            {
                out.push(l);
                placed += 1;
            }
        }
    }
    out
}
