//! Synthetic pictures for the tests (`docs/20-auto-crop.md` §13): scenes drawn analytically,
//! 4 × 4 samples per pixel, turned by a known roll, with a little fixed-seed noise.

/// Picture width of the tests: Apple Auto's picture is 768 px on the long edge (`docs/20` §4).
pub(crate) const W: usize = 768;
/// Picture height of the tests (3:2).
pub(crate) const H: usize = 512;

/// A xorshift64 generator with a fixed seed: the tests' only source of noise.
pub(crate) struct XorShift(u64);

impl XorShift {
    pub(crate) fn new(seed: u64) -> Self {
        Self(seed.max(1))
    }

    pub(crate) fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    /// Uniform in `[0, 1)`.
    pub(crate) fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Roughly normal with mean 0 and variance 1 (the sum of four uniforms).
    pub(crate) fn normal(&mut self) -> f64 {
        let s: f64 = (0..4).map(|_| self.uniform()).sum();
        (s - 2.0) * 3f64.sqrt()
    }
}

/// Renders `scene` on a `w × h` picture turned clockwise by `roll_deg` about its centre: picture
/// point `q` (relative to the centre) shows the scene at `R(−roll) q`, so a level line of the
/// scene lies at `+roll_deg` in the picture (y down). The scene takes coordinates in pixels
/// relative to the centre, x right and y down, and returns gamma-encoded display RGB in `[0, 1]`.
/// Gaussian noise of `noise` (in display values) is added per channel before 8-bit rounding.
pub(crate) fn render(
    w: usize,
    h: usize,
    roll_deg: f64,
    noise: f64,
    seed: u64,
    scene: impl Fn(f64, f64) -> [f64; 3],
) -> Vec<u8> {
    const SS: usize = 4;
    let (s, c) = roll_deg.to_radians().sin_cos();
    let (cx, cy) = (w as f64 / 2.0, h as f64 / 2.0);
    let mut rng = XorShift::new(seed);
    let mut out = Vec::with_capacity(w * h * 3);
    for y in 0..h {
        for x in 0..w {
            let mut acc = [0.0; 3];
            for sy in 0..SS {
                for sx in 0..SS {
                    let qx = x as f64 + (sx as f64 + 0.5) / SS as f64 - cx;
                    let qy = y as f64 + (sy as f64 + 0.5) / SS as f64 - cy;
                    let col = scene(c * qx + s * qy, -s * qx + c * qy);
                    for k in 0..3 {
                        acc[k] += col[k];
                    }
                }
            }
            for a in acc {
                let v = a / (SS * SS) as f64 + noise * rng.normal();
                out.push((v.clamp(0.0, 1.0) * 255.0).round() as u8);
            }
        }
    }
    out
}

/// Smooth value noise in about `[−1, 1]` on a unit lattice, hashed from `seed`.
pub(crate) fn value_noise(x: f64, y: f64, seed: u64) -> f64 {
    let hash = |i: i64, j: i64| {
        let mut v = (i as u64)
            .wrapping_mul(0x9E37_79B9_7F4A_7C15)
            .wrapping_add((j as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F))
            .wrapping_add(seed.wrapping_mul(0x1656_67B1_9E37_79F9));
        v ^= v >> 33;
        v = v.wrapping_mul(0xFF51_AFD7_ED55_8CCD);
        v ^= v >> 33;
        (v >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
    };
    let (i, j) = (x.floor(), y.floor());
    let (tx, ty) = (x - i, y - j);
    let (sx, sy) = (tx * tx * (3.0 - 2.0 * tx), ty * ty * (3.0 - 2.0 * ty));
    let (i, j) = (i as i64, j as i64);
    let top = hash(i, j) + sx * (hash(i + 1, j) - hash(i, j));
    let bottom = hash(i, j + 1) + sx * (hash(i + 1, j + 1) - hash(i, j + 1));
    top + sy * (bottom - top)
}

/// A clear sky with soft clouds.
pub(crate) fn sky(u: f64, v: f64) -> [f64; 3] {
    let t = 0.06 * value_noise(u / 70.0, v / 30.0, 7) + 0.03 * value_noise(u / 23.0, v / 11.0, 8);
    let g = -v / 2000.0;
    [0.62 + t - g, 0.76 + t - g, 0.93 + 0.5 * t]
}

/// The sea with low waves.
pub(crate) fn sea(u: f64, v: f64) -> [f64; 3] {
    let t = 0.03 * value_noise(u / 45.0, v / 5.0, 11) + 0.015 * value_noise(u / 13.0, v / 3.0, 12);
    [0.10 + t, 0.27 + t, 0.42 + 1.2 * t]
}

/// Where the sea's horizon lies in the open-sea scene, in pixels below the centre.
pub(crate) const SEA_HORIZON_V: f64 = 30.0;

/// An open sea: sky above a straight horizon [`SEA_HORIZON_V`] below the centre.
pub(crate) fn open_sea(u: f64, v: f64) -> [f64; 3] {
    if v < SEA_HORIZON_V {
        sky(u, v)
    } else {
        sea(u, v)
    }
}

/// A façade seen by a camera pitched up: a pinhole of `focal_px`, the wall 22 m away, windows on a
/// 3 m × 3.2 m grid. The building spans `left_m` to `right_m` across; beyond it is sky above the
/// horizon and, below, the sea when `sea_beyond`, else the ground.
pub(crate) fn facade(
    focal_px: f64,
    pitch_deg: f64,
    left_m: f64,
    right_m: f64,
    sea_beyond: bool,
) -> impl Fn(f64, f64) -> [f64; 3] {
    const DISTANCE_M: f64 = 22.0;
    const TOP_M: f64 = -9.0;
    const EYE_M: f64 = 1.6;
    const WALL: [f64; 3] = [0.78, 0.70, 0.58];
    const WINDOW: [f64; 3] = [0.16, 0.18, 0.22];
    const GROUND: [f64; 3] = [0.36, 0.34, 0.31];
    let (sp, cp) = pitch_deg.to_radians().sin_cos();
    move |u: f64, v: f64| {
        // The ray in world coordinates: x right, y down, z level and forward.
        let ry = v * cp - focal_px * sp;
        let rz = v * sp + focal_px * cp;
        if rz > 0.0 {
            let t = DISTANCE_M / rz;
            let (x, y) = (u * t, ry * t);
            if (left_m..=right_m).contains(&x) && (TOP_M..=EYE_M).contains(&y) {
                let cx = (x - left_m - 1.0).rem_euclid(3.0);
                let cy = (y - TOP_M - 1.0).rem_euclid(3.2);
                let in_window = cx < 1.4 && cy < 1.8 && x < right_m - 1.0 && y < EYE_M - 1.0;
                return if in_window { WINDOW } else { WALL };
            }
        }
        if ry < 0.0 {
            // Above the horizon: the same sky as the open sea, placed by its own coordinates.
            sky(u, v)
        } else if sea_beyond {
            sea(u, v)
        } else {
            GROUND
        }
    }
}

/// Pebbles: about 900 discs of 6 to 28 px in random greys and tints on a grey field, no straight
/// line anywhere.
pub(crate) fn pebbles(seed: u64) -> impl Fn(f64, f64) -> [f64; 3] {
    const BUCKET: f64 = 32.0;
    const SPAN: f64 = 560.0; // covers the turned picture
    let mut rng = XorShift::new(seed);
    let n_side = (2.0 * SPAN / BUCKET) as usize;
    let mut buckets: Vec<Vec<usize>> = vec![Vec::new(); n_side * n_side];
    let mut discs = Vec::new();
    for k in 0..900 {
        let x = -SPAN + 2.0 * SPAN * rng.uniform();
        let y = -SPAN + 2.0 * SPAN * rng.uniform();
        let r = 6.0 + 22.0 * rng.uniform();
        let g = 0.15 + 0.7 * rng.uniform();
        let tint = 0.1 * (rng.uniform() - 0.5);
        discs.push((x, y, r, [g + tint, g, g - tint]));
        let b0 = ((x - r + SPAN) / BUCKET).floor().max(0.0) as usize;
        let b1 = (((x + r + SPAN) / BUCKET).floor() as usize).min(n_side - 1);
        let c0 = ((y - r + SPAN) / BUCKET).floor().max(0.0) as usize;
        let c1 = (((y + r + SPAN) / BUCKET).floor() as usize).min(n_side - 1);
        for bj in c0..=c1 {
            for bi in b0..=b1 {
                buckets[bj * n_side + bi].push(k);
            }
        }
    }
    move |u: f64, v: f64| {
        let bi = ((u + SPAN) / BUCKET).floor();
        let bj = ((v + SPAN) / BUCKET).floor();
        if bi < 0.0 || bj < 0.0 || bi >= n_side as f64 || bj >= n_side as f64 {
            return [0.5; 3];
        }
        // The last disc drawn on top.
        let mut col = [0.5; 3];
        for &k in &buckets[bj as usize * n_side + bi as usize] {
            let (x, y, r, c) = discs[k];
            if (u - x) * (u - x) + (v - y) * (v - y) <= r * r {
                col = c;
            }
        }
        col
    }
}

/// A disc of the subject: orange with darker dots, `radius` px at `(x, y)` (centre-relative).
pub(crate) fn in_subject(u: f64, v: f64, x: f64, y: f64, radius: f64) -> Option<[f64; 3]> {
    let (du, dv) = (u - x, v - y);
    if du * du + dv * dv > radius * radius {
        return None;
    }
    let dot = (du / 9.0).rem_euclid(1.0) < 0.4 && (dv / 9.0).rem_euclid(1.0) < 0.4;
    Some(if dot {
        [0.60, 0.30, 0.08]
    } else {
        [0.95, 0.55, 0.15]
    })
}

/// A plain grey field with faint texture.
pub(crate) fn field(u: f64, v: f64) -> [f64; 3] {
    let t = 0.02 * value_noise(u / 40.0, v / 40.0, 21);
    [0.5 + t, 0.5 + t, 0.5 + t]
}
