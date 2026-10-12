//! Void-and-cluster blue-noise generator (Ulichney, 1993) for the 64 x 64 dither texture.
//!
//! The generator is deterministic and uses integer arithmetic for everything that decides a tie:
//! the Gaussian kernel (sigma = 1.5, radius 8, wrapped on the torus) is rounded to integer weights
//! once, energies are `i64` sums, ties go to the lowest index, and the start pattern comes from a
//! SplitMix64 stream with a fixed seed. The shipped texture (`encode/blue_noise_data.rs`) is the
//! output of [`generate`]; a test regenerates it and compares, so the table is reproducible but
//! nothing is generated at run time.

/// Side of the texture in texels.
pub const SIDE: usize = 64;
/// Number of texels (and of distinct ranks).
pub const TEXELS: usize = SIDE * SIDE;
/// Seed of the start pattern: the ASCII bytes of "YELLOWMO".
pub const SEED: u64 = 0x5945_4c4c_4f57_4d4f;

const SIGMA: f64 = 1.5;
const RADIUS: i32 = 8;
const WEIGHT_SCALE: f64 = 1_048_576.0; // 2^20

/// SplitMix64: a tiny, well-known deterministic generator (public domain).
struct SplitMix64(u64);

impl SplitMix64 {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
}

/// Integer Gaussian kernel, `(2 RADIUS + 1)^2` weights.
fn kernel() -> Vec<(i32, i32, i64)> {
    let mut taps = Vec::new();
    for dy in -RADIUS..=RADIUS {
        for dx in -RADIUS..=RADIUS {
            let d2 = f64::from(dx * dx + dy * dy);
            let w = ((-d2 / (2.0 * SIGMA * SIGMA)).exp() * WEIGHT_SCALE).round() as i64;
            if w > 0 {
                taps.push((dx, dy, w));
            }
        }
    }
    taps
}

struct Field {
    ones: Vec<bool>,
    energy: Vec<i64>,
    taps: Vec<(i32, i32, i64)>,
}

impl Field {
    fn new(ones: Vec<bool>) -> Self {
        let mut f = Self {
            energy: vec![0; TEXELS],
            ones: vec![false; TEXELS],
            taps: kernel(),
        };
        for (i, &on) in ones.iter().enumerate() {
            if on {
                f.set(i, true);
            }
        }
        f
    }

    fn splat(&mut self, index: usize, sign: i64) {
        let (x, y) = ((index % SIDE) as i32, (index / SIDE) as i32);
        let side = SIDE as i32;
        for &(dx, dy, w) in &self.taps {
            let xx = (x + dx).rem_euclid(side) as usize;
            let yy = (y + dy).rem_euclid(side) as usize;
            self.energy[yy * SIDE + xx] += sign * w;
        }
    }

    fn set(&mut self, index: usize, on: bool) {
        if self.ones[index] != on {
            self.ones[index] = on;
            self.splat(index, if on { 1 } else { -1 });
        }
    }

    /// The set texel with the highest energy (the tightest cluster); lowest index wins ties.
    fn tightest_cluster(&self) -> Option<usize> {
        let mut best: Option<(i64, usize)> = None;
        for i in 0..TEXELS {
            if self.ones[i] && best.is_none_or(|(e, _)| self.energy[i] > e) {
                best = Some((self.energy[i], i));
            }
        }
        best.map(|(_, i)| i)
    }

    /// The unset texel with the lowest energy (the largest void); lowest index wins ties.
    fn largest_void(&self) -> Option<usize> {
        let mut best: Option<(i64, usize)> = None;
        for i in 0..TEXELS {
            if !self.ones[i] && best.is_none_or(|(e, _)| self.energy[i] < e) {
                best = Some((self.energy[i], i));
            }
        }
        best.map(|(_, i)| i)
    }
}

/// Generates the 64 x 64 void-and-cluster rank texture: a permutation of `0..4096`, row-major.
#[must_use]
pub fn generate() -> Vec<u16> {
    let initial_ones = TEXELS / 10;

    // Phase 0: a seeded random start pattern with 10% ones (partial Fisher-Yates shuffle).
    let mut rng = SplitMix64(SEED);
    let mut order: Vec<usize> = (0..TEXELS).collect();
    for i in 0..initial_ones {
        let j = i + (rng.next() % (TEXELS - i) as u64) as usize;
        order.swap(i, j);
    }
    let mut start = vec![false; TEXELS];
    for &i in &order[..initial_ones] {
        start[i] = true;
    }

    // Phase 1: move ones from the tightest cluster to the largest void until it is stable.
    let mut field = Field::new(start);
    while let Some(cluster) = field.tightest_cluster() {
        field.set(cluster, false);
        let Some(void) = field.largest_void() else {
            break;
        };
        field.set(void, true);
        if void == cluster {
            break;
        }
    }
    let prototype = field.ones.clone();
    let ones = prototype.iter().filter(|&&b| b).count();

    let mut rank = vec![0_u16; TEXELS];

    // Phase 2: remove ones from the tightest cluster; the last one removed gets rank 0.
    let mut removal = Field::new(prototype.clone());
    for r in (0..ones).rev() {
        let Some(cluster) = removal.tightest_cluster() else {
            break;
        };
        removal.set(cluster, false);
        rank[cluster] = r as u16;
    }

    // Phase 3: add ones at the largest void, from the prototype, until the pattern is full.
    let mut fill = Field::new(prototype);
    for r in ones..TEXELS {
        let Some(void) = fill.largest_void() else {
            break;
        };
        rank[void] = r as u16;
        fill.set(void, true);
    }
    rank
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generator_produces_a_permutation() {
        let t = generate();
        assert_eq!(t.len(), TEXELS);
        let mut seen = vec![false; TEXELS];
        for &r in &t {
            assert!(!seen[usize::from(r)], "rank {r} repeated");
            seen[usize::from(r)] = true;
        }
    }
}

/// Writes the Rust source of the embedded table (used by the ignored `regenerate_table` test).
#[cfg(test)]
pub(crate) fn table_source() -> String {
    use std::fmt::Write as _;
    let t = generate();
    let mut s = String::new();
    s.push_str("//! The embedded 64 x 64 blue-noise rank texture (generated; do not edit).\n");
    s.push_str("//!\n");
    s.push_str("//! Produced by `encode::blue_noise::generate()` (void-and-cluster, sigma 1.5, seed `blue_noise::SEED`).\n");
    s.push_str("//! The test `embedded_table_matches_the_generator` fails if this file and the generator disagree.\n");
    s.push_str(
        "//! To regenerate: `cargo test -p ym-color --lib regenerate_table -- --ignored`.\n\n",
    );
    s.push_str("/// Ranks `0..4096`, row-major, each exactly once.\n");
    s.push_str("#[rustfmt::skip]\npub(super) const BLUE_NOISE_64: [u16; 4096] = [\n");
    for row in t.chunks(16) {
        s.push_str("   ");
        for r in row {
            let _ = write!(s, " {r},");
        }
        s.push('\n');
    }
    s.push_str("];\n");
    s
}

#[cfg(test)]
mod regenerate {
    #[test]
    #[ignore = "rewrites src/encode/blue_noise_data.rs from the generator"]
    fn regenerate_table() {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/src/encode/blue_noise_data.rs");
        std::fs::write(path, super::table_source()).expect("write the table");
    }
}
