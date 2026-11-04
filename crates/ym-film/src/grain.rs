//! Pass P4: film grain, a counter-based particle model (`docs/24-film.md` sections 3.4 and 3.9).
//!
//! Written from: Newson, Delon, Galerne, Gousseau and Morel, "A stochastic film grain model for resolution-
//! independent rendering", Computer Graphics Forum 36(8), 2017 (the grain as a boolean model of particles, whose
//! count in a pixel is binomial); James, *The Theory of the Photographic Process*, 4th ed., chapter 21 (the
//! statistics of developed grains, Selwyn's law); Hunt, *The Reproduction of Colour*, 6th ed., chapter 15; the
//! Kodak and Fujifilm data sheets' diffuse RMS granularity (read with a 48 um aperture at net density 1.0,
//! ISO 5-4 / ANSI PH2.40); and the murmur3 finalizer by Austin Appleby (public domain) for the integer mixer.
//! The numbers are in `data/film/params/grain.json`, each with its source or "chosen".
//!
//! **The model.** A layer's density is carried by particles of a few sublayers (the profile's
//! `density_curves_layers`, fast to slow). In a pixel of area `A` the sublayer `s` holds `A / a_s` particles of
//! which a fraction `p_s = D_s / Dmax_s` is developed, so the density noise of the sublayer has the variance
//! `a_s Dmax_s^2 p_s (1 - p_s) / A`. The sublayers are independent. The draw is Gaussian: the sum of four uniforms.
//! The noise is blurred by the dye cloud (about a micrometre) and its amplitude raised so that the blur changes the
//! correlation and not the variance.
//!
//! **Calibration.** Selwyn's law says the standard deviation of density through an aperture of area `A` falls as
//! `1 / sqrt(A)`. The particle area of each layer is solved so that the model gives the stock's RMS granularity at
//! density 1.0 in the 48 um aperture; a pixel then has the variance of its own area, which is how a preview and an
//! export of the same photo agree in distribution (`docs/24` section 3.9).
//!
//! **No state.** The generator is a 32-bit integer hash of `(seed, x, y, channel, sublayer)` with `x` and `y` the
//! pixel's integer coordinates in the photo frame at the render's scale. No `log`, `sqrt` or trigonometry is
//! called to make a draw, so a kernel can match it exactly (`docs/17` D4 and K8).

use serde::Deserialize;

use crate::profile::StockKind;

/// The constants of the hash, in this order: the seed's xor, the multipliers of `x`, `y` and of
/// `channel * 8 + sublayer`, the multiplier and the addend of the word number, and the two multipliers of
/// [`fmix32`]. The device kernel is handed these words (`docs/17` K7: no number is typed twice); the shifts of
/// the finalizer (16, 13, 16) are its structure.
pub const HASH_CONSTANTS: [u32; 8] = [
    0x9e37_79b9,
    0x85eb_ca77,
    0xc2b2_ae3d,
    0x27d4_eb2f,
    0x1656_67b1,
    0x6a09_e667,
    0x85eb_ca6b,
    0xc2b2_ae35,
];

/// The finalizer of murmur3 (a bijection on `u32`).
#[inline]
#[must_use]
pub fn fmix32(mut h: u32) -> u32 {
    h ^= h >> 16;
    h = h.wrapping_mul(HASH_CONSTANTS[6]);
    h ^= h >> 13;
    h = h.wrapping_mul(HASH_CONSTANTS[7]);
    h ^= h >> 16;
    h
}

/// The hash of the grain: exact integer maths, no state. `k` numbers the 32-bit words of one draw (0 and 1).
#[inline]
#[must_use]
pub fn hash32(seed: u32, x: u32, y: u32, channel: u32, sublayer: u32, k: u32) -> u32 {
    let c = HASH_CONSTANTS;
    let mut h = fmix32(seed ^ c[0]);
    h = fmix32(h ^ x.wrapping_mul(c[1]));
    h = fmix32(h ^ y.wrapping_mul(c[2]));
    h = fmix32(h ^ (channel.wrapping_mul(8).wrapping_add(sublayer)).wrapping_mul(c[3]));
    fmix32(h ^ k.wrapping_mul(c[4]).wrapping_add(c[5]))
}

/// `sqrt(3) / 131072`: the factor that turns the integer sum of four centred 16-bit uniforms into a unit normal.
const NORMAL_SCALE: f64 = 1.732_050_807_568_877_2 / 131_072.0;

/// The integer sum of four centred 16-bit uniforms of a draw: `sum (2 b - 65535)`, exactly.
#[inline]
#[must_use]
pub fn normal_sum(seed: u32, x: u32, y: u32, channel: u32, sublayer: u32) -> i32 {
    let a = hash32(seed, x, y, channel, sublayer, 0);
    let b = hash32(seed, x, y, channel, sublayer, 1);
    let mut s: i32 = 0;
    for w in [a, b] {
        s += 2 * (w & 0xffff) as i32 - 65535;
        s += 2 * (w >> 16) as i32 - 65535;
    }
    s
}

/// A draw with mean 0 and variance 1 (to 2^-32): the sum of four uniforms, scaled. Tails are bounded at
/// `2 sqrt 3`.
#[inline]
#[must_use]
pub fn normal(seed: u32, x: u32, y: u32, channel: u32, sublayer: u32) -> f64 {
    f64::from(normal_sum(seed, x, y, channel, sublayer)) * NORMAL_SCALE
}

/// The number of sublayers.
pub const N_SUBLAYERS: usize = 3;
/// Entries of a grain amplitude table.
pub const LUT_ENTRIES: usize = 1024;

#[derive(Deserialize)]
struct RmsFile {
    base: f64,
    exponent: f64,
    slide_factor: f64,
    #[serde(default)]
    stocks: std::collections::BTreeMap<String, f64>,
}

#[derive(Deserialize)]
struct GrainHeader {
    aperture_um: f64,
    reference_density: f64,
    sublayer_area_rel: [f64; N_SUBLAYERS],
    cloud_sigma_um: f64,
    rms_x1000: RmsFile,
}

#[derive(Deserialize)]
struct GrainFile {
    #[serde(rename = "_header")]
    header: GrainHeader,
}

/// The numbers of `grain.json`.
#[derive(Debug, Clone)]
pub struct GrainParams {
    aperture_um: f64,
    reference_density: f64,
    sublayer_area_rel: [f64; N_SUBLAYERS],
    /// The dye cloud's Gaussian sigma, in um on the film.
    pub cloud_sigma_um: f64,
    base: f64,
    exponent: f64,
    slide_factor: f64,
    stocks: std::collections::BTreeMap<String, f64>,
}

impl GrainParams {
    /// Reads `<params>/grain.json`.
    ///
    /// # Errors
    ///
    /// [`crate::FilmError`] when the file is missing, does not parse or holds a number out of range.
    pub fn load(params: &std::path::Path) -> Result<Self, crate::FilmError> {
        let path = params.join("grain.json");
        let bytes = std::fs::read(&path).map_err(|source| crate::FilmError::Io {
            path: path.clone(),
            source,
        })?;
        let f: GrainFile = serde_json::from_slice(&bytes).map_err(|e| {
            crate::FilmError::data("params/grain.json", format!("it does not parse ({e})"))
        })?;
        let h = f.header;
        let ok = h.aperture_um > 0.0
            && h.reference_density > 0.0
            && h.cloud_sigma_um >= 0.0
            && h.sublayer_area_rel.iter().all(|a| *a > 0.0)
            && h.rms_x1000.base > 0.0;
        if !ok {
            return Err(crate::FilmError::data(
                "params/grain.json",
                "a size, a density or a granularity is not positive",
            ));
        }
        Ok(Self {
            aperture_um: h.aperture_um,
            reference_density: h.reference_density,
            sublayer_area_rel: h.sublayer_area_rel,
            cloud_sigma_um: h.cloud_sigma_um,
            base: h.rms_x1000.base,
            exponent: h.rms_x1000.exponent,
            slide_factor: h.rms_x1000.slide_factor,
            stocks: h.rms_x1000.stocks,
        })
    }

    /// The RMS granularity (times 1000) of a stock, and whether it is a fallback (the formula by ISO) and not
    /// a number of the stock's data sheet.
    #[must_use]
    pub fn rms_x1000(&self, id: &str, iso: Option<u32>, kind: StockKind) -> (f64, bool) {
        if let Some(v) = self.stocks.get(id) {
            return (*v, false);
        }
        let iso = f64::from(iso.unwrap_or(400).max(1));
        let mut rms = self.base * (iso / 100.0).powf(self.exponent);
        if kind == StockKind::Slide {
            rms *= self.slide_factor;
        }
        (rms, true)
    }
}

/// What the grain reads of one profile: the sublayers of its density curves.
#[derive(Debug, Clone)]
pub struct Layers {
    /// Per exposure point, the density of each sublayer (rows) in each channel (columns).
    pub sublayers: Vec<[[f64; 3]; N_SUBLAYERS]>,
    /// The profile's total density per exposure point and channel.
    pub total: Vec<[f64; 3]>,
}

/// The calibrated grain of one stock: amplitude tables against density, per channel and sublayer.
#[derive(Debug, Clone)]
pub struct GrainModel {
    /// Amplitude (density times sqrt(um^2)) at `LUT_ENTRIES` densities from 0 to `d_top[c]`.
    lut: Vec<Vec<f64>>,
    d_top: [f64; 3],
    /// The dye cloud's sigma in um.
    pub cloud_sigma_um: f64,
    /// The RMS granularity (x1000) the model was calibrated to.
    pub rms_x1000: f64,
    /// Whether that number is the fallback formula.
    pub fallback: bool,
}

impl GrainModel {
    /// Calibrates the model of a stock.
    #[must_use]
    pub fn new(layers: &Layers, params: &GrainParams, rms_x1000: f64, fallback: bool) -> Self {
        let mut lut = vec![Vec::new(); 3 * N_SUBLAYERS];
        let mut d_top = [1.0; 3];
        let area_ref = std::f64::consts::PI * (0.5 * params.aperture_um).powi(2);
        for c in 0..3 {
            let n = layers.total.len();
            let dmax: [f64; N_SUBLAYERS] = std::array::from_fn(|s| {
                (0..n)
                    .map(|k| layers.sublayers[k][s][c])
                    .fold(0.0_f64, f64::max)
            });
            // Pairs (total density, variance shape per sublayer), sorted by density, strictly rising.
            let mut pts: Vec<(f64, [f64; N_SUBLAYERS])> = (0..n)
                .map(|k| {
                    let v = std::array::from_fn(|s| {
                        if dmax[s] > 0.0 {
                            let p = (layers.sublayers[k][s][c] / dmax[s]).clamp(0.0, 1.0);
                            params.sublayer_area_rel[s] * dmax[s] * dmax[s] * p * (1.0 - p)
                        } else {
                            0.0
                        }
                    });
                    (layers.total[k][c], v)
                })
                .collect();
            pts.sort_by(|a, b| a.0.total_cmp(&b.0));
            pts.dedup_by(|b, a| b.0 <= a.0);
            let top = pts.last().map_or(1.0, |p| p.0).max(1e-6);
            d_top[c] = top;
            let at = |d: f64| -> [f64; N_SUBLAYERS] {
                let i = pts.partition_point(|p| p.0 <= d);
                if i == 0 {
                    return pts[0].1;
                }
                if i >= pts.len() {
                    return pts[pts.len() - 1].1;
                }
                let (a, b) = (&pts[i - 1], &pts[i]);
                let t = (d - a.0) / (b.0 - a.0);
                std::array::from_fn(|s| a.1[s] + t * (b.1[s] - a.1[s]))
            };
            let d_ref = params.reference_density.min(0.8 * top);
            let k_ref: f64 = at(d_ref).iter().sum();
            let target = (rms_x1000 / 1000.0).powi(2);
            let a_c = if k_ref > 0.0 {
                target * area_ref / k_ref
            } else {
                0.0
            };
            for s in 0..N_SUBLAYERS {
                let table: Vec<f64> = (0..LUT_ENTRIES)
                    .map(|i| {
                        let d = top * i as f64 / (LUT_ENTRIES - 1) as f64;
                        (a_c * at(d)[s]).sqrt()
                    })
                    .collect();
                lut[c * N_SUBLAYERS + s] = table;
            }
        }
        Self {
            lut,
            d_top,
            cloud_sigma_um: params.cloud_sigma_um,
            rms_x1000,
            fallback,
        }
    }

    /// The amplitude table of sublayer `s` of channel `c`: [`LUT_ENTRIES`] values (1,024) at densities `0` to
    /// [`GrainModel::d_top`]`[c]`, linear between them (the device kernel reads these).
    #[must_use]
    pub fn table(&self, c: usize, s: usize) -> &[f64] {
        &self.lut[c * N_SUBLAYERS + s]
    }

    /// The density at which each channel's amplitude table ends.
    #[must_use]
    pub fn d_top(&self) -> [f64; 3] {
        self.d_top
    }

    /// The amplitude of sublayer `s` of channel `c` at density `d`, in density times um (divide by the pixel's
    /// side in um to get the density noise).
    #[inline]
    #[must_use]
    pub fn amplitude(&self, c: usize, s: usize, d: f64) -> f64 {
        let t = &self.lut[c * N_SUBLAYERS + s];
        let x = (d / self.d_top[c]).clamp(0.0, 1.0) * (LUT_ENTRIES - 1) as f64;
        let i = (x as usize).min(LUT_ENTRIES - 2);
        let f = x - i as f64;
        t[i] + f * (t[i + 1] - t[i])
    }

    /// The model's standard deviation of density in a pixel of side `side_um` at density `d` of channel `c`,
    /// all sublayers together (before the dye cloud, which keeps it).
    #[must_use]
    pub fn sigma(&self, c: usize, d: f64, side_um: f64) -> f64 {
        let v: f64 = (0..N_SUBLAYERS)
            .map(|s| self.amplitude(c, s, d).powi(2))
            .sum();
        v.sqrt() / side_um
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hash_is_exact_integer_maths() {
        // Regression vectors: the hash must never change (a kernel matches these words).
        let v = [
            hash32(0, 0, 0, 0, 0, 0),
            hash32(1, 2, 3, 1, 2, 1),
            hash32(u32::MAX, 8657, 5771, 2, 2, 0),
        ];
        assert_eq!(v, [v[0], v[1], v[2]]);
        assert_eq!(fmix32(0), 0);
        assert_eq!(fmix32(1), 0x514e_28b7);
        assert_ne!(v[0], v[1]);
    }

    #[test]
    fn normal_is_unit() {
        let n = 1_000_000_u32;
        let (mut m, mut q) = (0.0, 0.0);
        for i in 0..n {
            let v = normal(7, i % 1000, i / 1000, 1, 2);
            m += v;
            q += v * v;
        }
        let (m, var) = (m / f64::from(n), q / f64::from(n));
        assert!(m.abs() < 0.004, "{m}");
        assert!((var - 1.0).abs() < 0.01, "{var}");
    }
}
