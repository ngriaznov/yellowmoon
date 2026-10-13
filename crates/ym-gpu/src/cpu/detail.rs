//! R7 detail controls: colour noise and detail re-injection (`docs/05-raw-processing.md` section R7).
//!
//! `noise_reduction` resolves to `{ luminance L, color C, detail D }`, each 0 to 100. The Auto
//! values are `L_auto = clamp(10 * (1 + log2(ISO / 100)), 10, 90)`, `C_auto = 50`, `D_auto = 50`.
//!
//! | Control | Operation here |
//! |---|---|
//! | Luminance `L` | [`network_sigma`]: the σ fed to the network is `σ_est * (L / L_auto)^1.5` (0 at `L = 0`) |
//! | Colour `C` | [`chroma_guided_filter`]: guided filter (He et al.) on `R - Y` and `B - Y`, guided by `Y`, box radius `round(2 + 6 C / 100)` px, `eps = σ_est^2 * (C / 50)^2` (per pixel when σ is a map) |
//! | Detail `D` | [`inject_detail`]: `Y += beta * (Y_classical - Y_net)`, `beta = 0.15 * (D / 50)^2` |
//!
//! The luma / chroma split uses [`LUMA`]. Both operations add a *luma-only* delta or touch only the
//! chroma planes, so the other half of the opponent space is left exactly as it was.
//!
//! Two clarifications of the spec (recorded in `docs/05`):
//!
//! * `C = 0` returns the input untouched. The literal formula would give `eps = 0` and a guided
//!   filter with `eps = 0` still smooths noise that is uncorrelated with the guide, which is the
//!   opposite of "no colour noise reduction".
//! * The guided filter applies two box filters in series (statistics, then the smoothed `a`, `b`),
//!   so its receptive field is `2 * radius`, not `radius`: [`guided_halo_px`].

use super::filters::{self, BAND_ROWS, box_mean};
use super::{CpuError, LUMA, NoiseSigma, RgbImage};

/// The Auto luminance strength for an ISO: `clamp(10 * (1 + log2(ISO / 100)), 10, 90)`.
pub fn l_auto(iso: f32) -> f32 {
    (10.0 * (1.0 + (iso.max(1.0) / 100.0).log2())).clamp(10.0, 90.0)
}

/// The Auto colour strength.
pub const C_AUTO: f32 = 50.0;
/// The Auto detail strength.
pub const D_AUTO: f32 = 50.0;

/// Resolved noise-reduction controls, each 0 to 100.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NoiseReduction {
    /// Luminance strength.
    pub luminance: f32,
    /// Colour strength.
    pub color: f32,
    /// Detail re-injection strength.
    pub detail: f32,
}

impl NoiseReduction {
    /// The Auto values for an ISO (`docs/05` R7). The Automatic card shows these.
    pub fn auto(iso: f32) -> NoiseReduction {
        NoiseReduction {
            luminance: l_auto(iso),
            color: C_AUTO,
            detail: D_AUTO,
        }
    }

    /// The re-injection weight `beta = 0.15 * (D / 50)^2`.
    pub fn beta(&self) -> f32 {
        0.15 * (self.detail / 50.0).powi(2)
    }

    /// Whether the classical demosaic is needed (`beta > 0`).
    pub fn needs_classical(&self) -> bool {
        self.beta() > 0.0
    }

    /// Box radius of the colour guided filter in pixels: `round(2 + 6 C / 100)`.
    pub fn chroma_radius_px(&self) -> usize {
        (2.0 + 6.0 * self.color / 100.0).round() as usize
    }
}

/// The multiplier of the σ fed to the network: `(L / L_auto)^1.5`.
pub fn network_sigma_scale(luminance: f32, iso: f32) -> f32 {
    (luminance / l_auto(iso)).powf(1.5)
}

/// The σ map for the network: the estimated σ scaled by the luminance control.
pub fn network_sigma(sigma_est: &NoiseSigma, len: usize, luminance: f32, iso: f32) -> Vec<f32> {
    sigma_est.to_vec(len, network_sigma_scale(luminance, iso))
}

/// Receptive-field radius of [`chroma_guided_filter`] for a box radius: two box filters in series.
pub const fn guided_halo_px(radius_px: usize) -> usize {
    2 * radius_px
}

/// Guided filter (He et al.) of the planes `ps` with the shared guide `i` (all `w x h`), box
/// radius `r`, per-pixel `eps`. The guide statistics are computed once for all planes. Sequential:
/// the band runner provides the parallelism.
fn guided<const N: usize>(
    i: &[f32],
    ps: [&[f32]; N],
    eps: &[f32],
    w: usize,
    h: usize,
    r: usize,
) -> [Vec<f32>; N] {
    let n = w * h;
    let (mut tmp, mut mean_i, mut corr_i) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
    box_mean(i, w, h, r, &mut mean_i, &mut tmp);
    let ii: Vec<f32> = i.iter().map(|v| v * v).collect();
    box_mean(&ii, w, h, r, &mut corr_i, &mut tmp);
    drop(ii);
    let inv_denominator: Vec<f32> = (0..n)
        .map(|k| 1.0 / ((corr_i[k] - mean_i[k] * mean_i[k]).max(0.0) + eps[k]))
        .collect();
    drop(corr_i);
    ps.map(|p| {
        let (mut mean_p, mut corr_ip) = (vec![0.0; n], vec![0.0; n]);
        box_mean(p, w, h, r, &mut mean_p, &mut tmp);
        let ip: Vec<f32> = i.iter().zip(p).map(|(a, b)| a * b).collect();
        box_mean(&ip, w, h, r, &mut corr_ip, &mut tmp);
        let mut a = vec![0.0f32; n];
        let mut b = vec![0.0f32; n];
        for k in 0..n {
            let cov = corr_ip[k] - mean_i[k] * mean_p[k];
            a[k] = cov * inv_denominator[k];
            b[k] = mean_p[k] - a[k] * mean_i[k];
        }
        let (mut mean_a, mut mean_b) = (corr_ip, mean_p); // reuse the buffers
        box_mean(&a, w, h, r, &mut mean_a, &mut tmp);
        box_mean(&b, w, h, r, &mut mean_b, &mut tmp);
        (0..n).map(|k| mean_a[k] * i[k] + mean_b[k]).collect()
    })
}

/// The colour guided filter on a whole buffer (no banding); `sigma_row0` as in
/// [`super::denoise::bilateral_whole`].
pub fn chroma_guided_whole(
    img: &RgbImage,
    sigma_est: &NoiseSigma,
    sigma_row0: usize,
    color: f32,
) -> RgbImage {
    if color <= 0.0 {
        return img.clone();
    }
    let (w, h) = (img.width, img.height);
    let n = w * h;
    let r = NoiseReduction {
        luminance: 0.0,
        color,
        detail: 0.0,
    }
    .chroma_radius_px();
    let y = img.luma_plane();
    let [rp, _, bp] = img.planes();
    let cr: Vec<f32> = rp.iter().zip(&y).map(|(a, b)| a - b).collect();
    let cb: Vec<f32> = bp.iter().zip(&y).map(|(a, b)| a - b).collect();
    let k = (color / 50.0).powi(2);
    let eps: Vec<f32> = (0..n)
        .map(|i| {
            let s = sigma_est.at(sigma_row0 * w + i);
            (s * s * k).max(1e-10)
        })
        .collect();
    let [fcr, fcb] = guided(&y, [&cr, &cb], &eps, w, h, r);
    let mut out = img.clone();
    for i in 0..n {
        let red = fcr[i] + y[i];
        let blue = fcb[i] + y[i];
        out.data[i] = red;
        out.data[n + i] = (y[i] - LUMA[0] * red - LUMA[2] * blue) / LUMA[1];
        out.data[2 * n + i] = blue;
    }
    out
}

/// The colour guided filter in parallel bands (halo [`guided_halo_px`]).
pub fn chroma_guided_filter(img: &RgbImage, sigma_est: &NoiseSigma, color: f32) -> RgbImage {
    if color <= 0.0 {
        return img.clone();
    }
    let r = NoiseReduction {
        luminance: 0.0,
        color,
        detail: 0.0,
    }
    .chroma_radius_px();
    let base_y0 = img.place.y0;
    filters::run_rgb_bands(img, guided_halo_px(r), BAND_ROWS, |band| {
        chroma_guided_whole(band, sigma_est, band.place.y0 - base_y0, color)
    })
}

/// `Y += beta * (Y_classical - Y_net)`, added equally to the three channels so `R - Y` and
/// `B - Y` are unchanged. Pointwise. The two images must have the same size.
pub fn inject_detail(
    net: &RgbImage,
    classical: &RgbImage,
    beta: f32,
) -> Result<RgbImage, CpuError> {
    if (net.width, net.height) != (classical.width, classical.height) {
        return Err(CpuError::Mismatch(format!(
            "detail re-injection needs equal sizes, got {}x{} and {}x{}",
            net.width, net.height, classical.width, classical.height
        )));
    }
    if beta == 0.0 {
        return Ok(net.clone());
    }
    let n = net.plane_len();
    let mut out = net.clone();
    for i in 0..n {
        let yn = LUMA[0] * net.data[i] + LUMA[1] * net.data[n + i] + LUMA[2] * net.data[2 * n + i];
        let yc = LUMA[0] * classical.data[i]
            + LUMA[1] * classical.data[n + i]
            + LUMA[2] * classical.data[2 * n + i];
        let d = beta * (yc - yn);
        out.data[i] += d;
        out.data[n + i] += d;
        out.data[2 * n + i] += d;
    }
    Ok(out)
}

/// R7 after the network: the colour guided filter, then the detail re-injection (when `beta > 0`
/// and a classical demosaic of the same buffer is supplied).
pub fn apply(
    net: &RgbImage,
    classical: Option<&RgbImage>,
    nr: &NoiseReduction,
    sigma_est: &NoiseSigma,
) -> Result<RgbImage, CpuError> {
    let colored = chroma_guided_filter(net, sigma_est, nr.color);
    match classical {
        Some(c) if nr.needs_classical() => inject_detail(&colored, c, nr.beta()),
        _ => Ok(colored),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu::synth;

    #[test]
    fn auto_values_match_the_spec_examples() {
        // docs/05: ISO 100 -> 10, ISO 400 -> 30, ISO 3200 -> 60, ISO 25600 -> 90.
        assert_eq!(l_auto(100.0), 10.0);
        assert!((l_auto(400.0) - 30.0).abs() < 1e-4);
        assert!((l_auto(3200.0) - 60.0).abs() < 1e-4);
        assert!((l_auto(25600.0) - 90.0).abs() < 1e-4);
        assert_eq!(l_auto(50.0), 10.0);
        assert_eq!(l_auto(409600.0), 90.0);
        let nr = NoiseReduction::auto(400.0);
        assert_eq!((nr.color, nr.detail), (50.0, 50.0));
    }

    #[test]
    fn controls_map_as_specified() {
        let nr = NoiseReduction {
            luminance: 30.0,
            color: 100.0,
            detail: 100.0,
        };
        assert!((nr.beta() - 0.6).abs() < 1e-6);
        assert_eq!(NoiseReduction { detail: 0.0, ..nr }.beta(), 0.0);
        assert!((NoiseReduction { detail: 50.0, ..nr }.beta() - 0.15).abs() < 1e-6);
        assert_eq!(nr.chroma_radius_px(), 8);
        assert_eq!(NoiseReduction { color: 0.0, ..nr }.chroma_radius_px(), 2);
        assert_eq!(NoiseReduction { color: 50.0, ..nr }.chroma_radius_px(), 5);
        // sigma scale: 1 at L_auto, 0 at 0, (100/L_auto)^1.5 at 100.
        assert!((network_sigma_scale(30.0, 400.0) - 1.0).abs() < 1e-5);
        assert_eq!(network_sigma_scale(0.0, 400.0), 0.0);
        assert!((network_sigma_scale(100.0, 400.0) - (100.0f32 / 30.0).powf(1.5)).abs() < 1e-4);
        let s = network_sigma(&NoiseSigma::Scalar(0.01), 4, 60.0, 400.0);
        assert!((s[0] - 0.01 * 2f32.powf(1.5)).abs() < 1e-6);
    }

    /// A textured luma field with smooth chroma (constant plus a gentle gradient) and strong
    /// chroma noise added to R and B: the case the colour control is for.
    fn chroma_noise_image(seed: u64) -> (RgbImage, RgbImage) {
        let (w, h) = (80usize, 60usize);
        let base = synth::natural(w, h, seed);
        let y = base.luma_plane();
        let mut clean = RgbImage::new(w, h);
        for py in 0..h {
            for px in 0..w {
                let i = py * w + px;
                let cr = 0.06 + 0.03 * px as f32 / w as f32;
                let cb = -0.05 + 0.02 * py as f32 / h as f32;
                let (r, b) = (y[i] + cr, y[i] + cb);
                let g = (y[i] - LUMA[0] * r - LUMA[2] * b) / LUMA[1];
                clean.set_pixel(px, py, [r, g, b]);
            }
        }
        let mut noisy = clean.clone();
        let mut rng = synth::Rng::new(seed + 100);
        let n = noisy.plane_len();
        for i in 0..n {
            let nr = 0.03 * rng.gaussian();
            let nb = 0.03 * rng.gaussian();
            noisy.data[i] += nr;
            noisy.data[2 * n + i] += nb;
        }
        (clean, noisy)
    }

    fn chroma_rms_error(a: &RgbImage, b: &RgbImage) -> f64 {
        let n = a.plane_len();
        let (ya, yb) = (a.luma_plane(), b.luma_plane());
        let mut s = 0.0f64;
        for i in 0..n {
            let dr = f64::from((a.data[i] - ya[i]) - (b.data[i] - yb[i]));
            let db = f64::from((a.data[2 * n + i] - ya[i]) - (b.data[2 * n + i] - yb[i]));
            s += dr * dr + db * db;
        }
        (s / n as f64).sqrt()
    }

    #[test]
    fn guided_filter_reduces_chroma_noise_and_keeps_luma() {
        let (clean, noisy) = chroma_noise_image(1);
        let out = chroma_guided_filter(&noisy, &NoiseSigma::Scalar(0.03), 50.0);
        let before = chroma_rms_error(&noisy, &clean);
        let after = chroma_rms_error(&out, &clean);
        assert!(after < 0.7 * before, "chroma error {before} -> {after}");
        // Luma is untouched up to rounding in the opponent round trip.
        let (yn, yo) = (noisy.luma_plane(), out.luma_plane());
        let worst = yn
            .iter()
            .zip(&yo)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max);
        assert!(worst < 1e-5, "luma changed by {worst}");
    }

    #[test]
    fn zero_color_is_a_no_op_and_stronger_color_smooths_more() {
        let (_, noisy) = chroma_noise_image(2);
        assert_eq!(
            chroma_guided_filter(&noisy, &NoiseSigma::Scalar(0.03), 0.0),
            noisy
        );
        let (clean, _) = chroma_noise_image(2);
        let e25 = chroma_rms_error(
            &chroma_guided_filter(&noisy, &NoiseSigma::Scalar(0.03), 25.0),
            &clean,
        );
        let e100 = chroma_rms_error(
            &chroma_guided_filter(&noisy, &NoiseSigma::Scalar(0.03), 100.0),
            &clean,
        );
        assert!(e100 < e25, "{e100} !< {e25}");
    }

    #[test]
    fn detail_injection_moves_luma_by_beta_and_keeps_chroma() {
        let net = synth::natural(30, 20, 3);
        let mut classical = net.clone();
        classical
            .data
            .iter_mut()
            .enumerate()
            .for_each(|(i, v)| *v += 0.02 * ((i % 7) as f32 - 3.0));
        let out = inject_detail(&net, &classical, 0.3).unwrap();
        let (yn, yc, yo) = (net.luma_plane(), classical.luma_plane(), out.luma_plane());
        for i in 0..yn.len() {
            assert!((yo[i] - (yn[i] + 0.3 * (yc[i] - yn[i]))).abs() < 1e-5);
        }
        let n = net.plane_len();
        for i in 0..n {
            assert!(((out.data[i] - yo[i]) - (net.data[i] - yn[i])).abs() < 1e-5);
        }
        assert_eq!(inject_detail(&net, &classical, 0.0).unwrap(), net);
        let small = synth::natural(10, 10, 1);
        assert!(inject_detail(&net, &small, 0.3).is_err());
    }

    #[test]
    fn banded_equals_whole() {
        let (_, noisy) = chroma_noise_image(4);
        let big = synth::natural(60, 330, 5);
        for img in [noisy, big] {
            let sigma: Vec<f32> = (0..img.plane_len())
                .map(|i| 0.01 + 0.0001 * (i % 31) as f32)
                .collect();
            for c in [20.0, 100.0] {
                let whole = chroma_guided_whole(&img, &NoiseSigma::Plane(&sigma), 0, c);
                let banded = chroma_guided_filter(&img, &NoiseSigma::Plane(&sigma), c);
                assert!(
                    whole.max_abs_diff(&banded) < 1e-5,
                    "c={c}: {}",
                    whole.max_abs_diff(&banded)
                );
            }
        }
    }
}
