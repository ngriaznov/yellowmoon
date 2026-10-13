//! R8 capture sharpening, Richardson-Lucy with a Gaussian PSF (`docs/05-raw-processing.md` section R8).
//!
//! Richardson-Lucy deconvolution of the linear luminance (weights [`LUMA`]), clean-room from the
//! published description of the method (no RawTherapee code was read).
//!
//! * `sharpen` resolves to `{ amount A in 0..=300, radius rho in 0.3..=2.5 px }`; iterations
//!   `n = round(A / 30)` (0 to 10).
//! * **PSF**: Gaussian, standard deviation `sigma = rho` at the frame centre and `1.25 rho` in the
//!   corners, linear in `r^2` with `r` the distance from the centre over the half-diagonal:
//!   `sigma(r) = rho (1 + 0.25 r^2)`. The kernel is truncated at `ceil(3 sigma_max)` pixels and
//!   renormalised per pixel. It varies slowly, so both the forward blur and its transpose are
//!   computed as gathers with the kernel of the pixel being written (the usual space-variant
//!   approximation); blurs are separable (horizontal then vertical, each with the local kernel).
//! * **Damping** (Lucy-style) against noise amplification: the correction ratio `d / e` (data over
//!   re-blurred estimate) is replaced by `1 + w (d - e) / e` with `w = r^2 / (r^2 + (lambda sigma_n)^2)`,
//!   `r = d - e` and `lambda =` [`DAMPING_SIGMAS`]. Residuals well below the local noise `sigma_n`
//!   (flat areas) get `w ~ 0`; real structure gets `w ~ 1`, i.e. plain RL. The damping threshold is
//!   proportional to σ as the spec asks. `sigma_n = 0` gives plain RL.
//! * **Auto**: [`auto_sharpen`]: `rho = 0.8` (0.7 at f/8 and above, 0.9 below f/2); `A = 120` at base
//!   ISO falling linearly in log ISO to 60 at ISO 12800.
//! * Luminance only: the result adds `Y' - Y` to all three channels, leaving `R - Y`, `B - Y` as they were.
//!
//! Receptive field: each iteration blurs twice, so the exact radius is `2 n ceil(3 sigma_max)`;
//! the scheduler caps it at 24 px ([`super::tiling::sharpen_halo`]). Tiles are exact when the
//! exact radius is within the cap (always for the Auto values), and differ by far less than 1e-4
//! beyond it for photographic content.

use self::diffraction::DiffractionParams;
use super::filters::{self, BAND_ROWS};
use super::tiling::sharpen_halo;
use super::{LUMA, NoiseSigma, Placement, RgbImage, mirror};

pub mod diffraction;

/// The R8 capture-sharpening method of a frame and its controls: exactly one runs
/// (`docs/algorithms/03-diffraction-deconvolution.md` section 1.2).
///
/// Richardson-Lucy is the default. The choice is internal (no settings-schema change): the caller
/// that builds a [`Chain`](super::chain::Chain) decides, and falls back to Richardson-Lucy when
/// the diffraction stage would be inactive (no usable pixel pitch or f-number, spec section 2.2).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum CaptureSharpen {
    /// Richardson-Lucy with a Gaussian PSF ([`richardson_lucy`]).
    RichardsonLucy(SharpenParams),
    /// Diffraction deconvolution ([`diffraction::DiffractionPlan`]).
    Diffraction(DiffractionParams),
}

impl CaptureSharpen {
    /// The halo the tile scheduler uses for this method: Richardson-Lucy's `2 n ceil(3 sigma)`
    /// capped at [`SHARPEN_HALO_CAP_PX`](super::tiling::SHARPEN_HALO_CAP_PX), or the diffraction
    /// stage's `10 + R + r_G` (at most 25), which that cap does not apply to.
    #[must_use]
    pub fn halo_px(&self) -> usize {
        match self {
            CaptureSharpen::RichardsonLucy(p) => p.halo_px(),
            CaptureSharpen::Diffraction(p) => p.halo_px(),
        }
    }
}

impl From<SharpenParams> for CaptureSharpen {
    fn from(p: SharpenParams) -> Self {
        CaptureSharpen::RichardsonLucy(p)
    }
}

impl From<DiffractionParams> for CaptureSharpen {
    fn from(p: DiffractionParams) -> Self {
        CaptureSharpen::Diffraction(p)
    }
}

/// Residuals below this many noise σ are damped (`w = 0.5` at exactly this residual).
pub const DAMPING_SIGMAS: f32 = 1.0;
/// Lowest value an estimate or blurred estimate may take in the ratio, to avoid division by zero.
const FLOOR: f32 = 1e-4;
/// Number of tabulated kernels between `sigma_min` and `sigma_max`.
const KERNEL_LEVELS: usize = 33;

/// Smallest PSF radius `rho` of the Detail tab, in pixels.
pub const RADIUS_MIN_PX: f32 = 0.3;
/// Largest PSF radius `rho` of the Detail tab, in pixels.
pub const RADIUS_MAX_PX: f32 = 2.5;

/// Resolved sharpening controls.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SharpenParams {
    /// Amount `A`, 0 to 300.
    pub amount: f32,
    /// Radius `rho` in pixels, [`RADIUS_MIN_PX`] to [`RADIUS_MAX_PX`]; values outside are clamped
    /// (a zero radius would make the Gaussian kernel NaN) and NaN reads as the Auto radius 0.8.
    pub radius_px: f32,
}

impl SharpenParams {
    /// Iterations `round(A / 30)`, clamped to 0..=10 (a NaN amount gives none).
    pub fn iterations(&self) -> usize {
        ((self.amount / 30.0).round() as i64).clamp(0, 10) as usize
    }

    /// The radius `rho` actually used: [`radius_px`](Self::radius_px) clamped to the allowed range.
    pub fn effective_radius_px(&self) -> f32 {
        if self.radius_px.is_nan() {
            0.8
        } else {
            self.radius_px.clamp(RADIUS_MIN_PX, RADIUS_MAX_PX)
        }
    }

    /// The largest PSF σ in the frame (the corner value `1.25 rho`).
    pub fn sigma_max_px(&self) -> f32 {
        1.25 * self.effective_radius_px()
    }

    /// The halo the scheduler uses for these parameters.
    pub fn halo_px(&self) -> usize {
        sharpen_halo(self.iterations(), self.sigma_max_px())
    }
}

/// The Auto sharpening for a shot: `rho` from the aperture (`None`: 0.8), `A` from the ISO (`base_iso`
/// is the camera's base ISO, 100 for most Sony bodies).
pub fn auto_sharpen(iso: f32, base_iso: f32, f_number: Option<f32>) -> SharpenParams {
    let radius_px = match f_number {
        Some(f) if f >= 8.0 => 0.7,
        Some(f) if f < 2.0 => 0.9,
        _ => 0.8,
    };
    let t =
        ((iso.max(base_iso) / base_iso).log2() / (12800.0f32 / base_iso).log2()).clamp(0.0, 1.0);
    SharpenParams {
        amount: 120.0 - 60.0 * t,
        radius_px,
    }
}

/// PSF σ at normalised squared radius `r2`: `rho (1 + 0.25 r2)`.
#[inline]
pub fn psf_sigma(rho: f32, r2: f32) -> f32 {
    rho * (1.0 + 0.25 * r2)
}

/// Tabulated normalised Gaussian kernels for σ between `lo` and `hi`.
struct KernelBank {
    radius: usize,
    lo: f32,
    inv_step: f32,
    /// `KERNEL_LEVELS` rows of `2 * radius + 1` taps.
    taps: Vec<f32>,
}

impl KernelBank {
    fn new(lo: f32, hi: f32, radius: usize) -> KernelBank {
        let width = 2 * radius + 1;
        let step = if hi > lo {
            (hi - lo) / (KERNEL_LEVELS - 1) as f32
        } else {
            1.0
        };
        let mut taps = vec![0.0f32; KERNEL_LEVELS * width];
        for l in 0..KERNEL_LEVELS {
            let sigma = lo + step * l as f32;
            let row = &mut taps[l * width..(l + 1) * width];
            for (k, t) in row.iter_mut().enumerate() {
                let d = k as f32 - radius as f32;
                *t = (-0.5 * d * d / (sigma * sigma)).exp();
            }
            let s: f32 = row.iter().sum();
            row.iter_mut().for_each(|t| *t /= s);
        }
        KernelBank {
            radius,
            lo,
            inv_step: if hi > lo { 1.0 / step } else { 0.0 },
            taps,
        }
    }

    /// The kernel for `sigma` as `(row_a, row_b, frac)`: tap `k` is `(1 - frac) a[k] + frac b[k]`.
    #[inline]
    fn at(&self, sigma: f32) -> (&[f32], &[f32], f32) {
        let width = 2 * self.radius + 1;
        let u = ((sigma - self.lo) * self.inv_step).clamp(0.0, (KERNEL_LEVELS - 1) as f32);
        let l = (u as usize).min(KERNEL_LEVELS - 2);
        (
            &self.taps[l * width..(l + 1) * width],
            &self.taps[(l + 1) * width..(l + 2) * width],
            u - l as f32,
        )
    }
}

/// Per-pixel kernels, interpolated from the bank once per call and stored k-major (`taps[k * n + i]`
/// for tap `k` of pixel `i`) so that every pass below is a run of contiguous multiply-adds.
fn pixel_taps(psf_sigma: &[f32], bank: &KernelBank) -> Vec<f32> {
    let n = psf_sigma.len();
    let width = 2 * bank.radius + 1;
    let mut taps = vec![0.0f32; width * n];
    for (i, &sigma) in psf_sigma.iter().enumerate() {
        let (a, b, f) = bank.at(sigma);
        for k in 0..width {
            taps[k * n + i] = (1.0 - f) * a[k] + f * b[k];
        }
    }
    taps
}

/// Space-variant separable blur: `out = V(H(src))`, each pass with the kernel of the pixel written.
fn blur(src: &[f32], w: usize, h: usize, taps: &[f32], r: usize, tmp: &mut [f32], out: &mut [f32]) {
    let n = w * h;
    let width = 2 * r + 1;
    let mut padded = vec![0.0f32; w + 2 * r];
    for y in 0..h {
        let row = &src[y * w..(y + 1) * w];
        for (px, p) in padded.iter_mut().enumerate() {
            *p = row[mirror(px as isize - r as isize, w)];
        }
        let acc = &mut tmp[y * w..(y + 1) * w];
        acc.fill(0.0);
        for k in 0..width {
            let t = &taps[k * n + y * w..k * n + (y + 1) * w];
            let s = &padded[k..k + w];
            for ((a, &tk), &sv) in acc.iter_mut().zip(t).zip(s) {
                *a += tk * sv;
            }
        }
    }
    for y in 0..h {
        let acc = &mut out[y * w..(y + 1) * w];
        acc.fill(0.0);
        for k in 0..width {
            let yy = mirror(y as isize + k as isize - r as isize, h);
            let t = &taps[k * n + y * w..k * n + (y + 1) * w];
            let s = &tmp[yy * w..(yy + 1) * w];
            for ((a, &tk), &sv) in acc.iter_mut().zip(t).zip(s) {
                *a += tk * sv;
            }
        }
    }
}

/// Richardson-Lucy on a whole buffer (no banding).
///
/// `sigma_noise` is the luminance noise σ (linear units) indexed from `sigma_row0`; `place` of
/// `img` decides the PSF σ at each pixel.
pub fn richardson_lucy_whole(
    img: &RgbImage,
    params: &SharpenParams,
    sigma_noise: &NoiseSigma,
    sigma_row0: usize,
) -> RgbImage {
    let n_iter = params.iterations();
    if n_iter == 0 || img.width == 0 || img.height == 0 {
        return img.clone();
    }
    let (w, h) = (img.width, img.height);
    let n = w * h;
    let place: Placement = img.place;
    let y_orig = img.luma_plane();
    let d: Vec<f32> = y_orig.iter().map(|v| v.max(0.0)).collect();

    // PSF σ per pixel from the absolute position.
    let (cx, cy) = place.centre();
    let inv_hd2 = 1.0 / (place.half_diagonal() * place.half_diagonal());
    let mut psf = vec![0.0f32; n];
    for y in 0..h {
        let dy = (place.y0 + y) as f64 + 0.5 - cy;
        for x in 0..w {
            let dx = (place.x0 + x) as f64 + 0.5 - cx;
            let r2 = ((dx * dx + dy * dy) * inv_hd2) as f32;
            psf[y * w + x] = psf_sigma(params.effective_radius_px(), r2);
        }
    }
    let sigma_max = params.sigma_max_px();
    let radius = (3.0 * sigma_max).ceil() as usize;
    let bank = KernelBank::new(params.effective_radius_px(), sigma_max, radius);
    let taps = pixel_taps(&psf, &bank);

    let mut u = d.clone();
    let (mut e, mut q, mut c, mut tmp) = (
        vec![0.0f32; n],
        vec![0.0f32; n],
        vec![0.0f32; n],
        vec![0.0f32; n],
    );
    for _ in 0..n_iter {
        blur(&u, w, h, &taps, radius, &mut tmp, &mut e);
        for i in 0..n {
            let ee = e[i].max(FLOOR);
            let resid = d[i] - e[i];
            let sn = sigma_noise.at(sigma_row0 * w + i);
            let lam = DAMPING_SIGMAS * sn;
            let wgt = if lam > 0.0 {
                resid * resid / (resid * resid + lam * lam)
            } else {
                1.0
            };
            q[i] = 1.0 + wgt * resid / ee;
        }
        blur(&q, w, h, &taps, radius, &mut tmp, &mut c);
        for i in 0..n {
            u[i] = (u[i] * c[i]).max(0.0);
        }
    }
    let mut out = img.clone();
    for i in 0..n {
        let delta = u[i] - d[i];
        out.data[i] += delta;
        out.data[n + i] += delta;
        out.data[2 * n + i] += delta;
    }
    out
}

/// Richardson-Lucy in parallel bands with the capped halo.
pub fn richardson_lucy(
    img: &RgbImage,
    params: &SharpenParams,
    sigma_noise: &NoiseSigma,
) -> RgbImage {
    if params.iterations() == 0 {
        return img.clone();
    }
    let base_y0 = img.place.y0;
    filters::run_rgb_bands(img, params.halo_px(), BAND_ROWS, |band| {
        richardson_lucy_whole(band, params, sigma_noise, band.place.y0 - base_y0)
    })
}

/// Luma weights used by the stage (re-exported for the WGSL parity tests).
pub const LUMA_WEIGHTS: [f32; 3] = LUMA;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu::synth;

    #[test]
    fn auto_values_follow_the_spec() {
        let a = auto_sharpen(100.0, 100.0, Some(5.6));
        assert_eq!((a.amount, a.radius_px), (120.0, 0.8));
        assert_eq!(a.iterations(), 4);
        assert_eq!(auto_sharpen(100.0, 100.0, Some(8.0)).radius_px, 0.7);
        assert_eq!(auto_sharpen(100.0, 100.0, Some(11.0)).radius_px, 0.7);
        assert_eq!(auto_sharpen(100.0, 100.0, Some(1.8)).radius_px, 0.9);
        assert_eq!(auto_sharpen(100.0, 100.0, Some(2.0)).radius_px, 0.8);
        assert_eq!(auto_sharpen(100.0, 100.0, None).radius_px, 0.8);
        // 60 at ISO 12800, linear in log ISO in between: ISO 1131 (geometric middle of 100..12800) -> 90.
        assert!((auto_sharpen(12800.0, 100.0, None).amount - 60.0).abs() < 1e-3);
        assert!((auto_sharpen(1131.37, 100.0, None).amount - 90.0).abs() < 0.1);
        assert_eq!(auto_sharpen(51200.0, 100.0, None).amount, 60.0);
        assert_eq!(auto_sharpen(50.0, 100.0, None).amount, 120.0);
    }

    #[test]
    fn iteration_count_and_psf_profile() {
        let p = |a| SharpenParams {
            amount: a,
            radius_px: 1.0,
        };
        assert_eq!(p(0.0).iterations(), 0);
        assert_eq!(p(14.0).iterations(), 0);
        assert_eq!(p(15.0).iterations(), 1);
        assert_eq!(p(120.0).iterations(), 4);
        assert_eq!(p(300.0).iterations(), 10);
        assert_eq!(psf_sigma(0.8, 0.0), 0.8);
        assert!((psf_sigma(0.8, 1.0) - 1.0).abs() < 1e-6);
        assert!((psf_sigma(0.8, 0.5) - 0.9).abs() < 1e-6);
        assert_eq!(
            SharpenParams {
                amount: 120.0,
                radius_px: 0.8
            }
            .halo_px(),
            24
        );
    }

    #[test]
    fn out_of_range_radius_is_clamped_not_nan() {
        let img = synth::natural(40, 30, 3);
        for radius in [0.0f32, -1.0, 100.0, f32::NAN] {
            let p = SharpenParams {
                amount: 90.0,
                radius_px: radius,
            };
            let r = p.effective_radius_px();
            assert!(
                (RADIUS_MIN_PX..=RADIUS_MAX_PX).contains(&r),
                "{radius} -> {r}"
            );
            let out = richardson_lucy(&img, &p, &NoiseSigma::Scalar(0.01));
            assert!(out.data.iter().all(|v| v.is_finite()), "radius {radius}");
        }
        assert_eq!(
            SharpenParams {
                amount: 90.0,
                radius_px: 0.0
            }
            .effective_radius_px(),
            RADIUS_MIN_PX
        );
    }

    #[test]
    fn zero_amount_is_identity() {
        let img = synth::natural(30, 20, 1);
        let out = richardson_lucy(
            &img,
            &SharpenParams {
                amount: 0.0,
                radius_px: 1.0,
            },
            &NoiseSigma::Scalar(0.0),
        );
        assert_eq!(out, img);
    }

    fn blur_image(img: &RgbImage, sigma: f32) -> RgbImage {
        let (w, h) = (img.width, img.height);
        let mut out = img.clone();
        let k = filters::gaussian_kernel(sigma, (3.0 * sigma).ceil() as usize);
        let r = (k.len() / 2) as isize;
        for c in 0..3 {
            let src = img.plane(c).to_vec();
            let mut tmp = vec![0.0f32; w * h];
            for y in 0..h {
                for x in 0..w {
                    tmp[y * w + x] = (-r..=r)
                        .map(|d| k[(d + r) as usize] * src[y * w + mirror(x as isize + d, w)])
                        .sum();
                }
            }
            let dst = out.plane_mut(c);
            for y in 0..h {
                for x in 0..w {
                    dst[y * w + x] = (-r..=r)
                        .map(|d| k[(d + r) as usize] * tmp[mirror(y as isize + d, h) * w + x])
                        .sum();
                }
            }
        }
        out
    }

    #[test]
    fn restores_detail_lost_to_a_known_blur() {
        let sharp = synth::fine_text(96, 64, 2);
        let blurred = blur_image(&sharp, 0.9);
        let params = SharpenParams {
            amount: 120.0,
            radius_px: 0.9 / 1.0,
        };
        let out = richardson_lucy(&blurred, &params, &NoiseSigma::Scalar(0.0));
        let mse = |a: &RgbImage, b: &RgbImage| {
            a.data
                .iter()
                .zip(&b.data)
                .map(|(x, y)| f64::from(x - y).powi(2))
                .sum::<f64>()
                / a.data.len() as f64
        };
        let before = mse(&blurred, &sharp);
        let after = mse(&out, &sharp);
        assert!(
            after < 0.75 * before,
            "RL did not help: {before} -> {after}"
        );
    }

    #[test]
    fn leaves_chroma_untouched() {
        let img = synth::natural(48, 40, 6);
        let out = richardson_lucy(
            &img,
            &SharpenParams {
                amount: 90.0,
                radius_px: 0.8,
            },
            &NoiseSigma::Scalar(0.005),
        );
        let n = img.plane_len();
        let (y0, y1) = (img.luma_plane(), out.luma_plane());
        let mut luma_changed = 0.0f32;
        for i in 0..n {
            for c in [0usize, 2] {
                let a = img.data[c * n + i] - y0[i];
                let b = out.data[c * n + i] - y1[i];
                assert!((a - b).abs() < 2e-5, "chroma changed at {i}: {a} vs {b}");
            }
            luma_changed = luma_changed.max((y1[i] - y0[i]).abs());
        }
        assert!(luma_changed > 1e-3, "nothing was sharpened");
    }

    #[test]
    fn damping_keeps_flat_noise_from_growing() {
        let mut img = RgbImage::new(64, 64);
        img.data.fill(0.3);
        synth::add_noise(&mut img.data, 0.01, 9);
        let p = SharpenParams {
            amount: 120.0,
            radius_px: 0.8,
        };
        let plain = richardson_lucy(&img, &p, &NoiseSigma::Scalar(0.0));
        let damped = richardson_lucy(&img, &p, &NoiseSigma::Scalar(0.01));
        let sd = |im: &RgbImage| {
            let y = im.luma_plane();
            let m = y.iter().sum::<f32>() / y.len() as f32;
            (y.iter().map(|v| (v - m).powi(2)).sum::<f32>() / y.len() as f32).sqrt()
        };
        assert!(
            sd(&damped) < sd(&plain),
            "damped {} !< plain {}",
            sd(&damped),
            sd(&plain)
        );
        assert!(
            sd(&damped) < 1.5 * sd(&img),
            "damping failed to hold the noise: {} vs {}",
            sd(&damped),
            sd(&img)
        );
    }

    #[test]
    fn banded_equals_whole_when_the_halo_is_exact() {
        let img = synth::natural(60, 300, 3);
        let p = SharpenParams {
            amount: 120.0,
            radius_px: 0.8,
        };
        let whole = richardson_lucy_whole(&img, &p, &NoiseSigma::Scalar(0.004), 0);
        let banded = richardson_lucy(&img, &p, &NoiseSigma::Scalar(0.004));
        assert!(
            whole.max_abs_diff(&banded) < 1e-5,
            "{}",
            whole.max_abs_diff(&banded)
        );
    }
}
