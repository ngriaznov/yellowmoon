//! R8 diffraction deconvolution, specified in `docs/algorithms/03-diffraction-deconvolution.md`
//! (RA-07).
//!
//! A second capture-sharpening method beside Richardson-Lucy (`super::richardson_lucy`). The blur
//! that diffraction adds depends only on the f-number, the wavelength and the pixel pitch, so it is
//! computed, not guessed: [`DiffractionPlan::new`] builds a polychromatic Airy-pattern PSF once per
//! frame, and [`DiffractionPlan::apply`] removes it from luma in one noise-weighted local
//! least-squares step. There is no iteration and no amount control (`strength` 1 is the specified
//! behaviour). Integration: `docs/algorithms/INTEGRATION.md` sections 3, 5 and 7.
//!
//! # Units
//!
//! The constants of the algorithm are tuned for a 16-bit scale, so every sample is read as
//! `x_dn = S * x` with `S = 65535` ([`WORKING_SCALE`]) and the
//! result is written back as `out = in * k + off / S` (spec section 1.4), so identities (flat
//! fields, `strength = 0`, an inactive stage) are bit-exact. The noise weight is built from
//! [`NoiseDn`] (spec section 2.3). Following the specs of the package, a `NaN` or infinite
//! `NoiseDn::r` (what [`NoiseDn::from_model`] gives for a negative or infinite variance) is read
//! as 0, so the luma noise floor sits at its 1 DN minimum; a negative `g` reads as 0.
//!
//! # Precision and deviations from the reference
//!
//! The port follows the C++ development oracle (not in this repository) operation by operation:
//! the planner is `f64`; the kernel taps and `sigma_G` are stored as `f32` and the stored values
//! are used afterwards; planes are `f32`; every window sum, convolution and product accumulates
//! in `f64` in the same order as the reference (rows first, then columns; kernel rows then
//! columns) and is rounded once to `f32`; the per-pixel solve and the transfer are `f64`. No
//! `mul_add` is used. The 13 image vectors in `tests/cpu_data/diffraction_deconvolution/` agree bit
//! for bit (tolerance 2e-5) and the 10 planner vectors to 1.3e-8 (tolerance 1e-6). Two things are
//! not literal copies:
//!
//! * the loops are organised by rows (padded rows, one accumulator row per output row) so they
//!   vectorise; each output sample still sums the same terms in the same order;
//! * [`DiffractionPlan::apply`] cuts the frame into 128-row bands with the plan's exact halo, like
//!   the other stages (`filters::run_rgb_bands`): because every window is read through the mirror
//!   rule and the halo is the exact dependency radius, a band, a tile with that much context and the
//!   whole frame give the same bits.
//!
//! # Halo
//!
//! `halo = 10 + R + r_G` (spec section 1.5, at most [`MAX_HALO_PX`] = 25). It is not capped at
//! `SHARPEN_HALO_CAP_PX`, which belongs to Richardson-Lucy. The kernel, `R`, `sigma_G` and so the
//! halo depend only on `(p_eff, N)`; the planner caches them by those two numbers
//! ([`DiffractionParams::halo_px`] is then cheap enough for the tile scheduler to call per tile).
//!
//! # Bessel J1
//!
//! The one piece the spec leaves to the implementer. [`bessel_j1`] is written here from
//! Abramowitz and Stegun: the power series 9.1.10 below `|x| = 12` and the Hankel asymptotic
//! expansion 9.2.5 above (terms summed until they start to grow). No library and no GPL code is
//! involved, so there is no new dependency or notice. Its error is measured in the tests against
//! tabulated values (at most 1e-12) and against Bessel's integral on `[0, 120]` (8.6e-13; the spec
//! asks for 1e-10).
//!
//! # Evaluation
//!
//! [`eval`] holds the edge, MTF and noise measures of the A/B of spec section 6.3, used by the
//! `ra07_ab` example and the local real-file comparison; the render path does not call them.
//!
//! # Not covered here
//!
//! The choice of pixel pitch and the fallback to Richardson-Lucy when the stage is inactive are the
//! caller's (`ym-core`, spec section 2.2): [`DiffractionPlan::new`] returns `None` for an inactive
//! stage, and a [`Chain`](crate::cpu::chain::Chain) that is given an inactive Diffraction stage
//! leaves the image unchanged.

use std::f64::consts::PI;
use std::sync::{Arc, Mutex, PoisonError};

use crate::cpu::filters::{self, BAND_ROWS};
use crate::cpu::noise::{NoiseDn, WORKING_SCALE};
use crate::cpu::{RgbImage, mirror};

pub mod eval;

/// The luma weights of the specification (`0.299, 0.587, 0.114` on balanced linear camera RGB).
/// Not [`LUMA`](crate::cpu::LUMA) (Rec. 709) of the other stages: see spec section 7.1, item 3.
pub const SPEC_LUMA: [f32; 3] = [0.299, 0.587, 0.114];

/// The largest halo any parameter set can need: `10 + R + r_G` with `R <= 7` and `r_G <= 8`.
pub const MAX_HALO_PX: usize = 25;

/// Smallest f-number the stage accepts.
pub const F_NUMBER_MIN: f64 = 1.0;
/// Largest f-number the stage accepts.
pub const F_NUMBER_MAX: f64 = 50.0;

/// Radial samples of the Airy profile (index 0 to 499).
const PROFILE_SAMPLES: usize = 500;
/// Size of the quadrant grid of the 2-D PSF (radius up to 7).
const GRID: usize = 8;
/// Largest kernel radius `R`.
const MAX_RADIUS: usize = 7;
/// Largest radius of the Gaussian high-pass.
const MAX_GAUSS_RADIUS: usize = 8;
/// Residual gate: pixels with `Q <= GATE * P` are left alone (spec section 3.9).
const GATE: f64 = 0.5;
/// Fraction of white above which a channel counts as near clipping (spec section 2.4).
const CLIP_FRACTION: f64 = 0.8;

/// R8 diffraction deconvolution controls (spec section 2.1).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DiffractionParams {
    /// Sensor pixel pitch (x, y), micrometres. The larger of the two is used.
    pub pixel_pitch_um: [f64; 2],
    /// Lens f-number, valid from [`F_NUMBER_MIN`] to [`F_NUMBER_MAX`]; anything else (zero, `NaN`)
    /// makes the stage inactive.
    pub f_number: f64,
    /// Image pixels per sensor pixel, `(0, 1]`; a value outside the range (or `NaN`) reads as 1.
    pub scale: f64,
    /// Blend between input and corrected luma, 0 to 3, clamped; `NaN` reads as 1. 1 is the
    /// specified behaviour and 0 returns the input exactly.
    pub strength: f32,
}

impl DiffractionParams {
    /// Square pixels of `pixel_pitch_um` micrometres at full resolution and the specified strength.
    #[must_use]
    pub const fn new(pixel_pitch_um: f64, f_number: f64) -> Self {
        Self {
            pixel_pitch_um: [pixel_pitch_um, pixel_pitch_um],
            f_number,
            scale: 1.0,
            strength: 1.0,
        }
    }

    /// The same controls for an image at `scale` image pixels per sensor pixel.
    #[must_use]
    pub const fn with_scale(mut self, scale: f64) -> Self {
        self.scale = scale;
        self
    }

    /// The same controls with another strength.
    #[must_use]
    pub const fn with_strength(mut self, strength: f32) -> Self {
        self.strength = strength;
        self
    }

    /// Whether the pitch and the f-number pass the gate of spec section 3.1 (the gains `g0` are
    /// checked by [`DiffractionPlan::new`]).
    #[must_use]
    pub fn is_valid(&self) -> bool {
        let [px, py] = self.pixel_pitch_um;
        px > 0.0
            && py > 0.0
            && px.is_finite()
            && py.is_finite()
            && self.f_number >= F_NUMBER_MIN
            && self.f_number <= F_NUMBER_MAX
    }

    /// `scale` as used: itself when in `(0, 1]`, else 1.
    #[must_use]
    pub fn effective_scale(&self) -> f64 {
        if self.scale > 0.0 && self.scale <= 1.0 {
            self.scale
        } else {
            1.0
        }
    }

    /// `p_eff = max(pitch_x, pitch_y) / scale`, micrometres.
    #[must_use]
    pub fn effective_pitch_um(&self) -> f64 {
        cmax(self.pixel_pitch_um[0], self.pixel_pitch_um[1]) / self.effective_scale()
    }

    /// `strength` as used: clamped to `[0, 3]`, `NaN` reads as 1.
    #[must_use]
    pub fn effective_strength(&self) -> f64 {
        if self.strength.is_nan() {
            1.0
        } else {
            f64::from(self.strength.clamp(0.0, 3.0))
        }
    }

    /// The halo the tile scheduler uses for these controls: `10 + R + r_G` of the plan, 0 when the
    /// stage is inactive (it then returns its input and needs no context). It does not depend on
    /// the gains or the noise, so it is known before the plan is.
    #[must_use]
    pub fn halo_px(&self) -> usize {
        if !self.is_valid() {
            return 0;
        }
        geometry(self.effective_pitch_um(), self.f_number).map_or(0, |g| g.halo_px)
    }
}

/// What depends only on `(p_eff, N)`: the PSF kernel, the high-pass width, the halo.
#[derive(Debug, PartialEq)]
struct Geometry {
    radius: usize,
    kernel: Vec<f32>,
    gauss_sigma: f32,
    gauss_radius: usize,
    halo_px: usize,
    step_um: f64,
    r80_um: f64,
    support_um: f64,
}

/// Everything derived once per frame: the kernel, `sigma_G`, the radii and the noise constants
/// (spec section 2.7).
#[derive(Clone, Debug)]
pub struct DiffractionPlan {
    geometry: Arc<Geometry>,
    pitch_eff_um: f64,
    sigma0_dn: f64,
    sigma1_dn: f64,
    clip_threshold_dn: f64,
    luma: [f64; 3],
    g0: [f64; 3],
    strength: f64,
    /// Factor on the 9x9 clip-penalty mean in the gate (spec section 7.1, item 1): 1 is the
    /// specified behaviour.
    gate_factor: f64,
}

impl DiffractionPlan {
    /// Plans the stage. `None` when it is inactive (spec section 3.1): a pitch that is not positive
    /// and finite, an f-number outside `[1, 50]` or `NaN`, a gain that is not positive and finite,
    /// or any intermediate value that cannot happen for valid inputs.
    ///
    /// `g0` are the R4 gains (`RawParams::g0`), `noise` the model on the working scale
    /// ([`NoiseDn::from_model`]) and `luma` the luma weights ([`SPEC_LUMA`] by default).
    #[must_use]
    pub fn new(
        p: &DiffractionParams,
        g0: [f32; 3],
        noise: &NoiseDn,
        luma: [f32; 3],
    ) -> Option<Self> {
        if !p.is_valid() {
            return None;
        }
        let g0 = g0.map(f64::from);
        if g0.iter().any(|&g| !g.is_finite() || g <= 0.0) {
            return None;
        }
        let luma = luma.map(f64::from);
        let scale = p.effective_scale();
        let pitch_eff_um = p.effective_pitch_um();
        let geometry = geometry(pitch_eff_um, p.f_number)?;

        // Noise of luma on the working scale (spec section 2.3). A NaN or infinite `r` reads as 0.
        let r_dn = if noise.r.is_finite() { noise.r } else { 0.0 };
        let d = luma[0] * g0[0] + luma[1] * g0[1] + luma[2] * g0[2];
        let s = cmin(2.0 * scale, 1.0);
        let sigma0_dn = cmax(d * r_dn * s.sqrt(), 1.0);
        let sigma1_dn = cmax(d * noise.g * s, 0.0);
        if !sigma0_dn.is_finite() || !sigma1_dn.is_finite() {
            return None;
        }
        Some(Self {
            geometry,
            pitch_eff_um,
            sigma0_dn,
            sigma1_dn,
            clip_threshold_dn: CLIP_FRACTION * WORKING_SCALE,
            luma,
            g0,
            strength: p.effective_strength(),
            gate_factor: 1.0,
        })
    }

    /// The noise-gate variant of spec section 7.1, item 1 (an A/B candidate; **not** the specified
    /// behaviour, which is [`new`](Self::new)'s plan): `with_noise_gate(true)` scales the gate by the
    /// white-noise level of this PSF, so that flat noise is no longer opened up by the wider kernels
    /// of small apertures. With `q_w = 1 - 2 K0 + sum K^2` ([`white_noise_residual`](Self::white_noise_residual))
    /// and `f = max(1, 2 q_w)` ([`scaled_gate_factor`](Self::scaled_gate_factor)) the gate threshold becomes
    /// `t = 0.5 f P` and the gain reaches 1 at `Q = f P`. `with_noise_gate(false)` restores the
    /// specified gate. At f/8 and below `f = 1` for a 3.76 um pitch (`q_w` is 0.40 and under), so the
    /// two agree there.
    #[must_use]
    pub fn with_noise_gate(mut self, scaled: bool) -> Self {
        self.gate_factor = if scaled {
            self.scaled_gate_factor()
        } else {
            1.0
        };
        self
    }

    /// `q_w = 1 - 2 K0 + sum K^2` with `K0` the centre tap: the value the noise-normalised residual
    /// `Q` takes for white noise of exactly the modelled variance (spec section 3.8). 0.07 at f/2.8,
    /// 0.40 at f/8, 0.75 at f/16 and 0.94 at f/50 for a 3.76 um pitch.
    #[must_use]
    pub fn white_noise_residual(&self) -> f64 {
        let k = &self.geometry.kernel;
        let kw = 2 * self.geometry.radius + 1;
        let k0 = f64::from(k[self.geometry.radius * kw + self.geometry.radius]);
        let sum_sq: f64 = k.iter().map(|&v| f64::from(v) * f64::from(v)).sum();
        1.0 - 2.0 * k0 + sum_sq
    }

    /// `f = max(1, 2 q_w)`, the factor of the [noise-gate variant](Self::with_noise_gate).
    #[must_use]
    pub fn scaled_gate_factor(&self) -> f64 {
        cmax(1.0, 2.0 * self.white_noise_residual())
    }

    /// The factor on the clip-penalty mean in the gate: 1 for the specified behaviour.
    #[must_use]
    pub fn gate_factor(&self) -> f64 {
        self.gate_factor
    }

    /// The R4 gains `g0` the plan was made with, exactly as they were given to
    /// [`new`](Self::new). The correction needs them again (the noise floor is `sigma0 = D * r`
    /// with `D` the luma of the gains), so a consumer that runs the stage elsewhere, like the GPU
    /// kernel, reads them from the plan and cannot use other ones.
    #[must_use]
    pub fn g0(&self) -> [f32; 3] {
        // They were `f32` values widened to `f64`: narrowing gives them back exactly.
        self.g0.map(|g| g as f32)
    }

    /// The luma weights the plan was made with, exactly as they were given to
    /// [`new`](Self::new) (see [`g0`](Self::g0)).
    #[must_use]
    pub fn luma(&self) -> [f32; 3] {
        self.luma.map(|w| w as f32)
    }

    /// The PSF kernel radius `R`, 1 to 7.
    #[must_use]
    pub fn radius(&self) -> usize {
        self.geometry.radius
    }

    /// The PSF kernel, `(2R + 1)^2` taps, row-major, non-negative, summing to 1.
    #[must_use]
    pub fn kernel(&self) -> &[f32] {
        &self.geometry.kernel
    }

    /// Width of the high-pass Gaussian `sigma_G`, 0.2 to 3 pixels.
    #[must_use]
    pub fn gauss_sigma(&self) -> f32 {
        self.geometry.gauss_sigma
    }

    /// Radius of the high-pass Gaussian `r_G = min(floor(3.5 sigma_G + 1), 8)`.
    #[must_use]
    pub fn gauss_radius(&self) -> usize {
        self.geometry.gauss_radius
    }

    /// The halo: `10 + R + r_G`, at most [`MAX_HALO_PX`].
    #[must_use]
    pub fn halo_px(&self) -> usize {
        self.geometry.halo_px
    }

    /// Luma noise floor `sigma0` in DN (at least 1).
    #[must_use]
    pub fn sigma0_dn(&self) -> f64 {
        self.sigma0_dn
    }

    /// Luma shot-noise slope `sigma1` in DN.
    #[must_use]
    pub fn sigma1_dn(&self) -> f64 {
        self.sigma1_dn
    }

    /// The 80 % encircled-energy radius of the PSF, micrometres (diagnostic).
    #[must_use]
    pub fn r80_um(&self) -> f64 {
        self.geometry.r80_um
    }

    /// The support radius of the PSF after truncation, micrometres (diagnostic).
    #[must_use]
    pub fn support_um(&self) -> f64 {
        self.geometry.support_um
    }

    /// The radial step of the planner, micrometres (diagnostic).
    #[must_use]
    pub fn step_um(&self) -> f64 {
        self.geometry.step_um
    }

    /// The effective pixel pitch `max(pitch) / scale`, micrometres (diagnostic).
    #[must_use]
    pub fn pitch_eff_um(&self) -> f64 {
        self.pitch_eff_um
    }

    /// The strength as used, 0 to 3.
    #[must_use]
    pub fn strength(&self) -> f64 {
        self.strength
    }

    /// Runs the stage on a whole buffer without cutting it into bands (the reference form; use
    /// [`apply`](Self::apply) for frames). Same size, layout, units and placement as the input; luma
    /// only is changed.
    #[must_use]
    pub fn apply_whole(&self, img: &RgbImage) -> RgbImage {
        let n = img.plane_len();
        if n == 0 {
            return img.clone();
        }
        let (l, y) = self.corrected_luma(img);
        let mut out = img.clone();
        for ix in 0..n {
            let yv = f64::from(y[ix]);
            let k = cmin(cmax(l[ix], 1.0) / cmax(yv, 1.0), 1.5);
            let off = (l[ix] - yv * k) / WORKING_SCALE;
            for c in 0..3 {
                out.data[c * n + ix] = (f64::from(img.data[c * n + ix]) * k + off) as f32;
            }
        }
        out
    }

    /// Runs the stage on a frame or tile in parallel bands of rows with the exact halo. The result
    /// does not depend on the thread count, and equals [`apply_whole`](Self::apply_whole) and the
    /// same stage on a tile that carries [`halo_px`](Self::halo_px) pixels of context.
    #[must_use]
    pub fn apply(&self, img: &RgbImage) -> RgbImage {
        if img.plane_len() == 0 {
            return img.clone();
        }
        filters::run_rgb_bands(img, self.halo_px(), BAND_ROWS, |band| {
            self.apply_whole(band)
        })
    }

    /// The corrected luma `L_s` (DN, `f64`) and the input luma `Y` (DN, rounded to `f32`) of a
    /// buffer, one value per pixel: spec sections 3.4 to 3.11. [`apply_whole`](Self::apply_whole)
    /// carries `L_s` into the three channels; the luma of its output equals `L_s / S`.
    #[must_use]
    pub fn corrected_luma(&self, img: &RgbImage) -> (Vec<f64>, Vec<f32>) {
        let (w, h) = (img.width, img.height);
        let n = w * h;
        let s = WORKING_SCALE;
        let a = self.luma;
        let g0 = self.g0;
        let (s0, s1) = (self.sigma0_dn, self.sigma1_dn);
        let t_clip = self.clip_threshold_dn;
        let g = &*self.geometry;
        let [rp, gp, bp] = img.planes();

        // 3.4 luma, inverse noise variance, clip penalty.
        let mut luma = vec![0.0f32; n];
        let mut wgt = vec![0.0f32; n];
        let mut penalty = vec![0.0f32; n];
        for ix in 0..n {
            let xr = s * f64::from(rp[ix]);
            let xg = s * f64::from(gp[ix]);
            let xb = s * f64::from(bp[ix]);
            let y = (a[0] * xr + a[1] * xg + a[2] * xb) as f32;
            luma[ix] = y;
            let wi = (1.0 / (s0 * s0 + s1 * cmax(f64::from(y), 0.0))) as f32;
            wgt[ix] = wi;
            let mut e = [
                cmax(xr / g0[0] - t_clip, 0.0) * a[0],
                cmax(xg / g0[1] - t_clip, 0.0) * a[1],
                cmax(xb / g0[2] - t_clip, 0.0) * a[2],
            ];
            sort3(&mut e);
            let excess = 25.0 * e[0] + 5.0 * e[1] + e[2];
            penalty[ix] = (1.0 + f64::from(wi) * excess * excess) as f32;
        }

        // 3.5 envelope estimates, 3.6 dark and bright layers.
        let (lower, upper) = envelopes(&luma, w, h);
        let lm = box_mean(&lower, w, h, 2, true);
        let hm = box_mean(&upper, w, h, 2, true);
        drop((lower, upper));
        let mut dark = vec![0.0f32; n];
        let mut bright = vec![0.0f32; n];
        for ix in 0..n {
            let y = f64::from(luma[ix]);
            let l = f64::from(lm[ix]);
            let u = f64::from(hm[ix]);
            if y <= l {
                dark[ix] = luma[ix];
                bright[ix] = 0.0;
            } else if y >= u {
                dark[ix] = 0.0;
                bright[ix] = luma[ix];
            } else {
                let t = (y - l) / (u - l);
                bright[ix] = (u * t) as f32;
                dark[ix] = (y - f64::from(bright[ix])) as f32;
            }
        }
        drop((lm, hm));

        // 3.7 high-pass, reblur, residual.
        let sg = f64::from(g.gauss_sigma);
        let rg = g.gauss_radius;
        let gy = gauss_blur(&luma, w, h, sg, rg);
        let hp: Vec<f32> = luma.iter().zip(&gy).map(|(&y, &b)| y - b).collect();
        drop(gy);
        let f1 = convolve_psf(&dark, w, h, &g.kernel, g.radius);
        let f2 = convolve_psf(&bright, w, h, &g.kernel, g.radius);
        let reblur: Vec<f32> = f1.iter().zip(&f2).map(|(&p, &q)| p + q).collect();
        let g_reblur = gauss_blur(&reblur, w, h, sg, rg);
        let f3: Vec<f32> = reblur.iter().zip(&g_reblur).map(|(&z, &b)| z - b).collect();
        let resid: Vec<f32> = luma.iter().zip(&reblur).map(|(&y, &z)| y - z).collect();
        drop((g_reblur, reblur));

        // 3.8 weighted moments over 9x9.
        let mut prod = vec![0.0f64; n];
        let mut moment = |u: &[f32], v: &[f32], clamp: bool| -> Vec<f32> {
            for (((p, &wi), &ui), &vi) in prod.iter_mut().zip(&wgt).zip(u).zip(v) {
                *p = f64::from(wi) * f64::from(ui) * f64::from(vi);
            }
            box_mean(&prod, w, h, 4, clamp)
        };
        let s11 = moment(&f1, &f1, true);
        let s22 = moment(&f2, &f2, true);
        let s33 = moment(&f3, &f3, true);
        let s12 = moment(&f1, &f2, false);
        let s13 = moment(&f1, &f3, false);
        let s23 = moment(&f2, &f3, false);
        let t1 = moment(&f1, &resid, false);
        let t2 = moment(&f2, &resid, false);
        let t3 = moment(&f3, &resid, false);
        let q = moment(&resid, &resid, true);
        let p_bar = box_mean(&penalty, w, h, 4, true);
        drop((prod, f1, f2, f3, resid, wgt, penalty));

        // 3.9 solve.
        let mut k1 = vec![0.0f32; n];
        let mut k2 = vec![0.0f32; n];
        let mut k3 = vec![0.0f32; n];
        let mut kw = vec![0.0f32; n];
        for ix in 0..n {
            let m = Moments {
                s11: f64::from(s11[ix]),
                s12: f64::from(s12[ix]),
                s13: f64::from(s13[ix]),
                s22: f64::from(s22[ix]),
                s23: f64::from(s23[ix]),
                s33: f64::from(s33[ix]),
                t1: f64::from(t1[ix]),
                t2: f64::from(t2[ix]),
                t3: f64::from(t3[ix]),
                q: f64::from(q[ix]),
                p: f64::from(p_bar[ix]) * self.gate_factor,
            };
            let sol = solve_pixel(&m);
            k1[ix] = sol.cd as f32;
            k2[ix] = sol.cb as f32;
            k3[ix] = sol.ch as f32;
            kw[ix] = sol.wgt as f32;
        }
        drop((s11, s12, s13, s22, s23, s33, t1, t2, t3, q, p_bar));

        // 3.10 smoothing and correction, 3.11 strength.
        let m1 = box_mean(&k1, w, h, 2, false);
        let m2 = box_mean(&k2, w, h, 2, false);
        let m3 = box_mean(&k3, w, h, 2, false);
        let mw = box_mean(&kw, w, h, 2, false);
        let mut out = vec![0.0f64; n];
        for ix in 0..n {
            let y = f64::from(luma[ix]);
            let num = f64::from(dark[ix]) * f64::from(m1[ix])
                + f64::from(bright[ix]) * f64::from(m2[ix])
                + f64::from(hp[ix]) * f64::from(m3[ix]);
            let l = y + num / f64::from(mw[ix]);
            out[ix] = y + self.strength * (l - y);
        }
        (out, luma)
    }
}

// ---------------------------------------------------------------------------------------------
// Planning: PSF, kernel radius, high-pass sigma (spec sections 3.1 to 3.3)
// ---------------------------------------------------------------------------------------------

/// Number of geometries the planner keeps (`(p_eff, N)` pairs).
const CACHE_ENTRIES: usize = 8;

type CacheKey = (u64, u64);
type CacheEntry = (CacheKey, Option<Arc<Geometry>>);

/// The planner's results by `(p_eff, N)`, most recent first. The geometry is a pure function of
/// the key, so the cache changes speed only.
static CACHE: Mutex<Vec<CacheEntry>> = Mutex::new(Vec::new());

/// The geometry for `(p_eff, N)`, from the cache or planned. Both numbers must already pass the
/// gate (`p_eff > 0` and finite, `N` in `[1, 50]`).
fn geometry(pitch_eff_um: f64, f_number: f64) -> Option<Arc<Geometry>> {
    let key = (pitch_eff_um.to_bits(), f_number.to_bits());
    {
        let mut cache = CACHE.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some(at) = cache.iter().position(|(k, _)| *k == key) {
            let entry = cache.remove(at);
            let found = entry.1.clone();
            cache.insert(0, entry);
            return found;
        }
    }
    // Planned outside the lock (a few milliseconds): two threads that ask for the same key at the
    // same time both plan it, and `remember` hands both the entry that got in first.
    let planned = plan_geometry(pitch_eff_um, f_number).map(Arc::new);
    let mut cache = CACHE.lock().unwrap_or_else(PoisonError::into_inner);
    remember(&mut cache, key, planned)
}

/// Puts a planned geometry into `cache` (most recent first, at most [`CACHE_ENTRIES`] entries) and
/// returns the one every caller of `key` must use: the entry that is already there when another
/// thread planned the same key and got in first, else `planned` itself. Returning `planned` in the
/// first case would give this thread an `Arc` that is not the cached one, so two plans of the same
/// geometry would not share their kernel.
fn remember(
    cache: &mut Vec<CacheEntry>,
    key: CacheKey,
    planned: Option<Arc<Geometry>>,
) -> Option<Arc<Geometry>> {
    if let Some((_, existing)) = cache.iter().find(|(k, _)| *k == key) {
        return existing.clone();
    }
    cache.insert(0, (key, planned.clone()));
    cache.truncate(CACHE_ENTRIES);
    planned
}

/// The spectral weight `w(lambda)` of the polychromatic PSF: 1 from 540 nm, a linear ramp up to
/// 1.9875 at 461 nm below (spec section 2.6).
fn spectral_weight(lambda_nm: u32) -> f64 {
    if lambda_nm >= 540 {
        1.0
    } else {
        (620.0 - f64::from(lambda_nm)) / 80.0
    }
}

/// Plans the PSF kernel, its radius, `sigma_G` and the halo (spec sections 3.1 to 3.3).
#[allow(clippy::too_many_lines, clippy::needless_range_loop)] // one planner, kept in the order of the spec; the index loops are its formulas
fn plan_geometry(pitch_eff: f64, f_number: f64) -> Option<Geometry> {
    let n_f = f_number;
    let step = cmin(0.25 * pitch_eff, 0.25 * (n_f * 0.42 / PI));

    // Radial polychromatic Airy profile, 1 nm steps over 461..619 nm.
    let mut prof = [0.0f64; PROFILE_SAMPLES];
    let mut wsum = 0.0f64;
    for lam in 461u32..=619 {
        let wl = spectral_weight(lam);
        let lam_um = f64::from(lam) * 1e-3;
        for (k, p) in prof.iter_mut().enumerate().skip(1) {
            let x = PI * k as f64 * step / (lam_um * n_f);
            let t = bessel_j1(x) / x;
            *p += 4.0 * wl * t * t;
        }
        wsum += wl;
    }
    prof[0] = wsum;

    // 80 % encircled-energy radius, in samples.
    let c0 = wsum / 6.0;
    let mut total = c0;
    for (k, p) in prof.iter().enumerate().skip(1) {
        total += k as f64 * p;
    }
    let e80 = 0.8 * total;
    let (mut prev, mut cur) = (c0, c0);
    let mut idx: Option<usize> = None;
    for ix in 0..PROFILE_SAMPLES - 1 {
        cur = prev + (ix + 1) as f64 * prof[ix + 1];
        if cur >= e80 {
            idx = Some(ix);
            break;
        }
        prev = cur;
    }
    let idx = idx?;
    let r80 = idx as f64 + (e80 - prev) / cmax(cur - prev, 0.01 * total);
    let r80_um = r80 * step;

    // Truncation and cos^2 taper.
    let support = cmin(4.0 * r80 * step, 6.0 * pitch_eff);
    let nsamp = (support / step) as i64;
    if nsamp < 1 {
        return None;
    }
    let nsamp = (nsamp as usize).min(PROFILE_SAMPLES);
    for (k, p) in prof.iter_mut().enumerate().take(nsamp) {
        let c = (k as f64 * (step / support) * (0.5 * PI)).cos();
        *p *= c * c;
    }

    // Bilinear splat of the rings onto the first quadrant of the pixel grid.
    let mut grid = [[0.0f64; GRID]; GRID];
    let mut add = |row: i64, col: i64, v: f64| {
        if row < 0 || col < 0 || row >= GRID as i64 || col >= GRID as i64 {
            return;
        }
        grid[row as usize][col as usize] += v;
    };
    let mut splat = |pr: f64, pc: f64, v: f64| {
        let fr0 = pr.floor();
        let fc0 = pc.floor();
        let (ir, ic) = (fr0 as i64, fc0 as i64);
        let fr = pr - fr0;
        let fc = pc - fc0;
        add(ir, ic, v * (1.0 - fr) * (1.0 - fc));
        add(ir, ic + 1, v * (1.0 - fr) * fc);
        add(ir + 1, ic, v * fr * (1.0 - fc));
        add(ir + 1, ic + 1, v * fr * fc);
    };
    for k in 1..nsamp {
        let rr = k as f64 * step / pitch_eff; // pixels
        let m = (6.0 * rr + 0.999) as i64;
        if m < 1 {
            continue;
        }
        let v = k as f64 * prof[k] / m as f64;
        for j in 0..m {
            let th = j as f64 * (0.5 * PI) / m as f64;
            let x = rr * th.cos();
            let y = rr * th.sin();
            splat(y, x, v); // (row, col)
            splat(-x, y, v); // rotated by 90 degrees
            splat(-y, -x, v); // 180
            splat(x, -y, v); // 270
        }
    }
    for r in 0..GRID {
        for c in r..GRID {
            let v = cmax(0.0, 0.5 * (grid[r][c] + grid[c][r]));
            grid[r][c] = v;
            grid[c][r] = v;
        }
    }

    // Kernel radius: square energy and ring maxima.
    let mut energy = [0.0f64; GRID]; // energy inside the (2r+1)^2 square
    let mut ring_peak = [0.0f64; GRID]; // peak of ring r
    let mut tail_peak = [0.0f64; GRID]; // peak of rings r..7
    energy[0] = grid[0][0];
    ring_peak[0] = cmax(grid[0][0], 0.0);
    for r in 1..GRID {
        let mut ring = 4.0 * grid[r][0] + 4.0 * grid[r][r];
        let mut mx = cmax(grid[r][0], 0.0);
        for c in 1..r {
            ring += 8.0 * grid[r][c];
            mx = cmax(mx, grid[r][c]);
        }
        energy[r] = energy[r - 1] + ring;
        ring_peak[r] = cmax(mx, grid[r][r]);
    }
    tail_peak[GRID - 1] = ring_peak[GRID - 1];
    for r in (0..GRID - 1).rev() {
        tail_peak[r] = cmax(ring_peak[r], tail_peak[r + 1]);
    }
    let mut radius = MAX_RADIUS;
    for r in 1..MAX_RADIUS {
        if energy[r] >= 0.99 * energy[GRID - 1] && tail_peak[r] < 0.002 * tail_peak[0] {
            radius = r;
            break;
        }
    }
    if energy[radius].is_nan() || energy[radius] <= 0.0 {
        return None;
    }
    let kw = 2 * radius + 1;
    let mut kernel = vec![0.0f32; kw * kw];
    for ix in 0..=radius {
        for j in 0..=radius {
            let v = (grid[ix][j] / energy[radius]) as f32;
            kernel[(radius + ix) * kw + radius + j] = v;
            kernel[(radius + ix) * kw + radius - j] = v;
            kernel[(radius - ix) * kw + radius + j] = v;
            kernel[(radius - ix) * kw + radius - j] = v;
        }
    }

    // sigma_G from the 80 % half-width of the line-spread function at 8 angles.
    let n0 = (17 * radius + 16) / 4;
    let mut x80_sum = 0.0f64;
    for a in 0..8 {
        let th = f64::from(a) * PI / 16.0;
        let (sn, cs) = (th.sin(), th.cos());
        let mut bins = vec![0.0f64; 2 * n0 + 2];
        for dy in -(radius as i64)..=radius as i64 {
            for dx in -(radius as i64)..=radius as i64 {
                let pos = 3.0 * (sn * dy as f64 + cs * dx as f64);
                let fpos = (pos + n0 as f64).floor();
                let bi = fpos as usize;
                let fr = pos + n0 as f64 - fpos;
                let kv = f64::from(
                    kernel[(dy + radius as i64) as usize * kw + (dx + radius as i64) as usize],
                );
                bins[bi] += (1.0 - fr) * kv;
                bins[bi + 1] += fr * kv;
            }
        }
        let raw = bins.clone();
        for m in 1..2 * n0 {
            bins[m] = (raw[m - 1] + raw[m] + raw[m + 1]) / 3.0;
        }
        let c0 = bins[n0];
        let mut c = c0 + bins[n0 - 1] + bins[n0 + 1];
        let mut sx = 1.0 / 3.0;
        let mut sxx = 1.0 / 9.0;
        let mut sc = c0 + c;
        let mut sxc = c / 3.0;
        let mut cnt = 2.0f64;
        for m in 2..n0 {
            c += bins[n0 - m] + bins[n0 + m];
            if c > 0.85 {
                break;
            }
            let x = m as f64 / 3.0;
            sx += x;
            sxx += x * x;
            sc += c;
            sxc += x * c;
            cnt += 1.0;
        }
        let d = sxx * cnt - sx * sx;
        let slope = (cnt * sxc - sx * sc) / d;
        let icpt = (sxx * sc - sx * sxc) / d;
        x80_sum += (0.8 - icpt) / slope;
    }
    let sg = 0.75 * (x80_sum / 8.0);
    if !sg.is_finite() {
        return None;
    }
    let sg = cmin(cmax(sg, 0.2), 3.0);
    let gauss_sigma = sg as f32;
    let gauss_radius =
        ((3.5 * f64::from(gauss_sigma) + 1.0).floor() as usize).min(MAX_GAUSS_RADIUS);
    Some(Geometry {
        radius,
        kernel,
        gauss_sigma,
        gauss_radius,
        halo_px: 10 + radius + gauss_radius,
        step_um: step,
        r80_um,
        support_um: support,
    })
}

// ---------------------------------------------------------------------------------------------
// Bessel J1
// ---------------------------------------------------------------------------------------------

/// The Bessel function of the first kind of order 1.
///
/// The power series (Abramowitz and Stegun 9.1.10) below `|x| = 12`, the Hankel asymptotic
/// expansion (9.2.5) above, summed until its terms start to grow. Odd in `x`. Absolute error
/// below 1e-10 on `[0, 120]` (the planner needs `x <= 114`); see the module documentation.
#[must_use]
pub fn bessel_j1(x: f64) -> f64 {
    let ax = x.abs();
    let r = if ax < 12.0 {
        let half = 0.5 * ax;
        let h2 = half * half;
        let mut term = half;
        let mut sum = half;
        for k in 1..100 {
            let k = f64::from(k);
            term *= -h2 / (k * (k + 1.0));
            sum += term;
            if term.abs() <= 1e-17 * sum.abs() {
                break;
            }
        }
        sum
    } else {
        // J1(z) ~ sqrt(2 / (pi z)) (P cos(chi) - Q sin(chi)), chi = z - 3 pi / 4, with
        // P = sum (-1)^k a_2k / z^2k and Q = sum (-1)^k a_2k+1 / z^(2k+1),
        // a_k = (4 - 1)(4 - 9)...(4 - (2k - 1)^2) / (k! 8^k).
        let (mut p, mut q) = (1.0f64, 0.0f64);
        let (mut coef, mut xpow, mut last) = (1.0f64, 1.0f64, f64::MAX);
        for k in 1..60u32 {
            let odd = f64::from(2 * k - 1);
            coef *= (4.0 - odd * odd) / (8.0 * f64::from(k));
            xpow *= ax;
            let t = coef / xpow;
            if t.abs() >= last {
                break; // the asymptotic series has started to grow
            }
            last = t.abs();
            match k % 4 {
                1 => q += t,
                2 => p -= t,
                3 => q -= t,
                _ => p += t,
            }
            if t.abs() < 1e-17 {
                break;
            }
        }
        let chi = ax - 0.75 * PI;
        (2.0 / (PI * ax)).sqrt() * (p * chi.cos() - q * chi.sin())
    };
    if x < 0.0 { -r } else { r }
}

// ---------------------------------------------------------------------------------------------
// Image operations (spec sections 3.5 to 3.8, 3.10)
// ---------------------------------------------------------------------------------------------

/// `std::max` of C++ (`a < b ? b : a`): keeps `a` when either is `NaN`.
#[inline(always)]
fn cmax(a: f64, b: f64) -> f64 {
    if a < b { b } else { a }
}

/// `std::min` of C++ (`b < a ? b : a`): keeps `a` when either is `NaN`.
#[inline(always)]
fn cmin(a: f64, b: f64) -> f64 {
    if b < a { b } else { a }
}

/// Sorts three values ascending.
#[inline(always)]
fn sort3(e: &mut [f64; 3]) {
    if e[1] < e[0] {
        e.swap(0, 1);
    }
    if e[2] < e[1] {
        e.swap(1, 2);
    }
    if e[1] < e[0] {
        e.swap(0, 1);
    }
}

/// Row `y` of `plane` read through the mirror rule into `padded`, `r` samples beyond each end, as
/// `f64`.
fn padded_row<T: Copy + Into<f64>>(plane: &[T], w: usize, y: usize, r: usize, padded: &mut [f64]) {
    let row = &plane[y * w..(y + 1) * w];
    for (px, p) in padded.iter_mut().enumerate() {
        *p = row[mirror(px as isize - r as isize, w)].into();
    }
}

/// Mean over a `(2r + 1)` square window (mirror rule), accumulated in `f64` (rows, then columns),
/// rounded once to `f32` and optionally clamped at zero (`NaN` reads as 0).
fn box_mean<T: Copy + Into<f64>>(
    src: &[T],
    w: usize,
    h: usize,
    r: usize,
    clamp_zero: bool,
) -> Vec<f32> {
    let n = w * h;
    let width = 2 * r + 1;
    let mut rows = vec![0.0f64; n];
    let mut padded = vec![0.0f64; w + 2 * r];
    for y in 0..h {
        padded_row(src, w, y, r, &mut padded);
        let acc = &mut rows[y * w..(y + 1) * w];
        for k in 0..width {
            for (a, &p) in acc.iter_mut().zip(&padded[k..k + w]) {
                *a += p;
            }
        }
    }
    let area = (width * width) as f64;
    let mut out = vec![0.0f32; n];
    let mut acc = vec![0.0f64; w];
    for y in 0..h {
        acc.fill(0.0);
        for k in 0..width {
            let yy = mirror(y as isize + k as isize - r as isize, h);
            for (a, &v) in acc.iter_mut().zip(&rows[yy * w..(yy + 1) * w]) {
                *a += v;
            }
        }
        for (o, &s) in out[y * w..(y + 1) * w].iter_mut().zip(&acc) {
            let mut m = s / area;
            if clamp_zero && (m.is_nan() || m < 0.0) {
                m = 0.0;
            }
            *o = m as f32;
        }
    }
    out
}

/// Separable Gaussian with taps `exp(-i^2 / (2 sigma^2))` for `|i| <= r`, normalised to sum 1.
fn gauss_blur(src: &[f32], w: usize, h: usize, sigma: f64, r: usize) -> Vec<f32> {
    let width = 2 * r + 1;
    let mut taps = vec![0.0f64; width];
    let mut sum = 0.0f64;
    for (k, t) in taps.iter_mut().enumerate() {
        let ix = k as i64 - r as i64;
        *t = (-((ix * ix) as f64) / (2.0 * sigma * sigma)).exp();
        sum += *t;
    }
    for t in &mut taps {
        *t /= sum;
    }
    let n = w * h;
    let mut rows = vec![0.0f64; n];
    let mut padded = vec![0.0f64; w + 2 * r];
    for y in 0..h {
        padded_row(src, w, y, r, &mut padded);
        let acc = &mut rows[y * w..(y + 1) * w];
        for (k, &t) in taps.iter().enumerate() {
            for (a, &p) in acc.iter_mut().zip(&padded[k..k + w]) {
                *a += t * p;
            }
        }
    }
    let mut out = vec![0.0f32; n];
    let mut acc = vec![0.0f64; w];
    for y in 0..h {
        acc.fill(0.0);
        for (k, &t) in taps.iter().enumerate() {
            let yy = mirror(y as isize + k as isize - r as isize, h);
            for (a, &v) in acc.iter_mut().zip(&rows[yy * w..(yy + 1) * w]) {
                *a += t * v;
            }
        }
        for (o, &s) in out[y * w..(y + 1) * w].iter_mut().zip(&acc) {
            *o = s as f32;
        }
    }
    out
}

/// 2-D convolution with the (symmetric) PSF kernel of radius `r`.
fn convolve_psf(src: &[f32], w: usize, h: usize, kernel: &[f32], r: usize) -> Vec<f32> {
    let kw = 2 * r + 1;
    let mut out = vec![0.0f32; w * h];
    let mut padded = vec![0.0f64; w + 2 * r];
    let mut acc = vec![0.0f64; w];
    for y in 0..h {
        acc.fill(0.0);
        for dy in 0..kw {
            let sy = mirror(y as isize + dy as isize - r as isize, h);
            padded_row(src, w, sy, r, &mut padded);
            for dx in 0..kw {
                let k = f64::from(kernel[dy * kw + dx]);
                for (a, &p) in acc.iter_mut().zip(&padded[dx..dx + w]) {
                    *a += k * p;
                }
            }
        }
        for (o, &s) in out[y * w..(y + 1) * w].iter_mut().zip(&acc) {
            *o = s as f32;
        }
    }
    out
}

/// Lower and upper envelope estimates over a 5x5 window (spec section 3.5).
fn envelopes(luma: &[f32], w: usize, h: usize) -> (Vec<f32>, Vec<f32>) {
    let n = w * h;
    let pad = filters::pad_mirror(luma, w, h, 2);
    let pw = w + 4;
    let mut lower = vec![0.0f32; n];
    let mut upper = vec![0.0f32; n];
    for y in 0..h {
        for x in 0..w {
            let c = f64::from(luma[y * w + x]);
            let (mut sp, mut sn) = (0.0f64, 0.0f64);
            let (mut np, mut nn) = (0i64, 0i64);
            for dy in 0..5 {
                for &v in &pad[(y + dy) * pw + x..(y + dy) * pw + x + 5] {
                    let d = f64::from(v) - c;
                    if d >= 0.0 {
                        sp += d;
                        np += 1;
                    }
                    if d <= 0.0 {
                        sn += d;
                        nn += 1;
                    }
                }
            }
            let gp = 2.0 * sp / ((np * (np - 1)) as f64 + 1e-20);
            let gn = -2.0 * sn / ((nn * (nn - 1)) as f64 + 1e-20);
            let top = c + gp * (np - 1) as f64;
            let bottom = c - gn * (nn - 1) as f64;
            let mid = 0.5 * (top + bottom);
            let pull = 12.5 * cmin(cmin(gp, gn), 0.04 * (top - bottom));
            lower[y * w + x] = cmin(bottom + pull, mid) as f32;
            upper[y * w + x] = cmax(top - pull, mid) as f32;
        }
    }
    (lower, upper)
}

// ---------------------------------------------------------------------------------------------
// The per-pixel solve (spec section 3.9)
// ---------------------------------------------------------------------------------------------

/// The weighted moments of one pixel (spec section 3.8).
struct Moments {
    s11: f64,
    s12: f64,
    s13: f64,
    s22: f64,
    s23: f64,
    s33: f64,
    t1: f64,
    t2: f64,
    t3: f64,
    q: f64,
    /// 9x9 mean of the clip penalty.
    p: f64,
}

/// The coefficients of one pixel, already multiplied by `gain * wgt`, and the confidence weight.
struct Solution {
    cd: f64,
    cb: f64,
    ch: f64,
    wgt: f64,
}

/// The constrained, ridge-regularised 3-parameter least squares with the noise gate.
fn solve_pixel(m: &Moments) -> Solution {
    let t = GATE * m.p;
    if m.q <= t {
        return Solution {
            cd: 0.0,
            cb: 0.0,
            ch: 0.0,
            wgt: 1.0 + (t - m.q) / t,
        };
    }
    let s11 = cmax(m.s11, 1e-15);
    let s22 = cmax(m.s22, 1e-15);
    let s33 = cmax(m.s33, 1e-15);
    let (s12, s13, s23) = (m.s12, m.s13, m.s23);
    let (t1, t2, t3) = (m.t1, m.t2, m.t3);

    let lambda = 0.0005 * (s11 + s22 + s33);
    let (m11, m22, m33) = (s11 + lambda, s22 + lambda, s33 + lambda);
    let cof11 = m22 * m33 - s23 * s23;
    let cof12 = s13 * s23 - s12 * m33;
    let cof13 = s12 * s23 - s13 * m22;
    let cof22 = m11 * m33 - s13 * s13;
    let cof23 = s12 * s13 - s23 * m11;
    let cof33 = m11 * m22 - s12 * s12;
    let det = m11 * cof11 + s12 * cof12 + s13 * cof13;

    let (mut cd, mut cb);
    let mut ch = (t1 * cof13 + t2 * cof23 + t3 * cof33) / det;
    let mut tie = false;
    if (0.0..1.0).contains(&ch) {
        cd = (t1 * cof11 + t2 * cof12 + t3 * cof13) / det;
        cb = (t1 * cof12 + t2 * cof22 + t3 * cof23) / det;
        tie = cd > cb;
    } else {
        ch = cmin(cmax(ch, 0.0), 1.0);
        let t1p = t1 - s13 * ch;
        let t2p = t2 - s23 * ch;
        let d2 = s11 * s22 - s12 * s12;
        cd = (s22 * t1p - s12 * t2p) / d2;
        cb = (s11 * t2p - s12 * t1p) / d2;
        if cd > cb {
            let c = (t1p + t2p) / (s11 + 2.0 * s12 + s22);
            cd = c;
            cb = c;
        }
    }
    if tie {
        // Re-solve with cd = cb = c, unknowns (c, ch).
        let a = s11 + 2.0 * s12 + s22;
        let b = s13 + s23;
        let ts = t1 + t2;
        let d = a * s33 - b * b;
        ch = (a * t3 - b * ts) / d;
        let c;
        if (0.0..1.0).contains(&ch) {
            c = (s33 * ts - b * t3) / d;
        } else {
            ch = cmin(cmax(ch, 0.0), 1.0);
            c = (ts - b * ch) / a;
        }
        cd = c;
        cb = c;
    }
    if cd < -0.5 {
        cd = -0.5;
        cb = cmax(cb, -0.5);
    }
    if !cd.is_finite() || !cb.is_finite() || !ch.is_finite() {
        // A degenerate system: leave the pixel alone.
        return Solution {
            cd: 0.0,
            cb: 0.0,
            ch: 0.0,
            wgt: 1.0,
        };
    }

    // Weighted residual energy after the fit (diagonal clamped as above, no ridge).
    let e = m.q - 2.0 * (cd * t1 + cb * t2 + ch * t3)
        + s11 * cd * cd
        + s22 * cb * cb
        + s33 * ch * ch
        + 2.0 * (s12 * cd * cb + s13 * cd * ch + s23 * cb * ch);

    let mut gain = if m.q >= m.p {
        1.0
    } else {
        cmax((m.q - t) / (m.p - t), 0.0)
    };
    let k = 1.0 / GATE;
    let lo = 0.25 * m.q + k;
    if e > lo {
        let hi = 0.5 * m.q + k;
        gain *= cmax((hi - e) / (hi - lo), 0.0);
    }
    let mut wgt = 1.0;
    if e > k {
        wgt = cmax((k / e) * (k / e), 0.01);
    }
    let f = gain * wgt;
    Solution {
        cd: cd * f,
        cb: cb * f,
        ch: ch * f,
        wgt,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bessel_j1_matches_tabulated_values() {
        // Values to 16 digits (NIST DLMF / scipy.special.j1).
        let table = [
            (0.5, 0.2422684576748739),
            (1.0, 0.4400505857449335),
            (2.0, 0.576_724_807_756_873_3),
            (5.0, -0.327_579_137_591_465_2),
            (10.0, 0.043_472_746_168_861_44),
            (11.9, -0.228_983_249_661_924_04),
            (12.0, -0.223_447_104_490_627_66),
            (12.1, -0.215_748_973_376_924_8),
            (20.0, 0.066_833_124_175_850_02),
            (50.0, -0.097_511_828_125_175_16),
            (100.0, -0.077_145_352_014_112_17),
        ];
        let mut worst = 0.0f64;
        for (x, want) in table {
            worst = worst.max((bessel_j1(x) - want).abs());
            // Odd in x.
            assert_eq!(bessel_j1(-x), -bessel_j1(x));
        }
        assert!(worst < 1e-12, "worst error {worst:e}");
        assert_eq!(bessel_j1(0.0), 0.0);
    }

    #[test]
    fn the_spec_constants_are_consistent() {
        assert_eq!(MAX_HALO_PX, 10 + MAX_RADIUS + MAX_GAUSS_RADIUS);
        assert_eq!(spectral_weight(461), 1.9875);
        assert_eq!(spectral_weight(539), 1.0125);
        assert_eq!(spectral_weight(540), 1.0);
        assert_eq!(spectral_weight(619), 1.0);
    }

    #[test]
    fn params_read_out_of_range_values_as_the_spec_says() {
        let p = DiffractionParams::new(3.76, 8.0);
        assert_eq!(p.effective_scale(), 1.0);
        assert_eq!(p.with_scale(0.25).effective_scale(), 0.25);
        for bad in [0.0, -1.0, 1.5, f64::NAN, f64::INFINITY] {
            assert_eq!(p.with_scale(bad).effective_scale(), 1.0, "{bad}");
        }
        assert_eq!(p.with_strength(f32::NAN).effective_strength(), 1.0);
        assert_eq!(p.with_strength(-2.0).effective_strength(), 0.0);
        assert_eq!(p.with_strength(9.0).effective_strength(), 3.0);
        assert_eq!(p.with_strength(0.5).effective_strength(), 0.5);
        let unequal = DiffractionParams {
            pixel_pitch_um: [3.76, 4.0],
            ..p
        };
        assert_eq!(unequal.effective_pitch_um(), 4.0);
        assert_eq!(p.with_scale(0.25).effective_pitch_um(), 15.04);
    }

    #[test]
    fn the_gate_refuses_bad_pitch_and_f_number() {
        let g0 = [2.0, 1.0, 1.5];
        let noise = NoiseDn::ZERO;
        let plan = |p: DiffractionParams| DiffractionPlan::new(&p, g0, &noise, SPEC_LUMA);
        assert!(plan(DiffractionParams::new(3.76, 8.0)).is_some());
        assert!(plan(DiffractionParams::new(3.76, 1.0)).is_some());
        assert!(plan(DiffractionParams::new(3.76, 50.0)).is_some());
        for n in [0.0, 0.99, 50.01, f64::NAN, -8.0, f64::INFINITY] {
            assert!(plan(DiffractionParams::new(3.76, n)).is_none(), "f/{n}");
            assert_eq!(DiffractionParams::new(3.76, n).halo_px(), 0);
        }
        for pitch in [0.0, -3.0, f64::NAN, f64::INFINITY] {
            assert!(
                plan(DiffractionParams::new(pitch, 8.0)).is_none(),
                "{pitch}"
            );
        }
        let mut one_bad = DiffractionParams::new(3.76, 8.0);
        one_bad.pixel_pitch_um = [3.76, 0.0];
        assert!(plan(one_bad).is_none());
        for bad in [0.0, -1.0, f32::NAN, f32::INFINITY] {
            assert!(
                DiffractionPlan::new(
                    &DiffractionParams::new(3.76, 8.0),
                    [2.0, bad, 1.5],
                    &noise,
                    SPEC_LUMA
                )
                .is_none(),
                "g0 {bad}"
            );
        }
    }

    #[test]
    fn a_nan_or_infinite_noise_floor_reads_as_zero() {
        let p = DiffractionParams::new(3.76, 8.0);
        let g0 = [2.0, 1.0, 1.5];
        let zero = DiffractionPlan::new(&p, g0, &NoiseDn::new(0.0, 0.0), SPEC_LUMA).unwrap();
        assert_eq!(zero.sigma0_dn(), 1.0, "the 1 DN floor");
        for r in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let plan = DiffractionPlan::new(&p, g0, &NoiseDn::new(1.0, r), SPEC_LUMA).unwrap();
            assert_eq!(plan.sigma0_dn(), 1.0, "r = {r}");
        }
        // A negative gain reads as no shot noise.
        let plan = DiffractionPlan::new(&p, g0, &NoiseDn::new(-1.0, 5.0), SPEC_LUMA).unwrap();
        assert_eq!(plan.sigma1_dn(), 0.0);
    }

    #[test]
    fn the_plan_gives_back_the_gains_and_luma_weights_it_was_made_with() {
        let p = DiffractionParams::new(3.76, 8.0);
        let noise = NoiseDn::new(1.0, 5.0);
        // Values with no exact short form, so that a round trip through f64 and back is tested.
        let g0 = [2.1_f32, 1.0, 1.7_f32 / 3.0];
        let luma = [0.2126_f32, 0.7152, 0.0722];
        let plan = DiffractionPlan::new(&p, g0, &noise, luma).unwrap();
        assert_eq!(plan.g0().map(f32::to_bits), g0.map(f32::to_bits));
        assert_eq!(plan.luma().map(f32::to_bits), luma.map(f32::to_bits));
        // The gate variant keeps them.
        let gated = plan.with_noise_gate(true);
        assert_eq!(gated.g0(), g0);
        assert_eq!(gated.luma(), luma);
    }

    #[test]
    fn a_plan_that_lost_the_race_gets_the_cached_geometry() {
        // Two threads plan the same key; the second to take the lock must use the first one's
        // entry, not its own. (The race itself is in `tests/cpu_diffraction_cache.rs`.)
        let key = (3.76_f64.to_bits(), 8.0_f64.to_bits());
        let first = plan_geometry(3.76, 8.0).map(Arc::new);
        let second = plan_geometry(3.76, 8.0).map(Arc::new);
        assert!(!Arc::ptr_eq(
            first.as_ref().unwrap(),
            second.as_ref().unwrap()
        ));
        let mut cache: Vec<CacheEntry> = Vec::new();
        let won = remember(&mut cache, key, first.clone());
        let lost = remember(&mut cache, key, second);
        assert!(Arc::ptr_eq(won.as_ref().unwrap(), first.as_ref().unwrap()));
        assert!(
            Arc::ptr_eq(lost.as_ref().unwrap(), first.as_ref().unwrap()),
            "the second caller must get the entry that is cached"
        );
        assert_eq!(cache.len(), 1, "the loser added nothing");
        // A geometry that cannot be planned is cached as such, and a later plan does not replace it.
        let none_key = (0.5_f64.to_bits(), 8.0_f64.to_bits());
        assert!(remember(&mut cache, none_key, None).is_none());
        assert!(remember(&mut cache, none_key, first.clone()).is_none());
        // Most recent first, and no more than the cache holds.
        for i in 0..20_u64 {
            remember(&mut cache, (i, i), first.clone());
        }
        assert_eq!(cache.len(), CACHE_ENTRIES);
        assert_eq!(cache[0].0, (19, 19));
    }

    #[test]
    fn the_geometry_is_cached_by_pitch_and_f_number() {
        let a = geometry(3.76, 8.0).unwrap();
        let b = geometry(3.76, 8.0).unwrap();
        assert!(Arc::ptr_eq(&a, &b), "the second call must hit the cache");
        let other = geometry(3.76, 11.0).unwrap();
        assert!(!Arc::ptr_eq(&a, &other));
        assert_eq!(*a, *plan_geometry(3.76, 8.0).as_ref().unwrap());
        // More keys than entries: still correct.
        for i in 0..20 {
            let g = geometry(2.0 + 0.1 * f64::from(i), 8.0).unwrap();
            assert_eq!(*g, plan_geometry(2.0 + 0.1 * f64::from(i), 8.0).unwrap());
        }
    }
}
