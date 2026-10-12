//! Per-pixel sigma map for the network input and the DNG NoiseProfile values
//! (`docs/06-noise.md` sections Model and DNG NoiseProfile).
//!
//! The network sees the four CFA planes at half resolution, balanced with the initial gains `g0`
//! (`docs/05-raw-processing.md` R4) and multiplied by the vignetting gain of R3. The map says how
//! noisy each of those values is:
//!
//! ```text
//! x_hat = box5x5(plane)                              smoothed signal, never the noisy value
//! x     = x_hat / (g0_c * G)                         back to normalised units, before R3 and R4
//! sigma = g0_c * G * sqrt(a max(x - delta_c, 0) + b + sigma_row^2 + q^2/12)
//! ```
//!
//! where `q` is the per-photosite quantisation step from the decoder (normalised units, before the
//! gains) or the model's default step when the file has none. Gains scale the noise exactly as they
//! scale the signal, which is why `sigma` carries the same factor as the plane.
//!
//! The signature is written for a GPU port: borrowed planes and parameters in, a plane out, no
//! allocation of anything but scratch rows, and [`SigmaParams`] is a plain 48-byte struct that can
//! be uploaded as a uniform. The kernel will be checked against this reference.
//!
//! **One plane or four.** `docs/07-ml-models.md` gives the network a single sigma plane. The
//! per-position sigmas differ by the balanced gains, so [`sigma_planes`] returns all four and
//! [`sigma_quad`] reduces them to one with the quadrature (root-mean-square) mean over the four
//! positions of each 2 x 2 cell. The training sampler must use the same reduction.

use bytemuck::{Pod, Zeroable};

use crate::model::{NUM_PLANES, NoiseModel};

/// Radius of the box filter that smooths the signal (5 x 5).
pub const BOX_RADIUS: usize = 2;

/// Model parameters and gains in the layout a GPU uniform buffer wants (12 floats, 48 bytes).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub struct SigmaParams {
    /// Shot-noise gain `a`.
    pub a: f32,
    /// `b + sigma_row^2`.
    pub k0: f32,
    /// Quantisation step used where no per-photosite map is given, normalised units.
    pub q_default: f32,
    /// Unused; keeps the struct a multiple of 16 bytes.
    pub reserved: f32,
    /// Per-plane offsets `delta_c`.
    pub delta: [f32; NUM_PLANES],
    /// Per-plane balancing gains `g0` of R4 (green = 1).
    pub g0: [f32; NUM_PLANES],
}

impl SigmaParams {
    /// Parameters for `model` and the initial gains `g0`.
    pub fn new(model: &NoiseModel, g0: [f32; NUM_PLANES]) -> Self {
        Self {
            a: model.a as f32,
            k0: (model.b + model.sigma_row * model.sigma_row) as f32,
            q_default: model.q as f32,
            reserved: 0.0,
            delta: model.delta.map(|d| d as f32),
            g0,
        }
    }
}

/// Why a sigma map could not be computed.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum SigmaMapError {
    /// A buffer does not have the size its description says.
    #[error("{what} has {got} values, expected {want}")]
    Size {
        /// Which buffer.
        what: &'static str,
        /// Values found.
        got: usize,
        /// Values expected.
        want: usize,
    },
    /// The plane dimensions are so large that their sizes cannot be represented.
    #[error("{width} x {height} planes are too large")]
    TooLarge {
        /// Plane width.
        width: usize,
        /// Plane height.
        height: usize,
    },
}

/// Four half-resolution CFA planes, plane after plane: index `(p * height + y) * width + x`.
#[derive(Clone, Copy, Debug)]
pub struct HalfPlanes<'a> {
    /// `4 * width * height` values.
    pub data: &'a [f32],
    /// Plane width (half the mosaic width).
    pub width: usize,
    /// Plane height (half the mosaic height).
    pub height: usize,
}

/// Quantisation steps for the sigma map, normalised units before the gains.
#[derive(Clone, Copy, Debug)]
pub enum QstepMap<'a> {
    /// No map: every photosite uses [`SigmaParams::q_default`].
    None,
    /// One step per photosite, laid out like [`HalfPlanes`].
    Packed(&'a [f32]),
    /// One half-resolution plane shared by the four positions.
    Single(&'a [f32]),
}

fn check(what: &'static str, got: usize, want: usize) -> Result<(), SigmaMapError> {
    if got == want {
        Ok(())
    } else {
        Err(SigmaMapError::Size { what, got, want })
    }
}

/// 5 x 5 box mean of one `w x h` plane with edge replication, written to `out`.
fn box5(plane: &[f32], w: usize, h: usize, out: &mut [f32], scratch: &mut [f32]) {
    let r = BOX_RADIUS as isize;
    let norm = 1.0 / ((2 * BOX_RADIUS + 1) as f32);
    // Horizontal pass.
    for y in 0..h {
        let row = &plane[y * w..(y + 1) * w];
        let dst = &mut scratch[y * w..(y + 1) * w];
        for (x, d) in dst.iter_mut().enumerate() {
            let mut s = 0.0;
            for k in -r..=r {
                s += row[(x as isize + k).clamp(0, w as isize - 1) as usize];
            }
            *d = s * norm;
        }
    }
    // Vertical pass.
    for y in 0..h {
        let dst = &mut out[y * w..(y + 1) * w];
        dst.fill(0.0);
        for k in -r..=r {
            let src = &scratch[(y as isize + k).clamp(0, h as isize - 1) as usize * w..][..w];
            for (d, s) in dst.iter_mut().zip(src) {
                *d += *s;
            }
        }
        for d in dst.iter_mut() {
            *d *= norm;
        }
    }
}

fn validate(
    signal: &HalfPlanes<'_>,
    qstep: &QstepMap<'_>,
    gain: Option<&[f32]>,
) -> Result<usize, SigmaMapError> {
    let too_large = || SigmaMapError::TooLarge {
        width: signal.width,
        height: signal.height,
    };
    let n = signal
        .width
        .checked_mul(signal.height)
        .ok_or_else(too_large)?;
    let total = n.checked_mul(NUM_PLANES).ok_or_else(too_large)?;
    check("signal planes", signal.data.len(), total)?;
    match qstep {
        QstepMap::None => {}
        QstepMap::Packed(q) => check("packed qstep planes", q.len(), total)?,
        QstepMap::Single(q) => check("qstep plane", q.len(), n)?,
    }
    if let Some(g) = gain {
        check("gain plane", g.len(), n)?;
    }
    Ok(n)
}

/// Computes the four sigma planes (`out` has `4 * width * height` values, laid out like the input).
///
/// `signal` is the network input (balanced and vignetting-corrected); `gain` is the optional
/// half-resolution vignetting **gain**, the multiplier that R3 applied to each photosite
/// (`VignetteProfile::gain_at` in `ym-gpu`, which is `1 / G(r)` in the notation of
/// `docs/05-raw-processing.md`: at least 1 where a lens darkens the corners; 1 where absent). The
/// signal was multiplied by it, so the noise of the corrected photosite is multiplied by it too.
///
/// # Errors
///
/// Returns [`SigmaMapError::Size`] when a buffer has the wrong length.
pub fn sigma_planes(
    params: &SigmaParams,
    signal: &HalfPlanes<'_>,
    qstep: QstepMap<'_>,
    gain: Option<&[f32]>,
    out: &mut [f32],
) -> Result<(), SigmaMapError> {
    let n = validate(signal, &qstep, gain)?;
    check("sigma output", out.len(), n * NUM_PLANES)?;
    let (w, h) = (signal.width, signal.height);
    let mut smooth = vec![0.0f32; n];
    let mut scratch = vec![0.0f32; n];
    for p in 0..NUM_PLANES {
        box5(
            &signal.data[p * n..(p + 1) * n],
            w,
            h,
            &mut smooth,
            &mut scratch,
        );
        let dst = &mut out[p * n..(p + 1) * n];
        for i in 0..n {
            let g_total = params.g0[p] * gain.map_or(1.0, |g| g[i]);
            let inv = if g_total > 0.0 { 1.0 / g_total } else { 1.0 };
            let x = smooth[i] * inv;
            let q = match qstep {
                QstepMap::None => params.q_default,
                QstepMap::Packed(q) => q[p * n + i],
                QstepMap::Single(q) => q[i],
            };
            let var = params.a * (x - params.delta[p]).max(0.0) + params.k0 + q * q * (1.0 / 12.0);
            dst[i] = g_total.max(0.0) * var.sqrt();
        }
    }
    Ok(())
}

/// One sigma plane for the network (`out` has `width * height` values): the root-mean-square of the
/// four position sigmas of [`sigma_planes`].
///
/// # Errors
///
/// Returns [`SigmaMapError::Size`] when a buffer has the wrong length.
pub fn sigma_quad(
    params: &SigmaParams,
    signal: &HalfPlanes<'_>,
    qstep: QstepMap<'_>,
    gain: Option<&[f32]>,
    out: &mut [f32],
) -> Result<(), SigmaMapError> {
    let n = validate(signal, &qstep, gain)?;
    check("sigma output", out.len(), n)?;
    let mut planes = vec![0.0f32; NUM_PLANES * n];
    sigma_planes(params, signal, qstep, gain, &mut planes)?;
    for i in 0..n {
        let s2: f32 = (0..NUM_PLANES).map(|p| planes[p * n + i].powi(2)).sum();
        out[i] = (0.25 * s2).sqrt();
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn model() -> NoiseModel {
        NoiseModel {
            a: 2e-4,
            b: 4e-8,
            delta: [1e-4, 0.0, 0.0, -1e-4],
            sigma_row: 1e-4,
            q: 1.0 / 16383.0,
        }
    }

    #[test]
    fn params_are_48_bytes_and_carry_the_model() {
        assert_eq!(std::mem::size_of::<SigmaParams>(), 48);
        let m = model();
        let p = SigmaParams::new(&m, [2.0, 1.0, 1.0, 1.5]);
        assert_relative_eq!(
            f64::from(p.k0),
            m.b + m.sigma_row * m.sigma_row,
            max_relative = 1e-6
        );
        assert_eq!(bytemuck::bytes_of(&p).len(), 48);
    }

    #[test]
    fn flat_plane_gives_the_model_sigma_scaled_by_the_gain() {
        let m = model();
        let (w, h) = (16, 12);
        let g0 = [2.0f32, 1.0, 1.0, 1.5];
        let params = SigmaParams::new(&m, g0);
        let mut data = vec![0.0f32; 4 * w * h];
        let level = 0.3;
        for p in 0..4 {
            data[p * w * h..(p + 1) * w * h].fill(level * g0[p]); // balanced values
        }
        let mut out = vec![0.0f32; 4 * w * h];
        sigma_planes(
            &params,
            &HalfPlanes {
                data: &data,
                width: w,
                height: h,
            },
            QstepMap::None,
            None,
            &mut out,
        )
        .unwrap();
        for p in 0..4 {
            let want = f64::from(g0[p]) * m.sigma(f64::from(level), p);
            for i in 0..w * h {
                assert_relative_eq!(f64::from(out[p * w * h + i]), want, max_relative = 2e-4);
            }
        }
    }

    #[test]
    fn signal_is_smoothed_before_the_square_root() {
        let m = NoiseModel::new(1e-3, 0.0);
        let (w, h) = (21, 21);
        let params = SigmaParams::new(&m, [1.0; 4]);
        let mut data = vec![0.0f32; 4 * w * h];
        data[10 * w + 10] = 1.0; // impulse in plane 0
        let mut out = vec![0.0f32; 4 * w * h];
        sigma_planes(
            &params,
            &HalfPlanes {
                data: &data,
                width: w,
                height: h,
            },
            QstepMap::None,
            None,
            &mut out,
        )
        .unwrap();
        // Inside the 5 x 5 box the smoothed value is 1/25; outside it is zero.
        let inside = f64::from(out[8 * w + 8]);
        assert_relative_eq!(inside, (1e-3f64 / 25.0).sqrt(), max_relative = 1e-4);
        assert_eq!(out[7 * w + 7], 0.0);
        assert_eq!(out[10 * w + 13], 0.0);
    }

    #[test]
    fn qstep_planes_and_gain_enter_the_formula() {
        let m = NoiseModel {
            a: 0.0,
            b: 0.0,
            ..NoiseModel::new(0.0, 0.0)
        };
        let (w, h) = (8, 8);
        let params = SigmaParams::new(&m, [2.0, 1.0, 1.0, 1.0]);
        let data = vec![0.5f32; 4 * w * h];
        let q = 0.01f32;
        let gain = vec![1.25f32; w * h];
        let mut out = vec![0.0f32; 4 * w * h];
        let sig = HalfPlanes {
            data: &data,
            width: w,
            height: h,
        };
        sigma_planes(
            &params,
            &sig,
            QstepMap::Single(&vec![q; w * h]),
            Some(&gain),
            &mut out,
        )
        .unwrap();
        // Only quantisation noise: sigma = g0 G q / sqrt(12).
        assert_relative_eq!(
            f64::from(out[0]),
            2.0 * 1.25 * f64::from(q) / 12f64.sqrt(),
            max_relative = 1e-5
        );
        assert_relative_eq!(
            f64::from(out[w * h]),
            1.25 * f64::from(q) / 12f64.sqrt(),
            max_relative = 1e-5
        );
        let mut packed = vec![0.0f32; 4 * w * h];
        packed[..w * h].fill(0.02);
        packed[w * h..].fill(q);
        sigma_planes(&params, &sig, QstepMap::Packed(&packed), None, &mut out).unwrap();
        assert_relative_eq!(
            f64::from(out[0]),
            2.0 * 0.02 / 12f64.sqrt(),
            max_relative = 1e-5
        );
        assert_relative_eq!(
            f64::from(out[3 * w * h]),
            f64::from(q) / 12f64.sqrt(),
            max_relative = 1e-5
        );
    }

    #[test]
    fn quad_is_the_rms_of_the_four_planes() {
        let m = model();
        let (w, h) = (9, 9);
        let params = SigmaParams::new(&m, [2.0, 1.0, 1.0, 1.6]);
        let data: Vec<f32> = (0..4 * w * h)
            .map(|i| 0.1 + 0.001 * (i % 37) as f32)
            .collect();
        let sig = HalfPlanes {
            data: &data,
            width: w,
            height: h,
        };
        let mut planes = vec![0.0f32; 4 * w * h];
        let mut quad = vec![0.0f32; w * h];
        sigma_planes(&params, &sig, QstepMap::None, None, &mut planes).unwrap();
        sigma_quad(&params, &sig, QstepMap::None, None, &mut quad).unwrap();
        for i in 0..w * h {
            let s2: f32 = (0..4).map(|p| planes[p * w * h + i].powi(2)).sum();
            assert_relative_eq!(quad[i], (s2 / 4.0).sqrt(), max_relative = 1e-6);
        }
    }

    #[test]
    fn vignetting_gain_is_a_multiplier_of_signal_and_noise() {
        // R3 multiplies a photosite by 1/G(r); the sigma map must undo it to find the raw level
        // (a model evaluated at the raw signal) and then scale the sigma by the same factor.
        let m = NoiseModel::new(1e-3, 4e-8);
        let (w, h) = (12, 12);
        let params = SigmaParams::new(&m, [1.0; 4]);
        let raw = 0.1f64;
        let gain = 2.0f32;
        let data = vec![(raw as f32) * gain; 4 * w * h];
        let sig = HalfPlanes {
            data: &data,
            width: w,
            height: h,
        };
        let mut out = vec![0.0f32; 4 * w * h];
        sigma_planes(
            &params,
            &sig,
            QstepMap::None,
            Some(&vec![gain; w * h]),
            &mut out,
        )
        .unwrap();
        assert_relative_eq!(
            f64::from(out[6 * w + 6]),
            f64::from(gain) * m.sigma(raw, 0),
            max_relative = 1e-5
        );
    }

    #[test]
    fn dimensions_that_overflow_are_reported_not_wrapped() {
        let params = SigmaParams::new(&model(), [1.0; 4]);
        let sig = HalfPlanes {
            data: &[],
            width: usize::MAX / 2 + 1,
            height: 3,
        };
        let mut out = [];
        assert!(matches!(
            sigma_planes(&params, &sig, QstepMap::None, None, &mut out),
            Err(SigmaMapError::TooLarge { .. })
        ));
        let sig = HalfPlanes {
            data: &[],
            width: usize::MAX / 2,
            height: 2,
        };
        assert!(matches!(
            sigma_quad(&params, &sig, QstepMap::None, None, &mut out),
            Err(SigmaMapError::TooLarge { .. })
        ));
    }

    #[test]
    fn wrong_sizes_are_reported() {
        let params = SigmaParams::new(&model(), [1.0; 4]);
        let data = vec![0.0f32; 4 * 4 * 4];
        let sig = HalfPlanes {
            data: &data,
            width: 4,
            height: 4,
        };
        let mut out = vec![0.0f32; 10];
        assert!(matches!(
            sigma_planes(&params, &sig, QstepMap::None, None, &mut out),
            Err(SigmaMapError::Size { .. })
        ));
        let mut out = vec![0.0f32; 64];
        assert!(sigma_planes(&params, &sig, QstepMap::Single(&[0.0; 3]), None, &mut out).is_err());
    }
}
