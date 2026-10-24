//! The per-photo data of L3, made once from the lens plane (`docs/21` section 3, "Planes and lists").
//!
//! * the **tail plane**: the restored light `R` of the lens plane, halved by 2 x 2 box means until its long
//!   edge is at most 512 px, convolved once, by FFT, with the glow's tail part and the veiling kernel (both
//!   already weighted by their fractions of light), per channel;
//! * the **source list** of the highlights `H` of the lens plane;
//! * the glow kernels (for the core each tile gathers) and the lens's spike tables projected on the camera.
//!
//! The data do not depend on the amounts, so a slider moves nothing here.

use crate::consts::{CLIP_RAMP_PLANE, PSF_WAVELENGTHS, SPIKE_SIZE, TAIL_PLANE_LONG_EDGE};
use crate::error::LensError;
use crate::image::ImageRef;
use crate::l3::fft::{Convolver, Kernel2d};
use crate::l3::glow::GlowProfiles;
use crate::l3::restore::Restore;
use crate::l3::sources::{self, Source};
use crate::plan::LensPlan;

/// Subsamples per side for the kernel pixels nearest the centre, where the tail changes within a pixel.
const NEAR_SUBSAMPLES: usize = 8;

/// Pixels from the centre (per axis) that are subsampled.
const NEAR_PIXELS: isize = 2;

/// The data of L3 that every tile of a photo shares.
#[derive(Clone, Debug, Default)]
pub(crate) struct FlarePlanes {
    /// The glow and veil light per channel on the tail plane, row-major.
    tail: [Vec<f64>; 3],
    tail_size: (usize, usize),
    sources: Vec<Source>,
    glow: Option<GlowProfiles>,
    /// The lens's spike table per camera channel, `129 x 129`.
    spikes: [Vec<f64>; 3],
}

/// Halves a plane by 2 x 2 means over the pixels that exist.
fn halve(src: &[f64], w: usize, h: usize) -> (Vec<f64>, usize, usize) {
    let (nw, nh) = (w.div_ceil(2), h.div_ceil(2));
    let mut out = vec![0.0; nw * nh];
    for y in 0..nh {
        for x in 0..nw {
            let mut s = 0.0;
            let mut n = 0.0;
            for dy in 0..2 {
                for dx in 0..2 {
                    let (sx, sy) = (2 * x + dx, 2 * y + dy);
                    if sx < w && sy < h {
                        s += src[sy * w + sx];
                        n += 1.0;
                    }
                }
            }
            out[y * nw + x] = s / n;
        }
    }
    (out, nw, nh)
}

/// The restored light of the lens plane, reduced to the tail plane's size: three planes and their size.
fn reduced_restored(
    plane: &ImageRef,
    rs: &Restore,
    cancel: &dyn Fn() -> bool,
) -> Result<([Vec<f64>; 3], usize, usize), LensError> {
    let (w, h) = (plane.width, plane.height);
    let n = w * h;
    let restored_at = |i: usize, j: usize| -> [f64; 3] {
        let at = j * w + i;
        rs.restored([
            f64::from(plane.data[at]),
            f64::from(plane.data[n + at]),
            f64::from(plane.data[2 * n + at]),
        ])
    };
    if w.max(h) <= TAIL_PLANE_LONG_EDGE {
        let mut out: [Vec<f64>; 3] = std::array::from_fn(|_| vec![0.0; n]);
        for j in 0..h {
            for i in 0..w {
                let r = restored_at(i, j);
                for c in 0..3 {
                    out[c][j * w + i] = r[c];
                }
            }
        }
        return Ok((out, w, h));
    }
    // The first halving reads the restored light straight from the plane.
    if cancel() {
        return Err(LensError::Cancelled);
    }
    let (mut tw, mut th) = (w.div_ceil(2), h.div_ceil(2));
    let mut cur: [Vec<f64>; 3] = std::array::from_fn(|_| vec![0.0; tw * th]);
    for y in 0..th {
        for x in 0..tw {
            let mut s = [0.0; 3];
            let mut cnt = 0.0;
            for dy in 0..2 {
                for dx in 0..2 {
                    let (sx, sy) = (2 * x + dx, 2 * y + dy);
                    if sx < w && sy < h {
                        let r = restored_at(sx, sy);
                        for c in 0..3 {
                            s[c] += r[c];
                        }
                        cnt += 1.0;
                    }
                }
            }
            for c in 0..3 {
                cur[c][y * tw + x] = s[c] / cnt;
            }
        }
    }
    while tw.max(th) > TAIL_PLANE_LONG_EDGE {
        if cancel() {
            return Err(LensError::Cancelled);
        }
        let mut next: [Vec<f64>; 3] = std::array::from_fn(|_| Vec::new());
        let (mut nw, mut nh) = (tw, th);
        for c in 0..3 {
            let (v, a, b) = halve(&cur[c], tw, th);
            next[c] = v;
            nw = a;
            nh = b;
        }
        cur = next;
        tw = nw;
        th = nh;
    }
    Ok((cur, tw, th))
}

/// The sum of a function of the distance over the pixel grid within `radius_px`, with the pixels nearest the
/// centre integrated over their area, and the values inside the window `[-hx, hx] x [-hy, hy]` kept.
///
/// Returns the values (row-major in the window) and the sum over the whole disc.
fn disc_kernel(
    radius_px: isize,
    hx: usize,
    hy: usize,
    half_diag: f64,
    f: &dyn Fn(f64) -> f64,
) -> (Vec<f64>, f64) {
    let side = 2 * hx + 1;
    let mut data = vec![0.0; side * (2 * hy + 1)];
    let mut sum = 0.0;
    let inv = 1.0 / half_diag;
    for dy in -radius_px..=radius_px {
        for dx in -radius_px..=radius_px {
            let v = if dx.abs() <= NEAR_PIXELS && dy.abs() <= NEAR_PIXELS {
                let n = NEAR_SUBSAMPLES;
                let mut s = 0.0;
                for sy in 0..n {
                    for sx in 0..n {
                        let ox = dx as f64 - 0.5 + (sx as f64 + 0.5) / n as f64;
                        let oy = dy as f64 - 0.5 + (sy as f64 + 0.5) / n as f64;
                        s += f((ox * ox + oy * oy).sqrt() * inv);
                    }
                }
                s / (n * n) as f64
            } else {
                f(((dx * dx + dy * dy) as f64).sqrt() * inv)
            };
            sum += v;
            if dx.unsigned_abs() <= hx && dy.unsigned_abs() <= hy {
                let q = (dy + hy as isize) as usize * side + (dx + hx as isize) as usize;
                data[q] = v;
            }
        }
    }
    (data, sum)
}

/// The tail plane: per channel, the glow's tail part and the veil convolved with the restored light.
fn tail_plane(
    plan: &LensPlan,
    glow: &GlowProfiles,
    restored: &[Vec<f64>; 3],
    tw: usize,
    th: usize,
    cancel: &dyn Fn() -> bool,
) -> Result<[Vec<f64>; 3], LensError> {
    let fl = &plan.lens().flare;
    let half_diag = 0.5 * ((tw * tw + th * th) as f64).sqrt();
    let reach = glow.reach().max(glow.veil_radius());
    let radius_px = (libm::ceil(reach * half_diag) as isize).max(1);
    let hx = (radius_px as usize).min(tw.saturating_sub(1));
    let hy = (radius_px as usize).min(th.saturating_sub(1));
    let conv = Convolver::new(tw, th, hx, hy);
    let m_core = glow.core_mass();
    let (a_g, a_v) = (fl.glow.fraction, fl.veil.fraction);

    let (veil, veil_sum) = disc_kernel(radius_px, hx, hy, half_diag, &|r| glow.veil(r));
    let mut out: [Vec<f64>; 3] = std::array::from_fn(|_| Vec::new());
    for c in 0..3 {
        if cancel() {
            return Err(LensError::Cancelled);
        }
        let (tail, tail_sum) = disc_kernel(radius_px, hx, hy, half_diag, &|r| glow.tail(c, r));
        let glow_scale = if tail_sum > 0.0 {
            a_g * (1.0 - m_core[c]) / tail_sum
        } else {
            0.0
        };
        let veil_scale = if veil_sum > 0.0 { a_v / veil_sum } else { 0.0 };
        let data: Vec<f64> = tail
            .iter()
            .zip(&veil)
            .map(|(g, v)| glow_scale * g + veil_scale * v)
            .collect();
        let kernel = Kernel2d { hx, hy, data };
        if cancel() {
            return Err(LensError::Cancelled);
        }
        let spectrum = conv.kernel_spectrum(&kernel);
        drop(kernel);
        if cancel() {
            return Err(LensError::Cancelled);
        }
        out[c] = conv.convolve(&restored[c], tw, th, &spectrum);
    }
    Ok(out)
}

impl FlarePlanes {
    /// The data of a photo without lens effects: nothing to draw.
    pub(crate) fn empty() -> Self {
        FlarePlanes::default()
    }

    /// Makes the data of L3 from the lens plane `plane` (the whole frame at the lens plane's scale).
    ///
    /// # Errors
    ///
    /// [`LensError::Cancelled`] when `cancel` returns true between steps.
    pub(crate) fn build(
        plan: &LensPlan,
        plane: &ImageRef,
        cancel: &dyn Fn() -> bool,
    ) -> Result<Self, LensError> {
        let axis = plan.frame().axis;
        let rs = Restore::new(plan, CLIP_RAMP_PLANE);
        let sources = sources::find(plane, &rs, axis, cancel)?;
        if cancel() {
            return Err(LensError::Cancelled);
        }
        let glow = GlowProfiles::new(plan);
        let (restored, tw, th) = reduced_restored(plane, &rs, cancel)?;
        let tail = tail_plane(plan, &glow, &restored, tw, th, cancel)?;
        // The spike tables on the camera's channels.
        let lens = plan.lens();
        let weights = plan.camera().weights();
        let spikes: [Vec<f64>; 3] = std::array::from_fn(|c| {
            let mut t = vec![0.0; SPIKE_SIZE * SPIKE_SIZE];
            for j in 0..PSF_WAVELENGTHS {
                let w = weights[c][j];
                for (d, s) in t.iter_mut().zip(lens.spikes(j)) {
                    *d += w * f64::from(*s);
                }
            }
            t
        });
        Ok(FlarePlanes {
            tail,
            tail_size: (tw, th),
            sources,
            glow: Some(glow),
            spikes,
        })
    }

    /// The glow kernels.
    pub(crate) fn glow(&self) -> Option<&GlowProfiles> {
        self.glow.as_ref()
    }

    /// The sources, brightest first.
    pub(crate) fn sources(&self) -> &[Source] {
        &self.sources
    }

    /// The tail plane's channel `c`, row-major, with its size.
    pub(crate) fn tail(&self, c: usize) -> (&[f64], usize, usize) {
        (&self.tail[c], self.tail_size.0, self.tail_size.1)
    }

    /// The spike tables per camera channel.
    pub(crate) fn spikes(&self) -> &[Vec<f64>; 3] {
        &self.spikes
    }

    /// Bytes the data hold.
    pub(crate) fn bytes(&self) -> usize {
        self.tail.iter().map(|t| t.len() * 8).sum::<usize>()
            + self.spikes.iter().map(|t| t.len() * 8).sum::<usize>()
            + self.sources.len() * std::mem::size_of::<Source>()
            + std::mem::size_of::<Self>()
    }
}
