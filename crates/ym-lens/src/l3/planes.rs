//! The per-photo data of L3, made once from the lens plane (`docs/21` section 3, "Planes and lists").
//!
//! * the **tail plane**: the restored light `R` of the lens plane, halved by 2 x 2 box means until its long
//!   edge is at most 512 px, convolved once, by FFT, with the glow's tail part and the veiling kernel (both
//!   already weighted by their fractions of light), per channel;
//! * the **source list** of the highlights `H` of the lens plane;
//! * the glow kernels (for the core each tile gathers) and the lens's spike tables projected on the camera.
//!
//! The data do not depend on the amounts, so a slider moves nothing here.

use crate::consts::{
    CLIP_RAMP_PLANE, GLOW_BOOST_SHARE, GLOW_HAZE_FROM_HD, GLOW_HAZE_TO_HD, HAZE_NEUTRAL,
    SPIKE_SIZE, TAIL_PLANE_LONG_EDGE,
};
use crate::error::LensError;
use crate::image::ImageRef;
use crate::l3::fft::{Convolver, Kernel2d};
use crate::l3::glow::GlowProfiles;
use crate::l3::lights;
use crate::l3::profile::smoothstep;
use crate::l3::restore::Restore;
use crate::l3::sources::{self, Source, SourceView};
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
    /// The lights whole, one source per connected component (`l3::sources::SourceLists`).
    lights: Vec<Source>,
    glow: Option<GlowProfiles>,
    /// The lens's spike table per camera channel, `129 x 129`.
    spikes: [Vec<f64>; 3],
    /// The share of the small lights' extra headroom the photo keeps (`l3::lights`), 1 for most photos.
    boost_scale: f64,
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

/// A function of a pixel `(i, j)` of the lens plane to three values (red, green, blue).
type PixelFn<'a> = &'a dyn Fn(usize, usize) -> [f64; 3];

/// The values `value` of the lens plane's pixels, reduced to the tail plane's size by 2 x 2 box means: three
/// planes and their size.
fn reduced(
    plane: &ImageRef,
    value: PixelFn,
    cancel: &dyn Fn() -> bool,
) -> Result<([Vec<f64>; 3], usize, usize), LensError> {
    let (w, h) = (plane.width, plane.height);
    let n = w * h;
    if w.max(h) <= TAIL_PLANE_LONG_EDGE {
        let mut out: [Vec<f64>; 3] = std::array::from_fn(|_| vec![0.0; n]);
        for j in 0..h {
            for i in 0..w {
                let r = value(i, j);
                for c in 0..3 {
                    out[c][j * w + i] = r[c];
                }
            }
        }
        return Ok((out, w, h));
    }
    // The first halving reads the values straight from the plane.
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
                        let r = value(sx, sy);
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

/// The restored light of the lens plane, reduced to the tail plane's size, and the extra light of the small
/// clipped lights (what their extra headroom adds, [`GLOW_BOOST_SHARE`] of it in EV), the same size, or `None`
/// when no light is boosted: three planes each and their size.
fn reduced_restored(
    plane: &ImageRef,
    rs: &Restore,
    boost: Option<&[f32]>,
    cancel: &dyn Fn() -> bool,
) -> Result<RestoredPlanes, LensError> {
    let (w, h) = (plane.width, plane.height);
    let n = w * h;
    let pixel = |i: usize, j: usize| -> [f64; 3] {
        let at = j * w + i;
        [
            f64::from(plane.data[at]),
            f64::from(plane.data[n + at]),
            f64::from(plane.data[2 * n + at]),
        ]
    };
    let (base, tw, th) = reduced(plane, &|i, j| rs.restored(pixel(i, j)), cancel)?;
    let extra = match boost {
        Some(b) if b.iter().any(|v| *v > 0.0) => {
            let ev = |i: usize, j: usize| f64::from(b[j * w + i]) * GLOW_BOOST_SHARE;
            let (x, _, _) = reduced(
                plane,
                &|i, j| {
                    let x = pixel(i, j);
                    let (lit, plain) = (rs.restored_ev(x, ev(i, j)), rs.restored(x));
                    [lit[0] - plain[0], lit[1] - plain[1], lit[2] - plain[2]]
                },
                cancel,
            )?;
            Some(x)
        }
        _ => None,
    };
    Ok(RestoredPlanes {
        base,
        extra,
        width: tw,
        height: th,
    })
}

/// The restored light on the tail plane's grid.
struct RestoredPlanes {
    /// The restored light `R`.
    base: [Vec<f64>; 3],
    /// What the small lights' extra headroom adds to it.
    extra: Option<[Vec<f64>; 3]>,
    width: usize,
    height: usize,
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
///
/// The tail is two things. Near a light it is the light's halo, in the light's colour (the glow's tail up to
/// `GLOW_HAZE_FROM_HD`, fading out by `GLOW_HAZE_TO_HD`, per channel). Far from every light it is a haze: the
/// veil and the glow's far tail are one kernel for the three channels, convolved with each channel's light, and
/// `HAZE_NEUTRAL` of it is moved to the mean of the three channels, so a picture's average colour does not
/// tint its blacks. The two parts add up to the glow's tail and the veil as before, light for light.
///
/// The light a small clipped light holds beyond the plain headroom (`restored.extra`) has no core of its own
/// (the core is gathered per tile from the plainly restored light): it goes through the whole glow, the tail
/// kernel scaled to the glow's whole fraction, and the veil, so the glow and the veil take exactly their
/// fractions of all the light the lights hold.
fn tail_plane(
    plan: &LensPlan,
    glow: &GlowProfiles,
    restored: &RestoredPlanes,
    cancel: &dyn Fn() -> bool,
) -> Result<[Vec<f64>; 3], LensError> {
    let (tw, th) = (restored.width, restored.height);
    let half_diag = 0.5 * ((tw * tw + th * th) as f64).sqrt();
    let reach = glow.reach().max(glow.veil_radius());
    let radius_px = (libm::ceil(reach * half_diag) as isize).max(1);
    let hx = (radius_px as usize).min(tw.saturating_sub(1));
    let hy = (radius_px as usize).min(th.saturating_sub(1));
    let conv = Convolver::new(tw, th, hx, hy);
    let m_core = glow.core_mass();
    let st = plan.flare_strengths();
    let (a_g, a_v) = (st.glow, st.veil);

    let (veil, veil_sum) = disc_kernel(radius_px, hx, hy, half_diag, &|r| glow.veil(r));
    let veil_scale = if veil_sum > 0.0 { a_v / veil_sum } else { 0.0 };
    // The glow's tail per channel, split at the radius where it stops being the halo of a light and becomes
    // haze: `far(r)` is 0 up to `GLOW_HAZE_FROM_HD` and 1 from `GLOW_HAZE_TO_HD` (a smoothstep), and the two
    // parts add up to the tail.
    let far = |r: f64| smoothstep((r - GLOW_HAZE_FROM_HD) / (GLOW_HAZE_TO_HD - GLOW_HAZE_FROM_HD));
    struct Tail {
        near: Vec<f64>,
        far: Vec<f64>,
        sum: f64,
    }
    let tails: [Tail; 3] = std::array::from_fn(|c| {
        let (near, near_sum) = disc_kernel(radius_px, hx, hy, half_diag, &|r| {
            glow.tail(c, r) * (1.0 - far(r))
        });
        let (far_part, far_sum) =
            disc_kernel(radius_px, hx, hy, half_diag, &|r| glow.tail(c, r) * far(r));
        Tail {
            near,
            far: far_part,
            sum: near_sum + far_sum,
        }
    });
    // The glow's mass in the tail: the base light's (the core is gathered per tile) and the extra light's (whole).
    let mass_base = [0, 1, 2].map(|c| a_g * (1.0 - m_core[c]));
    let kernel = |data: Vec<f64>| Kernel2d { hx, hy, data };
    let scaled = |t: &[f64], sum: f64, mass: f64| -> Vec<f64> {
        let k = if sum > 0.0 { mass / sum } else { 0.0 };
        t.iter().map(|v| k * v).collect()
    };
    // The haze: the veil and the glow's far tail (the mean over the channels, so one kernel serves the three
    // channels) spread the light of the whole picture, and its colour is the picture's average colour, which is
    // not the light's: `HAZE_NEUTRAL` of it is the mean of its three channels.
    let haze_kernel = |mass: [f64; 3]| {
        let mut data: Vec<f64> = veil.iter().map(|v| veil_scale * v).collect();
        for c in 0..3 {
            let part = scaled(&tails[c].far, tails[c].sum, mass[c] / 3.0);
            for (d, p) in data.iter_mut().zip(&part) {
                *d += p;
            }
        }
        kernel(data)
    };
    let haze_base = conv.kernel_spectrum(&haze_kernel(mass_base));
    let haze_extra = restored
        .extra
        .as_ref()
        .map(|_| conv.kernel_spectrum(&haze_kernel([a_g; 3])));
    let mut haze: [Vec<f64>; 3] = std::array::from_fn(|_| Vec::new());
    for c in 0..3 {
        if cancel() {
            return Err(LensError::Cancelled);
        }
        let mut plane = conv.convolve(&restored.base[c], tw, th, &haze_base);
        if let (Some(extra), Some(spectrum)) = (&restored.extra, &haze_extra) {
            let more = conv.convolve(&extra[c], tw, th, spectrum);
            for (p, m) in plane.iter_mut().zip(&more) {
                *p += m;
            }
        }
        haze[c] = plane;
    }
    for i in 0..haze[0].len() {
        let mean = (haze[0][i] + haze[1][i] + haze[2][i]) / 3.0;
        for plane in &mut haze {
            plane[i] += HAZE_NEUTRAL * (mean - plane[i]);
        }
    }
    // The halo of a light: the glow's tail up to the haze, per channel, in the light's colour.
    let mut out: [Vec<f64>; 3] = std::array::from_fn(|_| Vec::new());
    for c in 0..3 {
        if cancel() {
            return Err(LensError::Cancelled);
        }
        let t = &tails[c];
        let spectrum = conv.kernel_spectrum(&kernel(scaled(&t.near, t.sum, mass_base[c])));
        let mut plane = conv.convolve(&restored.base[c], tw, th, &spectrum);
        if let Some(extra) = &restored.extra {
            let spectrum = conv.kernel_spectrum(&kernel(scaled(&t.near, t.sum, a_g)));
            if cancel() {
                return Err(LensError::Cancelled);
            }
            let more = conv.convolve(&extra[c], tw, th, &spectrum);
            for (p, m) in plane.iter_mut().zip(&more) {
                *p += m;
            }
        }
        for (p, v) in plane.iter_mut().zip(&haze[c]) {
            *p += v;
        }
        out[c] = plane;
    }
    Ok(out)
}

impl FlarePlanes {
    /// The data of a photo without lens effects: nothing to draw.
    pub(crate) fn empty() -> Self {
        FlarePlanes {
            boost_scale: 1.0,
            ..FlarePlanes::default()
        }
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
        let half_diag =
            0.5 * ((plane.width * plane.width + plane.height * plane.height) as f64).sqrt();
        let boost = lights::map(plane, &rs, half_diag, cancel)?;
        let boost_scale = boost.as_ref().map_or(1.0, |b| b.scale);
        // The ghosts and the glow's extra layer draw from the lights with the cap's scale, the starburst and the
        // streak from the lights as they are.
        let rs_ghost = rs.with_boost_scale(boost_scale);
        let ev = boost.as_ref().map(|b| b.ev.as_slice());
        let found = sources::find_all(plane, &rs, &rs_ghost, ev, axis, cancel)?;
        if cancel() {
            return Err(LensError::Cancelled);
        }
        let glow = GlowProfiles::new(plan);
        let restored = reduced_restored(plane, &rs_ghost, ev, cancel)?;
        let (tw, th) = (restored.width, restored.height);
        let tail = tail_plane(plan, &glow, &restored, cancel)?;
        // The spike tables on the camera's channels.
        let lens = plan.lens();
        let weights = plan.camera().weights();
        let spikes: [Vec<f64>; 3] = std::array::from_fn(|c| {
            let mut t = vec![0.0; SPIKE_SIZE * SPIKE_SIZE];
            for (j, w) in weights[c].iter().enumerate() {
                for (d, s) in t.iter_mut().zip(lens.spikes(j)) {
                    *d += w * f64::from(*s);
                }
            }
            t
        });
        Ok(FlarePlanes {
            tail,
            tail_size: (tw, th),
            sources: found.cells,
            lights: found.lights,
            glow: Some(glow),
            spikes,
            boost_scale,
        })
    }

    /// The share of the small lights' extra headroom the photo keeps, 0 to 1.
    pub(crate) fn boost_scale(&self) -> f64 {
        self.boost_scale
    }

    /// The glow kernels.
    pub(crate) fn glow(&self) -> Option<&GlowProfiles> {
        self.glow.as_ref()
    }

    /// The sources and the lights whole.
    pub(crate) fn source_view(&self) -> SourceView<'_> {
        SourceView {
            cells: &self.sources,
            lights: &self.lights,
        }
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
            + (self.sources.len() + self.lights.len()) * std::mem::size_of::<Source>()
            + std::mem::size_of::<Self>()
    }
}
