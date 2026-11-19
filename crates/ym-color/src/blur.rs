//! The Blur node's law and its exact kernels: the CPU oracle in f64 (`docs/33-blur.md` sections 2.2, 3.1 to 3.5).
//!
//! One law serves every type. With `X` the node's input in linear AWG4, `m` the node's mask and `h` the highlight
//! weight (1 except for Lens with Highlights above 0):
//!
//! ```text
//! omega = h m   (Respect mask edges on)        omega = h   (off)
//! N(p) = sum_q k_p(q) omega(q) X(q)            D(p) = sum_q k_p(q) omega(q)       q over the frame only
//! out  = X + m (N - X D) / (D + EPS)           m = 0 gives the input's bits, m = 1 gives N / (D + EPS)
//! ```
//!
//! The sums run over the frame: a tap outside it has weight 0, so the picture's edge renormalises (section 3.4).
//! The kernels here are **exact definitions**, summed directly in f64 in a fixed order: the pixel-integrated Gaussian
//! cut at 4 sigma, the pixel coverage of a disc or a regular polygon, the length of a line inside each pixel, and the
//! radial line of Zoom. Nothing here is fast; every fast method of BL-06 to BL-09 is held to this module.
//!
//! **Windows.** [`apply`] works on a window of the frame (the whole frame, or a tile grown by its halo and cut to the
//! frame). Taps outside the window have no weight, so a window's pixels at least the kernel's reach from the window's
//! cut edges equal the whole frame's bit for bit (the tile invariance of section 8): the taps are visited in a fixed
//! order, positions are computed in frame coordinates, and a weight of 0 is skipped.
//!
//! Not product code of the node's scheduling: the plan step, tiles and threads are BL-04, the device is BL-05.

use crate::consts::Vec3;
use crate::logc::{logc4_decode_rgb, logc4_encode_rgb};
use crate::warp::smoothstep;

pub mod lens;
pub mod soft;

/// Radius 100 is this share of the picture's long edge (`docs/33` 2.2).
pub const RADIUS_MAX: f64 = 0.04;
/// Motion's Length 100 is this share of the long edge.
pub const MOTION_MAX: f64 = 0.08;
/// Zoom's Amount 100: the scale factor spans this much (a pixel at distance `d` is spread over `0.1 d`).
pub const ZOOM_MAX: f64 = 0.10;
/// A Gaussian is cut at this many sigma.
pub const SIGMA_CUT: f64 = 4.0;
/// Keeps the quotient finite where no weight is in reach. Not a threshold: the term it guards is multiplied by `m`.
pub const EPS: f64 = 1e-12;
/// Lens' Highlights: the stops over middle grey where the weight starts to rise, and where it is full (`docs/33` 3.5).
pub const HIGHLIGHT_EV: [f64; 2] = [2.0, 4.0];
/// Lens' Highlights: the extra weight at full Highlights and 4 stops or more over middle grey (`1 + 15 H`).
pub const HIGHLIGHT_GAIN: f64 = 15.0;
/// The AWG4 luminance row (`docs/02` C9), used by the highlight weight.
pub const AWG4_LUMA: Vec3 = [0.2722287, 0.6740818, 0.0536895];
/// Supersampling per side for the aperture's pixel coverage (8 x 8, `docs/33` 3.5).
pub const COVERAGE_SS: usize = 8;
/// The fewest taps along a Zoom line (the oracle takes more when the line is long, see [`zoom_taps`]).
pub const ZOOM_TAPS_MIN: usize = 64;
/// The fewest taps of the Zoom oracle per pixel of line length (two, so the bilinear reads never undersample).
const ZOOM_TAPS_PER_PX: f64 = 2.0;

/// Why a blur could not run.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum BlurError {
    /// The picture, the mask and the stated size disagree.
    #[error(
        "the picture has {pixels} pixels and the mask {mask}, but the window is {width} x {height}"
    )]
    Size {
        /// Pixels in the picture slice.
        pixels: usize,
        /// Values in the mask slice.
        mask: usize,
        /// Window width, pixels.
        width: usize,
        /// Window height, pixels.
        height: usize,
    },
    /// The window does not lie inside the frame, or the frame is empty.
    #[error(
        "the window at ({x0}, {y0}) of {width} x {height} does not lie inside the {full_w} x {full_h} frame"
    )]
    Window {
        /// Window origin x in the frame.
        x0: usize,
        /// Window origin y in the frame.
        y0: usize,
        /// Window width.
        width: usize,
        /// Window height.
        height: usize,
        /// Frame width.
        full_w: usize,
        /// Frame height.
        full_h: usize,
    },
    /// A setting is not a finite number.
    #[error("the setting `{0}` is not a finite number")]
    NotFinite(&'static str),
}

/// A blur type with its panel numbers (`docs/33` 2.2): 0 to 100 sliders, degrees, and Zoom's centre in the photo frame.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BlurKind {
    /// A smooth blur: Radius 0 to 100.
    Soft {
        /// Radius, 0 to 100 (0 to 4 % of the long edge).
        radius: f64,
    },
    /// The blur of an out-of-focus lens.
    Lens {
        /// Radius, 0 to 100 (0 to 4 % of the long edge).
        radius: f64,
        /// 0 for a round aperture, else the number of blades (3 and more).
        blades: u32,
        /// Rotation of the polygon, degrees.
        rotation_deg: f64,
        /// Highlights, 0 to 100 (read as a share of 100).
        highlights: f64,
    },
    /// A line.
    Motion {
        /// Length, 0 to 100 (0 to 8 % of the long edge).
        length: f64,
        /// Angle, degrees, 0 along x, counter-clockwise on screen.
        angle_deg: f64,
    },
    /// A radial line, stronger away from the centre.
    Zoom {
        /// Amount, 0 to 100 (0 to 10 %).
        amount: f64,
        /// The centre as a fraction of the frame, `[0.5, 0.5]` the middle.
        center: [f64; 2],
    },
}

/// What to run: a type with its numbers, and whether the mask also weighs the sums.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BlurParams {
    /// The type and its numbers.
    pub kind: BlurKind,
    /// Respect mask edges: `omega = m` (on, the default) or `omega = 1` (off, the plain blend).
    pub respect_mask_edges: bool,
}

/// Where a window sits in the frame. The long edge of the **whole** frame sets every length in pixels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Frame {
    /// The frame's width, pixels.
    pub full_w: usize,
    /// The frame's height, pixels.
    pub full_h: usize,
    /// The window's left edge in the frame.
    pub x0: usize,
    /// The window's top edge in the frame.
    pub y0: usize,
}

impl Frame {
    /// The window is the whole frame.
    #[must_use]
    pub fn whole(width: usize, height: usize) -> Self {
        Self {
            full_w: width,
            full_h: height,
            x0: 0,
            y0: 0,
        }
    }

    /// The frame's long edge, pixels (the unit of `docs/33` 2.2).
    #[must_use]
    pub fn long_edge_px(&self) -> f64 {
        self.full_w.max(self.full_h) as f64
    }
}

// ------------------------------------------------------------------------------------------------ units and weights

/// Radius (0 to 100) in pixels of the aperture or the Gaussian's `R` (sigma is `R / 2`).
#[must_use]
pub fn radius_px(radius: f64, long_edge_px: f64) -> f64 {
    radius / 100.0 * RADIUS_MAX * long_edge_px
}

/// Motion's Length (0 to 100) in pixels.
#[must_use]
pub fn length_px(length: f64, long_edge_px: f64) -> f64 {
    length / 100.0 * MOTION_MAX * long_edge_px
}

/// Zoom's Amount (0 to 100) as the scale span `a` (0 to 0.10).
#[must_use]
pub fn zoom_span(amount: f64) -> f64 {
    amount / 100.0 * ZOOM_MAX
}

/// Lens' highlight weight `h` of a linear AWG4 triple: 1 up to 4 x middle grey, `1 + 15 H` from 16 x, smooth between
/// (`docs/33` 3.5). `highlights` is the panel's 0 to 100.
#[must_use]
pub fn highlight_weight(x: Vec3, highlights: f64) -> f64 {
    let h = highlights / 100.0;
    if h <= 0.0 {
        return 1.0;
    }
    let y = x[0] * AWG4_LUMA[0] + x[1] * AWG4_LUMA[1] + x[2] * AWG4_LUMA[2];
    let ev = (y.max(1e-6) / 0.18).log2();
    1.0 + HIGHLIGHT_GAIN * h * smoothstep(HIGHLIGHT_EV[0], HIGHLIGHT_EV[1], ev)
}

/// The sum weight `omega` of a pixel (`docs/33` 3.2).
#[must_use]
pub fn omega(m: f64, h: f64, respect_mask_edges: bool) -> f64 {
    if respect_mask_edges { h * m } else { h }
}

/// The blend of one pixel, in linear light: `X + m (N - X D) / (D + EPS)`, with the exact branches `m = 0` (the
/// input's bits) and `m = 1` (`N / (D + EPS)`).
#[must_use]
pub fn blend_linear(x: Vec3, m: f64, n: Vec3, d: f64) -> Vec3 {
    if m == 0.0 {
        return x;
    }
    let q = d + EPS;
    if m == 1.0 {
        return [n[0] / q, n[1] / q, n[2] / q];
    }
    [
        x[0] + m * ((n[0] - x[0] * d) / q),
        x[1] + m * ((n[1] - x[1] * d) / q),
        x[2] + m * ((n[2] - x[2] * d) / q),
    ]
}

// ------------------------------------------------------------------------------------------------ exact kernels

/// The error function (Maclaurin series below 2.5, continued fraction above; about 1e-13).
#[must_use]
pub fn erf(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    let ax = x.abs();
    let v = if ax < 2.5 {
        let mut term = ax;
        let mut sum = ax;
        let x2 = ax * ax;
        for n in 1..200 {
            let nf = f64::from(n);
            term *= -x2 / nf;
            let add = term / (2.0 * nf + 1.0);
            sum += add;
            if add.abs() < 1e-17 {
                break;
            }
        }
        sum * std::f64::consts::FRAC_2_SQRT_PI
    } else if ax > 27.0 {
        1.0
    } else {
        let mut f = ax;
        for k in (1..80).rev() {
            f = ax + (f64::from(k) * 0.5) / f;
        }
        1.0 - (-ax * ax).exp() / (std::f64::consts::PI.sqrt() * f)
    };
    if x < 0.0 { -v } else { v }
}

/// The 1-D weights of the pixel-integrated Gaussian cut at 4 sigma and renormalised: `g(i) = Phi((i + 1/2) / sigma) -
/// Phi((i - 1/2) / sigma)` for `|i| <= ceil(4 sigma)`. Returns `2 half + 1` weights, index `half` the centre.
#[must_use]
pub fn gauss_weights(sigma: f64) -> Vec<f64> {
    if sigma <= 1e-6 {
        return vec![1.0];
    }
    gauss_weights_cut(sigma, (SIGMA_CUT * sigma).ceil() as i64)
}

/// [`gauss_weights`] cut at `half` taps each side instead of `ceil(4 sigma)` (renormalised).
#[must_use]
pub fn gauss_weights_cut(sigma: f64, half: i64) -> Vec<f64> {
    if sigma <= 1e-6 || half <= 0 {
        return vec![1.0];
    }
    let phi = |v: f64| 0.5 * (1.0 + erf(v / (sigma * std::f64::consts::SQRT_2)));
    let mut w: Vec<f64> = (-half..=half)
        .map(|i| phi(i as f64 + 0.5) - phi(i as f64 - 0.5))
        .collect();
    let total: f64 = w.iter().sum();
    for v in &mut w {
        *v /= total;
    }
    w
}

/// A square table of weights centred on the pixel: `(2 half + 1)^2` entries, row-major, the row the y offset.
#[derive(Debug, Clone, PartialEq)]
pub struct Table {
    /// Half width: offsets run from `-half` to `half`.
    pub half: i64,
    /// The weights, summing to 1.
    pub weights: Vec<f64>,
}

impl Table {
    fn identity() -> Self {
        Self {
            half: 0,
            weights: vec![1.0],
        }
    }

    fn normalized(half: i64, mut weights: Vec<f64>) -> Self {
        let total: f64 = weights.iter().sum();
        if total.is_nan() || total <= 0.0 || !total.is_finite() {
            return Self::identity();
        }
        for w in &mut weights {
            *w /= total;
        }
        Self { half, weights }
    }

    /// The weight at offset `(dx, dy)`, 0 outside the table.
    #[must_use]
    pub fn at(&self, dx: i64, dy: i64) -> f64 {
        if dx.abs() > self.half || dy.abs() > self.half {
            return 0.0;
        }
        let side = 2 * self.half + 1;
        self.weights[((dy + self.half) * side + dx + self.half) as usize]
    }
}

/// Whether the point `(px, py)` (pixels from the aperture's centre, y down) is inside a disc (`n = 0`) or a regular
/// `n`-gon of circumradius `r` rotated by `rot_deg`.
fn in_aperture(px: f64, py: f64, n: u32, rot_deg: f64, r: f64) -> bool {
    if n == 0 {
        return px * px + py * py <= r * r;
    }
    let rot = rot_deg.to_radians();
    let apothem = r * (std::f64::consts::PI / f64::from(n)).cos();
    (0..n).all(|i| {
        let a = rot + (f64::from(i) + 0.5) * 2.0 * std::f64::consts::PI / f64::from(n);
        px * a.cos() + py * a.sin() <= apothem
    })
}

/// The coverage of each pixel by a disc (`blades = 0`) or a regular polygon of circumradius `r`: the area of the pixel
/// inside the aperture (8 x 8 supersampled, each in `0..=1`), not normalised. Returns the half width and the
/// `(2 half + 1)^2` row-major values.
#[must_use]
pub fn aperture_coverage(r: f64, blades: u32, rotation_deg: f64) -> (i64, Vec<f64>) {
    let half = r.max(0.0).ceil() as i64 + 1;
    let side = (2 * half + 1) as usize;
    let ss = COVERAGE_SS;
    let mut weights = vec![0.0; side * side];
    for (iy, row) in weights.chunks_mut(side).enumerate() {
        for (ix, cell) in row.iter_mut().enumerate() {
            let cx = (ix as i64 - half) as f64;
            let cy = (iy as i64 - half) as f64;
            let mut hits = 0u32;
            for sy in 0..ss {
                for sx in 0..ss {
                    let gx = (sx as f64 + 0.5) / ss as f64 - 0.5;
                    let gy = (sy as f64 + 0.5) / ss as f64 - 0.5;
                    if in_aperture(cx + gx, cy + gy, blades, rotation_deg, r) {
                        hits += 1;
                    }
                }
            }
            *cell = f64::from(hits) / (ss * ss) as f64;
        }
    }
    (half, weights)
}

/// The weights of an aperture's [`aperture_coverage`], normalised to sum 1. A radius so small that no sample falls
/// inside gives the identity table.
#[must_use]
pub fn aperture_table(r: f64, blades: u32, rotation_deg: f64) -> Table {
    if r <= 1e-6 {
        return Table::identity();
    }
    let (half, weights) = aperture_coverage(r, blades, rotation_deg);
    Table::normalized(half, weights)
}

/// The length of the line through the pixel's centre inside each pixel (exactly, by clipping the segment against the
/// pixel's square), normalised to sum 1. `angle_deg` 0 runs along x, counter-clockwise on screen (y up).
#[must_use]
pub fn line_table(length: f64, angle_deg: f64) -> Table {
    if length <= 1e-6 {
        return Table::identity();
    }
    let half = (length / 2.0).ceil() as i64 + 1;
    let side = (2 * half + 1) as usize;
    let a = angle_deg.to_radians();
    let dir = [a.cos(), -a.sin()];
    let mut weights = vec![0.0; side * side];
    for (iy, row) in weights.chunks_mut(side).enumerate() {
        for (ix, cell) in row.iter_mut().enumerate() {
            let c = [(ix as i64 - half) as f64, (iy as i64 - half) as f64];
            let (mut t0, mut t1) = (-length / 2.0, length / 2.0);
            for k in 0..2 {
                let (lo, hi) = (c[k] - 0.5, c[k] + 0.5);
                if dir[k] == 0.0 {
                    if !(lo <= 0.0 && 0.0 <= hi) {
                        t1 = t0;
                    }
                } else {
                    let (a0, a1) = (lo / dir[k], hi / dir[k]);
                    t0 = t0.max(a0.min(a1));
                    t1 = t1.min(a0.max(a1));
                }
            }
            *cell = (t1 - t0).max(0.0);
        }
    }
    Table::normalized(half, weights)
}

/// The number of taps along a Zoom line of `line_px` pixels: at least [`ZOOM_TAPS_MIN`], and two per pixel of line.
#[must_use]
pub fn zoom_taps(line_px: f64) -> usize {
    ZOOM_TAPS_MIN.max((ZOOM_TAPS_PER_PX * line_px).ceil() as usize)
}

// ------------------------------------------------------------------------------------------------ the sums

/// A weighted plane: `[omega X (3), omega]`.
/// The four planes of a pixel: `omega X` (three) and `omega`.
pub type Plane4 = [f64; 4];

fn check_finite(params: &BlurParams) -> Result<(), BlurError> {
    let fin = |v: f64, name: &'static str| {
        if v.is_finite() {
            Ok(())
        } else {
            Err(BlurError::NotFinite(name))
        }
    };
    match params.kind {
        BlurKind::Soft { radius } => fin(radius, "radius"),
        BlurKind::Lens {
            radius,
            rotation_deg,
            highlights,
            ..
        } => {
            fin(radius, "radius")?;
            fin(rotation_deg, "rotation_deg")?;
            fin(highlights, "highlights")
        }
        BlurKind::Motion { length, angle_deg } => {
            fin(length, "length")?;
            fin(angle_deg, "angle_deg")
        }
        BlurKind::Zoom { amount, center } => {
            fin(amount, "amount")?;
            fin(center[0], "center")?;
            fin(center[1], "center")
        }
    }
}

/// Whether the type's amount is 0 or less: the node is the identity and changes no bit.
#[must_use]
pub fn is_identity(params: &BlurParams) -> bool {
    match params.kind {
        BlurKind::Soft { radius } | BlurKind::Lens { radius, .. } => radius <= 0.0,
        BlurKind::Motion { length, .. } => length <= 0.0,
        BlurKind::Zoom { amount, .. } => amount <= 0.0,
    }
}

fn clean_mask(m: f64) -> f64 {
    if m.is_nan() { 0.0 } else { m.clamp(0.0, 1.0) }
}

fn gather_table(p: &[Plane4], w: usize, h: usize, table: &Table) -> Vec<Plane4> {
    let side = (2 * table.half + 1) as usize;
    let mut out = vec![[0.0; 4]; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut acc = [0.0; 4];
            for dy in -table.half..=table.half {
                let yy = y as i64 + dy;
                if yy < 0 || yy >= h as i64 {
                    continue;
                }
                let row = ((dy + table.half) as usize) * side;
                for dx in -table.half..=table.half {
                    let xx = x as i64 + dx;
                    if xx < 0 || xx >= w as i64 {
                        continue;
                    }
                    let k = table.weights[row + (dx + table.half) as usize];
                    if k == 0.0 {
                        continue;
                    }
                    let q = &p[yy as usize * w + xx as usize];
                    for c in 0..4 {
                        acc[c] += k * q[c];
                    }
                }
            }
            out[y * w + x] = acc;
        }
    }
    out
}

/// Two 1-D passes (rows, then columns) of the separable Gaussian, taps outside the window skipped.
fn gather_separable(p: &[Plane4], w: usize, h: usize, g: &[f64]) -> Vec<Plane4> {
    let half = (g.len() / 2) as i64;
    let mut mid = vec![[0.0; 4]; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut acc = [0.0; 4];
            for dx in -half..=half {
                let xx = x as i64 + dx;
                if xx < 0 || xx >= w as i64 {
                    continue;
                }
                let k = g[(dx + half) as usize];
                let q = &p[y * w + xx as usize];
                for c in 0..4 {
                    acc[c] += k * q[c];
                }
            }
            mid[y * w + x] = acc;
        }
    }
    let mut out = vec![[0.0; 4]; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut acc = [0.0; 4];
            for dy in -half..=half {
                let yy = y as i64 + dy;
                if yy < 0 || yy >= h as i64 {
                    continue;
                }
                let k = g[(dy + half) as usize];
                let q = &mid[yy as usize * w + x];
                for c in 0..4 {
                    acc[c] += k * q[c];
                }
            }
            out[y * w + x] = acc;
        }
    }
    out
}

/// The bilinear read of the planes at the frame position `(qx, qy)`; what is outside the window reads 0.
fn bilinear(p: &[Plane4], w: usize, h: usize, frame: &Frame, qx: f64, qy: f64) -> Plane4 {
    let fx0 = qx.floor();
    let fy0 = qy.floor();
    let (fx, fy) = (qx - fx0, qy - fy0);
    let ix = fx0 as i64 - frame.x0 as i64;
    let iy = fy0 as i64 - frame.y0 as i64;
    let at = |x: i64, y: i64| -> Plane4 {
        if x < 0 || y < 0 || x >= w as i64 || y >= h as i64 {
            [0.0; 4]
        } else {
            p[y as usize * w + x as usize]
        }
    };
    let (a, b, c, d) = (
        at(ix, iy),
        at(ix + 1, iy),
        at(ix, iy + 1),
        at(ix + 1, iy + 1),
    );
    let mut out = [0.0; 4];
    for k in 0..4 {
        out[k] = a[k] * (1.0 - fx) * (1.0 - fy)
            + b[k] * fx * (1.0 - fy)
            + c[k] * (1.0 - fx) * fy
            + d[k] * fx * fy;
    }
    out
}

/// The exact radial line of Zoom: `n` equal-weight bilinear taps at `c + v (1 + a t_i)`, `t_i = (i + 1/2)/n - 1/2`.
fn gather_zoom(
    p: &[Plane4],
    w: usize,
    h: usize,
    frame: &Frame,
    a: f64,
    center: [f64; 2],
) -> Vec<Plane4> {
    let cx = center[0] * frame.full_w as f64 - 0.5;
    let cy = center[1] * frame.full_h as f64 - 0.5;
    let mut out = vec![[0.0; 4]; w * h];
    for y in 0..h {
        for x in 0..w {
            let fx = (x + frame.x0) as f64;
            let fy = (y + frame.y0) as f64;
            let (vx, vy) = (fx - cx, fy - cy);
            let line = a * vx.hypot(vy);
            let n = zoom_taps(line);
            let mut acc = [0.0; 4];
            for i in 0..n {
                let t = (i as f64 + 0.5) / n as f64 - 0.5;
                let s = 1.0 + a * t;
                let s4 = bilinear(p, w, h, frame, cx + vx * s, cy + vy * s);
                for c in 0..4 {
                    acc[c] += s4[c];
                }
            }
            for v in &mut acc {
                *v /= n as f64;
            }
            out[y * w + x] = acc;
        }
    }
    out
}

/// The sums `N` and `D` of the law (`docs/33` 3.2) of a window of linear AWG4 pixels `x` and its mask `m`, by the **exact**
/// kernels (the oracle).
///
/// # Errors
///
/// [`BlurError`] when the sizes disagree, the window leaves the frame or a setting is not finite.
pub fn sums(
    x: &[Vec3],
    m: &[f64],
    width: usize,
    height: usize,
    params: &BlurParams,
    frame: &Frame,
) -> Result<(Vec<Vec3>, Vec<f64>), BlurError> {
    sums_with(x, m, width, height, params, frame, false)
}

/// [`sums`] by the type's fast method where it has one (`docs/33` 3.5; Soft's coarse grid, [`soft`]; Lens' cells, [`lens`]) and by
/// the exact kernel where its task has not landed. Held to [`sums`] by the bounds of `docs/33` 3.7.
///
/// # Errors
///
/// As [`sums`].
pub fn sums_fast(
    x: &[Vec3],
    m: &[f64],
    width: usize,
    height: usize,
    params: &BlurParams,
    frame: &Frame,
) -> Result<(Vec<Vec3>, Vec<f64>), BlurError> {
    sums_with(x, m, width, height, params, frame, true)
}

fn sums_with(
    x: &[Vec3],
    m: &[f64],
    width: usize,
    height: usize,
    params: &BlurParams,
    frame: &Frame,
    fast: bool,
) -> Result<(Vec<Vec3>, Vec<f64>), BlurError> {
    if x.len() != width * height || m.len() != width * height {
        return Err(BlurError::Size {
            pixels: x.len(),
            mask: m.len(),
            width,
            height,
        });
    }
    if frame.x0 + width > frame.full_w
        || frame.y0 + height > frame.full_h
        || frame.full_w == 0
        || frame.full_h == 0
    {
        return Err(BlurError::Window {
            x0: frame.x0,
            y0: frame.y0,
            width,
            height,
            full_w: frame.full_w,
            full_h: frame.full_h,
        });
    }
    check_finite(params)?;
    let u = frame.long_edge_px();
    let hl = match params.kind {
        BlurKind::Lens { highlights, .. } => highlights,
        _ => 0.0,
    };
    let planes: Vec<Plane4> = x
        .iter()
        .zip(m)
        .map(|(xv, &mv)| {
            let w = omega(
                clean_mask(mv),
                highlight_weight(*xv, hl),
                params.respect_mask_edges,
            );
            [xv[0] * w, xv[1] * w, xv[2] * w, w]
        })
        .collect();
    let out = match params.kind {
        BlurKind::Soft { radius } => {
            let sigma = radius_px(radius, u) / 2.0;
            if fast {
                soft::gather(&planes, width, height, frame, sigma)
            } else {
                gather_separable(&planes, width, height, &gauss_weights(sigma))
            }
        }
        BlurKind::Lens {
            radius,
            blades,
            rotation_deg,
            ..
        } => {
            let r = radius_px(radius, u);
            if fast {
                lens::gather(
                    &planes,
                    (frame.x0, frame.y0, width, height),
                    (frame.full_w, frame.full_h),
                    (r, blades, rotation_deg),
                )
            } else {
                let t = aperture_table(r, blades, rotation_deg);
                gather_table(&planes, width, height, &t)
            }
        }
        BlurKind::Motion { length, angle_deg } => {
            let t = line_table(length_px(length, u), angle_deg);
            gather_table(&planes, width, height, &t)
        }
        BlurKind::Zoom { amount, center } => {
            gather_zoom(&planes, width, height, frame, zoom_span(amount), center)
        }
    };
    Ok((
        out.iter().map(|q| [q[0], q[1], q[2]]).collect(),
        out.iter().map(|q| q[3]).collect(),
    ))
}

/// The node in linear AWG4: `out = X + m (N - X D) / (D + EPS)` for every pixel of the window, with the exact
/// branches. A type whose amount is 0 returns the input unchanged.
///
/// # Errors
///
/// As [`sums`].
pub fn apply_linear(
    x: &[Vec3],
    m: &[f64],
    width: usize,
    height: usize,
    params: &BlurParams,
    frame: &Frame,
) -> Result<Vec<Vec3>, BlurError> {
    apply_linear_with(x, m, width, height, params, frame, false)
}

/// [`apply_linear`] by the fast methods ([`sums_fast`]).
///
/// # Errors
///
/// As [`sums`].
pub fn apply_linear_fast(
    x: &[Vec3],
    m: &[f64],
    width: usize,
    height: usize,
    params: &BlurParams,
    frame: &Frame,
) -> Result<Vec<Vec3>, BlurError> {
    apply_linear_with(x, m, width, height, params, frame, true)
}

fn apply_linear_with(
    x: &[Vec3],
    m: &[f64],
    width: usize,
    height: usize,
    params: &BlurParams,
    frame: &Frame,
    fast: bool,
) -> Result<Vec<Vec3>, BlurError> {
    check_finite(params)?;
    if is_identity(params) {
        // Still validate the sizes, so an identity call fails like any other on a bad window.
        if x.len() != width * height || m.len() != width * height {
            return Err(BlurError::Size {
                pixels: x.len(),
                mask: m.len(),
                width,
                height,
            });
        }
        return Ok(x.to_vec());
    }
    let (n, d) = sums_with(x, m, width, height, params, frame, fast)?;
    Ok((0..x.len())
        .map(|i| blend_linear(x[i], clean_mask(m[i]), n[i], d[i]))
        .collect())
}

/// The node on LogC4 AWG4 pixels (the graph's working encoding): decode, [`apply_linear`], encode. A pixel with
/// `m = 0` and a node at amount 0 return the input's bits (no encode of a decode). The exact kernels: the oracle.
///
/// # Errors
///
/// As [`sums`].
pub fn apply(
    v: &[Vec3],
    m: &[f64],
    width: usize,
    height: usize,
    params: &BlurParams,
    frame: &Frame,
) -> Result<Vec<Vec3>, BlurError> {
    apply_with(v, m, width, height, params, frame, false)
}

/// [`apply`] by the fast methods ([`sums_fast`]): what the CPU run of the render calls. The oracle [`apply`] stays the
/// reference.
///
/// # Errors
///
/// As [`sums`].
pub fn apply_fast(
    v: &[Vec3],
    m: &[f64],
    width: usize,
    height: usize,
    params: &BlurParams,
    frame: &Frame,
) -> Result<Vec<Vec3>, BlurError> {
    apply_with(v, m, width, height, params, frame, true)
}

fn apply_with(
    v: &[Vec3],
    m: &[f64],
    width: usize,
    height: usize,
    params: &BlurParams,
    frame: &Frame,
    fast: bool,
) -> Result<Vec<Vec3>, BlurError> {
    check_finite(params)?;
    if is_identity(params) {
        if v.len() != width * height || m.len() != width * height {
            return Err(BlurError::Size {
                pixels: v.len(),
                mask: m.len(),
                width,
                height,
            });
        }
        return Ok(v.to_vec());
    }
    let x: Vec<Vec3> = v.iter().map(|p| logc4_decode_rgb(*p)).collect();
    let (n, d) = sums_with(&x, m, width, height, params, frame, fast)?;
    Ok((0..v.len())
        .map(|i| {
            let mv = clean_mask(m[i]);
            if mv == 0.0 {
                v[i]
            } else {
                logc4_encode_rgb(blend_linear(x[i], mv, n[i], d[i]))
            }
        })
        .collect())
}

// ------------------------------------------------------------------------------------------------ the reach

/// The cell factor `f = 2^L` of the fast methods (`docs/33` 3.8): the power of two at or above the long edge over 256 (32 at
/// 6000 px and 64 at 8600 px, the table's numbers). The oracle needs none of it, but the scheduler's reach already holds
/// three cells of it, so that a fast method (BL-06 to BL-09) reads no more than the reach the plan grew the tile by.
#[must_use]
pub fn cell_factor(long_edge_px: f64) -> usize {
    let n = (long_edge_px / 256.0).ceil().max(1.0);
    if n.is_finite() && n < 1e9 {
        (n as usize).next_power_of_two()
    } else {
        1
    }
}

/// The reach of the node in output pixels (`docs/33` 3.8): how far from a pixel the node reads the picture and its mask,
/// for a frame of `full_w x full_h` pixels. A scalar for the whole tile: Zoom's is the worst case, the farthest corner from
/// the centre. 0 for a node at its identity.
///
/// | Type | Reach |
/// |---|---|
/// | Soft | `ceil(2 R) + 3 f`, or what [`soft::reads_px`] needs when that is more |
/// | Lens | `ceil(R) + 3 f_L`, `f_L` = [`lens::cell_size`] (the smallest cell with `R / f_L <= 8`) |
/// | Motion | `ceil(l / 2) + 1 + 3 f` |
/// | Zoom | `ceil(a / 2 d_max) + f` |
///
/// with `f` = [`cell_factor`]. Each is at least what the exact kernels read (the tests hold it to the tables' half widths).
#[must_use]
pub fn reach_px(params: &BlurParams, full_w: usize, full_h: usize) -> usize {
    if is_identity(params) || check_finite(params).is_err() {
        return 0;
    }
    let u = full_w.max(full_h) as f64;
    let f = cell_factor(u);
    let cells = |r: f64| r.max(0.0).ceil() as usize;
    match params.kind {
        BlurKind::Soft { radius } => {
            // The table's number, or what the fast method reads when that is more (`soft::reads_px`).
            let r = radius_px(radius, u);
            (cells(2.0 * r) + 3 * f).max(soft::reads_px(r / 2.0))
        }
        // Lens' cell is its own (`lens::level_for`: the smallest with `R / f <= 8`), so 3 cells of it hold what the cell method
        // reads (`R + 2.5 f`).
        BlurKind::Lens { radius, .. } => {
            let r = radius_px(radius, u);
            cells(r) + 3 * lens::cell_size(r)
        }
        BlurKind::Motion { length, .. } => cells(length_px(length, u) / 2.0) + 1 + 3 * f,
        BlurKind::Zoom { amount, center } => {
            let c = [
                center[0].clamp(0.0, 1.0) * full_w as f64,
                center[1].clamp(0.0, 1.0) * full_h as f64,
            ];
            let far = |p: f64, size: usize| p.max(size as f64 - p);
            let d_max = far(c[0], full_w).hypot(far(c[1], full_h));
            cells(zoom_span(amount) / 2.0 * d_max) + f
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    include!("blur_tests.rs");
}
