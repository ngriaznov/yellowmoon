//! The cine rendering: stage C12, the display transform (`docs/02-colour-pipeline.md`, *Cine rendering
//! (C12, default)*).
//!
//! Scene-linear AWG4 goes in, display-linear RGB in the primaries of an [`OutputTarget`] comes out
//! (1.0 is the 100 nit SDR white; C13 encodes it). The model is ours: a luminance tone curve, hue-dependent
//! chroma, hue and lightness terms in a cube-root opponent space, and a soft gamut stage on each channel of
//! the output. Nothing in it is read from a file; the 57 constants are [`consts`], generated from
//! `training/cine/constants.json`, and the executable reference is `training/cine/cine_ref.py`.
//!
//! * [`CineRendering::forward`] is the rendering, f64, about 30 transcendental calls a pixel.
//! * [`CineRendering::inverse`] is the numerical inverse the DNG of a look with its own display rendering
//!   needs (`docs/09`): a closed-form display side, a first-use table and a chord iteration for the core,
//!   a closed-form scene side, and the nearest-reachable rule for display colours the rendering cannot
//!   produce ([`CineInverse::chroma_fraction`]).
//!
//! Both are pure functions of their input: the same colour gives the same result on every thread. The
//! table of the inverse is built once per process, on first use, from the forward model alone.

pub mod consts;
mod inverse;
mod real;

use std::sync::LazyLock;

use self::real::Real;
use super::OutputTarget;
use crate::consts::Vec3;
use crate::consts::linalg::{self, Mat3};
use crate::gamut::{LinearAwg4, matrices};

pub use self::inverse::CineInverse;

// ---- numeric settings (all fixed, none fitted; `docs/02` C12, The equations) ---------------------------

/// A scene channel is held to plus or minus this before anything else (22 stops over grey).
pub const SCENE_LIMIT: f64 = 1.0e6;
/// Smallest scene luminance the colour terms see.
pub const Y_MIN: f64 = 1.0e-9;
/// Smallest display luminance the gamut stage divides by.
pub const Y2_MIN: f64 = 1.0e-12;
/// Width of the smooth cube root.
pub const DELTA: f64 = 1.0e-4;
/// Keeps the luminance floor's square root away from 0.
pub const EPS_FLOOR: f64 = 1.0e-30;
/// Keeps the chroma length away from 0.
pub const EPS_CHROMA: f64 = 1.0e-14;
/// The smallest display channel the inverse accepts: an exact 0 has no finite pre-image, so a smaller
/// channel is raised to this (`docs/02`, Inverse).
pub const D_MIN: f64 = 1.0e-6;

/// Scene middle grey.
const GREY: f64 = 0.18;

/// Values derived from the constants once: exponentials the equations use at every pixel.
struct Derived {
    /// `a`, `c`, `b` of the tone curve `H = (1 + c Ys^-a)^-b`.
    a: f64,
    c: f64,
    b: f64,
    /// `e^lgpk`: the chroma gain in the deepest shadows.
    g_peak: f64,
    /// `gb * lYc`.
    gb_lyc: f64,
    /// `e^(2 lchih)`.
    env2: f64,
    /// `kl = e^lklo`, `kh = e^lkhi`.
    kl: f64,
    kh: f64,
    /// `ln 0.18` and `1 / ln 2`: `u = log2(Ys / 0.18)`.
    ln_grey: f64,
    inv_ln2: f64,
    /// D65 at Y = 1.
    xn: f64,
    zn: f64,
    /// AWG4 to XYZ and back.
    awg4_to_xyz: Mat3,
    xyz_to_awg4: Mat3,
}

static DERIVED: LazyLock<Derived> = LazyLock::new(|| {
    let m = matrices();
    let d65 = crate::consts::xy_to_xyz(crate::consts::D65_XY);
    Derived {
        a: consts::LA.exp(),
        c: consts::LC.exp(),
        b: consts::LB.exp(),
        g_peak: consts::LGPK.exp(),
        gb_lyc: consts::GB * consts::LYC,
        env2: (2.0 * consts::LCHIH).exp(),
        kl: consts::LKLO.exp(),
        kh: consts::LKHI.exp(),
        ln_grey: GREY.ln(),
        inv_ln2: std::f64::consts::LN_2.recip(),
        xn: d65[0] / d65[1],
        zn: d65[2] / d65[1],
        awg4_to_xyz: m.awg4_to_xyz,
        xyz_to_awg4: m.xyz_to_awg4,
    }
});

fn derived() -> &'static Derived {
    &DERIVED
}

// ---- the model's pieces ---------------------------------------------------------------------------------

/// `softclip(z, w)`: the identity up to `1 - w`, a C1 quadratic that reaches exactly 1 with slope 0 at
/// `1 + w`, and 1 above.
fn soft_clip<T: Real>(z: T, w: f64) -> T {
    let (z0, z1) = (1.0 - w, 1.0 + w);
    let t = z.clamp_c(z0, z1) - T::c(z0);
    (z - t * t.sc(1.0 / (4.0 * w))).min_c(1.0)
}

fn sigmoid<T: Real>(x: T) -> T {
    T::c(1.0) / (T::c(1.0) + (-x).exp())
}

/// `k1 cos + l1 sin + k2 cos 2 + l2 sin 2 + k3 cos 3 + l3 sin 3` of the unit direction `(c, s)`, with the
/// double and triple angles as polynomials (no `atan2`).
fn harmonics<T: Real>(c: T, s: T, k: &[f64; 6]) -> T {
    let c2 = c * c - s * s;
    let s2 = (c * s).sc(2.0);
    let c3 = c * c2 - s * s2;
    let s3 = s * c2 + c * s2;
    c.sc(k[0]) + s.sc(k[1]) + c2.sc(k[2]) + s2.sc(k[3]) + c3.sc(k[4]) + s3.sc(k[5])
}

/// The terms of step 5, in logarithmic form where that is what the core needs.
struct Terms<T> {
    ln_yt: T,
    lr: T,
    gh: T,
    g0: T,
    dt: T,
}

/// Steps 4 and 5 for `(ln Ys, ca, cb)`.
fn terms<T: Real>(ly: T, ca: T, cb: T) -> Terms<T> {
    let d = derived();
    let chi = (ca * ca + cb * cb + T::c(EPS_CHROMA)).sqrt();
    let cs = ca / chi;
    let sn = cb / chi;
    let u = (ly - T::c(d.ln_grey)).sc(d.inv_ln2);
    // Step 4. H = (1 + c Ys^-a)^-b, then the soft clip at the top.
    let h = (-(T::c(1.0) + (-(ly.sc(d.a))).exp().sc(d.c)).ln().sc(d.b)).exp();
    let yt = soft_clip(h.sc(consts::KAP), consts::WC);
    // Step 5. G0 = e^lgpk / (1 + (Ys / e^lYc)^gb), written with the logarithm of Ys.
    let g0 = T::c(d.g_peak) / (T::c(1.0) + (ly.sc(consts::GB) - T::c(d.gb_lyc)).exp());
    let bh = sigmoid((u - T::c(1.5)).sc(1.0 / 1.5));
    let env = chi * chi / (chi * chi + T::c(d.env2));
    let gh = env * (harmonics(cs, sn, &consts::GH_B0) + bh * harmonics(cs, sn, &consts::GH_BH));
    let dt = env * (harmonics(cs, sn, &consts::DT_B0) + bh * harmonics(cs, sn, &consts::DT_BH));
    let rh = env * (harmonics(cs, sn, &consts::RH_B0) + bh * harmonics(cs, sn, &consts::RH_BH));
    // chi^rp / (chi^rp + e^(rp r_chi)) = 1 / (1 + e^(rp (r_chi - ln chi))).
    let chenv = T::c(1.0) / (T::c(1.0) + (T::c(consts::R_CHI) - chi.ln()).sc(consts::RP).exp());
    let lr =
        chenv * (sigmoid(-(u - T::c(consts::R_YD))).sc(consts::R_DARK) + T::c(consts::R_HI)) + rh;
    Terms {
        ln_yt: yt.ln(),
        lr,
        gh,
        g0,
        dt,
    }
}

/// Steps 4 to 6: `(ln Ys, ca, cb)` of the scene colour (after the floor and the chroma limit) to
/// `(ln Y2, ca2, cb2)` of the display colour before the gamut stage.
fn core<T: Real>(ly: T, ca: T, cb: T) -> [T; 3] {
    let t = terms(ly, ca, cb);
    let g = t.g0 * t.gh.exp();
    let (cd, sd) = (t.dt.cos(), t.dt.sin());
    [
        t.ln_yt + t.lr,
        g * (ca * cd - cb * sd),
        g * (ca * sd + cb * cd),
    ]
}

/// The smooth cube root `r ((1 + d^2) / (r^2 + d^2))^(1/3)`: odd, smooth at 0, 1 at 1.
fn smooth_cbrt(r: f64) -> f64 {
    r * ((1.0 + DELTA * DELTA) / (r * r + DELTA * DELTA)).cbrt()
}

/// The neutral tone curve: scene luminance (grey 0.18) to display luminance, before the gamut stage.
fn tone(ys: f64) -> f64 {
    let d = derived();
    let h = (-d.b * (1.0 + d.c * (-d.a * ys.ln()).exp()).ln()).exp();
    soft_clip(consts::KAP * h, consts::WC)
}

/// Step 2: the luminance floor, a smooth maximum of `Y` and `rho` times the distance of the chromaticity
/// from the neutral axis.
fn luminance_floor(x: f64, y: f64, z: f64) -> f64 {
    let d = derived();
    let yf = consts::RHO * ((x - d.xn * y).powi(2) + (z - d.zn * y).powi(2) + EPS_FLOOR).sqrt();
    let w = consts::WFRAC * yf;
    0.5 * (y + yf + ((y - yf) * (y - yf) + w * w).sqrt())
}

/// Step 3's soft limit of the relative chroma: identity up to `chi_k`, tending to `chi_m`.
fn chroma_limit(ca: f64, cb: f64) -> (f64, f64) {
    let (k, m) = (consts::CHI_K, consts::CHI_M);
    let chi = (ca * ca + cb * cb + EPS_CHROMA).sqrt();
    let e = (chi - k).max(0.0);
    let limited = chi.min(k) + (m - k) * (e / (m - k)).min(20.0).tanh();
    let s = limited / chi;
    (ca * s, cb * s)
}

/// Steps 0 to 3: scene AWG4 to `(ln Ys, ca, cb)` after the floor and the chroma limit.
fn scene_polar(x: Vec3) -> Vec3 {
    let d = derived();
    // NaN counts as 0, as everywhere in the chain; the limit keeps every later product inside f32 too.
    let x = x.map(|v| {
        if v.is_nan() {
            0.0
        } else {
            v.clamp(-SCENE_LIMIT, SCENE_LIMIT)
        }
    });
    let xyz = linalg::mul_vec(&d.awg4_to_xyz, x);
    let ys = luminance_floor(xyz[0], xyz[1], xyz[2]).max(Y_MIN);
    let ca = smooth_cbrt(xyz[0] / (d.xn * ys)) - 1.0;
    let cb = 1.0 - smooth_cbrt(xyz[2] / (d.zn * ys));
    let (ca, cb) = chroma_limit(ca, cb);
    [ys.ln(), ca, cb]
}

/// Step 7: the soft floor and ceiling on each channel of the output, relative to the display luminance.
fn gamut_stage(rgb: Vec3, y2: f64) -> Vec3 {
    let d = derived();
    let yd = y2.max(Y2_MIN);
    rgb.map(|v| {
        let r = v / yd;
        let r = r.max(d.kl) + d.kl * ((r - d.kl).min(0.0) / d.kl).exp_m1();
        soft_clip((1.0 + d.kh) * r * yd, d.kh)
    })
}

/// The cine rendering for one output: the matrices of its primaries, ready to use.
///
/// Cheap to build and `Copy`; the constants and the table of the inverse are shared by the process.
#[derive(Debug, Clone, Copy)]
pub struct CineRendering {
    output: OutputTarget,
    /// The output's RGB to XYZ and back (D65 white).
    to_xyz: Mat3,
    from_xyz: Mat3,
}

impl CineRendering {
    /// The rendering for `output`. `Viewport` and `DisplayP3` are the same colours.
    #[must_use]
    pub fn new(output: OutputTarget) -> Self {
        let m = matrices();
        let (to_xyz, from_xyz) = match output {
            OutputTarget::Srgb => (m.rec709_to_xyz, m.xyz_to_rec709),
            OutputTarget::Viewport | OutputTarget::DisplayP3 => (m.p3_to_xyz, m.xyz_to_p3),
            OutputTarget::Rec2020 => (m.rec2020_to_xyz, m.xyz_to_rec2020),
        };
        Self {
            output,
            to_xyz,
            from_xyz,
        }
    }

    /// The output this rendering is for.
    #[must_use]
    pub fn output(&self) -> OutputTarget {
        self.output
    }

    /// The matrix from XYZ (D65) to this output's own primaries, the one the gamut stage works in. The GPU
    /// kernel takes it in its uniform block.
    #[must_use]
    pub fn xyz_to_output(&self) -> Mat3 {
        self.from_xyz
    }

    /// The forward map: scene-linear AWG4 to display-linear RGB in the output's primaries, 1.0 being the
    /// 100 nit white. The result is finite and in `[0, 1]` for every input; NaN counts as 0.
    #[must_use]
    pub fn forward(&self, awg4: LinearAwg4) -> Vec3 {
        let d = derived();
        let q = scene_polar(awg4.0);
        let [ly2, ca2, cb2] = core::<f64>(q[0], q[1], q[2]);
        let y2 = ly2.exp();
        let xyz2 = [
            d.xn * y2 * (1.0 + ca2).powi(3),
            y2,
            d.zn * y2 * (1.0 - cb2).powi(3),
        ];
        gamut_stage(linalg::mul_vec(&self.from_xyz, xyz2), y2)
    }

    /// The inverse: the scene colour that renders to `display` (display-linear RGB in this output's
    /// primaries), or, for a display colour the rendering cannot produce, the pre-image of the nearest
    /// reachable one (`docs/02`, Inverse). Channels are held to `[D_MIN, 1]` first; NaN counts as 0. The
    /// result is always finite and always a colour [`CineRendering::forward`] produces.
    ///
    /// The first call in a process builds the table (0.4 to 0.7 s); after that a call costs about 1.5 us.
    #[must_use]
    pub fn inverse(&self, display: Vec3) -> CineInverse {
        inverse::invert(self, display)
    }
}

/// Every scalar a GPU kernel needs besides [`consts`]: the numeric settings and the values the equations
/// derive from the constants once (`a`, `c`, `b` of the tone curve, `kl`, `kh`, the D65 `XN` and `ZN` and so
/// on), as `(name, value)` pairs in f64. `ym-gpu` writes them into the shader text (`CINE_<NAME>`), so the
/// kernel and this module apply the same numbers.
#[must_use]
pub fn kernel_scalars() -> Vec<(&'static str, f64)> {
    let d = derived();
    vec![
        ("SCENE_LIMIT", SCENE_LIMIT),
        ("Y_MIN", Y_MIN),
        ("Y2_MIN", Y2_MIN),
        ("DELTA2", DELTA * DELTA),
        ("ONE_PLUS_DELTA2", 1.0 + DELTA * DELTA),
        ("EPS_FLOOR", EPS_FLOOR),
        ("EPS_CHROMA", EPS_CHROMA),
        ("XN", d.xn),
        ("ZN", d.zn),
        ("LN_GREY", d.ln_grey),
        ("INV_LN2", d.inv_ln2),
        ("A", d.a),
        ("C", d.c),
        ("B", d.b),
        ("KAP", consts::KAP),
        ("WC", consts::WC),
        ("G_PEAK", d.g_peak),
        ("GB", consts::GB),
        ("GB_LYC", d.gb_lyc),
        ("ENV2", d.env2),
        ("RP", consts::RP),
        ("R_CHI", consts::R_CHI),
        ("R_DARK", consts::R_DARK),
        ("R_YD", consts::R_YD),
        ("R_HI", consts::R_HI),
        ("KL", d.kl),
        ("KH", d.kh),
        ("RHO", consts::RHO),
        ("WFRAC", consts::WFRAC),
        ("CHI_K", consts::CHI_K),
        ("CHI_M", consts::CHI_M),
    ]
}

/// The six hue-harmonic arrays of [`consts`] as `(name, [c1, s1, c2, s2, c3, s3])`, for the same purpose as
/// [`kernel_scalars`].
#[must_use]
pub fn kernel_harmonics() -> [(&'static str, [f64; 6]); 6] {
    [
        ("GH_B0", consts::GH_B0),
        ("GH_BH", consts::GH_BH),
        ("DT_B0", consts::DT_B0),
        ("DT_BH", consts::DT_BH),
        ("RH_B0", consts::RH_B0),
        ("RH_BH", consts::RH_BH),
    ]
}

/// The scene luminance whose neutral colour the tone curve maps to display 1.0 (the white level,
/// `0.18 * 2^uw`, +10.4 stops over grey): the smallest grey that renders to exactly white.
#[must_use]
pub fn white_scene_level() -> f64 {
    inverse::tone_inverse(1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn derived_values_follow_the_constants() {
        let d = derived();
        assert!((d.a - consts::LA.exp()).abs() < 1e-15);
        assert!((d.xn - 0.950_455_927_051_671_6).abs() < 1e-15);
        assert!((d.zn - 1.089_057_750_759_878_4).abs() < 1e-15);
        assert!(d.kh > 0.0 && d.kl > d.kh);
    }

    #[test]
    fn soft_clip_is_identity_below_and_one_above() {
        let w = 0.15;
        assert_eq!(soft_clip(0.5_f64, w), 0.5);
        assert_eq!(soft_clip(1.0 + w, w), 1.0);
        assert_eq!(soft_clip(3.0_f64, w), 1.0);
        // C1 at both junctions: the value and the slope agree across them.
        let eps = 1e-7;
        let (lo, hi) = (1.0 - w, 1.0 + w);
        let slope = |z: f64| (soft_clip(z + eps, w) - soft_clip(z - eps, w)) / (2.0 * eps);
        assert!((slope(lo - 2.0 * eps) - slope(lo + 2.0 * eps)).abs() < 1e-5);
        assert!(slope(hi - 2.0 * eps).abs() < 1e-5);
    }

    #[test]
    fn the_white_level_is_ten_point_four_stops_over_grey() {
        let stops = (white_scene_level() / GREY).log2();
        // `kap` is listed to 9 digits, which moves the start of the flat top by about 2e-8 stops.
        assert!((stops - consts::UW).abs() < 1e-6, "{stops}");
    }
}
