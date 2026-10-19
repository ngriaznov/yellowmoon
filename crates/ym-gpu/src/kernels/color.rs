//! The whole per-pixel colour chain C1 to C13 as one compute pass (`docs/02-colour-pipeline.md`).
//!
//! [`PreparedChain::new`] turns an immutable `ym_color::display::ColorChain` into the GPU form: the
//! per-frame uniform block (fused matrix, C7 (the highlight range and the toe), C9, C9b, C10 and C12
//! parameters (the white lift of step 1b among them), the ACES 2 scalars and the flags, about 2 KB),
//! and four **persistent tables** that [`ChainTables`] keeps on the device and refills only when their
//! content changes:
//!
//! * the 64 x 64 dither thresholds, once per device;
//! * the ACES 2 tables, once per device and output;
//! * the four 1024-entry curve tables (16 KB), once per distinct set of curves;
//! * the look and the display cubes (f32, never rounded to f16, never hardware-filtered), in an LRU
//!   keyed by the content of the LUT (a 65-point cube is 4.4 MB, a 129-point one 34 MB).
//!
//! Nothing here computes colour: every number comes from the `ym_color` modules the CPU chain uses, so
//! the kernel and `ColorChain` cannot drift apart without a parity test failing.
//!
//! # Persistent tables and their keys
//!
//! [`PreparedChain::new`] finds the tables of the device thread's [`ChainTables`] (one per device,
//! kept in a thread-local of the thread that owns the device, so callers change nothing): preparing a
//! chain whose curves, look and display are the ones of the previous chain uploads nothing and costs
//! microseconds; a curve edit uploads the 16 KB of the curves; a look change uploads the new cube once
//! and finds it again later. The keys are *contents*, never identities that could go stale: the
//! curves by `CurveSet` equality, a LUT by a 128-bit hash of its entries (with the address of the
//! `Lut` as a shortcut, valid while the cache holds the `Arc` the `Lut` lives in), the ACES 2 tables
//! by the output. [`release_tables`] drops the whole set (the pool ledger counts it as resident until
//! then).
//!
//! # Precision decision (LUTs and curves)
//!
//! A LUT texture or curve table stored as f16 rounds values in [0.5, 1) by up to 2.4e-4, which alone
//! exceeds the 2e-4 parity target of `docs/02`. Cubes, shapers and curve tables are therefore **f32
//! storage buffers**, and the tetrahedral interpolation is done by hand in the shader. Image
//! intermediates follow the device's storage mode (f32 by default; the f16 modes belong to the legacy
//! kernels and their tests, and a device image ([`GpuImage`]) needs the f32 mode).
//!
//! # Outputs
//!
//! [`ColorOutput`] selects what the host wrappers write: the value at any stage (for parity checks
//! against `ColorChain::trace` and for the DNG path), the encoded f32 picture, 8-bit RGBA with the
//! deterministic blue-noise dither at absolute pixel coordinates (tiles agree with the whole image),
//! 10 to 16-bit planes, or the Before view as RGBA8. The device entry points
//! ([`PreparedChain::encode_stage`], [`PreparedChain::encode_rgba8`]) read a window of a [`GpuImage`]
//! and write into another one at an offset (a slab of an export), or into an RGBA8 buffer.
//!
//! A source image with a **fourth plane** is the sampler's output: that plane marks the pixels
//! outside the photo (the letterbox). Such pixels are not computed: a stage output has 0 there and
//! carries the mask into its own fourth plane, an RGBA8 output has the background colour.
//!
//! # The colour warp (C9b)
//!
//! The pins of `ChainParams::warp` travel in the uniform block, not in a table:
//! [`ChainTables::prepare`] packs what `ym_color::warp::Warp::new` prepares (the CPU's own clamping,
//! pivot code, number of flow steps, and the rule that still pins, and locks without a moving pin,
//! are left out) into `warp_n` (the moving and locked counts and the flow's steps), `warp_mv` (three
//! `vec4` rows a moving pin) and `warp_lk` (one row a lock), 1040 bytes for the 16 pins of
//! `WARP_MAX_PINS`. A pin moves on every frame of a drag, so a
//! persistent table would be refilled as often as the uniform is written; the uniform needs no
//! further storage binding (K3), no cache key and no upload of its own. No moving pin gives
//! `warp_n.x = 0` and the kernel skips the stage, so a chain without a warp renders the bits it
//! rendered before the stage existed.

use std::any::Any;
use std::cell::RefCell;
use std::sync::{Arc, Weak};

use bytemuck::{Pod, Zeroable};
use ym_color::aces2::{Aces2Transform, REF_LUMINANCE};
use ym_color::consts::Vec3;
use ym_color::consts::linalg::{self, Mat3};
use ym_color::curve::{CurveSet, TABLE_SIZE};
use ym_color::display::{
    BeforePoint, ChainParams, ColorChain, DisplayLutPrimaries, DisplayLutTransfer,
    DisplayTransform, OutputTarget,
};
use ym_color::encode::{OutputSpace, dither_threshold};
use ym_color::exposure::white::WhiteLift;
use ym_color::gamut::matrices;
use ym_color::grade::{
    BALANCE_CENTER_STOPS, GradeParams, ZONE_CENTER_STOPS, contrast_slope, logc4_at_stops,
    saturation_factor,
};
use ym_color::look::{LookInput, intensity_fraction};
use ym_color::lut::Lut;
use ym_color::tone::ToneParams;
use ym_color::warp::{WARP_MAX_PINS, Warp, WarpParams};

use super::shader::{KernelId, table};
use super::util::{OutKind, PixelData, STORAGE_IN, STORAGE_OUT, even, image_bytes, sample_bytes};
use super::util::{storage_buffer, unpack};
use crate::cpu::RgbImage;
use crate::cpu::tiling::Rect;
use crate::device::{GpuContext, GpuError, StorageMode};
use crate::image::GpuImage;
use crate::pool::PooledBuffer;
use crate::queue::CancelToken;
use crate::stream::{Counters, PassList, Pending, ReadId, upload_bytes};
use crate::tiles::{Tile, TilePlan};

/// The stage whose value is written (the stop of the chain).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Stop {
    /// The fused matrix applied: linear AWG4 (C1 to C6).
    C6,
    /// After the highlight range and the toe: linear AWG4.
    C7,
    /// LogC4 encode.
    C8,
    /// After the primary grade.
    C9,
    /// After the colour warp (`StageTrace::c9b_logc4`), in LogC4.
    C9b,
    /// After the look (log-to-log looks).
    C10,
    /// After the curves: the DNG stop point, in LogC4.
    C11,
    /// C11 decoded back to linear AWG4 (what the DNG writer takes).
    DngLinearAwg4,
    /// Display-linear RGB in the output's primaries (C12).
    C12,
    /// The encoded picture in [0, 1], not quantised (C13).
    C13,
    /// The Before view at the chain's `BeforePoint`: the camera default rendering encoded for the output
    /// ([`BeforePoint::CameraDefault`]), or the LogC4 signal at C8 or C9 clamped to [0, 1].
    Before,
}

impl Stop {
    fn code(self) -> u32 {
        match self {
            Stop::C6 => 0,
            Stop::C7 => 1,
            Stop::C8 => 2,
            Stop::C9 => 3,
            // Added after the Before view's 9; the codes of the other stops did not change.
            Stop::C9b => 10,
            Stop::C10 => 4,
            Stop::C11 => 5,
            Stop::DngLinearAwg4 => 6,
            Stop::C12 => 7,
            Stop::C13 => 8,
            Stop::Before => 9,
        }
    }
}

/// What the colour kernel writes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ColorOutput {
    /// The value at a stage as planar f32.
    Stage(Stop),
    /// The encoded picture as 8-bit RGBA; `dither` applies the blue-noise dither at absolute pixel
    /// coordinates, otherwise values are rounded.
    Rgba8 {
        /// Whether to dither.
        dither: bool,
    },
    /// The Before view ([`Stop::Before`]) as 8-bit RGBA, quantised like [`ColorOutput::Rgba8`] (the camera
    /// default is already C13-encoded; the log reading points are shown as code values).
    BeforeRgba8 {
        /// Whether to dither.
        dither: bool,
    },
    /// The encoded picture as planar u16 with `bits` (1 to 16) significant bits, rounded.
    U16 {
        /// Bits per sample.
        bits: u32,
    },
}

/// A rendered image.
#[derive(Clone, Debug, PartialEq)]
pub struct ColorImage {
    /// Width in pixels.
    pub width: usize,
    /// Height in pixels.
    pub height: usize,
    /// The pixels.
    pub data: PixelData,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Default)]
struct LutDescU {
    cube: [u32; 4],
    shaper: [u32; 4],
    dmin: [f32; 4],
    dmax: [f32; 4],
    smin: [f32; 4],
    smax: [f32; 4],
}

/// Rows (`vec4<f32>`) of one moving pin of the colour warp in [`ChainU`]: [s.x, s.y, delta.x,
/// delta.y], [1 / (2 rho^2), shadows, highlights, pivot code], [exposure in stops, 0, 0, 0].
pub(crate) const WARP_ROWS_PER_MOVE: usize = 3;
/// Rows of the moving pins in [`ChainU`] (`WARP_MOVE_ROWS` in `color.wgsl`): room for every pin.
pub(crate) const WARP_MOVE_ROWS: usize = WARP_ROWS_PER_MOVE * WARP_MAX_PINS;
/// Rows of the locked pins in [`ChainU`] (`WARP_LOCK_ROWS` in `color.wgsl`), one a lock: [l.x, l.y,
/// 1 / (2 rho^2), 0]. At most `WARP_MAX_PINS - 1` are used (locks count only beside a moving pin);
/// the sixteenth row keeps the arithmetic of the sizes plain.
pub(crate) const WARP_LOCK_ROWS: usize = WARP_MAX_PINS;

/// The per-frame uniform block of `color.wgsl` (`ChainU`), 2096 bytes: window and mode, the chain's
/// parameters, the Before view's camera default (`cam0` to `cam2`: the 3x3 rows, `camt`: the tone curve's
/// `n`, `s`, `k` and white level), the ACES 2 scalars and the colour warp's pins (`warp_n`: the moving
/// and locked counts and the flow's steps, `warp_mv`, `warp_lk`: module notes). Field for field the WGSL struct (a test pins
/// the offsets).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ChainU {
    dims: [u32; 4],
    src: [u32; 4],
    origin: [u32; 4],
    mode: [u32; 4],
    flags: [u32; 4],
    lk: [u32; 4],
    dp: [u32; 4],
    sp: [u32; 4],
    m0: [f32; 4],
    m1: [f32; 4],
    m2: [f32; 4],
    tone: [f32; 4],
    grade0: [f32; 4],
    zone_amt: [f32; 4],
    zone_ctr: [f32; 4],
    bal0: [f32; 4],
    bal1: [f32; 4],
    bal_ctr: [f32; 4],
    fl: [f32; 4],
    cdl_slope: [f32; 4],
    cdl_offset: [f32; 4],
    cdl_power: [f32; 4],
    look_lut: LutDescU,
    disp_lut: LutDescU,
    disp_m0: [f32; 4],
    disp_m1: [f32; 4],
    disp_m2: [f32; 4],
    look_m0: [f32; 4],
    look_m1: [f32; 4],
    look_m2: [f32; 4],
    cam0: [f32; 4],
    cam1: [f32; 4],
    cam2: [f32; 4],
    camt: [f32; 4],
    aces: AcesU,
    warp_n: [u32; 4],
    warp_mv: [[f32; 4]; WARP_MOVE_ROWS],
    warp_lk: [[f32; 4]; WARP_LOCK_ROWS],
}

impl Default for ChainU {
    /// All zero (`Default` is not derived: the warp's arrays are longer than 32).
    fn default() -> Self {
        Self::zeroed()
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Default)]
struct AcesU {
    in_cam: [[f32; 4]; 3],
    in_aab: [[f32; 4]; 3],
    lim_aab: [[f32; 4]; 3],
    lim_cam: [[f32; 4]; 3],
    l2d: [[f32; 4]; 3],
    in_p: [f32; 4],
    in_q: [f32; 4],
    lim_p: [f32; 4],
    ts0: [f32; 4],
    ts1: [f32; 4],
    cc: [f32; 4],
    hue: [i32; 4],
}

fn f4(v: [f64; 3], w: f64) -> [f32; 4] {
    [v[0] as f32, v[1] as f32, v[2] as f32, w as f32]
}

fn rows4(m: &Mat3) -> [[f32; 4]; 3] {
    [f4(m[0], 0.0), f4(m[1], 0.0), f4(m[2], 0.0)]
}

/// The columns of a row-vector matrix (`v * M`) as three rows to dot with `v`.
fn cols4(m: &[[f64; 3]; 3]) -> [[f32; 4]; 3] {
    std::array::from_fn(|j| f4([m[0][j], m[1][j], m[2][j]], 0.0))
}

fn clamp_control(x: f64) -> f64 {
    if x.is_nan() {
        0.0
    } else {
        x.clamp(-100.0, 100.0)
    }
}

/// Which stage C10 runs, as `ColorChain::new` classifies the look.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum LookKind {
    None,
    LogToLog,
    Baked,
}

fn classify_look(params: &ChainParams) -> LookKind {
    match &params.look {
        None => LookKind::None,
        Some(s) if s.look.includes_display_rendering() => LookKind::Baked,
        Some(s) if s.look.is_identity() || intensity_fraction(s.intensity_pct) <= 0.0 => {
            LookKind::None
        }
        Some(_) => LookKind::LogToLog,
    }
}

/// Matrix from linear primaries `from` (as Rec.709, P3-D65 or Rec.2020) to the output space.
fn primaries_matrix(from: DisplayLutPrimaries, to: OutputSpace) -> Option<Mat3> {
    let m = matrices();
    let (from_xyz, same) = match from {
        DisplayLutPrimaries::Rec709 => (m.rec709_to_xyz, to == OutputSpace::Srgb),
        DisplayLutPrimaries::P3D65 => (m.p3_to_xyz, to == OutputSpace::DisplayP3),
        DisplayLutPrimaries::Rec2020 => (m.rec2020_to_xyz, to == OutputSpace::Rec2020),
    };
    if same {
        return None;
    }
    let xyz_to = match to {
        OutputSpace::Srgb => m.xyz_to_rec709,
        OutputSpace::DisplayP3 => m.xyz_to_p3,
        OutputSpace::Rec2020 => m.xyz_to_rec2020,
    };
    Some(linalg::mul(&xyz_to, &from_xyz))
}

/// The cube of `lut` (shaper first, then the cube) as the vec4 entries of a LUT buffer, and the
/// descriptor that finds them (offsets in vec4 units from the start of the buffer).
fn lut_data(lut: &Lut) -> (Vec<f32>, LutDescU) {
    let mut luts: Vec<f32> = Vec::new();
    let mut d = LutDescU::default();
    if let Some(shaper) = &lut.shaper {
        d.cube[2] = 1;
        d.cube[3] = (luts.len() / 4) as u32;
        d.shaper[0] = shaper.size() as u32;
        d.smin = f4(shaper.domain_min(), 0.0);
        d.smax = f4(shaper.domain_max(), 1.0);
        luts.extend(shaper.to_rgba_f32());
    }
    d.cube[0] = (luts.len() / 4) as u32;
    d.cube[1] = lut.cube.size() as u32;
    d.dmin = f4(lut.cube.domain_min(), 0.0);
    d.dmax = f4(lut.cube.domain_max(), 1.0);
    luts.extend(lut.cube.to_rgba_f32());
    (luts, d)
}

/// C7 into the block: the highlight range's knee, Sony clip and expansion (stops over grey) and the toe's
/// width in `tone` (`[knee, clip, tau, expansion]`), the ALEXA cap in `fl.z`. No range
/// (`HighlightRange::NONE`, from `ToneParams::IDENTITY`) is a knee, clip and cap of
/// `HIGHLIGHT_OFF_STOPS` (1e30, a finite f32: K8) and an expansion of 0. `fl.w` is the white lift's
/// ([`white_lift_uniform`]).
fn tone_uniform(u: &mut ChainU, tone: &ToneParams) {
    let range = tone.range();
    u.tone = [
        range.knee_stops as f32,
        range.clip_stops as f32,
        tone.toe_tau() as f32,
        range.expansion_stops as f32,
    ];
    u.fl[2] = range.cap_stops as f32;
}

/// C12 step 1b into the block: the white lift's amount in stops in `fl.w` and the reciprocal of its
/// ramp's span in `grade0.w` (the two lanes the range and the grade leave free). No lift
/// ([`WhiteLift::NONE`], the exact identity, and what `ColorChain::white_lift` gives for a display LUT) is
/// 0 in both, and the shader then skips the stage; with a lift the span is at least `MIN_SPAN_EV`, so the
/// reciprocal is at most 4 (finite: K8). Written after [`grade_uniform`], which fills the whole of
/// `grade0`.
fn white_lift_uniform(u: &mut ChainU, lift: WhiteLift) {
    if lift.is_identity() {
        u.fl[3] = 0.0;
        u.grade0[3] = 0.0;
    } else {
        u.fl[3] = lift.amount_ev() as f32;
        u.grade0[3] = (1.0 / lift.span_ev()) as f32;
    }
}

/// The grade parameters as the kernel wants them (the same preparation as `Grade::new`): the
/// contrast curve's slope at grey `k` in `grade0.x`, the brightness in `grade0.y` and the saturation
/// factor in `grade0.z` (below 1 the blend toward grey, above 1 the boost `F`; `saturation_factor`).
/// A control at 0 clears its flag bit, and the kernel skips it. `grade0.w` is the white lift's
/// ([`white_lift_uniform`]), written after this.
fn grade_uniform(u: &mut ChainU, g: &GradeParams) {
    let slope = contrast_slope(g.contrast);
    let saturation = saturation_factor(g.saturation);
    if slope != 1.0 {
        u.flags[0] |= 1;
    }
    if saturation != 1.0 {
        u.flags[0] |= 2;
    }
    u.grade0 = [
        slope as f32,
        (clamp_control(g.brightness) / 100.0) as f32,
        saturation as f32,
        0.0,
    ];
    let amounts = [g.shadows, g.highlights, g.black, g.white].map(|x| clamp_control(x) / 100.0);
    u.zone_amt = [
        amounts[0] as f32,
        amounts[1] as f32,
        amounts[2] as f32,
        amounts[3] as f32,
    ];
    let centers = ZONE_CENTER_STOPS.map(logc4_at_stops);
    u.zone_ctr = [
        centers[0] as f32,
        centers[1] as f32,
        centers[2] as f32,
        centers[3] as f32,
    ];
    let b = [
        g.balance_shadows.clamped(),
        g.balance_midtones.clamped(),
        g.balance_highlights.clamped(),
    ];
    u.bal0 = [b[0].a as f32, b[0].b as f32, b[1].a as f32, b[1].b as f32];
    u.bal1 = [b[2].a as f32, b[2].b as f32, 0.0, 0.0];
    let bc = BALANCE_CENTER_STOPS.map(logc4_at_stops);
    u.bal_ctr = [bc[0] as f32, bc[1] as f32, bc[2] as f32, 0.0];
}

/// The colour warp's pins as the kernel wants them: what `Warp::new` prepares, packed into the rows of
/// the uniform block (module notes). Nothing is computed here but the casts to f32.
fn warp_uniform(u: &mut ChainU, params: &WarpParams) {
    let warp = Warp::new(params);
    let (moves, locks) = (warp.moves(), warp.locks());
    // `Warp::new` keeps at most `WARP_MAX_PINS` pins in all, so both fit; the zips and the `min`s keep
    // the rows and the counts inside the arrays even if that ever changed.
    for (rows, m) in u
        .warp_mv
        .chunks_exact_mut(WARP_ROWS_PER_MOVE)
        .zip(moves.iter())
    {
        rows[0] = [
            m.source[0] as f32,
            m.source[1] as f32,
            m.delta[0] as f32,
            m.delta[1] as f32,
        ];
        rows[1] = [
            m.inv_two_sigma_sq as f32,
            m.shadows as f32,
            m.highlights as f32,
            m.pivot_code as f32,
        ];
        rows[2] = [m.exposure_ev as f32, 0.0, 0.0, 0.0];
    }
    for (row, l) in u.warp_lk.iter_mut().zip(locks.iter()) {
        *row = [
            l.at[0] as f32,
            l.at[1] as f32,
            l.inv_two_sigma_sq as f32,
            0.0,
        ];
    }
    u.warp_n = [
        moves.len().min(WARP_MAX_PINS) as u32,
        locks.len().min(WARP_LOCK_ROWS) as u32,
        warp.steps(),
        0,
    ];
}

/// Limiting RGB to encoding RGB for the ACES 2 output (`None` when they are the same primaries).
fn limit_to_display(target: OutputTarget) -> Option<Mat3> {
    let out = target.aces2();
    if out.limiting_primaries() == out.encoding_primaries() {
        return None;
    }
    let m = matrices();
    // Only the Rec.2020 output differs: P3-D65 limited, Rec.2020 encoded.
    Some(linalg::mul(&m.xyz_to_rec2020, &m.p3_to_xyz))
}

fn aces_uniform(target: OutputTarget) -> AcesU {
    let t = Aces2Transform::shared(target.aces2());
    let p = t.params();
    let mut u = AcesU {
        in_cam: cols4(&p.input.rgb_to_cam16_c),
        in_aab: cols4(&p.input.cone_response_to_aab),
        lim_aab: cols4(&p.limit.aab_to_cone_response),
        lim_cam: cols4(&p.limit.cam16_c_to_rgb),
        ..AcesU::default()
    };
    u.in_p = [
        p.input.f_l_n as f32,
        p.input.cz as f32,
        p.input.inv_cz as f32,
        p.input.a_w_j as f32,
    ];
    u.in_q = [
        p.input.inv_a_w_j as f32,
        p.limit_j_max as f32,
        p.model_gamma_inv as f32,
        p.tone_scale.forward_limit as f32,
    ];
    u.lim_p = [
        p.limit.inv_cz as f32,
        p.mid_j as f32,
        p.focus_dist as f32,
        p.lower_hull_gamma_inv as f32,
    ];
    u.ts0 = [
        p.tone_scale.n_r as f32,
        p.tone_scale.g as f32,
        p.tone_scale.t_1 as f32,
        p.tone_scale.s_2 as f32,
    ];
    u.ts1 = [
        p.tone_scale.m_2 as f32,
        (p.peak_luminance_nits / REF_LUMINANCE) as f32,
        0.0,
        0.0,
    ];
    u.cc = [
        p.sat as f32,
        p.sat_thr as f32,
        p.compr as f32,
        p.chroma_compress_scale as f32,
    ];
    let range = p.hue_linearity_search_range;
    match limit_to_display(target) {
        Some(m) => {
            u.l2d = rows4(&m);
            u.hue = [range[0], range[1], 1, 0];
        }
        None => u.hue = [range[0], range[1], 0, 0],
    }
    u
}

/// The ACES 2 tables, in the order of the shader: reach M, hue, cusp J, cusp M, upper hull gamma.
fn aces_tables(target: OutputTarget) -> [Vec<f64>; 5] {
    let p = Aces2Transform::shared(target.aces2()).params();
    [
        p.reach_m_table.clone(),
        p.hue_table.clone(),
        p.cusp_j_table.clone(),
        p.cusp_m_table.clone(),
        p.upper_hull_gamma_inv_table.clone(),
    ]
}

/// The 64 x 64 dither thresholds as the `tables` section the shaders read (`y * 64 + x`).
fn dither_section() -> Vec<f32> {
    let mut v = vec![0.0f32; table::DITHER_ENTRIES];
    for y in 0..64 {
        for x in 0..64 {
            v[y * 64 + x] = dither_threshold(x, y) as f32;
        }
    }
    v
}

// ---------------------------------------------------------------------------------------------
// The persistent tables
// ---------------------------------------------------------------------------------------------

/// Curve sets kept per device (the last ones edited: a curve drag walks through many).
const CURVE_CACHE_LEN: usize = 12;
/// LUTs kept per device, by count and by bytes (a look tab renders a handful of looks).
const LUT_CACHE_LEN: usize = 6;
const LUT_CACHE_BYTES: u64 = 192 << 20;

/// A 128-bit hash of a stream of words: two independent multiply-rotate mixers. It keys the LUT
/// cache, so an accidental collision would show the wrong look; 128 bits make that impossible in
/// practice, and the pointer shortcut avoids hashing a 34 MB cube on every frame.
struct Hash128(u64, u64);

impl Hash128 {
    fn new() -> Self {
        Self(0x243F_6A88_85A3_08D3, 0x1319_8A2E_0370_7344)
    }

    #[inline]
    fn word(&mut self, w: u64) {
        self.0 = (self.0 ^ w)
            .wrapping_mul(0x9E37_79B9_7F4A_7C15)
            .rotate_left(29);
        self.1 = (self.1.rotate_left(23) ^ w.wrapping_mul(0xC2B2_AE3D_27D4_EB4F))
            .wrapping_mul(0x1656_67B1_9E37_79F9);
    }

    fn vec3(&mut self, v: Vec3) {
        for c in v {
            self.word(c.to_bits());
        }
    }

    fn finish(self) -> u128 {
        let fin = |mut x: u64| {
            x ^= x >> 33;
            x = x.wrapping_mul(0xFF51_AFD7_ED55_8CCD);
            x ^= x >> 33;
            x = x.wrapping_mul(0xC4CE_B9FE_1A85_EC53);
            x ^ (x >> 33)
        };
        (u128::from(fin(self.0)) << 64) | u128::from(fin(self.1 ^ self.0.rotate_left(32)))
    }
}

/// The content hash of a LUT: sizes, domains and every entry of the shaper and the cube.
fn hash_lut(lut: &Lut) -> u128 {
    let mut h = Hash128::new();
    match &lut.shaper {
        Some(s) => {
            h.word(1);
            h.word(s.size() as u64);
            h.vec3(s.domain_min());
            h.vec3(s.domain_max());
            s.data().iter().for_each(|&e| h.vec3(e));
        }
        None => h.word(0),
    }
    h.word(lut.cube.size() as u64);
    h.vec3(lut.cube.domain_min());
    h.vec3(lut.cube.domain_max());
    lut.cube.data().iter().for_each(|&e| h.vec3(e));
    h.finish()
}

struct CurveEntry {
    set: CurveSet,
    /// Bit `i` set when table `i` (luma, R, G, B) is not the identity.
    flags: u32,
    buf: Arc<PooledBuffer>,
}

struct LutEntry {
    /// The address of the `Lut` when it was last seen: a shortcut that is valid while `_owner`
    /// keeps the allocation alive.
    ptr: usize,
    _owner: Arc<dyn Any + Send + Sync>,
    key: u128,
    buf: Arc<PooledBuffer>,
    desc: LutDescU,
    bytes: u64,
}

/// The device-resident tables of the colour chain (module notes): one per device, found through
/// [`PreparedChain::new`]. Create one yourself ([`ChainTables::new`]) to prepare chains against a
/// private set, for instance in a test that counts uploads.
#[derive(Default)]
pub struct ChainTables {
    dither: Option<Arc<PooledBuffer>>,
    empty: Option<Arc<PooledBuffer>>,
    aces: Vec<(OutputTarget, AcesU, Arc<PooledBuffer>)>,
    curves: Vec<CurveEntry>,
    luts: Vec<LutEntry>,
}

impl std::fmt::Debug for ChainTables {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ChainTables")
            .field("aces_outputs", &self.aces.len())
            .field("curve_sets", &self.curves.len())
            .field("luts", &self.luts.len())
            .finish_non_exhaustive()
    }
}

impl ChainTables {
    /// An empty set; every table is created on first use.
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of LUTs held.
    pub fn lut_count(&self) -> usize {
        self.luts.len()
    }

    /// Number of distinct curve sets held.
    pub fn curve_set_count(&self) -> usize {
        self.curves.len()
    }

    /// The dither thresholds, created once.
    fn dither(&mut self, ctx: &GpuContext) -> Result<Arc<PooledBuffer>, GpuError> {
        if let Some(b) = &self.dither {
            return Ok(Arc::clone(b));
        }
        let buf = Arc::new(storage_buffer(
            ctx,
            "dither thresholds",
            bytemuck::cast_slice(&dither_section()),
        )?);
        self.dither = Some(Arc::clone(&buf));
        Ok(buf)
    }

    /// The stand-in bound where a chain has no look or display LUT (never read).
    fn empty(&mut self, ctx: &GpuContext) -> Result<Arc<PooledBuffer>, GpuError> {
        if let Some(b) = &self.empty {
            return Ok(Arc::clone(b));
        }
        let buf = Arc::new(storage_buffer(
            ctx,
            "empty lut",
            bytemuck::cast_slice(&[0.0f32, 0.0, 0.0, 1.0]),
        )?);
        self.empty = Some(Arc::clone(&buf));
        Ok(buf)
    }

    /// The ACES 2 scalars and the table buffer (dither, then the five hue tables) of an output.
    fn aces(
        &mut self,
        ctx: &GpuContext,
        target: OutputTarget,
    ) -> Result<(AcesU, Arc<PooledBuffer>), GpuError> {
        if let Some((_, u, b)) = self.aces.iter().find(|(t, _, _)| *t == target) {
            return Ok((*u, Arc::clone(b)));
        }
        let tables = aces_tables(target);
        let mut data = vec![0.0f32; table::TOTAL];
        data[..table::DITHER_ENTRIES].copy_from_slice(&dither_section());
        for (k, t) in tables.iter().enumerate() {
            if t.len() != table::ACES_ENTRIES {
                return Err(GpuError::Chain(format!(
                    "ACES 2 table {k} has {} entries, the kernel is laid out for {}",
                    t.len(),
                    table::ACES_ENTRIES
                )));
            }
            let start = table::ACES + k * table::ACES_ENTRIES;
            for (i, v) in t.iter().enumerate() {
                data[start + i] = *v as f32;
            }
        }
        let buf = Arc::new(storage_buffer(
            ctx,
            "aces2 tables",
            bytemuck::cast_slice(&data),
        )?);
        let u = aces_uniform(target);
        self.aces.push((target, u, Arc::clone(&buf)));
        Ok((u, buf))
    }

    /// The curve tables of `set` and the bits of the tables that are not the identity.
    fn curves(
        &mut self,
        ctx: &GpuContext,
        set: &CurveSet,
    ) -> Result<(Arc<PooledBuffer>, u32), GpuError> {
        if let Some(i) = self.curves.iter().position(|e| e.set == *set) {
            let e = self.curves.remove(i);
            let out = (Arc::clone(&e.buf), e.flags);
            self.curves.push(e);
            return Ok(out);
        }
        let baked = set.bake();
        let mut data = vec![0.0f32; 4 * TABLE_SIZE];
        let mut flags = 0u32;
        for (i, t) in [&baked.luma, &baked.r, &baked.g, &baked.b]
            .iter()
            .enumerate()
        {
            data[i * TABLE_SIZE..(i + 1) * TABLE_SIZE].copy_from_slice(&t.to_f32());
            if !t.is_identity() {
                flags |= 1 << i;
            }
        }
        let buf = Arc::new(storage_buffer(
            ctx,
            "chain curves",
            bytemuck::cast_slice(&data),
        )?);
        self.curves.push(CurveEntry {
            set: set.clone(),
            flags,
            buf: Arc::clone(&buf),
        });
        while self.curves.len() > CURVE_CACHE_LEN {
            self.curves.remove(0);
        }
        Ok((buf, flags))
    }

    /// The buffer and descriptor of `lut`, uploaded the first time its content is seen. `owner` is
    /// the `Arc` the `Lut` lives in (kept so the address stays valid).
    fn lut(
        &mut self,
        ctx: &GpuContext,
        lut: &Lut,
        owner: Arc<dyn Any + Send + Sync>,
    ) -> Result<(Arc<PooledBuffer>, LutDescU), GpuError> {
        let ptr = std::ptr::from_ref(lut) as usize;
        if let Some(i) = self.luts.iter().position(|e| e.ptr == ptr) {
            return Ok(self.touch_lut(i));
        }
        let key = hash_lut(lut);
        if let Some(i) = self.luts.iter().position(|e| e.key == key) {
            // The same content under another address: remember the new owner.
            self.luts[i].ptr = ptr;
            self.luts[i]._owner = owner;
            return Ok(self.touch_lut(i));
        }
        let (data, desc) = lut_data(lut);
        let bytes = (data.len() * 4) as u64;
        let buf = Arc::new(storage_buffer(
            ctx,
            "chain lut",
            bytemuck::cast_slice(&data),
        )?);
        self.luts.push(LutEntry {
            ptr,
            _owner: owner,
            key,
            buf: Arc::clone(&buf),
            desc,
            bytes,
        });
        // Evict the least recently used while over the limits, never the one just added.
        while self.luts.len() > 1
            && (self.luts.len() > LUT_CACHE_LEN
                || self.luts.iter().map(|e| e.bytes).sum::<u64>() > LUT_CACHE_BYTES)
        {
            self.luts.remove(0);
        }
        Ok((buf, desc))
    }

    fn touch_lut(&mut self, i: usize) -> (Arc<PooledBuffer>, LutDescU) {
        let e = self.luts.remove(i);
        let out = (Arc::clone(&e.buf), e.desc);
        self.luts.push(e);
        out
    }

    /// Prepares `chain` against these tables: builds the per-frame block and finds (or uploads) the
    /// tables it needs.
    ///
    /// # Errors
    ///
    /// Allocation errors from the pool ([`GpuError::OutOfBudget`], [`GpuError::BufferTooLarge`]).
    pub fn prepare(
        &mut self,
        ctx: &GpuContext,
        chain: &ColorChain,
    ) -> Result<PreparedChain, GpuError> {
        let params = chain.params();
        let space = chain.space();
        let look_kind = classify_look(params);
        // C1 to C6 (the fused matrix) and C7.
        let mut u = ChainU {
            m0: f4(params.matrix[0], 0.0),
            m1: f4(params.matrix[1], 0.0),
            m2: f4(params.matrix[2], 0.0),
            ..ChainU::default()
        };
        tone_uniform(&mut u, &params.tone);
        // C9.
        grade_uniform(&mut u, &params.grade);
        // C9b.
        warp_uniform(&mut u, &params.warp);

        // C10 and the display-baked look.
        u.flags[2] = match look_kind {
            LookKind::None => 0,
            LookKind::LogToLog => 1,
            LookKind::Baked => 2,
        };
        let mut look_luts = self.empty(ctx)?;
        if let Some(setting) = &params.look
            && look_kind != LookKind::None
        {
            let look = &setting.look;
            u.fl[0] = intensity_fraction(setting.intensity_pct) as f32;
            u.lk[0] = u32::from(look.input() == LookInput::LogC3Ei800);
            if let Some(cdl) = look.cdl() {
                u.lk[1] = 1;
                u.cdl_slope = f4(cdl.slope, cdl.saturation);
                u.cdl_offset = f4(cdl.offset, 0.0);
                u.cdl_power = f4(cdl.power, 0.0);
            }
            if let Some(lut) = look.lut() {
                let owner = Arc::clone(look);
                let owner: Arc<dyn Any + Send + Sync> = owner;
                let (buf, desc) = self.lut(ctx, lut, owner)?;
                u.look_lut = desc;
                look_luts = buf;
            }
            if look_kind == LookKind::Baked {
                // Rec.709 (the look's LUT output) to the output primaries.
                if let Some(m) = primaries_matrix(DisplayLutPrimaries::Rec709, space) {
                    u.look_m0 = f4(m[0], 0.0);
                    u.look_m1 = f4(m[1], 0.0);
                    u.look_m2 = f4(m[2], 0.0);
                    u.dp[3] = 1;
                }
            }
        }

        // C11.
        let (curves, curve_flags) = self.curves(ctx, &params.curves)?;
        u.flags[1] = curve_flags;

        // C12.
        let mut disp_luts = self.empty(ctx)?;
        let tables;
        match &params.display {
            DisplayTransform::Aces2 => {
                u.flags[3] = 0;
                let (aces, buf) = self.aces(ctx, params.output)?;
                u.aces = aces;
                tables = buf;
            }
            DisplayTransform::ArriDisplayLut(arri) => {
                u.flags[3] = 1;
                u.dp[0] = u32::from(matches!(
                    arri.input(),
                    ym_color::display::DisplayLutInput::LogC3Ei800
                ));
                match arri.output().transfer {
                    DisplayLutTransfer::Gamma(g) => {
                        u.dp[1] = 1;
                        u.fl[1] = g as f32;
                    }
                    DisplayLutTransfer::Srgb => u.dp[1] = 0,
                }
                if let Some(m) = primaries_matrix(arri.output().primaries, space) {
                    u.disp_m0 = f4(m[0], 0.0);
                    u.disp_m1 = f4(m[1], 0.0);
                    u.disp_m2 = f4(m[2], 0.0);
                    u.dp[2] = 1;
                }
                let lut = arri.lut();
                let owner = Arc::clone(lut);
                let owner: Arc<dyn Any + Send + Sync> = owner;
                let (buf, desc) = self.lut(ctx, lut, owner)?;
                u.disp_lut = desc;
                disp_luts = buf;
                // The ACES 2 section is never read; the dither section is all this chain needs.
                tables = self.dither(ctx)?;
            }
        }
        // C12 step 1b: the white lift (ACES 2 path only: `ColorChain::white_lift` is the identity for a
        // display LUT), after the grade, whose row it shares.
        white_lift_uniform(&mut u, chain.white_lift());
        // C13.
        u.sp[0] = u32::from(space == OutputSpace::Rec2020);
        // The Before view: its reading point, the camera default (the 3x3 from the camera RGB entering the
        // chain to scene-linear Rec.709, and the tone curve's constants) and, for it, the Rec.709 to
        // output primaries matrix (`look_m*`, `dp.w`: the same matrix the display-baked look uses, set
        // whatever the look is; the final picture never reads it without that look).
        u.lk[3] = match chain.before_point() {
            BeforePoint::CameraLog => 0,
            BeforePoint::AfterGrade => 1,
            BeforePoint::CameraDefault => 2,
        };
        let camera = chain.camera_default();
        let tone = camera.tone();
        u.cam0 = f4(camera.matrix()[0], 0.0);
        u.cam1 = f4(camera.matrix()[1], 0.0);
        u.cam2 = f4(camera.matrix()[2], 0.0);
        u.camt = [
            tone.n() as f32,
            tone.s() as f32,
            tone.k() as f32,
            tone.white() as f32,
        ];
        if let Some(m) = primaries_matrix(DisplayLutPrimaries::Rec709, space) {
            u.look_m0 = f4(m[0], 0.0);
            u.look_m1 = f4(m[1], 0.0);
            u.look_m2 = f4(m[2], 0.0);
            u.dp[3] = 1;
        }

        Ok(PreparedChain {
            base: u,
            tables,
            curves,
            look_luts,
            disp_luts,
            output: params.output,
            look: look_kind,
            before: chain.before_point(),
            dither: chain.dither(),
        })
    }
}

thread_local! {
    /// The tables of the devices whose thread this is: one entry per live device.
    static TABLES: RefCell<Vec<(Weak<Counters>, ChainTables)>> = const { RefCell::new(Vec::new()) };
}

/// Runs `f` with the persistent tables of `ctx`'s device (created empty the first time). The set
/// lives in a thread-local of the thread that calls this, which is the device thread: a `GpuContext`
/// never leaves it. Entries of devices that no longer exist are dropped on the way.
fn with_tables<R>(
    ctx: &GpuContext,
    f: impl FnOnce(&mut ChainTables) -> Result<R, GpuError>,
) -> Result<R, GpuError> {
    TABLES.with(|cell| {
        let mut slots = cell.borrow_mut();
        slots.retain(|(owner, _)| owner.strong_count() > 0);
        let id = Arc::as_ptr(ctx.counters_shared());
        let at = match slots.iter().position(|(owner, _)| owner.as_ptr() == id) {
            Some(i) => i,
            None => {
                slots.push((Arc::downgrade(ctx.counters_shared()), ChainTables::new()));
                slots.len() - 1
            }
        };
        f(&mut slots[at].1)
    })
}

/// Drops the persistent tables of `ctx`'s device (the dither thresholds, the ACES 2 tables, the
/// curves and the cached LUTs): their buffers go back to the pool and are created again by the next
/// chain. Call from the device thread; a chain prepared earlier keeps working (it holds its own
/// references). The pool ledger counts the tables as in use until then, which a test about "the
/// ledger returns to its baseline" must account for.
pub fn release_tables(ctx: &GpuContext) {
    TABLES.with(|cell| {
        let id = Arc::as_ptr(ctx.counters_shared());
        cell.borrow_mut()
            .retain(|(owner, _)| owner.as_ptr() != id && owner.strong_count() > 0);
    });
}

/// The dither thresholds buffer of the device (4096 f32, `y * 64 + x`), shared by the encode kernel
/// and the fused tail of output sharpening.
pub(crate) fn dither_table(ctx: &GpuContext) -> Result<Arc<PooledBuffer>, GpuError> {
    with_tables(ctx, |t| t.dither(ctx))
}

/// The WGSL of the colour kernel: the ACES 2 transform, the chain, the C13 helpers, with the LUT
/// functions written once for the look cube and once for the display cube.
pub(super) fn body() -> String {
    let color = include_str!("../shaders/color.wgsl");
    let (before, rest) = color
        .split_once("//@BEGIN_LUT_FUNCS\n")
        .unwrap_or((color, ""));
    let (template, after) = rest.split_once("//@END_LUT_FUNCS\n").unwrap_or(("", ""));
    let mut s = String::with_capacity(color.len() * 2);
    s.push_str(include_str!("../shaders/aces2.wgsl"));
    s.push('\n');
    s.push_str(include_str!("../shaders/c13.wgsl"));
    s.push('\n');
    s.push_str(before);
    for (suffix, buffer) in [("_look", "look_luts"), ("_disp", "disp_luts")] {
        s.push_str(&template.replace("@P@", suffix).replace("@L@", buffer));
    }
    s.push_str(after);
    s
}

// ---------------------------------------------------------------------------------------------
// The prepared chain
// ---------------------------------------------------------------------------------------------

/// A colour chain prepared for the GPU: the per-frame uniform block and references to the
/// persistent tables. Immutable, cheap to clone (the tables are shared), reusable for any number of
/// tiles and images.
#[derive(Clone)]
pub struct PreparedChain {
    base: ChainU,
    tables: Arc<PooledBuffer>,
    curves: Arc<PooledBuffer>,
    look_luts: Arc<PooledBuffer>,
    disp_luts: Arc<PooledBuffer>,
    output: OutputTarget,
    look: LookKind,
    before: BeforePoint,
    dither: bool,
}

impl std::fmt::Debug for PreparedChain {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PreparedChain")
            .field("output", &self.output)
            .field("look", &self.look)
            .finish_non_exhaustive()
    }
}

/// Where a dispatch writes: the destination's row stride and plane length (samples), and the place of
/// the window's (0, 0) in it.
#[derive(Clone, Copy, Debug)]
struct DstLayout {
    stride: usize,
    plane: usize,
    at: (usize, usize),
}

/// What a dispatch needs beyond the window: the letterbox colour and whether the source has a mask.
#[derive(Clone, Copy, Debug, Default)]
struct MaskPaint {
    has_mask: bool,
    background: u32,
}

impl PreparedChain {
    /// Prepares `chain` against the persistent tables of `ctx`'s device (module notes): an unchanged
    /// chain uploads nothing.
    ///
    /// # Errors
    ///
    /// Allocation errors from the pool ([`GpuError::OutOfBudget`], [`GpuError::BufferTooLarge`]).
    pub fn new(ctx: &GpuContext, chain: &ColorChain) -> Result<Self, GpuError> {
        with_tables(ctx, |t| t.prepare(ctx, chain))
    }

    /// Prepares `chain` against `tables` instead of the device's own set.
    ///
    /// # Errors
    ///
    /// As [`PreparedChain::new`].
    pub fn with_tables(
        ctx: &GpuContext,
        tables: &mut ChainTables,
        chain: &ColorChain,
    ) -> Result<Self, GpuError> {
        tables.prepare(ctx, chain)
    }

    /// The output this chain encodes for.
    pub fn output(&self) -> OutputTarget {
        self.output
    }

    /// Whether the look runs as a log-to-log stage, as a display-baked replacement of C12, or not
    /// at all (for tests and the UI's "includes display rendering" tag).
    pub fn look_includes_display_rendering(&self) -> bool {
        self.look == LookKind::Baked
    }

    /// Whether the chain dithers 8-bit outputs (`ChainParams::dither`); pass it as
    /// `ColorOutput::Rgba8 { dither }` to follow the chain's setting.
    pub fn dither(&self) -> bool {
        self.dither
    }

    /// The Before view's reading point.
    pub fn before_point(&self) -> BeforePoint {
        self.before
    }

    /// The uniform block of one dispatch over `out.width x out.height` pixels, and the kind of
    /// output it makes.
    fn uniform_for(
        &self,
        output: ColorOutput,
        src: &SrcWindow,
        out: &OutWindow,
        dst: DstLayout,
        mask: MaskPaint,
    ) -> Result<(ChainU, OutKind), GpuError> {
        let (out_w, out_h) = (out.width, out.height);
        let (src_stride, src_plane, offset, origin) =
            (src.stride, src.plane, src.offset, out.origin);
        let mut u = self.base;
        u.dims = [
            out_w as u32,
            out_h as u32,
            dst.stride as u32,
            dst.plane as u32,
        ];
        u.src = [
            src_stride as u32,
            src_plane as u32,
            offset.0 as u32,
            offset.1 as u32,
        ];
        u.origin = [
            origin.0 as u32,
            origin.1 as u32,
            dst.at.0 as u32,
            dst.at.1 as u32,
        ];
        u.sp[1] = u32::from(mask.has_mask);
        u.sp[2] = mask.background;
        let (stop, kind, bits, dither) = match output {
            ColorOutput::Stage(s) => (s, OutKind::F32Planar, 0, false),
            ColorOutput::Rgba8 { dither } => (Stop::C13, OutKind::Rgba8, 8, dither),
            ColorOutput::BeforeRgba8 { dither } => (Stop::Before, OutKind::Rgba8, 8, dither),
            ColorOutput::U16 { bits } => {
                if !(1..=16).contains(&bits) {
                    return Err(GpuError::InvalidInput(format!(
                        "a U16 output needs 1 to 16 bits, got {bits}"
                    )));
                }
                (Stop::C13, OutKind::U16Planar, bits, false)
            }
        };
        u.mode = [stop.code(), kind.code(), bits, u32::from(dither)];
        Ok((u, kind))
    }

    /// The bindings of one dispatch (the layout of `color.wgsl`).
    fn entries<'a>(
        &'a self,
        ub: &'a PooledBuffer,
        src: &'a PooledBuffer,
        out: &'a PooledBuffer,
    ) -> [(u32, wgpu::BindingResource<'a>); 7] {
        [
            (0, ub.binding()),
            (1, self.tables.binding()),
            (2, self.curves.binding()),
            (3, self.look_luts.binding()),
            (4, self.disp_luts.binding()),
            (5, src.binding()),
            (6, out.binding()),
        ]
    }

    /// Encodes one dispatch of the kernel over an `out_w x out_h` window of `src` (planar image
    /// samples, `src_stride` per row, `src_plane` per plane) starting at `offset`. Returns the
    /// pending readback; the caller keeps `src` alive until it has finished.
    pub(crate) fn dispatch(
        &self,
        ctx: &GpuContext,
        src: &PooledBuffer,
        src_window: &SrcWindow,
        out_window: &OutWindow,
        output: ColorOutput,
    ) -> Result<PendingColor, GpuError> {
        let (out_w, out_h) = (out_window.width, out_window.height);
        let out_stride = even(out_w);
        let dst = DstLayout {
            stride: out_stride,
            plane: out_stride * out_h,
            at: (0, 0),
        };
        let (u, kind) =
            self.uniform_for(output, src_window, out_window, dst, MaskPaint::default())?;
        let out_bytes = kind.device_bytes(out_stride, out_h);
        let mut passes = PassList::new(ctx, "color chain");
        let ub = passes.uniform("chain uniform", &u)?;
        let out = passes.keep(ctx.pool().acquire("chain out", out_bytes, STORAGE_OUT)?);
        let (gx, gy) = pair_groups(out_w, out_h);
        passes.dispatch(
            KernelId::Color,
            "color chain",
            &self.entries(&ub, src, &out),
            (gx, gy, 1),
        )?;
        let read = passes.read(&out, 0, out_bytes)?;
        let pending = passes.submit()?;
        Ok(PendingColor {
            pending,
            read,
            kind,
            width: out_w,
            height: out_h,
            stride: out_stride,
        })
    }

    /// Renders `image` (planar camera RGB, balanced with the initial gains) in one dispatch.
    /// `origin` is the absolute position of the image's (0, 0), which the dither needs.
    ///
    /// # Errors
    ///
    /// Allocation, shader or readback errors; [`GpuError::InvalidInput`] for a bad bit depth.
    pub fn render(
        &self,
        ctx: &GpuContext,
        image: &RgbImage,
        origin: (usize, usize),
        output: ColorOutput,
    ) -> Result<ColorImage, GpuError> {
        let (w, h) = (image.width, image.height);
        let stride = even(w);
        let plane = stride * h;
        let src = upload_planar(ctx, image, Rect::whole(w, h))?;
        let pending = self.dispatch(
            ctx,
            &src,
            &SrcWindow {
                stride,
                plane,
                offset: (0, 0),
            },
            &OutWindow {
                width: w,
                height: h,
                origin,
            },
            output,
        )?;
        pending.finish(ctx)
    }

    /// Renders `image` tile by tile over `plan` (cores only, no halo needed: the chain is
    /// pointwise) with up to `depth` tiles in flight, handing each finished tile to `sink` in
    /// raster order. The dither uses absolute pixel coordinates, so the result equals the
    /// whole-image render exactly.
    ///
    /// # Errors
    ///
    /// [`GpuError::Cancelled`] when `cancel` fires, plus the errors of [`PreparedChain::render`].
    #[allow(clippy::too_many_arguments)] // the natural argument list; a struct would only move them
    pub fn render_tiled(
        &self,
        ctx: &GpuContext,
        image: &RgbImage,
        plan: &TilePlan,
        output: ColorOutput,
        depth: usize,
        cancel: &CancelToken,
        sink: &mut dyn FnMut(Tile, ColorImage) -> Result<(), GpuError>,
    ) -> Result<(), GpuError> {
        if (plan.frame().w, plan.frame().h) != (image.width, image.height) {
            return Err(GpuError::InvalidInput(format!(
                "tile plan is for {}x{}, the image is {}x{}",
                plan.frame().w,
                plan.frame().h,
                image.width,
                image.height
            )));
        }
        let depth = depth.max(1);
        let mut queue: std::collections::VecDeque<(Tile, PendingColor, PooledBuffer)> =
            std::collections::VecDeque::new();
        for &tile in plan.tiles() {
            cancel.check()?;
            let c = tile.compute;
            let stride = even(c.w);
            let plane = stride * c.h;
            let src = upload_planar(ctx, image, c)?;
            let (dx, dy) = tile.core_offset();
            let pending = self.dispatch(
                ctx,
                &src,
                &SrcWindow {
                    stride,
                    plane,
                    offset: (dx, dy),
                },
                &OutWindow {
                    width: tile.core.w,
                    height: tile.core.h,
                    origin: (tile.core.x, tile.core.y),
                },
                output,
            )?;
            queue.push_back((tile, pending, src));
            if queue.len() >= depth
                && let Some((t, p, _src)) = queue.pop_front()
            {
                sink(t, p.finish(ctx)?)?;
            }
        }
        while let Some((t, p, _src)) = queue.pop_front() {
            sink(t, p.finish(ctx)?)?;
        }
        Ok(())
    }

    // ---- device in, device out ----------------------------------------------------------------

    /// Records the colour pass over the `window` of `src` (camera RGB, three planes, or four with the
    /// sampler's letterbox mask) up to `stop`, writing three f32 planes into `dst` (a fourth one,
    /// holding the mask, when `src` has four), the window's pixel (0, 0) at `dst_at`. Nothing is read
    /// back and nothing is uploaded but the uniform block.
    ///
    /// This is the C1 to C12 stage of the view and of the export: the result stays on the device.
    ///
    /// # Errors
    ///
    /// [`GpuError::InvalidInput`] when the device does not store f32 images (a legacy f16 mode), when
    /// the window is empty or outside `src`, when `src` is not a three- or four-plane image or `dst`
    /// does not have the same number of planes, or when the window does not fit `dst` at `dst_at`.
    pub fn encode_stage(
        &self,
        passes: &mut PassList<'_>,
        src: &GpuImage,
        window: Rect,
        stop: Stop,
        dst: &GpuImage,
        dst_at: (usize, usize),
    ) -> Result<(), GpuError> {
        if dst.planes != src.planes {
            return Err(GpuError::InvalidInput(format!(
                "the destination has {} planes, the source {} (the mask plane follows the pixels)",
                dst.planes, src.planes
            )));
        }
        if dst_at.0 + window.w > dst.width || dst_at.1 + window.h > dst.height {
            return Err(GpuError::InvalidInput(format!(
                "a {} x {} window at {dst_at:?} does not fit the {} x {} destination",
                window.w, window.h, dst.width, dst.height
            )));
        }
        let layout = DstLayout {
            stride: dst.stride,
            plane: dst.plane_len(),
            at: dst_at,
        };
        self.encode_device(
            passes,
            src,
            window,
            ColorOutput::Stage(stop),
            layout,
            &dst.buf,
            MaskPaint {
                has_mask: src.planes == 4,
                background: 0,
            },
            (0, 0),
        )
    }

    /// Like [`encode_stage`](Self::encode_stage), writing the finished picture (`Stop::C13`, or the
    /// Before view with `before`) as RGBA8 words into `dst`, `stride` words per row, the window's
    /// pixel (0, 0) at `dst_at`. `origin` is the absolute position of the window's (0, 0) in the
    /// picture, which selects the dither texel; pixels the source's mask marks as outside the photo
    /// are `background`.
    ///
    /// # Errors
    ///
    /// As [`encode_stage`](Self::encode_stage); [`GpuError::InvalidInput`] when `dst` is too small.
    #[allow(clippy::too_many_arguments)] // the natural argument list; a struct would only move them
    pub fn encode_rgba8(
        &self,
        passes: &mut PassList<'_>,
        src: &GpuImage,
        window: Rect,
        before: bool,
        dither: bool,
        origin: (usize, usize),
        background: [u8; 3],
        dst: &PooledBuffer,
        stride: usize,
        dst_at: (usize, usize),
    ) -> Result<(), GpuError> {
        let words = (dst_at.1 + window.h) as u64 * stride as u64;
        if dst_at.0 + window.w > stride || dst.size() < words * 4 {
            return Err(GpuError::InvalidInput(format!(
                "a {} x {} window at {dst_at:?} does not fit {stride} words per row in a {} byte buffer",
                window.w,
                window.h,
                dst.size()
            )));
        }
        let layout = DstLayout {
            stride,
            plane: 0,
            at: dst_at,
        };
        let output = if before {
            ColorOutput::BeforeRgba8 { dither }
        } else {
            ColorOutput::Rgba8 { dither }
        };
        self.encode_device(
            passes,
            src,
            window,
            output,
            layout,
            dst,
            MaskPaint {
                has_mask: src.planes == 4,
                background: u32::from(background[0])
                    | (u32::from(background[1]) << 8)
                    | (u32::from(background[2]) << 16)
                    | 0xff00_0000,
            },
            origin,
        )
    }

    #[allow(clippy::too_many_arguments)] // shared by the two public entry points
    fn encode_device(
        &self,
        passes: &mut PassList<'_>,
        src: &GpuImage,
        window: Rect,
        output: ColorOutput,
        dst: DstLayout,
        out: &PooledBuffer,
        mask: MaskPaint,
        origin: (usize, usize),
    ) -> Result<(), GpuError> {
        let ctx = passes.context();
        if ctx.storage_mode() != StorageMode::F32 {
            return Err(GpuError::InvalidInput(format!(
                "device images are f32; this device stores {} images (a legacy test mode)",
                ctx.storage_mode().name()
            )));
        }
        if !matches!(src.planes, 3 | 4) {
            return Err(GpuError::InvalidInput(format!(
                "the colour pass reads a three- or four-plane image, not {} planes",
                src.planes
            )));
        }
        if window.is_empty() || window.x1() > src.width || window.y1() > src.height {
            return Err(GpuError::InvalidInput(format!(
                "the window {window:?} is empty or outside the {} x {} source",
                src.width, src.height
            )));
        }
        let limit = ctx.info().limits.max_compute_workgroups_per_dimension as usize;
        let (gx, gy) = pair_groups(window.w, window.h);
        if gx as usize > limit || gy as usize > limit {
            return Err(GpuError::InvalidInput(format!(
                "a {} x {} window needs more workgroups than the device allows ({limit})",
                window.w, window.h
            )));
        }
        let (u, _) = self.uniform_for(
            output,
            &SrcWindow {
                stride: src.stride,
                plane: src.plane_len(),
                offset: (window.x, window.y),
            },
            &OutWindow {
                width: window.w,
                height: window.h,
                origin,
            },
            dst,
            mask,
        )?;
        let ub = passes.uniform("chain uniform", &u)?;
        passes.dispatch(
            KernelId::Color,
            "color chain",
            &self.entries(&ub, &src.buf, out),
            (gx, gy, 1),
        )
    }
}

/// The workgroups of a dispatch over a `w x h` window: one thread per horizontal pair of pixels.
fn pair_groups(w: usize, h: usize) -> (u32, u32) {
    let gx = w.div_ceil(2).div_ceil(16) as u32;
    let gy = h.div_ceil(16) as u32;
    (gx.max(1), gy.max(1))
}

/// The size of the largest storage binding a colour pass over the `window` makes (K4 of
/// `docs/17-gpu.md`): a four-plane source or destination of f32 samples, 16 bytes a pixel.
pub fn max_binding_bytes(window: Rect) -> u64 {
    16 * even(window.w) as u64 * window.h as u64
}

/// Where the planes of an image sit in a source buffer: row stride, plane length (both in
/// samples) and the offset of the window to process.
#[derive(Clone, Copy, Debug)]
pub(crate) struct SrcWindow {
    pub stride: usize,
    pub plane: usize,
    pub offset: (usize, usize),
}

/// The region a dispatch writes: its size, and the absolute pixel position of its (0, 0), which
/// the dither needs.
#[derive(Clone, Copy, Debug)]
pub(crate) struct OutWindow {
    pub width: usize,
    pub height: usize,
    pub origin: (usize, usize),
}

/// A dispatch whose readback has not been collected yet.
pub(crate) struct PendingColor {
    pending: Pending,
    read: ReadId,
    kind: OutKind,
    width: usize,
    height: usize,
    stride: usize,
}

impl PendingColor {
    pub(crate) fn finish(self, _ctx: &GpuContext) -> Result<ColorImage, GpuError> {
        let done = self.pending.finish()?;
        Ok(ColorImage {
            width: self.width,
            height: self.height,
            data: unpack(
                self.kind,
                done.read(self.read),
                self.width,
                self.height,
                self.stride,
            ),
        })
    }
}

/// Uploads the `rect` of a planar image as three planes with an even row stride, in the device's
/// storage mode.
fn upload_planar(ctx: &GpuContext, image: &RgbImage, rect: Rect) -> Result<PooledBuffer, GpuError> {
    let mode = ctx.storage_mode();
    let stride = even(rect.w);
    let plane = stride * rect.h;
    let mut samples = vec![0.0f32; 3 * plane];
    for c in 0..3 {
        let src = image.plane(c);
        for y in 0..rect.h {
            let s = (rect.y + y) * image.width + rect.x;
            let d = c * plane + y * stride;
            samples[d..d + rect.w].copy_from_slice(&src[s..s + rect.w]);
        }
    }
    let bytes = image_bytes(mode, &samples);
    let buf = ctx
        .pool()
        .acquire("chain input", sample_bytes(mode, 3 * plane), STORAGE_IN)?;
    upload_bytes(ctx, &buf, 0, &bytes)?;
    Ok(buf)
}

#[cfg(test)]
mod tests {
    use std::mem::{offset_of, size_of};

    use naga::valid::{Capabilities, ValidationFlags, Validator};

    use super::*;
    use crate::kernels::shader_source;

    #[test]
    fn the_per_frame_block_is_about_2_kb() {
        // The per-frame `ChainU` (`docs/17-gpu.md` section 11) was meant to be about 1 KB: the
        // ACES 2 scalars live in the one uniform block (K3 allows one), so the block holds the
        // whole chain's parameters. It was 992 bytes; the Before view's camera default added 64
        // (the 3x3 rows and the curve constants), and the colour warp (C9b) 1040: the counts (16),
        // three rows for each of 16 moving pins (768) and one row for each of 16 locks (256). Still
        // far below the 64 KiB uniform binding the device asks for
        // (`max_uniform_buffer_binding_size` of `wgpu::Limits::default()`) and the 16 KiB of wgpu's
        // downlevel limits.
        assert_eq!(size_of::<ChainU>(), 2096);
        assert_eq!(size_of::<ChainU>() % 16, 0);
        assert_eq!(
            size_of::<ChainU>() - offset_of!(ChainU, warp_n),
            16 + 16 * (WARP_MOVE_ROWS + WARP_LOCK_ROWS)
        );
        assert_eq!(WARP_MOVE_ROWS, 48);
        assert_eq!(WARP_LOCK_ROWS, 16);
    }

    #[test]
    fn the_warp_rows_are_what_the_cpu_prepares() {
        use ym_color::warp::{WarpParams, WarpPin};
        let pins = vec![
            WarpPin {
                range: 0.05,
                shadows: 0.25,
                highlights: 0.75,
                pivot_ev: 1.5,
                exposure_ev: -0.5,
                ..WarpPin::moving([0.1, 0.02], [0.04, 0.09])
            },
            // A still pin: left out, so the next moving pin takes the second slot.
            WarpPin::moving([0.2, 0.2], [0.2, 0.2]),
            WarpPin::locked_at([-0.05, 0.03]),
            // A long move with the smallest range: the range stays, the flow takes more steps.
            WarpPin {
                range: 0.001,
                ..WarpPin::moving([0.0, -0.1], [0.0, 0.2])
            },
        ];
        let params = WarpParams { pins };
        let warp = Warp::new(&params);
        let mut u = ChainU::default();
        warp_uniform(&mut u, &params);
        // ceil(1.25 (0.0922 / (0.05 sqrt(e)) + 0.3 / (0.01 sqrt(e)))) = ceil(24.14) = 25 steps.
        assert_eq!(warp.steps(), 25);
        assert_eq!(u.warp_n, [2, 1, 25, 0]);
        for (k, m) in warp.moves().iter().enumerate() {
            let rows = &u.warp_mv[WARP_ROWS_PER_MOVE * k..][..WARP_ROWS_PER_MOVE];
            let want = [
                [m.source[0], m.source[1], m.delta[0], m.delta[1]],
                [m.inv_two_sigma_sq, m.shadows, m.highlights, m.pivot_code],
                [m.exposure_ev, 0.0, 0.0, 0.0],
            ];
            for (row, w) in rows.iter().zip(want) {
                assert_eq!(*row, w.map(|x| x as f32), "move {k}");
            }
        }
        // The second move keeps its range, clamped to 0.01 (the smallest).
        let inv = f64::from(u.warp_mv[WARP_ROWS_PER_MOVE + 1][0]);
        assert!((inv - 1.0 / (2.0 * 0.01 * 0.01)).abs() < 1e-3, "{inv}");
        let l = warp.locks()[0];
        assert_eq!(
            u.warp_lk[0],
            [
                l.at[0] as f32,
                l.at[1] as f32,
                l.inv_two_sigma_sq as f32,
                0.0
            ]
        );
        assert!(
            u.warp_mv[2 * WARP_ROWS_PER_MOVE..]
                .iter()
                .all(|r| *r == [0.0; 4])
        );
        assert!(u.warp_lk[1..].iter().all(|r| *r == [0.0; 4]));
        // No moving pin: nothing at all, the locks included (the kernel skips the stage).
        for pins in [
            vec![],
            vec![WarpPin::locked_at([0.1, 0.0])],
            vec![
                WarpPin::moving([0.1, 0.0], [0.1, 0.0]),
                WarpPin::locked_at([0.0, 0.1]),
            ],
        ] {
            let mut u = ChainU::default();
            warp_uniform(&mut u, &WarpParams { pins });
            assert_eq!(
                bytemuck::bytes_of(&u),
                bytemuck::bytes_of(&ChainU::default())
            );
        }
        // The maximum: 16 moving pins fill every row of `warp_mv`.
        let pins: Vec<WarpPin> = (0..20)
            .map(|i| {
                let a = f64::from(i) * 0.3;
                WarpPin::moving(
                    [0.1 * a.cos(), 0.1 * a.sin()],
                    [0.12 * a.cos(), 0.1 * a.sin()],
                )
            })
            .collect();
        let mut u = ChainU::default();
        warp_uniform(&mut u, &WarpParams { pins });
        // Sixteen moves of 0.02 |cos a| at the default range: ceil(3.98) = 4 steps.
        assert_eq!(u.warp_n, [16, 0, 4, 0]);
        assert!(
            u.warp_mv
                .chunks_exact(WARP_ROWS_PER_MOVE)
                .all(|r| r[1][0] > 0.0)
        );
    }

    #[test]
    fn the_tone_rows_carry_the_highlight_range_and_no_range_is_finite() {
        use ym_color::tone::{HIGHLIGHT_CAP_STOPS, HIGHLIGHT_OFF_STOPS};
        // The Auto clip of a Sony body, 3.58 stops over grey: knee 2.58, expansion half the way to the cap.
        let tone = ToneParams::new(0.5, 0.5, 0.18 * 3.58_f64.exp2(), 0.002).expect("tone");
        let mut u = ChainU::default();
        tone_uniform(&mut u, &tone);
        let r = tone.range();
        assert_eq!(
            u.tone,
            [
                r.knee_stops as f32,
                r.clip_stops as f32,
                tone.toe_tau() as f32,
                r.expansion_stops as f32
            ]
        );
        assert!((f64::from(u.tone[0]) - 2.58).abs() < 1e-6);
        assert!((f64::from(u.tone[3]) - 0.5 * (HIGHLIGHT_CAP_STOPS - 3.58)).abs() < 1e-6);
        assert_eq!(u.fl, [0.0, 0.0, HIGHLIGHT_CAP_STOPS as f32, 0.0]);
        // C7 off: knee and cap 1e30, finite in f32 (Metal's fast-math assumes no infinity), no
        // expansion, no toe.
        let mut u = ChainU::default();
        tone_uniform(&mut u, &ToneParams::IDENTITY);
        let off = HIGHLIGHT_OFF_STOPS as f32;
        assert!(off.is_finite() && off >= 1e30);
        assert_eq!(u.tone, [off, off, 0.0, 0.0]);
        assert_eq!(u.fl, [0.0, 0.0, off, 0.0]);
        assert!(
            bytemuck::cast_slice::<ChainU, f32>(std::slice::from_ref(&u))
                .iter()
                .all(|v| v.is_finite())
        );
    }

    #[test]
    fn the_white_lift_takes_the_lanes_the_range_and_the_grade_leave_free() {
        use ym_color::exposure::white::MIN_SPAN_EV;
        use ym_color::tone::HIGHLIGHT_CAP_STOPS;
        // Everything a chain writes into the rows the lift shares: the range (`fl.z`), the look and the
        // display LUT (`fl.x`, `fl.y`) and a grade (`grade0.xyz`).
        let tone = ToneParams::new(1.0, 0.5, 0.18 * 3.58_f64.exp2(), 0.002).expect("tone");
        let grade = GradeParams {
            contrast: 20.0,
            brightness: -10.0,
            saturation: 15.0,
            ..GradeParams::default()
        };
        let base = || {
            let mut u = ChainU::default();
            tone_uniform(&mut u, &tone);
            grade_uniform(&mut u, &grade);
            u.fl[0] = 0.7;
            u.fl[1] = 2.4;
            u
        };
        let lift = WhiteLift::new(1.15, 2.78).expect("lift");
        let mut u = base();
        let (fl, grade0) = (u.fl, u.grade0);
        white_lift_uniform(&mut u, lift);
        assert_eq!(u.fl, [fl[0], fl[1], HIGHLIGHT_CAP_STOPS as f32, 1.15]);
        assert_eq!(u.fl[2], fl[2], "the range's cap is kept");
        assert_eq!(&u.grade0[..3], &grade0[..3], "the grade is kept");
        assert_eq!(u.grade0[3], (1.0 / lift.span_ev()) as f32);
        assert_eq!(u.tone, base().tone, "the range's row is not touched");
        // A narrow ramp is at least MIN_SPAN_EV wide: the reciprocal is at most 4.
        let mut u = base();
        white_lift_uniform(&mut u, WhiteLift::new(3.0, 0.0).expect("lift"));
        assert_eq!(u.grade0[3], (1.0 / MIN_SPAN_EV) as f32);
        assert_eq!(u.grade0[3], 4.0);
        // No lift is 0 in both lanes (the shader skips the stage), whatever was there.
        let mut u = base();
        u.fl[3] = 9.0;
        u.grade0[3] = 9.0;
        white_lift_uniform(&mut u, WhiteLift::NONE);
        assert_eq!((u.fl[3], u.grade0[3]), (0.0, 0.0));
        assert_eq!(u.fl[2], HIGHLIGHT_CAP_STOPS as f32);
        // Every lane finite (K8), with the range off too.
        let mut u = ChainU::default();
        tone_uniform(&mut u, &ToneParams::IDENTITY);
        grade_uniform(&mut u, &GradeParams::default());
        white_lift_uniform(&mut u, WhiteLift::new(3.0, 1.0).expect("lift"));
        assert!(
            bytemuck::cast_slice::<ChainU, f32>(std::slice::from_ref(&u))
                .iter()
                .all(|v| v.is_finite())
        );
    }

    #[test]
    fn the_uniform_block_has_the_layout_of_the_wgsl_struct() {
        for mode in [
            StorageMode::F32,
            StorageMode::F16Native,
            StorageMode::F16Packed,
        ] {
            let source = shader_source(mode, KernelId::Color);
            let module = naga::front::wgsl::parse_str(&source)
                .unwrap_or_else(|e| panic!("{}", e.emit_to_string(&source)));
            let caps = match mode {
                StorageMode::F16Native => Capabilities::SHADER_FLOAT16,
                StorageMode::F16Packed => Capabilities::SHADER_FLOAT16_IN_FLOAT32,
                StorageMode::F32 => Capabilities::empty(),
            };
            Validator::new(ValidationFlags::all(), caps)
                .validate(&module)
                .unwrap_or_else(|e| panic!("{mode:?}: {e:?}"));
            let var = module
                .global_variables
                .iter()
                .map(|(_, v)| v)
                .find(|v| v.name.as_deref() == Some("CU"))
                .expect("the uniform CU");
            let naga::TypeInner::Struct { members, span } = &module.types[var.ty].inner else {
                panic!("CU is not a struct");
            };
            let fields: [(&str, usize); 38] = [
                ("dims", offset_of!(ChainU, dims)),
                ("src", offset_of!(ChainU, src)),
                ("origin", offset_of!(ChainU, origin)),
                ("mode", offset_of!(ChainU, mode)),
                ("flags", offset_of!(ChainU, flags)),
                ("lk", offset_of!(ChainU, lk)),
                ("dp", offset_of!(ChainU, dp)),
                ("sp", offset_of!(ChainU, sp)),
                ("m0", offset_of!(ChainU, m0)),
                ("m1", offset_of!(ChainU, m1)),
                ("m2", offset_of!(ChainU, m2)),
                ("tone", offset_of!(ChainU, tone)),
                ("grade0", offset_of!(ChainU, grade0)),
                ("zone_amt", offset_of!(ChainU, zone_amt)),
                ("zone_ctr", offset_of!(ChainU, zone_ctr)),
                ("bal0", offset_of!(ChainU, bal0)),
                ("bal1", offset_of!(ChainU, bal1)),
                ("bal_ctr", offset_of!(ChainU, bal_ctr)),
                ("fl", offset_of!(ChainU, fl)),
                ("cdl_slope", offset_of!(ChainU, cdl_slope)),
                ("cdl_offset", offset_of!(ChainU, cdl_offset)),
                ("cdl_power", offset_of!(ChainU, cdl_power)),
                ("look_lut", offset_of!(ChainU, look_lut)),
                ("disp_lut", offset_of!(ChainU, disp_lut)),
                ("disp_m0", offset_of!(ChainU, disp_m0)),
                ("disp_m1", offset_of!(ChainU, disp_m1)),
                ("disp_m2", offset_of!(ChainU, disp_m2)),
                ("look_m0", offset_of!(ChainU, look_m0)),
                ("look_m1", offset_of!(ChainU, look_m1)),
                ("look_m2", offset_of!(ChainU, look_m2)),
                ("cam0", offset_of!(ChainU, cam0)),
                ("cam1", offset_of!(ChainU, cam1)),
                ("cam2", offset_of!(ChainU, cam2)),
                ("camt", offset_of!(ChainU, camt)),
                ("aces", offset_of!(ChainU, aces)),
                ("warp_n", offset_of!(ChainU, warp_n)),
                ("warp_mv", offset_of!(ChainU, warp_mv)),
                ("warp_lk", offset_of!(ChainU, warp_lk)),
            ];
            assert_eq!(members.len(), fields.len());
            for (member, (name, offset)) in members.iter().zip(fields) {
                assert_eq!(member.name.as_deref(), Some(name));
                assert_eq!(member.offset as usize, offset, "offset of {name}");
            }
            assert_eq!(*span as usize, size_of::<ChainU>(), "size of the block");
            // The warp's arrays: as many `vec4<f32>` rows as the Rust arrays, 16 bytes apart.
            for (name, rows) in [("warp_mv", WARP_MOVE_ROWS), ("warp_lk", WARP_LOCK_ROWS)] {
                let member = members
                    .iter()
                    .find(|m| m.name.as_deref() == Some(name))
                    .expect("warp member");
                let naga::TypeInner::Array { base, size, stride } = &module.types[member.ty].inner
                else {
                    panic!("{name} is not an array");
                };
                assert_eq!(*stride, 16, "{name}: stride");
                assert!(
                    matches!(size, naga::ArraySize::Constant(n) if n.get() as usize == rows),
                    "{name}: {size:?} rows, the host packs {rows}"
                );
                assert!(
                    matches!(
                        module.types[*base].inner,
                        naga::TypeInner::Vector {
                            size: naga::VectorSize::Quad,
                            scalar: naga::Scalar::F32
                        }
                    ),
                    "{name}: rows of vec4<f32>"
                );
            }
        }
    }

    #[test]
    fn the_kernel_binds_the_buffers_the_host_binds() {
        // Bindings 0 to 6, one uniform and six storage buffers: the table of `entries`.
        let source = shader_source(StorageMode::F32, KernelId::Color);
        let module = naga::front::wgsl::parse_str(&source).expect("color.wgsl");
        let info = Validator::new(ValidationFlags::all(), Capabilities::empty())
            .validate(&module)
            .expect("valid");
        let used = info.get_entry_point(0);
        let mut bindings = std::collections::BTreeSet::new();
        for (handle, var) in module.global_variables.iter() {
            if let Some(b) = &var.binding
                && !used[handle].is_empty()
            {
                assert_eq!(b.group, 0);
                bindings.insert(b.binding);
            }
        }
        assert_eq!(bindings, (0..7).collect());
    }

    #[test]
    fn the_hash_sees_every_entry() {
        use ym_color::lut::Lut3d;
        let make = |bump: f64| {
            Lut::from_cube(
                Lut3d::bake(5, [0.0; 3], [1.0; 3], |c| {
                    [c[0] + bump * c[1], c[1], c[2] * 0.5 + 0.01]
                })
                .expect("cube"),
            )
        };
        let a = make(0.0);
        assert_eq!(hash_lut(&a), hash_lut(&make(0.0)));
        assert_ne!(hash_lut(&a), hash_lut(&make(1e-9)));
    }
}
