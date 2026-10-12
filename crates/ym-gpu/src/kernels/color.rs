//! The whole per-pixel colour chain C1 to C13 as one compute pass (`docs/02-colour-pipeline.md`).
//!
//! [`PreparedChain::new`] turns an immutable `ym_color::display::ColorChain` into the GPU form:
//! one uniform block (fused matrix, C7, C9, C10 and C12 parameters, flags), the four 1024-entry
//! curve tables and the dither thresholds in one f32 storage buffer, the look and display cubes in
//! another (f32, never rounded to f16, never hardware-filtered), and the ACES 2 uniform block built
//! from the CPU reference's own tables. Nothing here computes colour: every number comes from the
//! `ym_color` modules the CPU chain uses, so the kernel and `ColorChain` cannot drift apart without
//! a parity test failing.
//!
//! # Precision decision (LUTs and curves)
//!
//! A LUT texture or curve table stored as f16 rounds values in [0.5, 1) by up to 2.4e-4, which alone
//! exceeds the 2e-4 parity target of `docs/02`. Cubes, shapers and curve tables are therefore **f32
//! storage buffers** (a 65-point cube is 4.4 MB, a 129-point one 34 MB), and the tetrahedral
//! interpolation is done by hand in the shader. Only image intermediates are f16.
//!
//! # Outputs
//!
//! [`ColorOutput`] selects what is written: the value at any stage (for parity checks against
//! `ColorChain::trace` and for the DNG path), the encoded f32 picture, 8-bit RGBA with the
//! deterministic blue-noise dither at absolute pixel coordinates (tiles agree with the whole
//! image), or 10 to 16-bit planes.

use bytemuck::{Pod, Zeroable};
use ym_color::aces2::{Aces2Transform, REF_LUMINANCE};
use ym_color::consts::linalg::{self, Mat3};
use ym_color::curve::TABLE_SIZE;
use ym_color::display::{
    BeforePoint, ChainParams, ColorChain, DisplayLutPrimaries, DisplayLutTransfer,
    DisplayTransform, OutputTarget,
};
use ym_color::encode::{OutputSpace, dither_threshold};
use ym_color::gamut::matrices;
use ym_color::grade::{
    BALANCE_CENTER_STOPS, CONTRAST_EXPONENT, GradeParams, ZONE_CENTER_STOPS, logc4_at_stops,
};
use ym_color::look::{LookInput, intensity_fraction};
use ym_color::lut::Lut;

use super::shader::{KernelId, table};
use super::util::{
    OutKind, PixelData, STORAGE_IN, STORAGE_OUT, bind, even, groups, image_bytes, sample_bytes,
    storage_buffer, submit_with_readback, uniform_buffer, unpack,
};
use crate::cpu::RgbImage;
use crate::cpu::tiling::Rect;
use crate::device::{GpuContext, GpuError};
use crate::pool::PooledBuffer;
use crate::queue::CancelToken;
use crate::tiles::{Tile, TilePlan};

/// The stage whose value is written (the stop of the chain).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Stop {
    /// The fused matrix applied: linear AWG4 (C1 to C6).
    C6,
    /// After the roll-off and toe: linear AWG4.
    C7,
    /// LogC4 encode.
    C8,
    /// After the primary grade.
    C9,
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
    /// The Before view: the LogC4 signal at the chain's `BeforePoint`, clamped to [0, 1].
    Before,
}

impl Stop {
    fn code(self) -> u32 {
        match self {
            Stop::C6 => 0,
            Stop::C7 => 1,
            Stop::C8 => 2,
            Stop::C9 => 3,
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

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Default)]
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

/// Appends a LUT (shaper, then cube) to `luts` and returns its descriptor.
fn push_lut(luts: &mut Vec<f32>, lut: &Lut) -> LutDescU {
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
    d
}

/// The grade parameters as the kernel wants them (the same preparation as `Grade::new`).
fn grade_uniform(u: &mut ChainU, g: &GradeParams) {
    let contrast_gain = (CONTRAST_EXPONENT * clamp_control(g.contrast) / 100.0).exp2();
    let saturation_gain = 1.0 + clamp_control(g.saturation) / 100.0;
    if contrast_gain != 1.0 {
        u.flags[0] |= 1;
    }
    if saturation_gain != 1.0 {
        u.flags[0] |= 2;
    }
    u.grade0 = [
        contrast_gain as f32,
        (clamp_control(g.brightness) / 100.0) as f32,
        saturation_gain as f32,
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

/// A colour chain uploaded to the GPU: immutable, reusable for any number of tiles and images.
pub struct PreparedChain {
    base: ChainU,
    aces: PooledBuffer,
    tables: PooledBuffer,
    luts: PooledBuffer,
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

impl PreparedChain {
    /// Uploads `chain`'s parameters and tables.
    ///
    /// # Errors
    ///
    /// Allocation errors from the pool ([`GpuError::OutOfBudget`], [`GpuError::BufferTooLarge`]).
    pub fn new(ctx: &GpuContext, chain: &ColorChain) -> Result<Self, GpuError> {
        let params = chain.params();
        let space = chain.space();
        let look_kind = classify_look(params);
        // C1 to C6 (the fused matrix) and C7.
        let mut u = ChainU {
            m0: f4(params.matrix[0], 0.0),
            m1: f4(params.matrix[1], 0.0),
            m2: f4(params.matrix[2], 0.0),
            tone: [
                params.tone.knee() as f32,
                (params.tone.shoulder_limit() - params.tone.knee()) as f32,
                params.tone.toe_tau() as f32,
                0.0,
            ],
            ..ChainU::default()
        };
        // C9.
        grade_uniform(&mut u, &params.grade);

        let mut luts: Vec<f32> = Vec::new();
        // C10 and the display-baked look.
        u.flags[2] = match look_kind {
            LookKind::None => 0,
            LookKind::LogToLog => 1,
            LookKind::Baked => 2,
        };
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
                u.look_lut = push_lut(&mut luts, lut);
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
        let baked = params.curves.bake();
        for (bit, t) in [&baked.luma, &baked.r, &baked.g, &baked.b]
            .iter()
            .enumerate()
        {
            if !t.is_identity() {
                u.flags[1] |= 1 << bit;
            }
        }

        // C12.
        let mut aces_tables_data: Option<[Vec<f64>; 5]> = None;
        match &params.display {
            DisplayTransform::Aces2 => {
                u.flags[3] = 0;
                aces_tables_data = Some(aces_tables(params.output));
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
                u.disp_lut = push_lut(&mut luts, arri.lut());
            }
        }
        // C13.
        u.sp[0] = u32::from(space == OutputSpace::Rec2020);
        u.lk[3] = u32::from(chain.before_point() == BeforePoint::AfterGrade);

        // The ACES 2 uniform exists for every chain (the kernel binds it); it is only read when the
        // display transform is ACES 2.
        let aces = uniform_buffer(ctx, "aces2 params", &aces_uniform(params.output))?;

        // Tables: dither thresholds, curves, ACES 2 hue tables.
        let mut tables = vec![0.0f32; table::TOTAL];
        for y in 0..64 {
            for x in 0..64 {
                tables[table::DITHER + y * 64 + x] = dither_threshold(x, y) as f32;
            }
        }
        for (i, t) in [&baked.luma, &baked.r, &baked.g, &baked.b]
            .iter()
            .enumerate()
        {
            let start = table::CURVES + i * TABLE_SIZE;
            tables[start..start + TABLE_SIZE].copy_from_slice(&t.to_f32());
        }
        if let Some(data) = aces_tables_data {
            for (k, t) in data.iter().enumerate() {
                if t.len() != table::ACES_ENTRIES {
                    return Err(GpuError::Chain(format!(
                        "ACES 2 table {k} has {} entries, the kernel is laid out for {}",
                        t.len(),
                        table::ACES_ENTRIES
                    )));
                }
                let start = table::ACES + k * table::ACES_ENTRIES;
                for (i, v) in t.iter().enumerate() {
                    tables[start + i] = *v as f32;
                }
            }
        }
        let tables_buf = storage_buffer(ctx, "chain tables", bytemuck::cast_slice(&tables))?;
        if luts.is_empty() {
            luts.extend([0.0, 0.0, 0.0, 1.0]);
        }
        let luts_buf = storage_buffer(ctx, "chain luts", bytemuck::cast_slice(&luts))?;
        Ok(Self {
            base: u,
            aces,
            tables: tables_buf,
            luts: luts_buf,
            output: params.output,
            look: look_kind,
            before: chain.before_point(),
            dither: chain.dither(),
        })
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

    fn uniform_for(
        &self,
        output: ColorOutput,
        src: &SrcWindow,
        out: &OutWindow,
    ) -> Result<(ChainU, OutKind), GpuError> {
        let (out_w, out_h) = (out.width, out.height);
        let (src_stride, src_plane, offset, origin) =
            (src.stride, src.plane, src.offset, out.origin);
        let mut u = self.base;
        let out_stride = even(out_w);
        u.dims = [
            out_w as u32,
            out_h as u32,
            out_stride as u32,
            (out_stride * out_h) as u32,
        ];
        u.src = [
            src_stride as u32,
            src_plane as u32,
            offset.0 as u32,
            offset.1 as u32,
        ];
        u.origin = [origin.0 as u32, origin.1 as u32, 0, 0];
        let (stop, kind, bits, dither) = match output {
            ColorOutput::Stage(s) => (s, OutKind::F32Planar, 0, false),
            ColorOutput::Rgba8 { dither } => (Stop::C13, OutKind::Rgba8, 8, dither),
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

    /// Encodes one dispatch of the kernel over an `out_w x out_h` window of `src` (planar image
    /// samples, `src_stride` per row, `src_plane` per plane) starting at `offset`. Returns the
    /// pending readback and the buffers to keep alive.
    pub(crate) fn dispatch(
        &self,
        ctx: &GpuContext,
        src: &PooledBuffer,
        src_window: &SrcWindow,
        out_window: &OutWindow,
        output: ColorOutput,
    ) -> Result<PendingColor, GpuError> {
        let (u, kind) = self.uniform_for(output, src_window, out_window)?;
        let (out_w, out_h) = (out_window.width, out_window.height);
        let out_stride = even(out_w);
        let out_bytes = kind.device_bytes(out_stride, out_h);
        let ub = uniform_buffer(ctx, "chain uniform", &u)?;
        let out = ctx.pool().acquire("chain out", out_bytes, STORAGE_OUT)?;
        let pipeline = ctx.kernels().pipeline(ctx, KernelId::Color)?;
        let group = bind(
            ctx,
            &pipeline,
            "color chain",
            &[
                (0, ub.binding()),
                (1, self.aces.binding()),
                (2, self.tables.binding()),
                (3, self.luts.binding()),
                (4, src.binding()),
                (5, out.binding()),
            ],
        );
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("color chain"),
            });
        {
            let (gx, gy) = groups(out_stride, out_h);
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("color chain"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &group, &[]);
            pass.dispatch_workgroups(gx, gy, 1);
        }
        let readback = submit_with_readback(ctx, encoder, &out, out_bytes)?;
        Ok(PendingColor {
            readback,
            kind,
            width: out_w,
            height: out_h,
            stride: out_stride,
            _held: vec![ub, out],
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
    readback: super::util::Readback,
    kind: OutKind,
    width: usize,
    height: usize,
    stride: usize,
    _held: Vec<PooledBuffer>,
}

impl PendingColor {
    pub(crate) fn finish(self, ctx: &GpuContext) -> Result<ColorImage, GpuError> {
        let bytes = self.readback.finish(ctx)?;
        Ok(ColorImage {
            width: self.width,
            height: self.height,
            data: unpack(self.kind, &bytes, self.width, self.height, self.stride),
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
    ctx.queue.write_buffer(&buf, 0, &bytes);
    Ok(buf)
}
