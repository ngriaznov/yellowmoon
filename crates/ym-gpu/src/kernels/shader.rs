//! Generation of the final WGSL text: the storage-mode prelude, constants written from
//! `ym_color`, the shared helpers and the kernel body.

use std::fmt::Write as _;

use ym_color::consts::{LOGC3_EI800, LOGC4, LOGC4_GRAY, LOGC4_STOP, linalg::Mat3};
use ym_color::gamut::{awg4_luminance_row, matrices};
use ym_color::grade::{
    BRIGHTNESS_SIGMA, COLOR_BALANCE_STRENGTH, E1, E2, ZONE_CONTROL_SIGMA, ZONE_SIGMA,
};

use super::aces;
use crate::device::StorageMode;

/// The kernels of the crate.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum KernelId {
    /// Spike R1 + R4: normalise and gains.
    SpikeNormalize,
    /// Spike R6: bilinear demosaic.
    SpikeDemosaic,
    /// Spike C1 + C13: fixed matrix and sRGB.
    SpikeMatrix,
    /// The whole colour chain C1 to C13.
    Color,
    /// RA-09: the log-guided fast demosaic (four dispatches of one entry point).
    LogGuided,
    /// RA-09 output sharpening, pass 1: the luma plane `L`.
    OutputSharpenLuma,
    /// RA-09 output sharpening: the horizontal part `t` of the difference-form high-pass.
    OutputSharpenHighpassRows,
    /// RA-09 output sharpening: the vertical part `h` of the difference-form high-pass.
    OutputSharpenHighpassCols,
    /// RA-09 output sharpening: the numerator and denominator planes of the gain maps.
    OutputSharpenGainTerms,
    /// RA-09 output sharpening: one half of a separable blur of 1 to 4 planes.
    OutputSharpenBlur,
    /// RA-09 output sharpening: the stage's delta, added to `D` and `Hacc`.
    OutputSharpenAccumulate,
    /// RA-09 output sharpening: the halo limiter's 5 x 5 envelope window.
    OutputSharpenWindow,
    /// RA-09 output sharpening: the limiter's clamp and the ratio transfer to RGB.
    OutputSharpenTransfer,
    /// RA-09 diffraction deconvolution: luma, inverse noise variance and clip penalty.
    DiffractionLuma,
    /// RA-09 diffraction deconvolution: the 5 x 5 lower and upper envelope estimates.
    DiffractionEnvelope,
    /// RA-09 diffraction deconvolution: the dark and bright layers.
    DiffractionLayers,
    /// RA-09 diffraction deconvolution: the horizontal half of the Gaussian high-pass.
    DiffractionGaussRows,
    /// RA-09 diffraction deconvolution: the vertical half of the Gaussian high-pass.
    DiffractionGaussCols,
    /// RA-09 diffraction deconvolution: the PSF convolutions of the layers and the residual.
    DiffractionPsf,
    /// RA-09 diffraction deconvolution: the horizontal sums of the weighted moments.
    DiffractionMomentRows,
    /// RA-09 diffraction deconvolution: the vertical sums of the moments and the per-pixel solve.
    DiffractionSolve,
    /// RA-09 diffraction deconvolution: the coefficient smoothing, the correction and the transfer.
    DiffractionOutput,
}

impl KernelId {
    /// All kernels.
    pub const ALL: [KernelId; 22] = [
        KernelId::SpikeNormalize,
        KernelId::SpikeDemosaic,
        KernelId::SpikeMatrix,
        KernelId::Color,
        KernelId::LogGuided,
        KernelId::OutputSharpenLuma,
        KernelId::OutputSharpenHighpassRows,
        KernelId::OutputSharpenHighpassCols,
        KernelId::OutputSharpenGainTerms,
        KernelId::OutputSharpenBlur,
        KernelId::OutputSharpenAccumulate,
        KernelId::OutputSharpenWindow,
        KernelId::OutputSharpenTransfer,
        KernelId::DiffractionLuma,
        KernelId::DiffractionEnvelope,
        KernelId::DiffractionLayers,
        KernelId::DiffractionGaussRows,
        KernelId::DiffractionGaussCols,
        KernelId::DiffractionPsf,
        KernelId::DiffractionMomentRows,
        KernelId::DiffractionSolve,
        KernelId::DiffractionOutput,
    ];

    /// Stable name (also the pipeline label).
    pub const fn name(self) -> &'static str {
        match self {
            KernelId::SpikeNormalize => "spike_normalize",
            KernelId::SpikeDemosaic => "spike_demosaic",
            KernelId::SpikeMatrix => "spike_matrix",
            KernelId::Color => "color_chain",
            KernelId::LogGuided => "log_guided",
            KernelId::OutputSharpenLuma => "output_sharpen_luma",
            KernelId::OutputSharpenHighpassRows => "output_sharpen_highpass_rows",
            KernelId::OutputSharpenHighpassCols => "output_sharpen_highpass_cols",
            KernelId::OutputSharpenGainTerms => "output_sharpen_gain_terms",
            KernelId::OutputSharpenBlur => "output_sharpen_blur",
            KernelId::OutputSharpenAccumulate => "output_sharpen_accumulate",
            KernelId::OutputSharpenWindow => "output_sharpen_window",
            KernelId::OutputSharpenTransfer => "output_sharpen_transfer",
            KernelId::DiffractionLuma => "diffraction_luma",
            KernelId::DiffractionEnvelope => "diffraction_envelope",
            KernelId::DiffractionLayers => "diffraction_layers",
            KernelId::DiffractionGaussRows => "diffraction_gauss_rows",
            KernelId::DiffractionGaussCols => "diffraction_gauss_cols",
            KernelId::DiffractionPsf => "diffraction_psf",
            KernelId::DiffractionMomentRows => "diffraction_moment_rows",
            KernelId::DiffractionSolve => "diffraction_solve",
            KernelId::DiffractionOutput => "diffraction_output",
        }
    }

    fn body(self) -> String {
        match self {
            KernelId::SpikeNormalize => include_str!("../shaders/spike_normalize.wgsl").to_string(),
            KernelId::SpikeDemosaic => include_str!("../shaders/spike_demosaic.wgsl").to_string(),
            KernelId::SpikeMatrix => include_str!("../shaders/spike_matrix.wgsl").to_string(),
            KernelId::Color => format!(
                "{}\n{}",
                include_str!("../shaders/aces2.wgsl"),
                include_str!("../shaders/color.wgsl")
            ),
            KernelId::LogGuided => include_str!("../shaders/log_guided.wgsl").to_string(),
            KernelId::OutputSharpenLuma
            | KernelId::OutputSharpenHighpassRows
            | KernelId::OutputSharpenHighpassCols
            | KernelId::OutputSharpenGainTerms
            | KernelId::OutputSharpenBlur
            | KernelId::OutputSharpenAccumulate
            | KernelId::OutputSharpenWindow
            | KernelId::OutputSharpenTransfer => super::output_sharpen::body(self),
            KernelId::DiffractionLuma
            | KernelId::DiffractionEnvelope
            | KernelId::DiffractionLayers
            | KernelId::DiffractionGaussRows
            | KernelId::DiffractionGaussCols
            | KernelId::DiffractionPsf
            | KernelId::DiffractionMomentRows
            | KernelId::DiffractionSolve
            | KernelId::DiffractionOutput => super::diffraction::body(self),
        }
    }
}

/// Offsets (in f32 elements) of the sections of the `tables` storage buffer of the colour kernel.
pub(crate) mod table {
    /// Dither thresholds, 64 x 64.
    pub const DITHER: usize = 0;
    /// The four 1024-entry curve tables (luma, R, G, B).
    pub const CURVES: usize = 4096;
    /// First ACES 2 table: reach M.
    pub const ACES: usize = CURVES + 4 * 1024;
    /// Entries per ACES 2 hue table (`ym_color`'s `TOTAL_TABLE_SIZE`: wrap entries included).
    pub const ACES_ENTRIES: usize = ym_color::aces2::TOTAL_TABLE_SIZE;
    /// Total f32 elements.
    pub const TOTAL: usize = ACES + 5 * ACES_ENTRIES;
}

fn f(x: f64) -> String {
    let v = x as f32;
    let s = format!("{v:?}");
    // WGSL needs a '.' or an exponent in a float literal; `{:?}` of an f32 always has one.
    s
}

fn vec3_row(row: &[f64; 3]) -> String {
    format!("vec3<f32>({}, {}, {})", f(row[0]), f(row[1]), f(row[2]))
}

fn rows(name: &str, m: &Mat3) -> String {
    let mut s = String::new();
    for (i, r) in m.iter().enumerate() {
        let _ = writeln!(s, "const {name}_{i}: vec3<f32> = {};", vec3_row(r));
    }
    s
}

/// The generated constants block.
fn constants() -> String {
    let m = matrices();
    let mut s = String::new();
    let _ = writeln!(s, "const LOG10_2: f32 = {};", f(std::f64::consts::LOG10_2));
    let _ = writeln!(s, "const LOG2_10: f32 = {};", f(std::f64::consts::LOG2_10));
    for (n, v) in [
        ("LC4_A", LOGC4.a),
        ("LC4_B", LOGC4.b),
        ("LC4_C", LOGC4.c),
        ("LC4_S", LOGC4.s),
        ("LC4_T", LOGC4.t),
        ("LC3_CUT", LOGC3_EI800.cut),
        ("LC3_A", LOGC3_EI800.a),
        ("LC3_B", LOGC3_EI800.b),
        ("LC3_C", LOGC3_EI800.c),
        ("LC3_D", LOGC3_EI800.d),
        ("LC3_E", LOGC3_EI800.e),
        ("LC3_F", LOGC3_EI800.f),
        ("GRAY", LOGC4_GRAY),
        ("STOP", LOGC4_STOP),
        ("BRIGHTNESS_SIGMA", BRIGHTNESS_SIGMA),
        ("ZONE_CONTROL_SIGMA", ZONE_CONTROL_SIGMA),
        ("ZONE_SIGMA", ZONE_SIGMA),
        ("BALANCE_STRENGTH", COLOR_BALANCE_STRENGTH),
    ] {
        let _ = writeln!(s, "const {n}: f32 = {};", f(v));
    }
    let _ = writeln!(s, "const E1: vec3<f32> = {};", vec3_row(&E1));
    let _ = writeln!(s, "const E2: vec3<f32> = {};", vec3_row(&E2));
    let _ = writeln!(
        s,
        "const Y_AWG4: vec3<f32> = {};",
        vec3_row(&awg4_luminance_row())
    );
    s.push_str(&rows("AWG4_TO_AWG3", &m.awg4_to_awg3));
    s.push_str(&rows("AWG3_TO_AWG4", &m.awg3_to_awg4));
    s.push_str(&rows("AWG4_TO_AP0", &m.awg4_to_ap0));
    s.push_str(&rows("AP0_TO_AP1", &aces::ap0_to_ap1()));
    s.push_str(&rows("AP1_TO_AP0", &aces::ap1_to_ap0()));
    let _ = writeln!(s, "const T_DITHER: u32 = {}u;", table::DITHER);
    let _ = writeln!(s, "const T_CURVES: u32 = {}u;", table::CURVES);
    let _ = writeln!(s, "const T_ACES: u32 = {}u;", table::ACES);
    let _ = writeln!(s, "const T_ACES_N: u32 = {}u;", table::ACES_ENTRIES);
    // The geometry of the ACES 2 hue tables, from the CPU reference that builds them.
    let _ = writeln!(
        s,
        "const BASE_INDEX: u32 = {}u;",
        ym_color::aces2::BASE_INDEX
    );
    let _ = writeln!(
        s,
        "const UPPER_WRAP_INDEX: u32 = {}u;",
        ym_color::aces2::UPPER_WRAP_INDEX
    );
    s
}

/// The sample accessors of one image buffer in `mode`: `<name>_ld(i)` reads sample `i`,
/// `<name>_st2(i, a, b)` writes samples `i` and `i + 1` (`i` even). Stores clamp to the f16 range
/// so an extreme value saturates instead of becoming infinity.
fn accessors(mode: StorageMode, name: &str, read_only: bool) -> String {
    let full = accessors_rw(mode, name);
    if read_only {
        // Keep only the load function: a read-only binding cannot be stored to.
        full.split_once(&format!("fn {name}_st2"))
            .map_or(full.clone(), |(ld, _)| ld.to_string())
    } else {
        full
    }
}

fn accessors_rw(mode: StorageMode, name: &str) -> String {
    match mode {
        StorageMode::F32 => format!(
            "fn {name}_ld(i: u32) -> f32 {{ return {name}[i]; }}\n\
             fn {name}_st2(i: u32, a: f32, b: f32) {{ {name}[i] = a; {name}[i + 1u] = b; }}\n"
        ),
        StorageMode::F16Native => format!(
            "fn {name}_ld(i: u32) -> f32 {{ return f32({name}[i]); }}\n\
             fn {name}_st2(i: u32, a: f32, b: f32) {{\n\
             \t{name}[i] = f16(clamp(a, -65504.0, 65504.0));\n\
             \t{name}[i + 1u] = f16(clamp(b, -65504.0, 65504.0));\n}}\n"
        ),
        StorageMode::F16Packed => format!(
            "fn {name}_ld(i: u32) -> f32 {{\n\
             \tlet w = unpack2x16float({name}[i >> 1u]);\n\
             \treturn select(w.x, w.y, (i & 1u) == 1u);\n}}\n\
             fn {name}_st2(i: u32, a: f32, b: f32) {{\n\
             \t{name}[i >> 1u] = pack2x16float(clamp(vec2<f32>(a, b), vec2<f32>(-65504.0), vec2<f32>(65504.0)));\n}}\n"
        ),
    }
}

fn element_type(mode: StorageMode) -> &'static str {
    match mode {
        StorageMode::F32 => "f32",
        StorageMode::F16Native => "f16",
        StorageMode::F16Packed => "u32",
    }
}

/// Expands `ST_ELEM` and the `//@ACCESSORS name [ro]` lines of a kernel body.
fn expand(body: &str, mode: StorageMode) -> String {
    let mut out = String::with_capacity(body.len() + 1024);
    for line in body.lines() {
        if let Some(args) = line.trim().strip_prefix("//@ACCESSORS ") {
            let mut parts = args.split_whitespace();
            let name = parts.next().unwrap_or_default();
            let read_only = parts.next() == Some("ro");
            out.push_str(&accessors(mode, name, read_only));
        } else {
            out.push_str(&line.replace("ST_ELEM", element_type(mode)));
            out.push('\n');
        }
    }
    out
}

/// The complete WGSL module of `id` for a device using `mode`.
pub fn shader_source(mode: StorageMode, id: KernelId) -> String {
    let mut s = String::new();
    if mode == StorageMode::F16Native {
        s.push_str("enable f16;\n");
    }
    s.push_str(&constants());
    s.push_str(include_str!("../shaders/common.wgsl"));
    s.push('\n');
    s.push_str(&expand(&id.body(), mode));
    s
}
