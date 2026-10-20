//! Generation of the final WGSL text: the storage-mode prelude, constants written from
//! `ym_color`, the shared helpers and the kernel body.

use std::fmt::Write as _;

use ym_color::consts::{
    LOGC3_EI800, LOGC4, LOGC4_GRAY, LOGC4_STOP, REC709_LUMA_WEIGHTS, linalg::Mat3,
};
use ym_color::display::aces2_input_matrix;
use ym_color::exposure::TARGET_GREY;
use ym_color::exposure::white;
use ym_color::gamut::{awg4_luminance_row, matrices};
use ym_color::grade::{
    BRIGHTNESS_SIGMA, COLOR_BALANCE_STRENGTH, CONTRAST_BLACK, CONTRAST_COLOR,
    CONTRAST_NEUTRAL_CHROMA, CONTRAST_TOP, E1, E2, SATURATION_CODE_LIMIT, ZONE_CONTROL_SIGMA,
    ZONE_SIGMA,
};
use ym_color::tone::HIGHLIGHT_GREY;
use ym_color::warp::{
    WARP_EXP_LIMIT, WARP_MAX_PINS, WARP_MAX_STEPS, WARP_NEUTRAL_GUARD, WARP_TONE_HALF_WIDTH,
};

use super::aces;
use super::color::{WARP_LOCK_ROWS, WARP_MOVE_ROWS};
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
    /// The log-guided fast demosaic (four dispatches of one entry point).
    LogGuided,
    /// Output sharpening, pass 1: the luma plane `L`.
    OutputSharpenLuma,
    /// Output sharpening: the horizontal part `t` of the difference-form high-pass.
    OutputSharpenHighpassRows,
    /// Output sharpening: the vertical part `h` of the difference-form high-pass.
    OutputSharpenHighpassCols,
    /// Output sharpening: the numerator and denominator planes of the gain maps.
    OutputSharpenGainTerms,
    /// Output sharpening: one half of a separable blur of 1 to 4 planes.
    OutputSharpenBlur,
    /// Output sharpening: the stage's delta, added to `D` and `Hacc`.
    OutputSharpenAccumulate,
    /// Output sharpening: the halo limiter's 5 x 5 envelope window.
    OutputSharpenWindow,
    /// Output sharpening: the limiter's clamp and the ratio transfer to RGB.
    OutputSharpenTransfer,
    /// Diffraction deconvolution: luma, inverse noise variance and clip penalty.
    DiffractionLuma,
    /// Diffraction deconvolution: the 5 x 5 lower and upper envelope estimates.
    DiffractionEnvelope,
    /// Diffraction deconvolution: the dark and bright layers.
    DiffractionLayers,
    /// Diffraction deconvolution: the horizontal half of the Gaussian high-pass.
    DiffractionGaussRows,
    /// Diffraction deconvolution: the vertical half of the Gaussian high-pass.
    DiffractionGaussCols,
    /// Diffraction deconvolution: the PSF convolutions of the layers and the residual.
    DiffractionPsf,
    /// Diffraction deconvolution: the horizontal sums of the weighted moments.
    DiffractionMomentRows,
    /// Diffraction deconvolution: the vertical sums of the moments and the per-pixel solve.
    DiffractionSolve,
    /// Diffraction deconvolution: the coefficient smoothing, the correction and the transfer.
    DiffractionOutput,
    /// Adaptive ratio demosaic, pass 0: the padded, scaled mosaic and the native samples.
    AdaptiveRatioPrepare,
    /// Adaptive ratio demosaic, pass 1: green at the R and B sites.
    AdaptiveRatioGreen,
    /// Adaptive ratio demosaic, pass 2: the green refinement (with the CA weighting).
    AdaptiveRatioRefine,
    /// Adaptive ratio demosaic, pass 3 step 1: B at R sites and R at B sites.
    AdaptiveRatioStep1,
    /// Adaptive ratio demosaic, pass 3 step 2: R and B at G sites.
    AdaptiveRatioStep2,
    /// Adaptive ratio demosaic, pass 3 step 3: the luma proxy of the shadow sharpening.
    AdaptiveRatioLuma,
    /// Adaptive ratio demosaic, pass 3 step 3: the diagonal shadow sharpening.
    AdaptiveRatioSharpen,
    /// Adaptive ratio demosaic, shadow colour matching: the bilinear demosaic of one channel.
    AdaptiveRatioSimple,
    /// Adaptive ratio demosaic, shadow colour matching: the low-pass and squared error of one channel.
    AdaptiveRatioLowErr,
    /// Adaptive ratio demosaic, shadow colour matching: the noise-adaptive blend and the exit.
    AdaptiveRatioMatch,
    /// Adaptive ratio demosaic: the exit without shadow colour matching.
    AdaptiveRatioExit,
    /// The geometry sampler (orientation, straighten, crop, copy / Catmull-Rom / box).
    Sampler,
    /// C13 with dither and the letterbox to RGBA8 or 8 / 16-bit interleaved bands.
    Encode,
    /// The Catmull-Rom magnification of a view above 100 % with the paper, C13 fused.
    Magnify,
    /// R1 normalise, R3 vignetting and R4 gains of a rectangle, from the resident raw words.
    Prep,
    /// The R1 clip bits of a rectangle (one bit per photosite) from the words and the steps.
    PrepMask,
    /// R5 highlight reconstruction, the superpixel planes of a rectangle.
    HighlightsSuper,
    /// R5 highlight reconstruction, applying the chroma offsets.
    Highlights,
    /// R5b lateral chromatic aberration on the mosaic.
    MosaicCa,
    /// R9 lens resampling.
    Lens,
    /// The half-resolution noise sigma plane, from the resident raw words.
    Sigma,
    /// The k x k mosaic binning of the fit view, with the vignetting gain at the binned pixels.
    Bin,
    /// The sigma of the binned image, from the half-resolution sigma plane.
    BinSigma,
    /// The fixed-order reduction of the R5 chroma offsets (reduced path): one partial per workgroup.
    Reduce,
    /// The fold of the partials of the R5 chroma-offset reduction, in index order.
    ReduceFold,
    /// The noise sigma of a tile, expanded from the half-resolution plane (or filled).
    SigmaExpand,
    /// The 5 x 5 opponent bilateral denoise of R6.
    Bilateral,
    /// The R7 chroma guided filter, pass 1: horizontal window means of the six quantities.
    Guided,
    /// The R7 chroma guided filter, pass 2: vertical means and the coefficients a and b.
    GuidedCols,
    /// The R7 chroma guided filter, pass 3: horizontal means of the coefficients.
    GuidedSmoothRows,
    /// The R7 chroma guided filter, pass 4: vertical means and the output.
    GuidedOutput,
    /// R7 detail re-injection, the classical luma plane (the luma of the image before the denoise).
    DetailLuma,
    /// R7 detail re-injection: `Y += beta (Y_classical - Y)`.
    Detail,
    /// R8 Richardson-Lucy, the level of every pixel's PSF and the starting planes.
    RichardsonLucy,
    /// R8 Richardson-Lucy, the horizontal half of a blur with the pixel's kernel.
    RlBlurRows,
    /// R8 Richardson-Lucy, the vertical half of the forward blur and the damped ratio.
    RlBlurColsRatio,
    /// R8 Richardson-Lucy, the vertical half of the transposed blur and the update.
    RlBlurColsUpdate,
    /// R8 Richardson-Lucy, the core of the result.
    RlOutput,
    /// The Lanczos-3 resize of the export tail (one separable pass per dispatch).
    Resize,
    /// Copies a rectangle of a planar image into another (the crop of a tile, the parts of a
    /// cut demosaic core, the pieces of a window).
    Copy,
    /// A permanent stub: the one placeholder that stays. It writes a NaN into its single
    /// storage binding and nothing else, so the tests of the pass list, the pool and a lost device,
    /// which need *a* dispatch with one binding, have one now that every other kernel is real.
    /// No stage is ever wired to it.
    Probe,
}

impl KernelId {
    /// All kernels, placeholders included (see [`KernelId::is_placeholder`]), in the order a cold
    /// start needs them (`docs/17-gpu.md` section 7, the warm-up order): the kernels of the fit
    /// view of a mosaic photo (the reduced group: R1 to R5, the binning, the sigma, the light
    /// denoise, R9), the copy that puts the pieces of an image together, the view stages (the
    /// sampler, the colour chain, output sharpening, C13), the full-resolution raw chain (the
    /// demosaic, the denoise, R7, capture sharpening, diffraction), the export tail's resize, and
    /// last the spikes and the probe, which no render uses. [`KernelSet::kernels`] and `warm_up`
    /// compile in this order, so that what the first frame of the first photo waits for is compiled
    /// before what only a zoomed view or an export needs.
    pub const ALL: [KernelId; 63] = [
        // The fit view of a mosaic photo.
        KernelId::Prep,
        KernelId::PrepMask,
        KernelId::HighlightsSuper,
        KernelId::Highlights,
        KernelId::MosaicCa,
        KernelId::Sigma,
        KernelId::Bin,
        KernelId::BinSigma,
        KernelId::Reduce,
        KernelId::ReduceFold,
        KernelId::SigmaExpand,
        KernelId::Bilateral,
        KernelId::Lens,
        KernelId::Copy,
        // The view stages.
        KernelId::Sampler,
        KernelId::Color,
        KernelId::OutputSharpenLuma,
        KernelId::OutputSharpenHighpassRows,
        KernelId::OutputSharpenHighpassCols,
        KernelId::OutputSharpenGainTerms,
        KernelId::OutputSharpenBlur,
        KernelId::OutputSharpenAccumulate,
        KernelId::OutputSharpenWindow,
        KernelId::OutputSharpenTransfer,
        KernelId::Encode,
        KernelId::Magnify,
        // The full-resolution raw chain (the 100 % view and the exports).
        KernelId::AdaptiveRatioPrepare,
        KernelId::AdaptiveRatioGreen,
        KernelId::AdaptiveRatioRefine,
        KernelId::AdaptiveRatioStep1,
        KernelId::AdaptiveRatioStep2,
        KernelId::AdaptiveRatioLuma,
        KernelId::AdaptiveRatioSharpen,
        KernelId::AdaptiveRatioSimple,
        KernelId::AdaptiveRatioLowErr,
        KernelId::AdaptiveRatioMatch,
        KernelId::AdaptiveRatioExit,
        KernelId::LogGuided,
        KernelId::Guided,
        KernelId::GuidedCols,
        KernelId::GuidedSmoothRows,
        KernelId::GuidedOutput,
        KernelId::DetailLuma,
        KernelId::Detail,
        KernelId::RichardsonLucy,
        KernelId::RlBlurRows,
        KernelId::RlBlurColsRatio,
        KernelId::RlBlurColsUpdate,
        KernelId::RlOutput,
        KernelId::DiffractionLuma,
        KernelId::DiffractionEnvelope,
        KernelId::DiffractionLayers,
        KernelId::DiffractionGaussRows,
        KernelId::DiffractionGaussCols,
        KernelId::DiffractionPsf,
        KernelId::DiffractionMomentRows,
        KernelId::DiffractionSolve,
        KernelId::DiffractionOutput,
        // The export tail.
        KernelId::Resize,
        // No render uses these.
        KernelId::SpikeNormalize,
        KernelId::SpikeDemosaic,
        KernelId::SpikeMatrix,
        KernelId::Probe,
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
            KernelId::AdaptiveRatioPrepare => "adaptive_ratio_prepare",
            KernelId::AdaptiveRatioGreen => "adaptive_ratio_green",
            KernelId::AdaptiveRatioRefine => "adaptive_ratio_refine",
            KernelId::AdaptiveRatioStep1 => "adaptive_ratio_step1",
            KernelId::AdaptiveRatioStep2 => "adaptive_ratio_step2",
            KernelId::AdaptiveRatioLuma => "adaptive_ratio_luma",
            KernelId::AdaptiveRatioSharpen => "adaptive_ratio_sharpen",
            KernelId::AdaptiveRatioSimple => "adaptive_ratio_simple",
            KernelId::AdaptiveRatioLowErr => "adaptive_ratio_low_err",
            KernelId::AdaptiveRatioMatch => "adaptive_ratio_match",
            KernelId::AdaptiveRatioExit => "adaptive_ratio_exit",
            KernelId::Sampler => "sampler",
            KernelId::Encode => "encode",
            KernelId::Magnify => "magnify",
            KernelId::Prep => "prep",
            KernelId::PrepMask => "prep_mask",
            KernelId::HighlightsSuper => "highlights_super",
            KernelId::Highlights => "highlights",
            KernelId::MosaicCa => "mosaic_ca",
            KernelId::Lens => "lens",
            KernelId::Sigma => "sigma",
            KernelId::Bin => "bin",
            KernelId::BinSigma => "bin_sigma",
            KernelId::Reduce => "reduce",
            KernelId::ReduceFold => "reduce_fold",
            KernelId::SigmaExpand => "sigma_expand",
            KernelId::Bilateral => "bilateral",
            KernelId::Guided => "guided",
            KernelId::GuidedCols => "guided_cols",
            KernelId::GuidedSmoothRows => "guided_smooth_rows",
            KernelId::GuidedOutput => "guided_output",
            KernelId::DetailLuma => "detail_luma",
            KernelId::Detail => "detail",
            KernelId::RichardsonLucy => "richardson_lucy",
            KernelId::RlBlurRows => "rl_blur_rows",
            KernelId::RlBlurColsRatio => "rl_blur_cols_ratio",
            KernelId::RlBlurColsUpdate => "rl_blur_cols_update",
            KernelId::RlOutput => "rl_output",
            KernelId::Resize => "resize",
            KernelId::Copy => "copy",
            KernelId::Probe => "probe",
        }
    }

    /// Who a placeholder kernel is for, `None` for a real kernel.
    ///
    /// A placeholder's WGSL is a stub that writes NaN into its first binding and does nothing
    /// else; [`warm_up`](crate::device::GpuContext::warm_up) skips it, and the tests list it. A
    /// kernel registered before its implementation gets an arm here until its own `.wgsl` replaces
    /// the stub in `KernelId::body`.
    pub const fn placeholder_owner(self) -> Option<&'static str> {
        // Every other kernel is real. The probe stays a stub on purpose, so that the offline checks
        // always have a placeholder to check.
        match self {
            KernelId::Probe => Some("the offline shader checks"),
            _ => None,
        }
    }

    /// Whether this kernel is still a placeholder ([`KernelId::placeholder_owner`]).
    pub const fn is_placeholder(self) -> bool {
        self.placeholder_owner().is_some()
    }

    /// The warm-up set this kernel belongs to: never [`KernelSet::All`].
    pub const fn set(self) -> KernelSet {
        match self {
            KernelId::Color
            | KernelId::SpikeMatrix
            | KernelId::OutputSharpenLuma
            | KernelId::OutputSharpenHighpassRows
            | KernelId::OutputSharpenHighpassCols
            | KernelId::OutputSharpenGainTerms
            | KernelId::OutputSharpenBlur
            | KernelId::OutputSharpenAccumulate
            | KernelId::OutputSharpenWindow
            | KernelId::OutputSharpenTransfer
            | KernelId::Sampler
            | KernelId::Encode
            | KernelId::Magnify => KernelSet::View,
            KernelId::Resize => KernelSet::Tail,
            _ => KernelSet::Raw,
        }
    }

    fn body(self) -> String {
        match self {
            KernelId::SpikeNormalize => include_str!("../shaders/spike_normalize.wgsl").to_string(),
            KernelId::SpikeDemosaic => include_str!("../shaders/spike_demosaic.wgsl").to_string(),
            KernelId::SpikeMatrix => include_str!("../shaders/spike_matrix.wgsl").to_string(),
            KernelId::Color => super::color::body(),
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
            KernelId::AdaptiveRatioPrepare
            | KernelId::AdaptiveRatioGreen
            | KernelId::AdaptiveRatioRefine
            | KernelId::AdaptiveRatioStep1
            | KernelId::AdaptiveRatioStep2
            | KernelId::AdaptiveRatioLuma
            | KernelId::AdaptiveRatioSharpen
            | KernelId::AdaptiveRatioSimple
            | KernelId::AdaptiveRatioLowErr
            | KernelId::AdaptiveRatioMatch
            | KernelId::AdaptiveRatioExit => super::adaptive_ratio::body(self),
            KernelId::Sampler => super::sampler::body(),
            KernelId::Encode => super::encode::body(),
            KernelId::Magnify => super::magnify::body(),
            KernelId::SigmaExpand | KernelId::Bilateral => super::bilateral::body(self),
            KernelId::Guided
            | KernelId::GuidedCols
            | KernelId::GuidedSmoothRows
            | KernelId::GuidedOutput => super::guided::body(self),
            KernelId::DetailLuma | KernelId::Detail => super::detail::body(self),
            KernelId::RichardsonLucy
            | KernelId::RlBlurRows
            | KernelId::RlBlurColsRatio
            | KernelId::RlBlurColsUpdate
            | KernelId::RlOutput => super::richardson_lucy::body(self),
            KernelId::Prep | KernelId::PrepMask => super::prep::body(self),
            KernelId::HighlightsSuper
            | KernelId::Highlights
            | KernelId::Reduce
            | KernelId::ReduceFold => super::highlights::body(self),
            KernelId::MosaicCa => super::mosaic_ca::body(),
            KernelId::Lens => super::lens::body(),
            KernelId::Sigma => super::sigma::body(),
            KernelId::Bin | KernelId::BinSigma => super::bin::body(self),
            KernelId::Resize => super::resize::body(),
            KernelId::Copy => super::copy::body(),
            KernelId::Probe => placeholder_body(self),
        }
    }
}

/// Groups of kernels that are compiled together (`GpuContext::warm_up_set`): the app warms the
/// view kernels first, `ym` the sets an export needs while the photos decode.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum KernelSet {
    /// What a viewport frame needs: the colour chain, output sharpening, the sampler, C13.
    View,
    /// The raw chain: normalise, highlights, demosaic, denoise, detail, sharpen, lens, the fit path.
    Raw,
    /// What only the export tail uses: the resize.
    Tail,
    /// Every kernel.
    All,
}

impl KernelSet {
    /// Stable lower-case name.
    pub const fn name(self) -> &'static str {
        match self {
            KernelSet::View => "view",
            KernelSet::Raw => "raw",
            KernelSet::Tail => "tail",
            KernelSet::All => "all",
        }
    }

    /// Whether `id` is in this set.
    pub fn contains(self, id: KernelId) -> bool {
        self == KernelSet::All || id.set() == self
    }

    /// The kernels of this set, placeholders included, in [`KernelId::ALL`] order (the order a cold
    /// start needs them).
    pub fn kernels(self) -> impl Iterator<Item = KernelId> {
        KernelId::ALL
            .into_iter()
            .filter(move |&id| self.contains(id))
    }
}

/// The WGSL of a placeholder kernel: one storage binding it writes a NaN into, nothing else, so a
/// stage wired to it by mistake fails every parity test instead of passing by accident. The NaN is
/// built from a run-time value because WGSL refuses to evaluate one at shader-creation time.
fn placeholder_body(id: KernelId) -> String {
    let owner = id.placeholder_owner().unwrap_or("?");
    format!(
        "// PLACEHOLDER kernel `{name}`, a stub (owner: {owner}).\n\
         // It writes a NaN into binding 0 and nothing else.\n\
         @group(0) @binding(0) var<storage, read_write> placeholder_out: array<f32>;\n\
         \n\
         @compute @workgroup_size(16, 16)\n\
         fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{\n\
         \tif (gid.x == 0u && gid.y == 0u && arrayLength(&placeholder_out) > 0u) {{\n\
         \t\tplaceholder_out[0] = bitcast<f32>(0x7fc00000u + gid.x);\n\
         \t}}\n\
         }}\n",
        name = id.name(),
    )
}

/// Offsets (in f32 elements) of the sections of the `tables` storage buffer of the colour kernel
/// (the dither thresholds, then the ACES 2 tables of one output).
pub(crate) mod table {
    /// Dither thresholds, 64 x 64.
    pub const DITHER: usize = 0;
    /// Entries of the dither section.
    pub const DITHER_ENTRIES: usize = 64 * 64;
    /// First ACES 2 table: reach M. (The four 1024-entry curve tables are a buffer of their own, so
    /// that a curve edit refills 16 KB and nothing else.)
    pub const ACES: usize = DITHER_ENTRIES;
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
        ("CONTRAST_BLACK", CONTRAST_BLACK),
        ("CONTRAST_TOP", CONTRAST_TOP),
        ("CONTRAST_COLOR", CONTRAST_COLOR),
        ("CONTRAST_NEUTRAL", CONTRAST_NEUTRAL_CHROMA),
        ("SAT_CODE_LIMIT", SATURATION_CODE_LIMIT),
        ("HIGHLIGHT_GREY", HIGHLIGHT_GREY),
        ("WHITE_LIFT_KNEE", white::KNEE_EV),
        ("WHITE_LIFT_GREY", TARGET_GREY),
        ("WARP_NEUTRAL_GUARD", WARP_NEUTRAL_GUARD),
        ("WARP_TONE_HALF_WIDTH", WARP_TONE_HALF_WIDTH),
        ("WARP_EXP_LIMIT", WARP_EXP_LIMIT),
    ] {
        let _ = writeln!(s, "const {n}: f32 = {};", f(v));
    }
    // The colour warp's pin count, the sizes of its rows in the colour kernel's uniform block
    // (`kernels::color`, `ChainU`) and the most flow steps a frame takes.
    for (n, v) in [
        ("WARP_MAX_PINS", WARP_MAX_PINS),
        ("WARP_MOVE_ROWS", WARP_MOVE_ROWS),
        ("WARP_LOCK_ROWS", WARP_LOCK_ROWS),
        ("WARP_MAX_STEPS", WARP_MAX_STEPS as usize),
    ] {
        let _ = writeln!(s, "const {n}: u32 = {v}u;");
    }
    // The luma of C7's highlight range (`ym_color::tone::luma`) and of the white lift
    // (`ym_color::exposure::white::drive_luma`, the same weights).
    let _ = writeln!(
        s,
        "const LUMA_709: vec3<f32> = {};",
        vec3_row(&REC709_LUMA_WEIGHTS)
    );
    let _ = writeln!(s, "const E1: vec3<f32> = {};", vec3_row(&E1));
    let _ = writeln!(s, "const E2: vec3<f32> = {};", vec3_row(&E2));
    let _ = writeln!(
        s,
        "const Y_AWG4: vec3<f32> = {};",
        vec3_row(&awg4_luminance_row())
    );
    s.push_str(&rows("AWG4_TO_AWG3", &m.awg4_to_awg3));
    s.push_str(&rows("AWG3_TO_AWG4", &m.awg3_to_awg4));
    // The ACES 2 path's input matrix: AWG4 to AP0 with the display placement 2^k folded in
    // (`docs/02` C12 step 1a). The only use of this constant is `display_ours`, so the CPU reference
    // (`PreparedDisplay`) and the kernel apply the same numbers from one Rust constant.
    s.push_str(&rows("AWG4_TO_AP0", &aces2_input_matrix()));
    s.push_str(&rows("AP0_TO_AP1", &aces::ap0_to_ap1()));
    s.push_str(&rows("AP1_TO_AP0", &aces::ap1_to_ap0()));
    let _ = writeln!(s, "const T_DITHER: u32 = {}u;", table::DITHER);
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
