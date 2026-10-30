//! Generation of the final WGSL text: the storage-mode prelude, constants written from
//! `ym_color`, the shared helpers and the kernel body.

use std::fmt::Write as _;

use ym_color::consts::{
    D65_XY, LOGC3_EI800, LOGC4, LOGC4_GRAY, LOGC4_STOP, REC709_LUMA_WEIGHTS, linalg::Mat3,
    xy_to_xyz,
};
use ym_color::display::cine;
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
    /// L2, the lens's field optics: the box-reduced pyramid of the input, one level per dispatch.
    LensFieldPyramid,
    /// L2: the gather of every pixel's spread function (and the gain when there is no fringe shift).
    LensFieldGather,
    /// L2: each channel read at its fringe shift along the axis, times the gain.
    LensFieldFinish,
    /// L3, the lens's highlight optics: the restored light of the input (the clip test's boost).
    LensFlareRestore,
    /// L3: the nodes of the grids the slowly varying ghosts are drawn on (one dispatch per grid).
    LensFlareNodes,
    /// L3: the glow core, the tail plane and the drawn items of every pixel, mixed with Z.
    LensFlare,
    /// L1 and L2, the lens's field stages: the input with the colour of the light it holds.
    LensLight,
    /// L1, the lens's bokeh reshaping: the footprint means of the root colours of the blur map's cells (the upsampling's colour guide).
    LensReshapeGuide,
    /// L1, the lens's bokeh reshaping: the blur map at the pixels of the tile: the joint bilateral upsampling and its acceptance tests.
    LensReshapeBlur,
    /// L1, the lens's bokeh reshaping: the square root of the luminance, the field the sharp-edge test reads.
    LensReshapeLumRoot,
    /// L1, the lens's bokeh reshaping: the gradient energy of that field and of its 3 x 3 mean.
    LensReshapeGrad,
    /// L1, the lens's bokeh reshaping: the places whose 5 x 5 energy exceeds the mean's by the ratio of the sharp-edge test.
    LensReshapeFlags,
    /// L1, the lens's bokeh reshaping: the sharp flags dilated by 2 pixels.
    LensReshapeDilate,
    /// L1, the lens's bokeh reshaping: the balls' protection of their rim and light, the blur zeroed on the sharp places, the balls' floor and the occluded pixels.
    LensReshapeProtect,
    /// L1, the lens's bokeh reshaping: the horizontal counts of blurred pixels in the feather's window.
    LensReshapeCover,
    /// L1, the lens's bokeh reshaping: the stretch's ramp towards the border of the blur map.
    LensReshapeRamp,
    /// L1, the lens's bokeh reshaping: the picture with each ball's disc replaced by its ring's plane fit.
    LensReshapeTakeOut,
    /// L1, the lens's bokeh reshaping: the gather across the anamorphic axis.
    LensReshapeVGather,
    /// L1, the lens's bokeh reshaping: the gather along the tangent and the balls drawn back.
    LensReshapeFinal,
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
    /// R6's single-frame denoise by method A, pass 1: the working encoding, Y/Cb/Cr and the zeroed
    /// accumulators of the buffer.
    BmDctPrep,
    /// R6's single-frame denoise by method A: one pass of A.5 (block matching, the 3D transform, the hard
    /// threshold and the aggregation of every reference of the pass).
    BmDct,
    /// R6's single-frame denoise by method A, the last pass: `acc / w`, RGB, the blend and the decoding.
    BmDctOutput,
    /// R6's false-colour suppression after method A, the gate: the luma of the working encoding.
    FalseColorLuma,
    /// The false-colour gate: the alternation of the luma and its row sums.
    FalseColorRows,
    /// The false-colour gate: the column sums and the stripe energy.
    FalseColorEnergy,
    /// The false-colour gate: the dilated energy, the weight and the blend with R7's colour guided filter.
    FalseColorBlend,
    /// R6's coarse colour pass after method A: the mean Y, Cb and Cr of every 4 x 4 cell.
    CoarseChromaCells,
    /// The coarse colour pass: the bilateral filter of the cells' colour, the correction of every cell.
    CoarseChromaCorrect,
    /// The coarse colour pass: the corrections interpolated to the pixels and applied to Cb and Cr.
    CoarseChromaApply,
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
    /// demosaic, the single-frame denoise of method A and its false-colour suppression, R7, capture
    /// sharpening, diffraction), the export tail's resize, and last the spikes and the probe, which
    /// no render uses. [`KernelSet::kernels`] and `warm_up` compile in this order, so that what the
    /// first frame of the first photo waits for is compiled before what only a zoomed view or an
    /// export needs.
    pub const ALL: [KernelId; 92] = [
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
        // The lens's field optics (L2), between the raw group and the colour pass.
        KernelId::LensFieldPyramid,
        KernelId::LensFieldGather,
        KernelId::LensFieldFinish,
        KernelId::LensFlareRestore,
        KernelId::LensFlareNodes,
        KernelId::LensLight,
        KernelId::LensReshapeGuide,
        KernelId::LensReshapeBlur,
        KernelId::LensReshapeLumRoot,
        KernelId::LensReshapeGrad,
        KernelId::LensReshapeFlags,
        KernelId::LensReshapeDilate,
        KernelId::LensReshapeProtect,
        KernelId::LensReshapeCover,
        KernelId::LensReshapeRamp,
        KernelId::LensReshapeTakeOut,
        KernelId::LensReshapeVGather,
        KernelId::LensReshapeFinal,
        KernelId::LensFlare,
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
        KernelId::BmDctPrep,
        KernelId::BmDct,
        KernelId::BmDctOutput,
        KernelId::FalseColorLuma,
        KernelId::FalseColorRows,
        KernelId::FalseColorEnergy,
        KernelId::FalseColorBlend,
        KernelId::CoarseChromaCells,
        KernelId::CoarseChromaCorrect,
        KernelId::CoarseChromaApply,
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
            KernelId::LensFieldPyramid => "lens_field_pyramid",
            KernelId::LensFieldGather => "lens_field_gather",
            KernelId::LensFieldFinish => "lens_field_finish",
            KernelId::LensFlareRestore => "lens_flare_restore",
            KernelId::LensFlareNodes => "lens_flare_nodes",
            KernelId::LensFlare => "lens_flare",
            KernelId::LensLight => "lens_light",
            KernelId::LensReshapeGuide => "lens_reshape_guide",
            KernelId::LensReshapeBlur => "lens_reshape_blur",
            KernelId::LensReshapeLumRoot => "lens_reshape_lumroot",
            KernelId::LensReshapeGrad => "lens_reshape_grad",
            KernelId::LensReshapeFlags => "lens_reshape_flags",
            KernelId::LensReshapeDilate => "lens_reshape_dilate",
            KernelId::LensReshapeProtect => "lens_reshape_protect",
            KernelId::LensReshapeCover => "lens_reshape_cover",
            KernelId::LensReshapeRamp => "lens_reshape_ramp",
            KernelId::LensReshapeTakeOut => "lens_reshape_takeout",
            KernelId::LensReshapeVGather => "lens_reshape_vgather",
            KernelId::LensReshapeFinal => "lens_reshape_final",
            KernelId::Sigma => "sigma",
            KernelId::Bin => "bin",
            KernelId::BinSigma => "bin_sigma",
            KernelId::Reduce => "reduce",
            KernelId::ReduceFold => "reduce_fold",
            KernelId::SigmaExpand => "sigma_expand",
            KernelId::Bilateral => "bilateral",
            KernelId::BmDctPrep => "bm_dct_prep",
            KernelId::BmDct => "bm_dct",
            KernelId::BmDctOutput => "bm_dct_output",
            KernelId::FalseColorLuma => "false_color_luma",
            KernelId::FalseColorRows => "false_color_rows",
            KernelId::FalseColorEnergy => "false_color_energy",
            KernelId::FalseColorBlend => "false_color_blend",
            KernelId::CoarseChromaCells => "coarse_chroma_cells",
            KernelId::CoarseChromaCorrect => "coarse_chroma_correct",
            KernelId::CoarseChromaApply => "coarse_chroma_apply",
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
            | KernelId::Magnify
            | KernelId::LensFieldPyramid
            | KernelId::LensFieldGather
            | KernelId::LensFieldFinish
            | KernelId::LensFlareRestore
            | KernelId::LensFlareNodes
            | KernelId::LensLight
            | KernelId::LensReshapeGuide
            | KernelId::LensReshapeBlur
            | KernelId::LensReshapeLumRoot
            | KernelId::LensReshapeGrad
            | KernelId::LensReshapeFlags
            | KernelId::LensReshapeDilate
            | KernelId::LensReshapeProtect
            | KernelId::LensReshapeCover
            | KernelId::LensReshapeRamp
            | KernelId::LensReshapeTakeOut
            | KernelId::LensReshapeVGather
            | KernelId::LensReshapeFinal
            | KernelId::LensFlare => KernelSet::View,
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
            KernelId::BmDctPrep | KernelId::BmDct | KernelId::BmDctOutput => {
                super::bm_dct::body(self)
            }
            KernelId::FalseColorLuma
            | KernelId::FalseColorRows
            | KernelId::FalseColorEnergy
            | KernelId::FalseColorBlend => super::false_color::body(self),
            KernelId::CoarseChromaCells
            | KernelId::CoarseChromaCorrect
            | KernelId::CoarseChromaApply => super::coarse_chroma::body(self),
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
            KernelId::LensFieldPyramid | KernelId::LensFieldGather | KernelId::LensFieldFinish => {
                super::lens_field::body(self)
            }
            KernelId::LensFlareRestore | KernelId::LensFlareNodes | KernelId::LensFlare => {
                super::lens_flare::body(self)
            }
            KernelId::LensLight => super::lens_light::body(),
            KernelId::LensReshapeGuide
            | KernelId::LensReshapeBlur
            | KernelId::LensReshapeLumRoot
            | KernelId::LensReshapeGrad
            | KernelId::LensReshapeFlags
            | KernelId::LensReshapeDilate
            | KernelId::LensReshapeProtect
            | KernelId::LensReshapeCover
            | KernelId::LensReshapeRamp
            | KernelId::LensReshapeTakeOut
            | KernelId::LensReshapeVGather
            | KernelId::LensReshapeFinal => super::lens_reshape::body(self),
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

/// Offsets (in f32 elements) of the sections of the `tables` storage buffer of the colour kernel (the dither
/// thresholds; the four 1024-entry curve tables are a buffer of their own, so that a curve edit refills 16 KB
/// and nothing else).
pub(crate) mod table {
    /// Dither thresholds, 64 x 64.
    pub const DITHER: usize = 0;
    /// Entries of the dither section.
    pub const DITHER_ENTRIES: usize = 64 * 64;
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

/// The rows of the chromatic distance of the cine rendering's luminance floor (step 2 of C12): `dX = X - XN Y`
/// and `dZ = Z - ZN Y` as linear functions of the linear AWG4 triple `x`, `DX = row_X - XN row_Y` and
/// `DZ = row_Z - ZN row_Y`, formed in f64. Returned as `[[DX_r, DX_b], [DZ_r, DZ_b]]`: AWG4's white maps to
/// `(XN, 1, ZN)`, so each full row sums to 0 (to 2e-16) and `D . x = D . (x - x_g 1)` has no green entry. The
/// kernel applies them to the channel differences `(r - g, b - g)`, so that a neutral gives exactly 0.
pub(crate) fn cine_floor_rows() -> [[f64; 2]; 2] {
    let m = matrices().awg4_to_xyz;
    let d65 = xy_to_xyz(D65_XY);
    // The same two quotients as `ym_color::display::cine` (`XN`, `ZN` of `kernel_scalars`).
    let (xn, zn) = (d65[0] / d65[1], d65[2] / d65[1]);
    let row = |k: usize, n: f64| [m[k][0] - n * m[1][0], m[k][2] - n * m[1][2]];
    [row(0, xn), row(2, zn)]
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
    // The luma of C7's highlight range (`ym_color::tone::luma`).
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
    // The cine rendering (C12, `ym_color::display::cine`): its constants from the generated Rust table, the numeric
    // settings and the values the equations derive from them, and the six hue-harmonic arrays as two vec3 each.
    for (n, v) in cine::kernel_scalars() {
        let _ = writeln!(s, "const CINE_{n}: f32 = {};", f(v));
    }
    for (n, k) in cine::kernel_harmonics() {
        let _ = writeln!(
            s,
            "const CINE_{n}_A: vec3<f32> = vec3<f32>({}, {}, {});",
            f(k[0]),
            f(k[1]),
            f(k[2])
        );
        let _ = writeln!(
            s,
            "const CINE_{n}_B: vec3<f32> = vec3<f32>({}, {}, {});",
            f(k[3]),
            f(k[4]),
            f(k[5])
        );
    }
    // The rows of the cine luminance floor's chromatic distance, on the differences to the green channel.
    for (n, r) in ["DX", "DZ"].into_iter().zip(cine_floor_rows()) {
        let _ = writeln!(
            s,
            "const CINE_{n}: vec2<f32> = vec2<f32>({}, {});",
            f(r[0]),
            f(r[1])
        );
    }
    s.push_str(&rows("AWG4_TO_XYZ", &m.awg4_to_xyz));
    s.push_str(&rows("XYZ_TO_AWG4", &m.xyz_to_awg4));
    s.push_str(&rows("AWG4_TO_AWG3", &m.awg4_to_awg3));
    s.push_str(&rows("AWG3_TO_AWG4", &m.awg3_to_awg4));
    let _ = writeln!(s, "const T_DITHER: u32 = {}u;", table::DITHER);
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

#[cfg(test)]
mod tests {
    //! The luminance floor of the cine rendering (step 2 of C12, `cine.wgsl`) in a CPU model of the kernel's f32
    //! arithmetic, with the constants parsed from the generated shader text. Metal fuses `a * b + c` into a
    //! multiply-add and re-associates sums (`docs/17-gpu.md` section 13); the model rounds the sums of products
    //! three ways (`Fuse`) to show that the result of the new form does not depend on which one the compiler
    //! picks.

    use super::*;
    use crate::device::StorageMode;

    /// The numbers of `const NAME: type = value;` in the generated text (a scalar or the arguments of a vector).
    fn numbers(src: &str, name: &str) -> Vec<f32> {
        let head = format!("const {name}: ");
        let at = src
            .find(&head)
            .unwrap_or_else(|| panic!("{name} is not generated"));
        let value = src[at + head.len()..]
            .split_once('=')
            .and_then(|(_, v)| v.split_once(';'))
            .map(|(v, _)| v.trim())
            .expect("a constant with a value");
        let inner = value
            .split_once('(')
            .map_or(value, |(_, r)| r.trim_end_matches(')'));
        inner
            .split(',')
            .map(|t| t.trim().parse().expect("a float literal"))
            .collect()
    }

    /// How the compiler rounds a sum of products.
    #[derive(Clone, Copy, Debug)]
    enum Fuse {
        /// Every product and every sum rounded (IEEE, lavapipe).
        Strict,
        /// A chain of multiply-adds from the first term to the last.
        Forward,
        /// A chain of multiply-adds from the last term to the first.
        Backward,
    }

    const FUSES: [Fuse; 3] = [Fuse::Strict, Fuse::Forward, Fuse::Backward];

    fn dot<const N: usize>(r: [f32; N], v: [f32; N], fuse: Fuse) -> f32 {
        match fuse {
            Fuse::Strict => r.iter().zip(v).fold(0.0, |acc, (a, b)| acc + a * b),
            Fuse::Forward => (1..N).fold(r[0] * v[0], |acc, i| r[i].mul_add(v[i], acc)),
            Fuse::Backward => (0..N - 1)
                .rev()
                .fold(r[N - 1] * v[N - 1], |acc, i| r[i].mul_add(v[i], acc)),
        }
    }

    /// The f32 constants the kernel uses for step 2.
    struct Step2 {
        rows: [[f32; 3]; 3],
        xn: f32,
        zn: f32,
        dx: [f32; 2],
        dz: [f32; 2],
        rho: f32,
        wfrac: f32,
        eps_floor: f32,
        y_min: f32,
    }

    impl Step2 {
        fn new() -> Self {
            let src = constants();
            let one = |n: &str| numbers(&src, n)[0];
            let pair = |n: &str| <[f32; 2]>::try_from(numbers(&src, n)).expect("two numbers");
            let row = |n: &str| <[f32; 3]>::try_from(numbers(&src, n)).expect("three numbers");
            Self {
                rows: [
                    row("AWG4_TO_XYZ_0"),
                    row("AWG4_TO_XYZ_1"),
                    row("AWG4_TO_XYZ_2"),
                ],
                xn: one("CINE_XN"),
                zn: one("CINE_ZN"),
                dx: pair("CINE_DX"),
                dz: pair("CINE_DZ"),
                rho: one("CINE_RHO"),
                wfrac: one("CINE_WFRAC"),
                eps_floor: one("CINE_EPS_FLOOR"),
                y_min: one("CINE_Y_MIN"),
            }
        }

        /// The smooth maximum and the clamp of the kernel, from `Y` and the floor `yf`.
        fn ys(&self, y: f32, yf: f32) -> f32 {
            let w = self.wfrac * yf;
            let dy = y - yf;
            let root = (dy * dy + w * w).sqrt();
            let smooth_max = if dy < 0.0 {
                yf + 0.5 * w * w / (root - dy)
            } else {
                0.5 * (y + yf + root)
            };
            smooth_max.max(self.y_min)
        }

        fn yf(&self, dx: f32, dz: f32) -> f32 {
            self.rho * (dx * dx + dz * dz + self.eps_floor).sqrt()
        }

        /// The form before the fix: `dX = X - XN Y` from the triple's XYZ. `(dX, dZ, Ys)`.
        fn old(&self, x: [f32; 3], fuse: Fuse) -> (f32, f32, f32) {
            let [xx, y, zz] = self.rows.map(|r| dot(r, x, fuse));
            let (dx, dz) = match fuse {
                Fuse::Strict => (xx - self.xn * y, zz - self.zn * y),
                _ => ((-self.xn).mul_add(y, xx), (-self.zn).mul_add(y, zz)),
            };
            (dx, dz, self.ys(y, self.yf(dx, dz)))
        }

        /// The kernel's form: the rows `DX`, `DZ` on the differences to the green channel.
        fn new_form(&self, x: [f32; 3], fuse: Fuse) -> (f32, f32, f32) {
            let d = [x[0] - x[1], x[2] - x[1]];
            let (dx, dz) = (dot(self.dx, d, fuse), dot(self.dz, d, fuse));
            let y = dot(self.rows[1], x, fuse);
            (dx, dz, self.ys(y, self.yf(dx, dz)))
        }
    }

    const MAGNITUDES: [f32; 4] = [1e3, 1e4, 1e5, 1e6];

    #[test]
    fn the_floor_rows_sum_to_zero_and_are_the_chromatic_distance() {
        let m = matrices().awg4_to_xyz;
        let d65 = xy_to_xyz(D65_XY);
        let (xn, zn) = (d65[0] / d65[1], d65[2] / d65[1]);
        let scalars = cine::kernel_scalars();
        let scalar = |n: &str| scalars.iter().find(|(k, _)| *k == n).expect("a scalar").1;
        assert_eq!(scalar("XN"), xn);
        assert_eq!(scalar("ZN"), zn);
        // AWG4's white is D65: the full rows (with the green entry) sum to 0, so dropping the green entry costs
        // at most 2e-16 of the green channel, under the f64 reference's own rounding at the scene limit.
        let full = |k: usize, n: f64| [0, 1, 2].map(|j| m[k][j] - n * m[1][j]);
        for (row, rows) in [(full(0, xn), 0), (full(2, zn), 1)] {
            let sum: f64 = row.iter().sum();
            assert!(sum.abs() < 1e-15, "row {rows} sums to {sum:e}");
            assert_eq!(cine_floor_rows()[rows], [row[0], row[2]]);
        }
        // dX and dZ of a colour, from the differences and from X - XN Y.
        for x in [[0.18, 0.2, 0.05], [-3.0, 0.4, 7.0], [1e5, -2e4, 3e3]] {
            let g = x[1];
            let rows = cine_floor_rows();
            let (xyz_x, xyz_y, xyz_z) = (
                (0..3).map(|j| m[0][j] * x[j]).sum::<f64>(),
                (0..3).map(|j| m[1][j] * x[j]).sum::<f64>(),
                (0..3).map(|j| m[2][j] * x[j]).sum::<f64>(),
            );
            let scale = x.iter().map(|v| v.abs()).fold(0.0, f64::max);
            let from_diff = |r: [f64; 2]| r[0] * (x[0] - g) + r[1] * (x[2] - g);
            assert!((from_diff(rows[0]) - (xyz_x - xn * xyz_y)).abs() < 1e-15 * scale);
            assert!((from_diff(rows[1]) - (xyz_z - zn * xyz_y)).abs() < 1e-15 * scale);
        }
        // The kernel text uses them.
        let src = shader_source(StorageMode::F32, KernelId::Color);
        assert!(
            src.contains("let dx = dot(CINE_DX, d);") && src.contains("let dz = dot(CINE_DZ, d);")
        );
        assert!(!src.contains("CINE_XN * xyz.y"));
    }

    #[test]
    fn a_neutral_gives_exactly_zero_in_every_rounding_of_step_2() {
        let k = Step2::new();
        for sign in [-1.0f32, 1.0] {
            for mag in MAGNITUDES {
                let v = sign * mag;
                for fuse in FUSES {
                    let (dx, dz, ys) = k.new_form([v, v, v], fuse);
                    assert!(
                        dx == 0.0 && dz == 0.0,
                        "{v:e} {fuse:?}: dX {dx:e}, dZ {dz:e}"
                    );
                    // A negative neutral is at the bottom of the tone curve, a positive one keeps its luminance.
                    let y = dot(k.rows[1], [v; 3], fuse);
                    assert_eq!(ys, if v < 0.0 { k.y_min } else { y }, "{v:e} {fuse:?}");
                }
            }
        }
    }

    #[test]
    fn the_old_form_left_rounding_noise_that_decided_the_luminance() {
        // The model of the form before the fix: for a negative neutral Ys is the floor yf = rho * noise, which
        // is above Y_MIN by orders of magnitude as soon as the products are fused or the sums ordered
        // differently. (The strict model happens to cancel at -1e3 to -1e5 and not at -1e6; lavapipe's own sums
        // are not the model's, and it passed the gate at the fixture's -1e5.) This test keeps the model
        // honest: it must show the failure.
        let k = Step2::new();
        for mag in MAGNITUDES {
            let v = -mag;
            for fuse in FUSES {
                let (dx, dz, ys) = k.old([v, v, v], fuse);
                eprintln!(
                    "old form, neutral {v:+.0e} {fuse:?}: dX {dx:+.2e} dZ {dz:+.2e} Ys {ys:.2e} (Y_MIN {:.0e})",
                    k.y_min
                );
                if !matches!(fuse, Fuse::Strict) {
                    assert!(ys > 100.0 * k.y_min, "{v:e} {fuse:?}: Ys {ys:e}");
                }
            }
        }
        assert!(k.old([-1e6; 3], Fuse::Strict).2 > 1e-3);
        // At -1e5 the noise puts Ys at 2e-4 to 5e-4, the luminance behind the 1.6e-8 of the Mac failure
        // (the tone curve there is about Ys^4.3).
        assert!(k.old([-1e5; 3], Fuse::Forward).2 > 1e-4);
    }

    #[test]
    fn a_near_neutral_has_dx_and_dz_with_an_error_relative_to_its_own_differences() {
        let k = Step2::new();
        let m = matrices().awg4_to_xyz;
        let d65 = xy_to_xyz(D65_XY);
        let (xn, zn) = (d65[0] / d65[1], d65[2] / d65[1]);
        let mut worst = 0.0f64;
        for sign in [-1.0f32, 1.0] {
            for mag in MAGNITUDES {
                let v = sign * mag;
                for e in [1e-2f32, 1e-3, 1e-5, 1e-7] {
                    let triples = [
                        [v, v * (1.0 + e), v],
                        [v, v, v * (1.0 - e)],
                        [v * (1.0 + e), v, v * (1.0 - 2.0 * e)],
                        [v * (1.0 - e), v * (1.0 + e), v * (1.0 + 3.0 * e)],
                    ];
                    for x in triples {
                        let xd = x.map(f64::from);
                        let xyz = m.map(|r| (0..3).map(|j| r[j] * xd[j]).sum::<f64>());
                        let want = [xyz[0] - xn * xyz[1], xyz[2] - zn * xyz[1]];
                        // The size of the two terms the kernel adds: the error is relative to them.
                        let d = [xd[0] - xd[1], xd[2] - xd[1]];
                        let rows = cine_floor_rows();
                        let scale = [0, 1]
                            .map(|i| rows[i][0].abs() * d[0].abs() + rows[i][1].abs() * d[1].abs());
                        for fuse in FUSES {
                            let (dx, dz, _) = k.new_form(x, fuse);
                            for (got, want, scale) in
                                [(dx, want[0], scale[0]), (dz, want[1], scale[1])]
                            {
                                let err = (f64::from(got) - want).abs();
                                worst = worst.max(err / scale.max(f64::MIN_POSITIVE));
                                assert!(
                                    err <= 5e-7 * scale + 1e-15 * mag as f64,
                                    "{x:?} {fuse:?}: {got:e} against {want:e} (terms {scale:e})"
                                );
                            }
                        }
                    }
                }
            }
        }
        eprintln!("near-neutrals: worst error {worst:.2e} of the size of the terms of dX and dZ");
    }
}
