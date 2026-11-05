//! The whole per-pixel colour chain C1 to C13 as one compute pass (`docs/02-colour-pipeline.md`).
//!
//! [`PreparedChain::new`] turns an immutable `ym_color::display::ColorChain` into the GPU form: the
//! per-frame uniform block (fused matrix, C7 (the highlight range and the toe), C10 and C12 parameters
//! (the cine rendering's matrix from XYZ to the output), the flags and the pass, 560 bytes), the **node
//! records** (the plan's steps with their nodes, below) and three **persistent tables** that
//! [`ChainTables`] keeps on the device and refills only when their content changes:
//!
//! * the 64 x 64 dither thresholds, once per device;
//! * the curve tables, four 1024-entry tables (16 KB) for every node that has a curve, once per distinct list of
//!   curve sets;
//! * the look cube (f32, never rounded to f16, never hardware-filtered), in an LRU keyed by the content
//!   of the LUT (a 65-point cube is 4.4 MB, a 129-point one 34 MB).
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
//! `Lut` as a shortcut, valid while the cache holds the `Arc` the `Lut` lives in).
//! [`release_tables`] drops the whole set (the pool ledger counts it as resident until
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
//! carries the mask into its own fourth plane, an RGBA8 output has the background colour. A source with
//! **six planes** is the sampler's output with the photo-frame position planes
//! ([`encode_sample_at`](crate::kernels::sampler::encode_sample_at)): the fifth and sixth hold each pixel's position in the
//! photo, which the mask shapes of the nodes are placed in, and the chain needs the render's
//! [`PhotoFrame`] ([`PreparedChain::with_photo_frame`]); the destination of a stage has the first four.
//!
//! # The plan loop and the node records
//!
//! The nodes of `ColorChain`'s plan (`docs/23-nodes.md` section 5.4) run in the kernel: each step is a
//! **record** of the storage buffer at binding 6 (`NodeRecU`, 1200 bytes), in plan order. A record holds
//! the step's flags (bypassed or at its identity, and which of the trims, C9, C9b and C11 run), where the
//! step reads its picture and key and writes its own (a register, a key slot or a plane of the pass
//! buffers), the trims' gain, the grade's parameters, the colour warp's pins (what
//! `ym_color::warp::Warp::new` prepares: the CPU's own clamping, pivot code, number of flow steps, and the
//! rule that still pins, and locks without a moving pin, are left out; three `vec4` rows a moving pin and
//! one a lock, 1040 bytes for the 16 pins of `WARP_MAX_PINS`) and the first of the node's four curve
//! tables. **Main is a record like the others**: the default graph is a plan of one step, and the uniform
//! block holds none of the node fields. The flags are integers decided on the host in f64 (`is_identity`
//! of the CPU), so no comparison of a computed float decides whether a step runs (`docs/17` K8), and a
//! chain whose controls are all at their defaults runs the code of a chain that never had a grade.
//!
//! The records are written beside the uniform block for every dispatch (like it, they change on every
//! frame of a drag: a pin moves, a slider moves), so preparing a chain after a grade edit uploads
//! nothing. **The curves** are the exception: they are a persistent table with one set of four tables per
//! node that has a curve (identity sets are not stored), found by the list of the sets.
//!
//! **Passes.** A plan of more than one pass ([`ym_color::plan::Plan::passes`]: more than eight pictures or
//! sixteen keys alive at once) is one dispatch per pass. The values that cross from one pass to the next
//! are f32 planes of a scratch buffer (binding 8, `pbuf`: a picture is three planes of the window, a key
//! one), written by the pass that makes them and read by the later ones; the last pass writes the output.
//! A window whose planes would pass the design binding ([`crate::tiles::DESIGN_BINDING_BYTES`]) is cut
//! into bands of rows, which is exact because the chain is pointwise. A plan of one pass binds a stand-in.
//!
//! **Film nodes (F-08b, `docs/24` section 3.9c).** A Film node is a step whose picture the render made before the colour
//! pass (`kernels::film_stack`): its record has `N_FILM`, its Strength in `gain.x`, and when it does something
//! (`N_FILM_LIVE`) the first plane of its picture in `ctl.w`. The pictures are the **first planes of the pass buffers**
//! (three for each live Film node in plan order, the buffers that cross between passes after them), copied in from the
//! Film images by [`PreparedChain::with_film`] for each band of the window; a picture that does not cover the band is
//! zero there. The step computes its mask as a corrector does and blends `out = in + m (film - in)` with `m` the mask
//! times Strength, the node's key output; no new binding.
//!
//! **Stops.** [`PreparedChain::with_node_stop`] ends the plan after a node (the Output then runs on that
//! node's picture: a thumbnail) or at a node's mask `m` (grey code values: the mask view); kernel stops 11
//! and 12, the node's record in `stopn`.
//!
//! # Masks (N-10, `docs/23` section 5.4)
//!
//! The binding that holds the records is raw `vec4` rows (the kernel reads a record's fields with `nu` and
//! `nf` at the offsets `R_*`, written from `NodeRecU`), so that the **mask rows** can follow the records in
//! the same buffer: `MaskOpU`, 64 bytes a row, every row of every node in plan order, a record naming its
//! first row and their number (`wire.z`, `wire.w`). A row is the kind, the flags (subtract, invert), the
//! opacity and the eight numbers [`ym_color::mask::MaskOp`] derives in `f64`; the kernel's loop over a node's
//! rows is [`ym_color::mask::PreparedMasks::mask`] line by line, with no cap on the number of rows. A node
//! blends (`N_BLEND`) when it has rows, an Invert or a key input: a node with none keeps its own bits, as
//! before.

use std::any::Any;
use std::cell::RefCell;
use std::mem::{offset_of, size_of};
use std::sync::{Arc, Weak};

use bytemuck::{Pod, Zeroable};
use ym_color::consts::Vec3;
use ym_color::consts::linalg::{self, Mat3};
use ym_color::curve::{CurveSet, TABLE_SIZE};
use ym_color::display::cine::CineRendering;
use ym_color::display::{BeforePoint, ChainParams, ColorChain, OutputTarget};
use ym_color::encode::{OutputSpace, dither_threshold};
use ym_color::gamut::matrices;
use ym_color::grade::{
    BALANCE_CENTER_STOPS, GradeParams, ZONE_CENTER_STOPS, contrast_slope, logc4_at_stops,
    saturation_factor,
};
use ym_color::look::{LookInput, intensity_fraction};
use ym_color::lut::Lut;
use ym_color::mask::{MaskOp, OpKind, PhotoFrame};
use ym_color::plan::{BufferKind, KeyRef, MAIN_NODE, PictureRef, PlanResult};
use ym_color::tone::ToneParams;
use ym_color::warp::{WARP_MAX_PINS, Warp};

use super::copy::encode_copy_into;
use super::film::FilmImage;
use super::film_stack::FilmPictures;
use super::mask_raster::RasterPlanes;
use super::shader::{KernelId, table};
use super::util::{OutKind, PixelData, STORAGE_IN, STORAGE_OUT, even, image_bytes, sample_bytes};
use super::util::{storage_buffer, unpack};
use crate::cpu::RgbImage;
use crate::cpu::tiling::Rect;
use crate::device::{GpuContext, GpuError, StorageMode};
use crate::image::{GpuImage, IMAGE_USAGE};
use crate::pool::PooledBuffer;
use crate::queue::CancelToken;
use crate::stream::{Counters, PassList, Pending, ReadId, upload_bytes};
use crate::tiles::{DESIGN_BINDING_BYTES, Tile, TilePlan};

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
    /// After the look (log-to-log looks), which acts after the curve: the DNG stop point, in LogC4.
    C10,
    /// After the curves, in front of the look (`StageTrace::c11_logc4`).
    C11,
    /// C10 decoded back to linear AWG4 (what the DNG writer takes when C12 is our cine rendering).
    DngLinearAwg4,
    /// Display-linear RGB in the output's primaries (C12).
    C12,
    /// The encoded picture in [0, 1], not quantised (C13).
    C13,
    /// The Before view at the chain's `BeforePoint`: the camera default rendering encoded for the output
    /// ([`BeforePoint::CameraDefault`]), or the LogC4 signal at C8 or C9 clamped to [0, 1].
    Before,
    /// The guide of Refine edges: the Input's luma code (`encode(Y(lin))` of the linear values C7 made, held to
    /// [0, 1]) in all three planes, with no node in it (`ColorChain::guide_pixel`).
    Guide,
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
            Stop::Guide => 13,
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

/// Rows (`vec4<f32>`) of one moving pin of the colour warp in [`NodeRecU`]: [s.x, s.y, delta.x,
/// delta.y], [1 / (2 rho^2), shadows, highlights, pivot code], [exposure in stops, saturation, 0, 0].
pub(crate) const WARP_ROWS_PER_MOVE: usize = 3;
/// Rows of the moving pins in [`NodeRecU`] (`WARP_MOVE_ROWS` in `color.wgsl`): room for every pin.
pub(crate) const WARP_MOVE_ROWS: usize = WARP_ROWS_PER_MOVE * WARP_MAX_PINS;
/// Rows of the locked pins in [`NodeRecU`] (`WARP_LOCK_ROWS` in `color.wgsl`), one a lock: [l.x, l.y,
/// 1 / (2 rho^2), 0]. At most `WARP_MAX_PINS - 1` are used (locks count only beside a moving pin);
/// the sixteenth row keeps the arithmetic of the sizes plain.
pub(crate) const WARP_LOCK_ROWS: usize = WARP_MAX_PINS;

/// The step is not run: the node is bypassed, or every one of its steps is at its identity (`N_SKIP`).
pub(crate) const NODE_SKIP: u32 = 1;
/// The node's trims run (`N_TRIM`); the gain is in the record.
pub(crate) const NODE_TRIM: u32 = 2;
/// C9 runs (`N_GRADE`).
pub(crate) const NODE_GRADE: u32 = 4;
/// C9b runs (`N_WARP`).
pub(crate) const NODE_WARP: u32 = 8;
/// C11 runs (`N_CURVES`).
pub(crate) const NODE_CURVES: u32 = 16;
/// The node is Main (`N_MAIN`): the trace's stops C9, C9b and C11 are its values.
pub(crate) const NODE_MAIN: u32 = 32;
/// The node blends by its mask (`N_BLEND`): it has mask rows, an Invert or a key input. Without it `m` is 1
/// and the node's output is its own bits.
pub(crate) const NODE_BLEND: u32 = 64;
/// The node's Invert (`N_INVERT`): `m = 1 - m`.
pub(crate) const NODE_INVERT: u32 = 128;
/// A mask row reads the luma of the node's input (`N_LUMA`, a Brightness range).
pub(crate) const NODE_LUMA: u32 = 256;
/// The node is a Film node (`N_FILM`): its picture is not computed per pixel but read from the planes of the plan buffer
/// that hold the picture the Film passes made, and blended by the mask times Strength (`gain.x`). Set for every Film
/// step; [`NODE_FILM_LIVE`] says whether it has a picture.
pub(crate) const NODE_FILM: u32 = 512;
/// The planes of the plan buffer a live Film node takes: its LogC4 picture (three) and the exact display colour of each
/// of its pixels, the shadow (three, all 0 where there is none).
const FILM_PLANES: u32 = 6;

/// The Film node does something (`N_FILM_LIVE`: not bypassed, Strength above 0); `ctl.w` is the first plane of the plan
/// buffer that holds its picture.
pub(crate) const NODE_FILM_LIVE: u32 = 1024;
/// The mask row is subtracted (`MF_SUBTRACT`, in `MaskOpU::head.y`).
pub(crate) const MASK_SUBTRACT: u32 = 1;
/// The mask row is inverted (`MF_INVERT`).
pub(crate) const MASK_INVERT: u32 = 2;
/// `vec4` rows of a node record (`REC_ROWS`).
pub(crate) const REC_ROWS: u32 = (size_of::<NodeRecU>() / 16) as u32;
/// `vec4` rows of a mask row (`MASK_ROW_VEC4`).
pub(crate) const MASK_ROW_VEC4: u32 = (size_of::<MaskOpU>() / 16) as u32;

/// The row (`vec4`) of each field of a node record: `R_*` in `color.wgsl`, written from [`NodeRecU`].
pub(crate) fn record_rows() -> [(&'static str, u32); 12] {
    let row = |bytes: usize| (bytes / 16) as u32;
    [
        ("R_CTL", row(offset_of!(NodeRecU, ctl))),
        ("R_WIRE", row(offset_of!(NodeRecU, wire))),
        ("R_OUTS", row(offset_of!(NodeRecU, outs))),
        ("R_GAIN", row(offset_of!(NodeRecU, gain))),
        ("R_GRADE0", row(offset_of!(NodeRecU, grade0))),
        ("R_ZONE_AMT", row(offset_of!(NodeRecU, zone_amt))),
        ("R_ZONE_CTR", row(offset_of!(NodeRecU, zone_ctr))),
        ("R_BAL0", row(offset_of!(NodeRecU, bal0))),
        ("R_BAL1", row(offset_of!(NodeRecU, bal1))),
        ("R_BAL_CTR", row(offset_of!(NodeRecU, bal_ctr))),
        ("R_WARP_N", row(offset_of!(NodeRecU, warp_n))),
        ("R_WARP_MV", row(offset_of!(NodeRecU, warp_mv))),
    ]
}

/// The row of the locked pins of a record (`R_WARP_LK`).
pub(crate) const R_WARP_LK: u32 = (offset_of!(NodeRecU, warp_lk) / 16) as u32;
/// "Not written" in a record's outputs and "no key" in its inputs (`NODE_NONE`).
pub(crate) const NODE_NONE: u32 = u32::MAX;
/// A reference to a value is `kind << REF_SHIFT | index` (`REF_*` in `color.wgsl`).
pub(crate) const REF_SHIFT: u32 = 28;
/// The mask of the index of a reference.
pub(crate) const REF_INDEX: u32 = (1 << REF_SHIFT) - 1;
/// A reference to the Input's picture (index 0).
pub(crate) const REF_INPUT: u32 = 0;
/// A picture register of the pass.
pub(crate) const REF_REGISTER: u32 = 1;
/// A picture in the planes of `pbuf` (the index is its first plane).
pub(crate) const REF_PICTURE: u32 = 2;
/// A key slot of the pass.
pub(crate) const REF_SLOT: u32 = 3;
/// A key in a plane of `pbuf`.
pub(crate) const REF_KEY: u32 = 4;

/// The most steps a plan can have: one for each corrector, and a graph has 64 at most
/// (`docs/23` section 3.5).
const MAX_STEPS: usize = 64;

/// The per-frame uniform block of `color.wgsl` (`ChainU`), 608 bytes: window and mode, the chain's
/// parameters (the fused matrix, C7, the look), the cine rendering's matrix from XYZ to the output's
/// primaries (`cine_m0` to `cine_m2`), the Before view's camera default (`cam0` to `cam2`: the 3x3
/// rows, `camt`: the tone curve's `n`, `s`, `k` and white level) and the pass (`plan`, `stopn`: module
/// notes), the mask rows' place (`mrow`) and the photo frame (`photo`). The nodes' parameters are not in it
/// ([`NodeRecU`]). Field for field the WGSL struct (a test pins the offsets).
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
    fl: [f32; 4],
    cdl_slope: [f32; 4],
    cdl_offset: [f32; 4],
    cdl_power: [f32; 4],
    look_lut: LutDescU,
    cine_m0: [f32; 4],
    cine_m1: [f32; 4],
    cine_m2: [f32; 4],
    look_m0: [f32; 4],
    look_m1: [f32; 4],
    look_m2: [f32; 4],
    cam0: [f32; 4],
    cam1: [f32; 4],
    cam2: [f32; 4],
    camt: [f32; 4],
    gate: [f32; 4],
    /// x: the pass's first record, y: its steps, z: bit 0 set for the plan's last pass, w: the plan's
    /// result (a reference).
    plan: [u32; 4],
    /// x: the record at which kernel stops 11 and 12 end ([`NODE_NONE`]: none).
    stopn: [u32; 4],
    /// x: the first mask row in the records' buffer, in `vec4` rows (after the records).
    mrow: [u32; 4],
    /// The photo frame ([`PhotoFrame`]): the photo's width and height over its long edge, and the narrowest
    /// edge ramp ([`PhotoFrame::ramp`]) in units of the long edge.
    photo: [f32; 4],
    /// The raster planes (binding 7): x: words in a plane, y: the first row of this dispatch in the window, z: a
    /// plane's row stride in elements, w: the number of planes (0: none are bound).
    rst: [u32; 4],
}

/// One step of the plan with its node, as the storage buffer at binding 6 holds it (`NodeRec` in
/// `color.wgsl`, 1200 bytes; a test pins the offsets). The module notes describe the fields.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct NodeRecU {
    /// x: the `NODE_*` bits, y: the grade's bits (1 contrast, 2 saturation), z: the curve bits (1 luma,
    /// 2 r, 4 g, 8 b), w: the first of the node's four curve tables.
    ctl: [u32; 4],
    /// x: the picture the step reads, y: the key it reads ([`NODE_NONE`]: none), z: the number of its
    /// mask rows, w: the index of the first of them among all the mask rows.
    wire: [u32; 4],
    /// x: the register the step writes its picture to, y: the first plane of `pbuf` it writes it to, z:
    /// the key slot it writes its mask to, w: the plane of `pbuf` it writes it to ([`NODE_NONE`]: not
    /// written).
    outs: [u32; 4],
    /// The trims' gain in linear AWG4.
    gain: [f32; 4],
    grade0: [f32; 4],
    zone_amt: [f32; 4],
    zone_ctr: [f32; 4],
    bal0: [f32; 4],
    bal1: [f32; 4],
    bal_ctr: [f32; 4],
    /// C9b: x the moving pins (0: the stage is skipped), y the locked pins, z the flow's steps.
    warp_n: [u32; 4],
    warp_mv: [[f32; 4]; WARP_MOVE_ROWS],
    warp_lk: [[f32; 4]; WARP_LOCK_ROWS],
}

impl Default for NodeRecU {
    /// All zero (`Default` is not derived: the warp's arrays are longer than 32).
    fn default() -> Self {
        Self::zeroed()
    }
}

/// One mask row as the storage buffer after the records holds it (`docs/algorithms/06` section 3,
/// [`ym_color::mask::MaskOp`] narrowed to `f32`), 64 bytes: four `vec4` rows.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Default, Debug, PartialEq)]
pub(crate) struct MaskOpU {
    /// x: the kind (`MK_*`, [`ym_color::mask::OpKind::code`]), y: the `MASK_*` flags, z: the raster plane
    /// ([`NODE_NONE`]: none; for N-13), w: the opacity as `f32` bits.
    head: [u32; 4],
    /// The first four numbers of [`MaskOp::params`].
    p0: [f32; 4],
    /// The last four.
    p1: [f32; 4],
    /// Reserved (the raster kinds of N-13 to N-19).
    spare: [f32; 4],
}

impl MaskOpU {
    /// The row for a prepared mask op.
    fn of(op: &MaskOp) -> Self {
        let p = op.params.map(|v| v as f32);
        Self {
            head: [
                op.kind.code(),
                if op.subtract { MASK_SUBTRACT } else { 0 }
                    | if op.invert { MASK_INVERT } else { 0 },
                // A raster row names its plane; one with no plane (a kind this machine cannot make) is NODE_NONE.
                if op.kind == OpKind::Raster && op.params[0] < f64::from(u32::MAX) {
                    op.params[0] as u32
                } else {
                    NODE_NONE
                },
                (op.opacity as f32).to_bits(),
            ],
            p0: [p[0], p[1], p[2], p[3]],
            p1: [p[4], p[5], p[6], p[7]],
            spare: [0.0; 4],
        }
    }
}

impl Default for ChainU {
    /// All zero.
    fn default() -> Self {
        Self::zeroed()
    }
}

fn f4(v: [f64; 3], w: f64) -> [f32; 4] {
    [v[0] as f32, v[1] as f32, v[2] as f32, w as f32]
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

/// Matrix from linear Rec.709 primaries (a display-baked look's LUT output) to the output space,
/// `None` when the output is Rec.709.
fn rec709_to_output(to: OutputSpace) -> Option<Mat3> {
    let m = matrices();
    if to == OutputSpace::Srgb {
        return None;
    }
    let xyz_to = match to {
        OutputSpace::Srgb => m.xyz_to_rec709,
        OutputSpace::DisplayP3 => m.xyz_to_p3,
        OutputSpace::Rec2020 => m.xyz_to_rec2020,
    };
    Some(linalg::mul(&xyz_to, &m.rec709_to_xyz))
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
/// `HIGHLIGHT_OFF_STOPS` (1e30, a finite f32: K8) and an expansion of 0. The achromatic gate of the range (`ToneParams::gate`) goes into `gate.xy`: its
/// first edge and the reciprocal of its span.
fn tone_uniform(u: &mut ChainU, tone: &ToneParams) {
    let range = tone.range();
    u.tone = [
        range.knee_stops as f32,
        range.clip_stops as f32,
        tone.toe_tau() as f32,
        range.expansion_stops as f32,
    ];
    u.fl[2] = range.cap_stops as f32;
    u.gate[0] = tone.gate.from() as f32;
    u.gate[1] = tone.gate.inv_span() as f32;
}

/// The grade parameters as the kernel wants them (the same preparation as `Grade::new`): the
/// contrast curve's slope at grey `k` in `grade0.x`, the brightness in `grade0.y` and the saturation
/// factor in `grade0.z` (below 1 the blend toward grey, above 1 the boost `F`; `saturation_factor`).
/// A control at 0 clears its bit in `ctl.y`, and the kernel skips it. `grade0.w` is not used.
fn grade_record(r: &mut NodeRecU, g: &GradeParams) {
    let slope = contrast_slope(g.contrast);
    let saturation = saturation_factor(g.saturation);
    if slope != 1.0 {
        r.ctl[1] |= 1;
    }
    if saturation != 1.0 {
        r.ctl[1] |= 2;
    }
    r.grade0 = [
        slope as f32,
        (clamp_control(g.brightness) / 100.0) as f32,
        saturation as f32,
        0.0,
    ];
    let amounts = [g.shadows, g.highlights, g.black, g.white].map(|x| clamp_control(x) / 100.0);
    r.zone_amt = [
        amounts[0] as f32,
        amounts[1] as f32,
        amounts[2] as f32,
        amounts[3] as f32,
    ];
    let centers = ZONE_CENTER_STOPS.map(logc4_at_stops);
    r.zone_ctr = [
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
    r.bal0 = [b[0].a as f32, b[0].b as f32, b[1].a as f32, b[1].b as f32];
    r.bal1 = [b[2].a as f32, b[2].b as f32, 0.0, 0.0];
    let bc = BALANCE_CENTER_STOPS.map(logc4_at_stops);
    r.bal_ctr = [bc[0] as f32, bc[1] as f32, bc[2] as f32, 0.0];
}

/// The colour warp's pins as the kernel wants them: what `Warp::new` prepares, packed into the rows of
/// the record (module notes). Nothing is computed here but the casts to f32. A warp that moves
/// nothing writes nothing (all zero).
fn warp_record(r: &mut NodeRecU, warp: &Warp) {
    let (moves, locks) = (warp.moves(), warp.locks());
    // `Warp::new` keeps at most `WARP_MAX_PINS` pins in all, so both fit; the zips and the `min`s keep
    // the rows and the counts inside the arrays even if that ever changed.
    for (rows, m) in r
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
        rows[2] = [m.exposure_ev as f32, m.saturation as f32, 0.0, 0.0];
    }
    for (row, l) in r.warp_lk.iter_mut().zip(locks.iter()) {
        *row = [
            l.at[0] as f32,
            l.at[1] as f32,
            l.inv_two_sigma_sq as f32,
            0.0,
        ];
    }
    r.warp_n = [
        moves.len().min(WARP_MAX_PINS) as u32,
        locks.len().min(WARP_LOCK_ROWS) as u32,
        warp.steps(),
        0,
    ];
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
    /// The curve sets of the nodes that have a curve, in plan order (a node with identity curves has none).
    sets: Vec<CurveSet>,
    /// For each set, bit `i` set when its table `i` (luma, R, G, B) is not the identity.
    flags: Vec<u32>,
    /// Four tables of [`TABLE_SIZE`] entries for each set, in order (a stand-in of four floats when there are
    /// none: nothing reads it).
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
    /// Bound as the pass buffers of a plan of one pass (never read or written).
    stand_in: Option<Arc<PooledBuffer>>,
    curves: Vec<CurveEntry>,
    luts: Vec<LutEntry>,
}

impl std::fmt::Debug for ChainTables {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ChainTables")
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

    /// The stand-in bound where a chain has no look (never read).
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

    /// The stand-in bound as the pass buffers where a plan has one pass.
    fn stand_in(&mut self, ctx: &GpuContext) -> Result<Arc<PooledBuffer>, GpuError> {
        if let Some(b) = &self.stand_in {
            return Ok(Arc::clone(b));
        }
        let buf = Arc::new(storage_buffer(
            ctx,
            "plan stand-in",
            bytemuck::cast_slice(&[0.0f32; 4]),
        )?);
        self.stand_in = Some(Arc::clone(&buf));
        Ok(buf)
    }

    /// The curve tables of `sets` (the sets of the nodes that have a curve, in plan order: four tables
    /// each) and, for each set, the bits of its tables that are not the identity.
    fn curves(
        &mut self,
        ctx: &GpuContext,
        sets: &[CurveSet],
    ) -> Result<(Arc<PooledBuffer>, Vec<u32>), GpuError> {
        if let Some(i) = self.curves.iter().position(|e| e.sets == sets) {
            let e = self.curves.remove(i);
            let out = (Arc::clone(&e.buf), e.flags.clone());
            self.curves.push(e);
            return Ok(out);
        }
        let mut data = vec![0.0f32; 4 * TABLE_SIZE * sets.len().max(1)];
        let mut flags = Vec::with_capacity(sets.len());
        for (k, set) in sets.iter().enumerate() {
            let baked = set.bake();
            let mut bits = 0u32;
            for (i, t) in [&baked.luma, &baked.r, &baked.g, &baked.b]
                .iter()
                .enumerate()
            {
                let at = (4 * k + i) * TABLE_SIZE;
                data[at..at + TABLE_SIZE].copy_from_slice(&t.to_f32());
                if !t.is_identity() {
                    bits |= 1 << i;
                }
            }
            flags.push(bits);
        }
        if sets.is_empty() {
            data.truncate(4);
        }
        let buf = Arc::new(storage_buffer(
            ctx,
            "chain curves",
            bytemuck::cast_slice(&data),
        )?);
        self.curves.push(CurveEntry {
            sets: sets.to_vec(),
            flags: flags.clone(),
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

    /// Prepares `chain` against these tables: builds the per-frame block and the node records and finds
    /// (or uploads) the tables it needs.
    ///
    /// # Errors
    ///
    /// Allocation errors from the pool ([`GpuError::OutOfBudget`], [`GpuError::BufferTooLarge`]);
    /// [`GpuError::InvalidInput`] for a plan the kernel cannot hold (more than 64 steps, or a step whose
    /// node the chain has no parameters for).
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
        u.stopn = [NODE_NONE; 4];

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
                if let Some(m) = rec709_to_output(space) {
                    u.look_m0 = f4(m[0], 0.0);
                    u.look_m1 = f4(m[1], 0.0);
                    u.look_m2 = f4(m[2], 0.0);
                    u.dp[3] = 1;
                }
            }
        }

        // The nodes (the trims, C9, C9b and C11 of every step of the plan) and their curve tables.
        let plan = NodePlan::new(self, ctx, chain)?;

        // C12: the cine rendering needs the matrix from XYZ to the output's own primaries and nothing else
        // (its constants are written into the shader).
        let xyz_to_output = CineRendering::new(params.output).xyz_to_output();
        u.cine_m0 = f4(xyz_to_output[0], 0.0);
        u.cine_m1 = f4(xyz_to_output[1], 0.0);
        u.cine_m2 = f4(xyz_to_output[2], 0.0);
        let tables = self.dither(ctx)?;
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
        if let Some(m) = rec709_to_output(space) {
            u.look_m0 = f4(m[0], 0.0);
            u.look_m1 = f4(m[1], 0.0);
            u.look_m2 = f4(m[2], 0.0);
            u.dp[3] = 1;
        }

        let stand_in = self.stand_in(ctx)?;
        let no_rasters = self.empty(ctx)?;
        u.mrow[0] = plan.rows_at;
        Ok(PreparedChain {
            base: u,
            tables,
            curves: plan.curves,
            look_luts,
            stand_in,
            no_rasters,
            records: plan.records,
            passes: plan.passes,
            steps: plan.steps,
            result: plan.result,
            planes: plan.planes,
            films: plan.films,
            film_pictures: None,
            node_stop: None,
            photo: None,
            rasters: None,
            buffer_limit: DESIGN_BINDING_BYTES,
            output: params.output,
            look: look_kind,
            before: chain.before_point(),
            dither: chain.dither(),
        })
    }
}

/// A pass of the plan in the node records: its first record and its number of steps.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PassSpan {
    first: u32,
    count: u32,
}

/// Where a node is in the plan: its id, its record and its pass.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct StepAt {
    node: u32,
    record: u32,
    pass: usize,
}

/// The plan of a `ColorChain` in the form the kernel reads it: the records, the passes, the result and
/// the curve tables.
struct NodePlan {
    /// The records and then the mask rows, as bytes.
    records: Arc<Vec<u8>>,
    /// Where the mask rows start in `records`, in `vec4` rows.
    rows_at: u32,
    passes: Arc<Vec<PassSpan>>,
    steps: Arc<Vec<StepAt>>,
    result: u32,
    /// The planes of `pbuf` a pixel needs: six for each live Film node (first), then the sum over the plan's buffers
    /// (three for a picture, one for a key).
    planes: u32,
    /// The live Film nodes of the plan, in plan order: the picture of the `k`-th is at planes `6k..6k+3` of `pbuf`, its
    /// exact display colours (the shadow) at `6k+3..6k+6`.
    films: Arc<Vec<u32>>,
    curves: Arc<PooledBuffer>,
}

impl NodePlan {
    fn new(
        tables: &mut ChainTables,
        ctx: &GpuContext,
        chain: &ColorChain,
    ) -> Result<Self, GpuError> {
        let params = chain.params();
        let plan = &params.nodes.plan;
        if plan.step_count() > MAX_STEPS {
            return Err(GpuError::InvalidInput(format!(
                "the plan has {} steps, the colour kernel runs {MAX_STEPS} at most",
                plan.step_count()
            )));
        }
        // The Film nodes that do something keep their pictures in the first planes of the plan buffer (the render makes
        // them before the colour pass, `docs/24` section 3.9c); the buffers that cross from one pass to the next follow.
        let films: Vec<u32> = plan
            .steps()
            .filter(|s| {
                params
                    .films
                    .get(&s.node)
                    .is_some_and(|f| !f.bypass && f.strength > 0.0)
            })
            .map(|s| s.node)
            .collect();
        let mut plane_of = Vec::with_capacity(plan.buffers.len());
        let mut planes = FILM_PLANES * films.len() as u32;
        for kind in &plan.buffers {
            plane_of.push(planes);
            planes += match kind {
                BufferKind::Picture => 3,
                BufferKind::Key => 1,
            };
        }
        let picture_ref = |r: PictureRef| match r {
            PictureRef::Input => REF_INPUT << REF_SHIFT,
            PictureRef::Register(i) => (REF_REGISTER << REF_SHIFT) | u32::from(i),
            PictureRef::Buffer(b) => (REF_PICTURE << REF_SHIFT) | plane_of[usize::from(b)],
        };
        let key_ref = |r: KeyRef| match r {
            KeyRef::Slot(i) => (REF_SLOT << REF_SHIFT) | u32::from(i),
            KeyRef::Buffer(b) => (REF_KEY << REF_SHIFT) | plane_of[usize::from(b)],
        };
        // The parameters of every step's node.
        let mut nodes = Vec::with_capacity(plan.step_count());
        for step in plan.steps() {
            let n = params.node_params(step.node).ok_or_else(|| {
                GpuError::InvalidInput(format!(
                    "the plan runs node {}, which the chain has no parameters for",
                    step.node
                ))
            })?;
            nodes.push((*step, n));
        }
        // The curve tables: four for each node that runs a curve, in plan order.
        let sets: Vec<CurveSet> = nodes
            .iter()
            .filter(|(_, n)| !n.bypass && !n.curves.is_identity())
            .map(|(_, n)| n.curves.clone())
            .collect();
        let (curves, curve_bits) = tables.curves(ctx, &sets)?;
        let mut next_set = 0usize;
        let mut records = Vec::with_capacity(nodes.len());
        let mut mask_rows: Vec<MaskOpU> = Vec::new();
        for (step, n) in &nodes {
            let mut r = NodeRecU::default();
            let mut bits = 0u32;
            if step.node == MAIN_NODE {
                bits |= NODE_MAIN;
            }
            // The same four tests as the CPU's `PreparedNode::new`, in f64, once.
            let gain = n.trim.gain();
            let warp = Warp::new(&n.warp);
            let grade = !n.grade.is_identity();
            let curve = !n.curves.is_identity();
            if n.bypass || (gain.is_none() && !grade && warp.is_identity() && !curve) {
                bits |= NODE_SKIP;
            } else {
                if let Some(g) = gain {
                    bits |= NODE_TRIM;
                    r.gain = f4(g, 0.0);
                }
                if grade {
                    bits |= NODE_GRADE;
                    grade_record(&mut r, &n.grade);
                }
                if !warp.is_identity() {
                    bits |= NODE_WARP;
                    warp_record(&mut r, &warp);
                }
                if curve {
                    bits |= NODE_CURVES;
                    r.ctl[2] = curve_bits[next_set];
                    r.ctl[3] = 4 * next_set as u32;
                    next_set += 1;
                }
            }
            // A Film node: its picture is read from the plan buffer, its Strength is in the gain's first lane.
            if let Some(film) = params.films.get(&step.node) {
                bits |= NODE_FILM;
                r.gain = [film.strength as f32, 0.0, 0.0, 0.0];
                if let Some(k) = films.iter().position(|&n| n == step.node) {
                    bits |= NODE_FILM_LIVE;
                    r.ctl[3] = FILM_PLANES * k as u32;
                }
            }
            // The mask rows (`ym_color::mask`): the same rows, derived once, as the CPU reads. A node blends when
            // it has rows, an Invert or a key input.
            let masks = n.masks.prepare();
            if masks.blends(step.key.is_some()) {
                bits |= NODE_BLEND;
            }
            if masks.invert() {
                bits |= NODE_INVERT;
            }
            if masks.needs_luma() {
                bits |= NODE_LUMA;
            }
            let first_row = mask_rows.len() as u32;
            mask_rows.extend(masks.ops().iter().map(MaskOpU::of));
            r.ctl[0] = bits;
            r.wire = [
                picture_ref(step.picture),
                step.key.map_or(NODE_NONE, key_ref),
                masks.ops().len() as u32,
                first_row,
            ];
            let (picture_out, key_out) = (step.picture_out, step.key_out);
            r.outs = [
                picture_out
                    .and_then(|o| o.register)
                    .map_or(NODE_NONE, u32::from),
                picture_out
                    .and_then(|o| o.buffer)
                    .map_or(NODE_NONE, |b| plane_of[usize::from(b)]),
                key_out.and_then(|o| o.slot).map_or(NODE_NONE, u32::from),
                key_out
                    .and_then(|o| o.buffer)
                    .map_or(NODE_NONE, |b| plane_of[usize::from(b)]),
            ];
            records.push(r);
        }
        let mut first = 0u32;
        let mut passes = Vec::with_capacity(plan.passes.len());
        let mut steps = Vec::with_capacity(records.len());
        for (p, pass) in plan.passes.iter().enumerate() {
            let count = pass.steps.len() as u32;
            passes.push(PassSpan { first, count });
            for (k, step) in pass.steps.iter().enumerate() {
                steps.push(StepAt {
                    node: step.node,
                    record: first + k as u32,
                    pass: p,
                });
            }
            first += count;
        }
        if passes.is_empty() {
            // The Output reads the Input itself: one pass of no steps.
            passes.push(PassSpan { first: 0, count: 0 });
        }
        let result = match plan.result {
            PlanResult::Picture(r) => picture_ref(r),
            PlanResult::Mask(r) => key_ref(r),
        };
        // A plan with no step still binds one record (the buffer is never empty).
        if records.is_empty() {
            records.push(NodeRecU::default());
        }
        // The records, then the mask rows, as one buffer of `vec4` rows.
        let rows_at = records.len() as u32 * REC_ROWS;
        let mut bytes = Vec::with_capacity(
            records.len() * size_of::<NodeRecU>() + mask_rows.len() * size_of::<MaskOpU>(),
        );
        bytes.extend_from_slice(bytemuck::cast_slice(&records));
        bytes.extend_from_slice(bytemuck::cast_slice(&mask_rows));
        Ok(Self {
            records: Arc::new(bytes),
            rows_at,
            passes: Arc::new(passes),
            steps: Arc::new(steps),
            result,
            planes,
            films: Arc::new(films),
            curves,
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

/// Drops the persistent tables of `ctx`'s device (the dither thresholds, the curves and the cached LUTs): their buffers go back to the pool and are created again by the next
/// chain. Call from the device thread; a chain prepared earlier keeps working (it holds its own
/// references). The pool ledger counts the tables as in use until then, which a test about "the
/// ledger returns to its baseline" must account for.
pub fn release_tables(ctx: &GpuContext) {
    super::film_stack::release(ctx);
    TABLES.with(|cell| {
        let id = Arc::as_ptr(ctx.counters_shared());
        cell.borrow_mut()
            .retain(|(owner, _)| owner.as_ptr() != id && owner.strong_count() > 0);
    });
    super::mask_raster::release_planes(ctx);
}

/// The dither thresholds buffer of the device (4096 f32, `y * 64 + x`), shared by the encode kernel
/// and the fused tail of output sharpening.
pub(crate) fn dither_table(ctx: &GpuContext) -> Result<Arc<PooledBuffer>, GpuError> {
    with_tables(ctx, |t| t.dither(ctx))
}

/// The WGSL of the colour kernel: the cine rendering, the chain, the C13 helpers, with the LUT
/// functions written for the look cube.
pub(super) fn body() -> String {
    let color = include_str!("../shaders/color.wgsl");
    let (before, rest) = color
        .split_once("//@BEGIN_LUT_FUNCS\n")
        .unwrap_or((color, ""));
    let (template, after) = rest.split_once("//@END_LUT_FUNCS\n").unwrap_or(("", ""));
    let mut s = String::with_capacity(color.len() * 2);
    s.push_str(include_str!("../shaders/precise.wgsl"));
    s.push_str("fn px_zero() -> u32 {\n    return CU.flags.x;\n}\n\n");
    s.push_str(include_str!("../shaders/cine.wgsl"));
    s.push('\n');
    s.push_str(include_str!("../shaders/c13.wgsl"));
    s.push('\n');
    s.push_str(before);
    s.push_str(&template.replace("@P@", "_look").replace("@L@", "look_luts"));
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
    /// Bound as the pass buffers of a plan with none.
    stand_in: Arc<PooledBuffer>,
    /// Bound read-only as the raster planes of a chain that was given none (a different buffer from `stand_in`,
    /// which a dispatch writes: one buffer cannot be both in one dispatch).
    no_rasters: Arc<PooledBuffer>,
    /// The plan's steps with their nodes, in plan order, then the mask rows: the storage buffer at binding 6.
    records: Arc<Vec<u8>>,
    /// The passes, as spans of the records (one pass at least).
    passes: Arc<Vec<PassSpan>>,
    /// Where each step's node is.
    steps: Arc<Vec<StepAt>>,
    /// The plan's result, as a reference.
    result: u32,
    /// The planes of the pass buffers a pixel needs.
    planes: u32,
    /// The live Film nodes of the plan, in plan order (their pictures are the first planes of the pass buffers).
    films: Arc<Vec<u32>>,
    /// The pictures the Film nodes made, and the absolute pixel of the window's (0, 0) in the photo's extent
    /// ([`PreparedChain::with_film`]).
    film_pictures: Option<(Arc<FilmPictures>, (i64, i64))>,
    /// Where kernel stops 11 and 12 end, when the chain was given one.
    node_stop: Option<(NodeStop, StepAt)>,
    /// The render's photo frame, for the mask shapes ([`PreparedChain::with_photo_frame`]).
    photo: Option<PhotoFrame>,
    /// The raster planes of the window the chain colours ([`PreparedChain::with_rasters`]); a chain without them
    /// binds a stand-in and its raster rows select nothing.
    rasters: Option<RasterPlanes>,
    /// The most bytes the pass buffers of one dispatch may have ([`DESIGN_BINDING_BYTES`]).
    buffer_limit: u64,
    output: OutputTarget,
    look: LookKind,
    before: BeforePoint,
    dither: bool,
}

/// A Film picture with the place of a window's first pixel in it.
type FilmPicture = (Arc<FilmImage>, i64, i64);

/// A stop inside the plan's nodes (`docs/23` section 5.4): [`PreparedChain::with_node_stop`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum NodeStop {
    /// The nodes end after this node and the Output runs on its picture (a node's thumbnail).
    Picture(u32),
    /// The picture is this node's mask `m` as grey code values, after the Output (the mask view; the node's
    /// real `m` since N-10, 1 for a node with no mask rows, no Invert and no key).
    Mask(u32),
}

impl NodeStop {
    fn node(self) -> u32 {
        match self {
            NodeStop::Picture(n) | NodeStop::Mask(n) => n,
        }
    }

    /// The kernel's stop code.
    fn code(self) -> u32 {
        match self {
            NodeStop::Picture(_) => 11,
            NodeStop::Mask(_) => 12,
        }
    }
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

/// What a dispatch needs beyond the window: the letterbox colour, whether the source has a mask plane and
/// whether it has the photo-frame position planes.
#[derive(Clone, Copy, Debug, Default)]
struct MaskPaint {
    has_mask: bool,
    has_positions: bool,
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

    /// Ends the plan's nodes at `stop` for the outputs that run the Output (the encoded picture,
    /// [`ColorOutput::Rgba8`], [`ColorOutput::U16`] and the stage C13): the Output then runs on the
    /// picture after the node ([`NodeStop::Picture`]; a node's thumbnail), or the picture is the node's
    /// mask `m` as grey code values ([`NodeStop::Mask`]; the mask view). The other stops are the
    /// chain's reading points and are not changed.
    ///
    /// # Errors
    ///
    /// [`GpuError::InvalidInput`] when the node is not a step of the plan (a dead node has no picture in
    /// this plan: compile the plan to end at it).
    pub fn with_node_stop(&self, stop: NodeStop) -> Result<Self, GpuError> {
        let Some(at) = self.steps.iter().find(|s| s.node == stop.node()) else {
            return Err(GpuError::InvalidInput(format!(
                "node {} is not a step of the plan",
                stop.node()
            )));
        };
        let mut chain = self.clone();
        chain.node_stop = Some((stop, *at));
        Ok(chain)
    }

    /// The same chain for a render with this photo frame (the photo's aspect, and the size of one output pixel
    /// in units of its long edge, of which the shapes' narrowest edge ramp is a multiple:
    /// [`ym_color::mask::PhotoFrame`], from the sampler's grid). The mask shapes of
    /// the nodes need it, with a source that has the position planes ([`encode_sample_at`]); a chain with no
    /// shape ignores it.
    ///
    /// [`encode_sample_at`]: crate::kernels::sampler::encode_sample_at
    #[must_use]
    pub fn with_photo_frame(&self, frame: PhotoFrame) -> Self {
        let mut chain = self.clone();
        chain.photo = Some(frame);
        chain
    }

    /// The same chain with the raster planes of the window it colours (`encode_raster_planes`): a raster row of a
    /// node reads its plane at the pixel (`docs/23` section 5.5). The window the chain is given afterwards must
    /// be the planes' window (same width and height); a chain with raster rows and no planes selects nothing
    /// for them.
    #[must_use]
    pub fn with_rasters(&self, planes: RasterPlanes) -> Self {
        let mut chain = self.clone();
        chain.rasters = Some(planes);
        chain
    }

    /// The same chain with the pictures the Film nodes of its plan made (`docs/24` section 3.9c): the plan buffer's first
    /// planes take the part of each picture under the window being coloured, and a Film step blends it by its mask and
    /// Strength. `origin` is the absolute pixel (in the photo's extent, the coordinates the pictures are in) of the
    /// window's pixel (0, 0). A picture must cover the window (a pixel it does not cover is zero).
    #[must_use]
    pub fn with_film(&self, pictures: Arc<FilmPictures>, origin: (i64, i64)) -> Self {
        let mut chain = self.clone();
        chain.film_pictures = Some((pictures, origin));
        chain
    }

    /// The Film nodes of the plan that need a picture, in plan order.
    #[must_use]
    pub fn live_films(&self) -> &[u32] {
        &self.films
    }

    /// The passes of the plan (one dispatch each; one for the default graph).
    pub fn pass_count(&self) -> usize {
        self.passes.len()
    }

    /// The f32 planes of the window a pixel's values that cross from one pass to the next take (a
    /// picture is three, a key one); 0 for a plan of one pass.
    pub fn pass_buffer_planes(&self) -> u32 {
        self.planes
    }

    /// The same chain with another limit on the bytes of the pass buffers of one dispatch (a plan of
    /// several passes keeps the values that cross from one to the next in f32 planes of the window; a
    /// window that needs more is cut into bands of rows, which gives the same bits). The default is the
    /// design binding ([`DESIGN_BINDING_BYTES`]); a smaller limit is for tests and for a device whose
    /// bindings are smaller than the design's.
    #[must_use]
    pub fn with_pass_buffer_limit(&self, bytes: u64) -> Self {
        let mut chain = self.clone();
        chain.buffer_limit = bytes.max(4);
        chain
    }

    /// The uniform block of one dispatch over `out.width x out.height` pixels, and the kind of
    /// output it makes. The pass lanes (`plan`) are those of a single pass over all the records; a
    /// plan with several passes sets them for each ([`PreparedChain::record`]).
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
        if let Some(r) = &self.rasters {
            u.rst = [
                r.plane_words() as u32,
                r.row0 as u32,
                r.stride as u32,
                r.planes as u32,
            ];
        }
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
        u.sp[3] = u32::from(mask.has_positions);
        if mask.has_positions {
            let Some(frame) = self.photo else {
                return Err(GpuError::InvalidInput(
                    "a source with position planes needs the photo frame (PreparedChain::with_photo_frame)"
                        .to_owned(),
                ));
            };
            u.photo = [
                frame.aspect[0] as f32,
                frame.aspect[1] as f32,
                frame.ramp() as f32,
                0.0,
            ];
        }
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
        let mut code = stop.code();
        if stop == Stop::C13
            && let Some((node_stop, at)) = self.node_stop
        {
            code = node_stop.code();
            u.stopn[0] = at.record;
        }
        // The live Film nodes: the colour pass looks for their pictures' exact display colours (`stopn.y`).
        u.stopn[1] = self.films.len() as u32;
        u.mode = [code, kind.code(), bits, u32::from(dither)];
        Ok((u, kind))
    }

    /// The passes a dispatch with the kernel stop `code` runs, in order, and the index of the last one,
    /// which writes the output. Stops before the nodes (C6 to C8 and the Before view's two reading points
    /// and its camera default) need no pass but the first; Main's stages (C9, C9b, C11 and the Before
    /// view's "after grade") need the pass Main is in (Main reads the Input, so nothing before it); a
    /// stop inside the nodes (11, 12) needs the passes up to the node's; every other stop runs the plan.
    fn passes_for(&self, code: u32) -> std::ops::RangeInclusive<usize> {
        let last = self.passes.len() - 1;
        let main_pass = self
            .steps
            .iter()
            .find(|s| s.node == MAIN_NODE)
            .map_or(0, |s| s.pass);
        match code {
            0..=2 | 13 => 0..=0,
            3 | 5 | 10 => main_pass..=main_pass,
            9 => match self.before {
                BeforePoint::AfterGrade => main_pass..=main_pass,
                BeforePoint::CameraLog | BeforePoint::CameraDefault => 0..=0,
            },
            11 | 12 => match self.node_stop {
                Some((_, at)) => 0..=at.pass,
                None => 0..=last,
            },
            _ => 0..=last,
        }
    }

    /// The bindings of one dispatch (the layout of `color.wgsl`).
    fn entries<'a>(
        &'a self,
        ub: &'a PooledBuffer,
        src: &'a PooledBuffer,
        out: &'a PooledBuffer,
        records: &'a PooledBuffer,
        pass_buffers: &'a PooledBuffer,
    ) -> [(u32, wgpu::BindingResource<'a>); 9] {
        [
            (0, ub.binding()),
            (1, self.tables.binding()),
            (2, self.curves.binding()),
            (3, self.look_luts.binding()),
            (4, src.binding()),
            (5, out.binding()),
            (6, records.binding()),
            // The raster masks' u16 planes of the window (`docs/23` section 5.5), or a stand-in that is never read.
            (
                7,
                self.rasters
                    .as_ref()
                    .map_or(&*self.no_rasters, |r| &*r.buf)
                    .binding(),
            ),
            (8, pass_buffers.binding()),
        ]
    }

    /// Records the passes of the plan over the window: the node records once, then one dispatch for each
    /// pass (the last writes `out`), cut into bands of rows when the planes that cross from one pass to
    /// the next would pass the design binding. Returns the kind of output.
    #[allow(clippy::too_many_arguments)] // the natural argument list; a struct would only move them
    fn record(
        &self,
        passes: &mut PassList<'_>,
        src: &PooledBuffer,
        src_window: SrcWindow,
        out_window: OutWindow,
        output: ColorOutput,
        dst: DstLayout,
        mask: MaskPaint,
        out: &PooledBuffer,
    ) -> Result<OutKind, GpuError> {
        let ctx = passes.context();
        let (_, kind) = self.uniform_for(output, &src_window, &out_window, dst, mask)?;
        let records = passes.keep(storage_buffer(ctx, "node records", &self.records[..])?);
        if let Some(r) = &self.rasters {
            // The planes must outlive the submission; the window they cover is the one coloured.
            passes.hold(Arc::clone(&r.buf));
            if (r.width, r.height) != (out_window.width, out_window.height) {
                return Err(GpuError::InvalidInput(format!(
                    "the raster planes cover {} x {} pixels, the colour window is {} x {}",
                    r.width, r.height, out_window.width, out_window.height
                )));
            }
        }
        // The pictures of the Film nodes the plan runs: one for each, or the frame is not this chain's to make.
        let film_images = self.film_images_for()?;
        // Rows per band: the whole window, unless the pass buffers would not fit one binding.
        let plane_bytes_per_row = 4 * u64::from(self.planes) * out_window.width.max(1) as u64;
        let band_rows = if self.planes == 0 {
            out_window.height.max(1)
        } else {
            ((self.buffer_limit / plane_bytes_per_row) as usize).clamp(1, out_window.height.max(1))
        };
        let scratch_bytes = plane_bytes_per_row * band_rows.min(out_window.height) as u64;
        let scratch = if self.planes == 0 {
            Arc::clone(&self.stand_in)
        } else if film_images.is_empty() {
            passes.scratch("plan buffers", scratch_bytes)?
        } else {
            // The Film pictures are copied in, and a picture that does not cover the window is cleared first.
            let buffer = ctx
                .pool()
                .acquire("plan buffers", scratch_bytes.max(4), IMAGE_USAGE)?;
            passes.keep(buffer)
        };
        let limit = ctx.info().limits.max_compute_workgroups_per_dimension as usize;
        let mut r0 = 0;
        loop {
            let rows = band_rows.min(out_window.height - r0);
            let mut src_band = src_window;
            src_band.offset.1 += r0;
            let mut out_band = out_window;
            out_band.height = rows;
            out_band.origin.1 += r0;
            let mut dst_band = dst;
            dst_band.at.1 += r0;
            let (mut base, _) = self.uniform_for(output, &src_band, &out_band, dst_band, mask)?;
            // A band of the window reads the raster planes from its own first row.
            base.rst[1] += r0 as u32;
            let (gx, gy) = pair_groups(out_band.width, rows);
            if gx as usize > limit || gy as usize > limit {
                return Err(GpuError::InvalidInput(format!(
                    "a {} x {rows} window needs more workgroups than the device allows ({limit})",
                    out_band.width
                )));
            }
            let span = self.passes_for(base.mode[0]);
            let last = *span.end();
            self.copy_films(passes, &film_images, &scratch, (out_band.width, rows, r0))?;
            for p in span {
                let mut u = base;
                let pass = self.passes[p];
                u.plan = [pass.first, pass.count, u32::from(p == last), self.result];
                let ub = passes.uniform("chain uniform", &u)?;
                passes.dispatch(
                    KernelId::Color,
                    if matches!(output, ColorOutput::Stage(Stop::Guide)) {
                        "color chain guide"
                    } else {
                        "color chain"
                    },
                    &self.entries(&ub, src, out, &records, &scratch),
                    (gx, gy, 1),
                )?;
            }
            r0 += rows;
            if r0 >= out_window.height {
                break;
            }
        }
        Ok(kind)
    }

    /// The pictures of the live Film nodes, in plan order, each with the place of the window's first pixel in it; `None`
    /// for a node the pictures do not hold (nothing of the photo is under the window). An error when the chain has a
    /// live Film node and was given no pictures at all.
    fn film_images_for(&self) -> Result<Vec<Option<FilmPicture>>, GpuError> {
        if self.films.is_empty() {
            return Ok(Vec::new());
        }
        let Some((pictures, origin)) = &self.film_pictures else {
            return Err(GpuError::InvalidInput(format!(
                "the plan runs {} Film node(s) and the chain was given no pictures for them",
                self.films.len()
            )));
        };
        Ok(self
            .films
            .iter()
            .map(|&node| {
                pictures.get(node).map(|image| {
                    (
                        Arc::clone(image),
                        origin.0 - i64::from(image.rect.x0),
                        origin.1 - i64::from(image.rect.y0),
                    )
                })
            })
            .collect())
    }

    /// Puts the part of each Film picture under the band (rows `r0..r0 + rows` of a window `width` wide) into the first
    /// planes of the plan buffer; what a picture does not cover is zero.
    fn copy_films(
        &self,
        passes: &mut PassList<'_>,
        images: &[Option<FilmPicture>],
        scratch: &Arc<PooledBuffer>,
        (width, rows, r0): (usize, usize, usize),
    ) -> Result<(), GpuError> {
        if images.is_empty() {
            return Ok(());
        }
        let n = width * rows;
        // The pixels of the band a picture covers: window column x is picture column x + dx, row y is y + dy.
        let mut whole = true;
        let mut covers = Vec::with_capacity(images.len());
        for item in images {
            let Some((image, dx, dy)) = item else {
                whole = false;
                covers.push((0, 0, 0, 0));
                continue;
            };
            let x0 = (-dx).max(0);
            let x1 = (i64::from(image.rect.width()) - dx).min(width as i64);
            let y0 = (-(dy + r0 as i64)).max(0);
            let y1 = (i64::from(image.rect.height()) - dy - r0 as i64).min(rows as i64);
            if x0 != 0 || y0 != 0 || x1 != width as i64 || y1 != rows as i64 {
                whole = false;
            }
            // A picture without a shadow leaves its shadow planes zero: they are cleared.
            if image.shadow.is_none() {
                whole = false;
            }
            covers.push((x0, y0, x1, y1));
        }
        if !whole {
            passes.encoder().clear_buffer(
                scratch,
                0,
                Some(u64::from(FILM_PLANES) * (images.len() * n * 4) as u64),
            );
        }
        let per = FILM_PLANES as usize;
        for (k, (item, (x0, y0, x1, y1))) in images.iter().zip(covers).enumerate() {
            let Some((image, dx, dy)) = item else {
                continue;
            };
            if x1 <= x0 || y1 <= y0 {
                continue;
            }
            // The rectangle of the picture, and where it lands in the band.
            let rect = Rect::new(
                (x0 + dx) as usize,
                (y0 + dy + r0 as i64) as usize,
                (x1 - x0) as usize,
                (y1 - y0) as usize,
            );
            // The copy kernel places the rectangle's (0, 0) at the destination's `at`, which a plain buffer of planes
            // does not have: the base sample carries the offset instead.
            let base = per * k * n + y0 as usize * width + x0 as usize;
            encode_copy_into(passes, &image.img, rect, 3, scratch, (width, n, base))?;
            if let Some(shadow) = &image.shadow {
                encode_copy_into(passes, shadow, rect, 3, scratch, (width, n, base + 3 * n))?;
            }
        }
        Ok(())
    }

    /// Encodes the dispatches of the kernel over an `out_w x out_h` window of `src` (planar image
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
        let mut passes = PassList::new(ctx, "color chain");
        let kind = self
            .uniform_for(output, src_window, out_window, dst, MaskPaint::default())?
            .1;
        let out_bytes = kind.device_bytes(out_stride, out_h);
        let out = passes.keep(ctx.pool().acquire("chain out", out_bytes, STORAGE_OUT)?);
        self.record(
            &mut passes,
            src,
            *src_window,
            *out_window,
            output,
            dst,
            MaskPaint::default(),
            &out,
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

    /// Records the colour pass over the `window` of `src` (camera RGB, three planes, four with the
    /// sampler's letterbox mask, or six with the photo-frame position planes besides) up to `stop`, writing
    /// three f32 planes into `dst` (a fourth one, holding the mask, when `src` has four or six), the
    /// window's pixel (0, 0) at `dst_at`. Nothing is read back and nothing is uploaded but the uniform
    /// block and the node records.
    ///
    /// This is the C1 to C12 stage of the view and of the export: the result stays on the device.
    ///
    /// # Errors
    ///
    /// [`GpuError::InvalidInput`] when the device does not store f32 images (a legacy f16 mode), when
    /// the window is empty or outside `src`, when `src` is not a three-, four- or six-plane image or `dst`
    /// does not have the planes of `src` up to four, when `src` has the position planes and the chain has no
    /// photo frame, or when the window does not fit `dst` at `dst_at`.
    pub fn encode_stage(
        &self,
        passes: &mut PassList<'_>,
        src: &GpuImage,
        window: Rect,
        stop: Stop,
        dst: &GpuImage,
        dst_at: (usize, usize),
    ) -> Result<(), GpuError> {
        if dst.planes != src.planes.min(4) {
            return Err(GpuError::InvalidInput(format!(
                "the destination has {} planes, the source {} (the mask plane follows the pixels, the position planes do not)",
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
                has_mask: src.planes >= 4,
                has_positions: src.planes == 6,
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
                has_mask: src.planes >= 4,
                has_positions: src.planes == 6,
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
        if !matches!(src.planes, 3 | 4 | 6) {
            return Err(GpuError::InvalidInput(format!(
                "the colour pass reads a three-, four- or six-plane image, not {} planes",
                src.planes
            )));
        }
        if window.is_empty() || window.x1() > src.width || window.y1() > src.height {
            return Err(GpuError::InvalidInput(format!(
                "the window {window:?} is empty or outside the {} x {} source",
                src.width, src.height
            )));
        }
        self.record(
            passes,
            &src.buf,
            SrcWindow {
                stride: src.stride,
                plane: src.plane_len(),
                offset: (window.x, window.y),
            },
            OutWindow {
                width: window.w,
                height: window.h,
                origin,
            },
            output,
            dst,
            mask,
            out,
        )?;
        Ok(())
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
    fn the_per_frame_block_is_608_bytes_and_a_node_record_1200() {
        // The per-frame `ChainU` (`docs/17-gpu.md` section 11) was meant to be about 1 KB: the one uniform
        // block (K3 allows one) holds the frame's parameters. It was 992 bytes with the ACES 2 scalars; the
        // Before view's camera default added 64, and the colour warp (C9b) 1040 plus the grade's six rows
        // (1664 bytes in all, 2026-11-01). The nodes' parameters then moved into the storage buffer of node
        // records (N-04, `docs/23` section 5.4), so a plan of 16 nodes does not make the uniform grow: the
        // block lost the grade, the zones, the balance and the warp (1136 bytes) and gained the pass (32). The
        // masks (N-10) added the place of the mask rows and the photo frame (32 more), the raster masks (N-13) the
        // place of their planes (16 more); the rows themselves are in the storage buffer after the records,
        // 64 bytes each.
        // A record is the step's flags and wiring (3 rows), the gain (1), the grade (6), the warp's counts
        // (1) and its 16 moving (3 rows each) and 16 locked pins (1 row each): 75 rows of 16 bytes. Both are
        // far below the 64 KiB uniform binding the device asks for
        // (`max_uniform_buffer_binding_size` of `wgpu::Limits::default()`) and the 16 KiB of wgpu's
        // downlevel limits.
        assert_eq!(size_of::<ChainU>(), 608);
        assert_eq!(size_of::<ChainU>() % 16, 0);
        assert_eq!(size_of::<NodeRecU>(), 1200);
        assert_eq!(size_of::<NodeRecU>() % 16, 0);
        assert_eq!(
            size_of::<NodeRecU>() - offset_of!(NodeRecU, warp_n),
            16 + 16 * (WARP_MOVE_ROWS + WARP_LOCK_ROWS)
        );
        assert_eq!(WARP_MOVE_ROWS, 48);
        assert_eq!(WARP_LOCK_ROWS, 16);
        // A plan of the most steps (a graph has 64 correctors) is 75 KiB of records.
        assert_eq!(MAX_STEPS * size_of::<NodeRecU>(), 76_800);
        // A mask row is four rows of 16 bytes, and 64 rows a node (the settings' limit) at 64 nodes are 256 KiB.
        assert_eq!(size_of::<MaskOpU>(), 64);
        assert_eq!(MASK_ROW_VEC4, 4);
        assert_eq!(REC_ROWS, 75);
        assert_eq!(MAX_STEPS * 64 * size_of::<MaskOpU>(), 262_144);
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
                saturation: 0.35,
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
        let mut u = NodeRecU::default();
        warp_record(&mut u, &warp);
        // ceil(1.25 (0.0922 / (0.05 sqrt(e)) + 0.3 / (0.01 sqrt(e)))) = ceil(24.14) = 25 steps.
        assert_eq!(warp.steps(), 25);
        assert_eq!(u.warp_n, [2, 1, 25, 0]);
        for (k, m) in warp.moves().iter().enumerate() {
            let rows = &u.warp_mv[WARP_ROWS_PER_MOVE * k..][..WARP_ROWS_PER_MOVE];
            let want = [
                [m.source[0], m.source[1], m.delta[0], m.delta[1]],
                [m.inv_two_sigma_sq, m.shadows, m.highlights, m.pivot_code],
                [m.exposure_ev, m.saturation, 0.0, 0.0],
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
        // A pin that only changes saturation is packed (it acts): no move, no exposure, its
        // saturation in the second lane of the third row. Saturation 0 beside it is a still pin.
        let params = WarpParams {
            pins: vec![
                WarpPin::moving([0.2, 0.2], [0.2, 0.2]),
                WarpPin {
                    saturation: -0.6,
                    ..WarpPin::moving([0.1, 0.0], [0.1, 0.0])
                },
            ],
        };
        let mut u = NodeRecU::default();
        warp_record(&mut u, &Warp::new(&params));
        assert_eq!(u.warp_n, [1, 0, 1, 0]);
        assert_eq!(u.warp_mv[0], [0.1, 0.0, 0.0, 0.0]);
        assert_eq!(u.warp_mv[2], [0.0, -0.6f32, 0.0, 0.0]);
        // No moving pin: nothing at all, the locks included (the kernel skips the stage).
        for pins in [
            vec![],
            vec![WarpPin::locked_at([0.1, 0.0])],
            vec![
                WarpPin::moving([0.1, 0.0], [0.1, 0.0]),
                WarpPin::locked_at([0.0, 0.1]),
            ],
        ] {
            let mut u = NodeRecU::default();
            warp_record(&mut u, &Warp::new(&WarpParams { pins }));
            assert_eq!(
                bytemuck::bytes_of(&u),
                bytemuck::bytes_of(&NodeRecU::default())
            );
        }
        // The maximum: 16 moving pins fill every row of `warp_mv`.
        let pins: Vec<WarpPin> = (0..20)
            .map(|i| {
                let a = f64::from(i) * 0.3;
                WarpPin {
                    range: 0.04,
                    ..WarpPin::moving(
                        [0.1 * a.cos(), 0.1 * a.sin()],
                        [0.12 * a.cos(), 0.1 * a.sin()],
                    )
                }
            })
            .collect();
        let mut u = NodeRecU::default();
        warp_record(&mut u, &Warp::new(&WarpParams { pins }));
        // Sixteen moves of 0.02 |cos a| at range 0.04: ceil(3.98) = 4 steps.
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
        // The Auto clip of a Sony body, 3.58 stops over grey: knee 1.58 (two stops under the clip), expansion half the way to the cap.
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
        assert!((f64::from(u.tone[0]) - 1.58).abs() < 1e-6);
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
    fn the_gate_lanes_carry_the_edges_of_the_highlight_range() {
        use ym_color::tone::{ACHROMATIC_GATE_FROM, ACHROMATIC_GATE_TO, AchromaticGate};
        let tone = ToneParams::new(1.0, 0.5, 0.18 * 3.58_f64.exp2(), 0.002).expect("tone");
        let custom = AchromaticGate::new(0.2, 0.9).expect("gate");
        let mut u = ChainU::default();
        tone_uniform(&mut u, &tone.with_gate(custom));
        assert_eq!(u.gate, [0.2_f32, (1.0 / 0.7) as f32, 0.0, 0.0]);
        // The default edges (0.6 and 0.9: a span of 0.3).
        let mut u = ChainU::default();
        tone_uniform(&mut u, &tone);
        let inv = (1.0 / (ACHROMATIC_GATE_TO - ACHROMATIC_GATE_FROM)) as f32;
        let from = ACHROMATIC_GATE_FROM as f32;
        assert_eq!(u.gate, [from, inv, 0.0, 0.0]);
        assert_eq!(from, 0.6);
        // A gate that is off has edges over the largest purity (2) and every lane stays finite (K8).
        let mut u = ChainU::default();
        tone_uniform(&mut u, &tone.with_gate(AchromaticGate::OFF));
        assert_eq!(u.gate, [2.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn the_cine_rows_are_the_matrix_from_xyz_to_the_output() {
        // The row-major rows of `CineRendering::xyz_to_output`, one per output, in f32.
        for target in OutputTarget::ALL {
            let m = CineRendering::new(target).xyz_to_output();
            let rows = [f4(m[0], 0.0), f4(m[1], 0.0), f4(m[2], 0.0)];
            for row in rows {
                assert_eq!(row[3], 0.0);
                assert!(row[..3].iter().all(|v| v.is_finite()));
            }
            // D65 white maps to (1, 1, 1) in every output: the rows sum to the white's coordinates.
            let d65 = ym_color::consts::xy_to_xyz(ym_color::consts::D65_XY);
            for row in rows {
                let w: f32 = (0..3).map(|k| row[k] * d65[k] as f32).sum();
                assert!((w - 1.0).abs() < 1e-5, "{target:?}: {w}");
            }
        }
    }

    /// The members of the WGSL struct `name` (the type of the global variable `var_name`, or of its
    /// array's elements) as (name, offset) pairs, and its size.
    fn wgsl_struct(
        module: &naga::Module,
        var_name: &str,
    ) -> (Vec<(String, u32, naga::Handle<naga::Type>)>, u32) {
        let var = module
            .global_variables
            .iter()
            .map(|(_, v)| v)
            .find(|v| v.name.as_deref() == Some(var_name))
            .unwrap_or_else(|| panic!("the variable {var_name}"));
        let mut ty = var.ty;
        if let naga::TypeInner::Array { base, .. } = &module.types[ty].inner {
            ty = *base;
        }
        let naga::TypeInner::Struct { members, span } = &module.types[ty].inner else {
            panic!("{var_name} is not a struct");
        };
        (
            members
                .iter()
                .map(|m| (m.name.clone().unwrap_or_default(), m.offset, m.ty))
                .collect(),
            *span,
        )
    }

    #[test]
    fn the_uniform_block_and_the_node_record_have_the_layout_of_the_wgsl_structs() {
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
            let (members, span) = wgsl_struct(&module, "CU");
            let fields: [(&str, usize); 33] = [
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
                ("fl", offset_of!(ChainU, fl)),
                ("cdl_slope", offset_of!(ChainU, cdl_slope)),
                ("cdl_offset", offset_of!(ChainU, cdl_offset)),
                ("cdl_power", offset_of!(ChainU, cdl_power)),
                ("look_lut", offset_of!(ChainU, look_lut)),
                ("cine_m0", offset_of!(ChainU, cine_m0)),
                ("cine_m1", offset_of!(ChainU, cine_m1)),
                ("cine_m2", offset_of!(ChainU, cine_m2)),
                ("look_m0", offset_of!(ChainU, look_m0)),
                ("look_m1", offset_of!(ChainU, look_m1)),
                ("look_m2", offset_of!(ChainU, look_m2)),
                ("cam0", offset_of!(ChainU, cam0)),
                ("cam1", offset_of!(ChainU, cam1)),
                ("cam2", offset_of!(ChainU, cam2)),
                ("camt", offset_of!(ChainU, camt)),
                ("gate", offset_of!(ChainU, gate)),
                ("plan", offset_of!(ChainU, plan)),
                ("stopn", offset_of!(ChainU, stopn)),
                ("mrow", offset_of!(ChainU, mrow)),
                ("photo", offset_of!(ChainU, photo)),
                ("rst", offset_of!(ChainU, rst)),
            ];
            assert_eq!(members.len(), fields.len());
            for ((name, offset, _), (want, at)) in members.iter().zip(fields) {
                assert_eq!(name, want);
                assert_eq!(*offset as usize, at, "offset of {want}");
            }
            assert_eq!(span as usize, size_of::<ChainU>(), "size of the block");

            // The records and the mask rows after them are raw `vec4` rows of one storage buffer: the kernel reads a
            // record's field at the row `R_*` the constants block writes from the Rust struct's offsets, and a mask
            // row's four rows from `CU.mrow.x`.
            let var = module
                .global_variables
                .iter()
                .map(|(_, v)| v)
                .find(|v| v.name.as_deref() == Some("nodes"))
                .expect("nodes");
            let naga::TypeInner::Array { base, stride, size } = &module.types[var.ty].inner else {
                panic!("nodes is not an array");
            };
            assert_eq!(*stride, 16);
            assert!(matches!(size, naga::ArraySize::Dynamic));
            assert!(
                matches!(
                    module.types[*base].inner,
                    naga::TypeInner::Vector {
                        size: naga::VectorSize::Quad,
                        scalar: naga::Scalar::U32
                    }
                ),
                "rows of vec4<u32>"
            );
            let constant = |name: &str| -> u32 {
                let tag = format!("const {name}: u32 = ");
                let at = source
                    .find(&tag)
                    .unwrap_or_else(|| panic!("{name} in the source"));
                source[at + tag.len()..]
                    .split('u')
                    .next()
                    .and_then(|n| n.parse().ok())
                    .unwrap_or_else(|| panic!("{name} is a u32 literal"))
            };
            assert_eq!(constant("REC_ROWS") as usize, size_of::<NodeRecU>() / 16);
            assert_eq!(
                constant("MASK_ROW_VEC4") as usize,
                size_of::<MaskOpU>() / 16
            );
            for (name, row) in record_rows() {
                assert_eq!(constant(name), row, "{name}");
            }
            assert_eq!(
                constant("R_WARP_LK") as usize * 16,
                offset_of!(NodeRecU, warp_lk)
            );
            // The warp's arrays are as long as the Rust arrays: the record ends after the last locked pin.
            assert_eq!(
                constant("R_WARP_MV") + constant("WARP_MOVE_ROWS"),
                constant("R_WARP_LK")
            );
            assert_eq!(
                constant("R_WARP_LK") + constant("WARP_LOCK_ROWS"),
                constant("REC_ROWS")
            );
            // The mask row: the head (kind, flags, plane, opacity), then the numbers.
            assert_eq!(offset_of!(MaskOpU, head), 0);
            assert_eq!(offset_of!(MaskOpU, p0), 16);
            assert_eq!(offset_of!(MaskOpU, p1), 32);
            assert_eq!(constant("MK_LINEAR"), 0);
            assert_eq!(constant("MK_BRIGHTNESS"), 3);
        }
    }

    #[test]
    fn the_kernel_binds_the_buffers_the_host_binds() {
        // One uniform and eight storage buffers (K3 allows eight): bindings 0 to 8, the table of `entries`;
        // 7 is the raster masks' planes of N-13.
        let source = shader_source(StorageMode::F32, KernelId::Color);
        let module = naga::front::wgsl::parse_str(&source).expect("color.wgsl");
        let info = Validator::new(ValidationFlags::all(), Capabilities::empty())
            .validate(&module)
            .expect("valid");
        let used = info.get_entry_point(0);
        let mut bindings = std::collections::BTreeSet::new();
        let mut storage = 0;
        for (handle, var) in module.global_variables.iter() {
            if let Some(b) = &var.binding
                && !used[handle].is_empty()
            {
                assert_eq!(b.group, 0);
                bindings.insert(b.binding);
                storage += usize::from(matches!(var.space, naga::AddressSpace::Storage { .. }));
            }
        }
        assert_eq!(bindings, [0, 1, 2, 3, 4, 5, 6, 7, 8].into());
        assert_eq!(storage, 8, "eight storage buffers and one uniform");
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
