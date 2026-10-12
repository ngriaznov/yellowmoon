//! The `TileModel` trait, tile input and output shapes, model ids, versions and SHA-256 checks
//! (`docs/07-ml-models.md` sections Runtime and Parity and versioning).
//!
//! # Data layout
//!
//! Everything crossing the trait is **planar `f32`**, channel-major (`[channel][row][column]`).
//!
//! * [`TileInputs`]: `7 x h x w` at half resolution (`h`, `w` multiples of 4): channels 0 to 3 are the
//!   packed Bayer planes R, G1, G2, B (RGGB) balanced with `g0` (white = 1.0 before gains), channel 4 the
//!   sigma map, channel 5 the quantisation-step map, channel 6 the clip mask. All in the physical units of
//!   `docs/05` (the network applies its fixed input scaling itself, see `training/jdd/model.py`).
//! * [`TileOutput`]: `3 x 2h x 2w`, linear camera RGB in the same balanced units.
//!
//! # No hard GPU dependency
//!
//! `docs/07` passes a `&GpuQueue` to `run`. `ym-gpu`'s queue does not exist yet and `ym-ml` must not
//! depend on it (it would invert the crate graph and make the CPU runtime need wgpu). `run` therefore
//! takes an [`ExecContext`]: a small value that may carry a type-erased handle to the host's GPU objects.
//! `ym-core` wraps its `GpuQueue` with [`ExecContext::with_gpu`]; the burn backend downcasts it to the
//! wgpu setup it needs with [`ExecContext::gpu`], other backends ignore it. A pure-CPU caller passes
//! [`ExecContext::cpu`].

use std::any::Any;
use std::fmt;
use std::ops::Range;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// Number of input channels of the `jdd` network.
pub const IN_CHANNELS: usize = 7;
/// Number of output channels (linear RGB).
pub const OUT_CHANNELS: usize = 3;
/// Output pixels per input pixel along each axis (pixel shuffle x2).
pub const OUT_SCALE: usize = 2;
/// Tile height and width (half-resolution pixels) must be multiples of this (two stride-2 stages).
pub const TILE_MULTIPLE: usize = 4;

/// Channel indices of a [`TileInputs`] tensor.
pub mod channel {
    use std::ops::Range;
    /// The four packed Bayer planes R, G1, G2, B.
    pub const PLANES: Range<usize> = 0..4;
    /// Noise standard deviation map.
    pub const SIGMA: usize = 4;
    /// Quantisation-step map (zeros for lossless files).
    pub const QSTEP: usize = 5;
    /// Clip mask (1.0 where clipped).
    pub const CLIP: usize = 6;
}

/// Errors of the learned-model runtime.
#[derive(Debug, thiserror::Error)]
pub enum MlError {
    /// A weights or model file could not be read or is malformed.
    #[error("weights file: {0}")]
    Weights(String),
    /// A tensor in the weights file has the wrong name or shape for the architecture.
    #[error("weights tensor `{name}`: expected shape {expected:?}, found {found:?}")]
    TensorShape {
        /// Tensor name.
        name: String,
        /// Shape the architecture needs.
        expected: Vec<usize>,
        /// Shape in the file.
        found: Vec<usize>,
    },
    /// The architecture described by the file is not one this runtime implements.
    #[error("unsupported architecture: {0}")]
    Unsupported(String),
    /// A buffer or tile has the wrong size.
    #[error("shape mismatch: {0}")]
    Shape(String),
    /// The file's SHA-256 does not match the recorded one.
    #[error("checksum mismatch for {what}: expected {expected}, found {found}")]
    Checksum {
        /// What was hashed.
        what: String,
        /// Recorded digest (hex).
        expected: String,
        /// Computed digest (hex).
        found: String,
    },
    /// A backend (burn, ONNX Runtime) failed.
    #[error("{backend} backend: {detail}")]
    Backend {
        /// Backend name.
        backend: &'static str,
        /// What went wrong.
        detail: String,
    },
    /// The caller asked a tiled run to stop (the progress callback returned `false`).
    #[error("cancelled")]
    Cancelled,
    /// File access failed.
    #[error("i/o: {0}")]
    Io(#[from] std::io::Error),
}

/// Result alias of this crate.
pub type Result<T> = std::result::Result<T, MlError>;

/// Fixed input geometry of a tile model, in half-resolution pixels.
///
/// The output is `OUT_CHANNELS x (OUT_SCALE * height) x (OUT_SCALE * width)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TileShape {
    /// Input rows (half-resolution pixels, core plus halo on both sides).
    pub height: usize,
    /// Input columns (half-resolution pixels, core plus halo on both sides).
    pub width: usize,
}

impl TileShape {
    /// A square tile of `side` half-resolution pixels.
    pub const fn square(side: usize) -> Self {
        Self {
            height: side,
            width: side,
        }
    }

    /// Number of `f32` in a [`TileInputs`] of this shape.
    pub const fn input_len(&self) -> usize {
        IN_CHANNELS * self.height * self.width
    }

    /// Output rows (full resolution).
    pub const fn out_height(&self) -> usize {
        OUT_SCALE * self.height
    }

    /// Output columns (full resolution).
    pub const fn out_width(&self) -> usize {
        OUT_SCALE * self.width
    }

    /// Number of `f32` in a [`TileOutput`] of this shape.
    pub const fn output_len(&self) -> usize {
        OUT_CHANNELS * self.out_height() * self.out_width()
    }

    /// Check that both sides are positive multiples of [`TILE_MULTIPLE`].
    pub fn validate(&self) -> Result<()> {
        if self.height == 0
            || self.width == 0
            || !self.height.is_multiple_of(TILE_MULTIPLE)
            || !self.width.is_multiple_of(TILE_MULTIPLE)
        {
            return Err(MlError::Shape(format!(
                "tile {}x{} must be a positive multiple of {TILE_MULTIPLE} in both directions",
                self.height, self.width
            )));
        }
        Ok(())
    }
}

impl fmt::Display for TileShape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}x{}", self.height, self.width)
    }
}

/// Network input of one tile: planar `7 x height x width`.
#[derive(Clone, Debug, PartialEq)]
pub struct TileInputs {
    shape: TileShape,
    data: Vec<f32>,
}

impl TileInputs {
    /// An all-zero input of the given shape.
    pub fn zeros(shape: TileShape) -> Self {
        Self {
            shape,
            data: vec![0.0; shape.input_len()],
        }
    }

    /// Wrap planar data (`7 x h x w`, channel-major).
    pub fn new(shape: TileShape, data: Vec<f32>) -> Result<Self> {
        if data.len() != shape.input_len() {
            return Err(MlError::Shape(format!(
                "tile input of {shape} needs {} values, got {}",
                shape.input_len(),
                data.len()
            )));
        }
        Ok(Self { shape, data })
    }

    /// The tile shape.
    pub fn shape(&self) -> TileShape {
        self.shape
    }

    /// All values, channel-major.
    pub fn as_slice(&self) -> &[f32] {
        &self.data
    }

    /// All values, mutable.
    pub fn as_mut_slice(&mut self) -> &mut [f32] {
        &mut self.data
    }

    /// Consume into the raw vector.
    pub fn into_vec(self) -> Vec<f32> {
        self.data
    }

    fn plane_range(&self, c: usize) -> Range<usize> {
        let n = self.shape.height * self.shape.width;
        c * n..(c + 1) * n
    }

    /// Channel `c` (`0..7`) as a row-major `height x width` slice.
    pub fn plane(&self, c: usize) -> &[f32] {
        &self.data[self.plane_range(c)]
    }

    /// Channel `c` as a mutable slice.
    pub fn plane_mut(&mut self, c: usize) -> &mut [f32] {
        let r = self.plane_range(c);
        &mut self.data[r]
    }
}

/// Network output of one tile: planar `3 x 2*height x 2*width`.
#[derive(Clone, Debug, PartialEq)]
pub struct TileOutput {
    shape: TileShape,
    data: Vec<f32>,
}

impl TileOutput {
    /// An all-zero output for an input tile of `shape`.
    pub fn zeros(shape: TileShape) -> Self {
        Self {
            shape,
            data: vec![0.0; shape.output_len()],
        }
    }

    /// Wrap planar data (`3 x 2h x 2w`).
    pub fn new(shape: TileShape, data: Vec<f32>) -> Result<Self> {
        if data.len() != shape.output_len() {
            return Err(MlError::Shape(format!(
                "tile output for {shape} needs {} values, got {}",
                shape.output_len(),
                data.len()
            )));
        }
        Ok(Self { shape, data })
    }

    /// The (input) tile shape this output belongs to.
    pub fn shape(&self) -> TileShape {
        self.shape
    }

    /// All values, channel-major.
    pub fn as_slice(&self) -> &[f32] {
        &self.data
    }

    /// All values, mutable.
    pub fn as_mut_slice(&mut self) -> &mut [f32] {
        &mut self.data
    }

    /// Colour channel `c` (`0..3`) as a row-major `2*height x 2*width` slice.
    pub fn plane(&self, c: usize) -> &[f32] {
        let n = self.shape.out_height() * self.shape.out_width();
        &self.data[c * n..(c + 1) * n]
    }
}

/// Execution context handed to [`TileModel::run`]: an optional type-erased handle to the host's GPU
/// objects (see the module documentation). `Copy`, so it can be passed down freely.
#[derive(Clone, Copy, Default)]
pub struct ExecContext<'a> {
    gpu: Option<&'a (dyn Any + Send + Sync)>,
}

impl<'a> ExecContext<'a> {
    /// A context without GPU objects (CPU runtime, headless CLI).
    pub const fn cpu() -> Self {
        Self { gpu: None }
    }

    /// A context carrying the host's GPU handle (for example `ym-gpu`'s queue wrapper). Backends that
    /// can use it downcast with [`ExecContext::gpu`]; the others ignore it.
    pub fn with_gpu(handle: &'a (dyn Any + Send + Sync)) -> Self {
        Self { gpu: Some(handle) }
    }

    /// The GPU handle as `T`, if one was supplied and has that type.
    pub fn gpu<T: Any>(&self) -> Option<&'a T> {
        self.gpu.and_then(|g| g.downcast_ref::<T>())
    }

    /// Whether a GPU handle was supplied.
    pub fn has_gpu(&self) -> bool {
        self.gpu.is_some()
    }
}

impl fmt::Debug for ExecContext<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ExecContext")
            .field("has_gpu", &self.has_gpu())
            .finish()
    }
}

/// A network that maps one input tile to one output tile (`docs/07` section Runtime).
///
/// Implemented by the pure-Rust CPU runtime ([`crate::cpu_runtime::CpuJdd`]) and by the optional burn and
/// ort backends ([`crate::backends`]). Implementations are shared between threads; whatever interior
/// mutability a backend needs (an ONNX session, a GPU queue) lives behind a lock inside it.
pub trait TileModel: Send + Sync {
    /// The fixed input shape the model expects (from the ONNX graph for static-shape backends).
    fn input_shape(&self) -> TileShape;

    /// Margin (half-resolution pixels) a tile needs around the pixels it is asked to produce, from the
    /// weights header: half the receptive field rounded up to a multiple of 16 full-resolution pixels and
    /// never less than the worst one-sided reach (`training/jdd/receptive_field.py`, `docs/05` section
    /// Tiling and halos). Tiles are cropped, not blended.
    fn halo_half_px(&self) -> usize;

    /// Run the network. `inputs` must have exactly [`TileModel::input_shape`]; `out` is overwritten.
    fn run(&self, inputs: &TileInputs, out: &mut TileOutput, ctx: &ExecContext<'_>) -> Result<()>;

    /// Short backend name for logs and benchmark tables.
    fn name(&self) -> &str {
        "tile-model"
    }
}

/// Identity of a shipped model file: name, version and SHA-256 (the version goes into the export XMP).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelId {
    /// Model name, for example `jdd`.
    pub name: String,
    /// Version string from the weights header.
    pub version: String,
    /// Hex SHA-256 of the weights file bytes.
    pub sha256: String,
}

/// Hex SHA-256 of a byte slice.
pub fn sha256_hex(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

/// Check `bytes` against an expected hex digest.
pub fn verify_sha256(what: &str, bytes: &[u8], expected_hex: &str) -> Result<()> {
    let found = sha256_hex(bytes);
    if found.eq_ignore_ascii_case(expected_hex) {
        Ok(())
    } else {
        Err(MlError::Checksum {
            what: what.to_string(),
            expected: expected_hex.to_string(),
            found,
        })
    }
}

/// Architecture of a `jdd` network as written into the weights header by `training/jdd/weights_io.py`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct JddConfig {
    /// Channels of the three resolution levels.
    pub channels: [usize; 3],
    /// 3x3 convolutions per level.
    pub convs_per_level: usize,
    /// Kernel of the convolution in front of each pixel shuffle (1 or 3).
    pub up_kernel: usize,
    /// Kernel of the 12-channel head convolution (1 or 3).
    pub head_kernel: usize,
    /// Fixed scale applied to the four Bayer planes inside the network.
    pub plane_scale: f32,
    /// Fixed scale applied to the sigma map.
    pub sigma_scale: f32,
    /// Fixed scale applied to the quantisation-step map.
    pub qstep_scale: f32,
    /// Training-time initialisation detail; ignored at run time.
    #[serde(default)]
    pub head_init_scale: f32,
}

/// Largest channel count of any level a weights header may declare (the starting layout uses 96).
pub const MAX_LEVEL_CHANNELS: usize = 1024;
/// Largest number of convolutions per level a weights header may declare (the starting layout uses 2).
pub const MAX_CONVS_PER_LEVEL: usize = 16;

impl JddConfig {
    /// Reject configurations this runtime cannot or should not build: zero or absurd sizes, kernels other
    /// than 1 or 3, and input scales that are zero, negative or not finite. A weights file comes from
    /// outside the process, so none of these may reach an allocation or a division.
    pub fn validate(&self) -> Result<()> {
        if self
            .channels
            .iter()
            .any(|&c| c == 0 || c > MAX_LEVEL_CHANNELS)
            || self.convs_per_level == 0
            || self.convs_per_level > MAX_CONVS_PER_LEVEL
            || !matches!(self.up_kernel, 1 | 3)
            || !matches!(self.head_kernel, 1 | 3)
        {
            return Err(MlError::Unsupported(format!(
                "channels {:?}, convs per level {}, kernels up {} head {}",
                self.channels, self.convs_per_level, self.up_kernel, self.head_kernel
            )));
        }
        for (name, v) in [
            ("plane_scale", self.plane_scale),
            ("sigma_scale", self.sigma_scale),
            ("qstep_scale", self.qstep_scale),
        ] {
            if !(v.is_finite() && v > 0.0) {
                return Err(MlError::Weights(format!(
                    "config `{name}` must be finite and positive, got {v}"
                )));
            }
        }
        Ok(())
    }

    /// The per-channel input scale vector (`planes x4, sigma, qstep, clip`).
    pub fn input_scale(&self) -> [f32; IN_CHANNELS] {
        [
            self.plane_scale,
            self.plane_scale,
            self.plane_scale,
            self.plane_scale,
            self.sigma_scale,
            self.qstep_scale,
            1.0,
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shape_arithmetic_and_validation() {
        let s = TileShape {
            height: 96,
            width: 64,
        };
        assert_eq!(s.input_len(), 7 * 96 * 64);
        assert_eq!(s.output_len(), 3 * 192 * 128);
        assert!(s.validate().is_ok());
        assert!(TileShape::square(30).validate().is_err());
        assert!(TileShape::square(0).validate().is_err());
        assert_eq!(s.to_string(), "96x64");
    }

    #[test]
    fn buffers_check_their_length() {
        let s = TileShape::square(8);
        assert!(TileInputs::new(s, vec![0.0; 10]).is_err());
        assert!(TileOutput::new(s, vec![0.0; 10]).is_err());
        let mut t = TileInputs::zeros(s);
        t.plane_mut(channel::CLIP).fill(1.0);
        assert_eq!(t.plane(channel::CLIP).iter().sum::<f32>(), 64.0);
        assert_eq!(t.plane(0).iter().sum::<f32>(), 0.0);
        assert_eq!(TileOutput::zeros(s).plane(2).len(), 256);
    }

    #[test]
    fn exec_context_downcasts_the_gpu_handle() {
        struct FakeQueue(u32);
        let q = FakeQueue(7);
        let ctx = ExecContext::with_gpu(&q);
        assert!(ctx.has_gpu());
        assert_eq!(ctx.gpu::<FakeQueue>().map(|q| q.0), Some(7));
        assert!(ctx.gpu::<String>().is_none());
        assert!(!ExecContext::cpu().has_gpu());
    }

    fn good_config() -> JddConfig {
        JddConfig {
            channels: [32, 64, 96],
            convs_per_level: 2,
            up_kernel: 1,
            head_kernel: 1,
            plane_scale: 0.25,
            sigma_scale: 16.0,
            qstep_scale: 16.0,
            head_init_scale: 0.1,
        }
    }

    #[test]
    fn config_validation_rejects_values_that_would_panic_or_allocate_wildly() {
        assert!(good_config().validate().is_ok());
        let bad = [
            JddConfig {
                channels: [0, 64, 96],
                ..good_config()
            },
            JddConfig {
                channels: [32, 64, usize::MAX],
                ..good_config()
            },
            JddConfig {
                convs_per_level: 0,
                ..good_config()
            },
            JddConfig {
                convs_per_level: 1000,
                ..good_config()
            },
            JddConfig {
                up_kernel: 5,
                ..good_config()
            },
            JddConfig {
                head_kernel: 2,
                ..good_config()
            },
            JddConfig {
                plane_scale: 0.0,
                ..good_config()
            },
            JddConfig {
                sigma_scale: f32::NAN,
                ..good_config()
            },
            JddConfig {
                qstep_scale: -1.0,
                ..good_config()
            },
            JddConfig {
                plane_scale: f32::INFINITY,
                ..good_config()
            },
        ];
        for (i, c) in bad.iter().enumerate() {
            assert!(c.validate().is_err(), "bad config {i} was accepted");
        }
    }

    #[test]
    fn sha256_known_vector_and_verification() {
        // SHA-256("abc"), FIPS 180-2 test vector.
        let want = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";
        assert_eq!(sha256_hex(b"abc"), want);
        assert!(verify_sha256("test", b"abc", want).is_ok());
        assert!(matches!(
            verify_sha256("test", b"abd", want),
            Err(MlError::Checksum { .. })
        ));
    }
}
