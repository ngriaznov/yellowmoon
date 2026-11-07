//! Depth Anything V2-Small (encoder `vits`) in pure Rust on the CPU: the network behind the Depth mask. It turns a picture into
//! a relative inverse depth map (larger is nearer) the size of its network input.
//!
//! The architecture, the names of the weights and the pre-processing follow the code of github.com/DepthAnything/Depth-Anything-V2
//! (Apache-2.0; `depth_anything_v2/dpt.py`, `dinov2.py`, `dinov2_layers/`, `util/blocks.py`, `util/transform.py`), which is
//! itself Meta's DINOv2 (Apache-2.0) with a DPT head. The weights are the published `Depth-Anything-V2-Small` checkpoint
//! (Apache-2.0), converted once to f16 safetensors with the upstream state-dict names (`data/models/depth/`); this module is our own
//! code, written from a reading of the upstream one and checked against it (`tests/depth.rs`, the oracle fixtures).
//!
//! # Use
//!
//! ```no_run
//! use ym_ml::depth::{DepthAnything, prepare_input};
//!
//! # fn run(rgb01: &[f32], width: usize, height: usize) -> ym_ml::model::Result<()> {
//! let model = DepthAnything::load_bundled(std::path::Path::new("data"))?;
//! // The picture (sRGB values in 0..1, `[y][x][rgb]`) to the network size (short side 518 px, multiples of 14), normalised.
//! let input = prepare_input(rgb01, width, height)?;
//! let depth = model.infer(&input.data, input.width, input.height)?;
//! assert_eq!(depth.data.len(), depth.width * depth.height);
//! # Ok(()) }
//! ```
//!
//! # What runs
//!
//! 1. [`prepare_input`] (when the caller has the full picture): the aspect-keeping cubic resize and the ImageNet
//!    normalisation of `image2tensor`.
//! 2. The encoder (`encoder.rs`): the 14 x 14 patch embedding, the class token, the positional embedding resized to the patch
//!    grid, 12 blocks of global attention (6 heads) and MLP with LayerScale, and the final norm at the output of blocks 2, 5,
//!    8 and 11.
//! 3. The DPT head (`head.rs`): four projections and resize layers, four fusion blocks, the output convolutions; the map has
//!    the size of the input. It is not rescaled to the photo here.
//!
//! # Determinism
//!
//! Everything is `f32` in a fixed order with no fused multiply-add: every output element of every product is one chain of
//! operations, the sums of layer norms and softmaxes are `f64` in index order, `erf` and `exp` come from `libm`. The same
//! input gives the same bits on every run and on any number of threads (`sam2/kernels.rs` describes the scheme; the new
//! operations in `ops.rs` follow it).

use std::path::Path;

use safetensors::{Dtype, SafeTensors, tensor::TensorView};

use crate::cpu_runtime::read_float_tensors;
use crate::model::verify_sha256;
pub use crate::model::{MlError, Result};
use crate::sam2::Weights as Tensors;

mod encoder;
mod head;
mod ops;
mod preprocess;

pub use encoder::EncoderTrace;
use encoder::{BLOCKS, EMBED_DIM, Encoder, MLP_DIM, PATCH_PX, POS_GRID};
pub use head::HeadTrace;
use head::{FEATURES, Head, OUT_CHANNELS};
pub use preprocess::{
    IMAGE_MEAN, IMAGE_STD, LONG_SIDE_CAP_PX, NETWORK_SHORT_SIDE_PX, Prepared, capped_network_size,
    network_size, prepare_input,
};

/// The weights file in the model folder (`data/models/depth/`).
pub const WEIGHTS_FILE: &str = "depth_anything_v2_vits.f16.safetensors";
/// The side of a patch: the network input is a multiple of it on both sides.
pub const PATCH: usize = PATCH_PX;
/// The smallest side of a network input, in pixels (one patch).
pub const MIN_SIDE_PX: usize = PATCH_PX;
/// The largest side [`DepthAnything::infer`] takes, in pixels (the cost grows with the square of the tokens). [`prepare_input`]
/// stays under [`LONG_SIDE_CAP_PX`], which is far below it.
pub const MAX_SIDE_PX: usize = 4 * NETWORK_SHORT_SIDE_PX + 2 * PATCH_PX;

impl Tensors {
    /// `{prefix}.weight` as a 3 x 3 convolution `[cout][cin][3][3]`, with `{prefix}.bias` when `bias`.
    pub(crate) fn conv3(
        &mut self,
        prefix: &str,
        cout: usize,
        cin: usize,
        bias: bool,
    ) -> Result<ops::Conv3> {
        let weight = self.take(&format!("{prefix}.weight"), &[cout, cin, 3, 3])?;
        let bias = if bias {
            Some(self.take(&format!("{prefix}.bias"), &[cout])?)
        } else {
            None
        };
        Ok(ops::Conv3::from_oihw(&weight, cout, cin, bias.as_deref()))
    }

    /// `{prefix}.weight` as a `ConvTranspose2d(c, c, k, k)` `[c][c][k][k]` and `{prefix}.bias`.
    pub(crate) fn tconv(&mut self, prefix: &str, c: usize, k: usize) -> Result<ops::TConv> {
        let weight = self.take(&format!("{prefix}.weight"), &[c, c, k, k])?;
        let bias = self.take(&format!("{prefix}.bias"), &[c])?;
        Ok(ops::TConv::from_iokk(&weight, c, c, k, &bias))
    }
}

/// The tensors of the checkpoint that the network does not read: `mask_token` (used only by DINOv2's masked training) and the
/// first residual unit of `refinenet4` (the coarsest fusion block has no skip input, so `resConfUnit1` is never called,
/// `blocks.py` lines 131 to 135). A converted file may hold them or not.
fn is_unused(name: &str) -> bool {
    name == "pretrained.mask_token"
        || name.starts_with("depth_head.scratch.refinenet4.resConfUnit1.")
}

/// Every tensor of the checkpoint the network reads, `(upstream state-dict name, shape)`, in a fixed order. Convolution weights
/// are PyTorch's `[out][in][ky][kx]` (`ConvTranspose2d`: `[in][out][ky][kx]`), linear weights `[out][in]`.
pub fn tensor_specs() -> Vec<(String, Vec<usize>)> {
    let c = EMBED_DIM;
    let mut v: Vec<(String, Vec<usize>)> = Vec::new();
    let mut add = |name: String, shape: &[usize]| v.push((name, shape.to_vec()));
    add("pretrained.cls_token".into(), &[1, 1, c]);
    add(
        "pretrained.pos_embed".into(),
        &[1, 1 + POS_GRID * POS_GRID, c],
    );
    add(
        "pretrained.patch_embed.proj.weight".into(),
        &[c, 3, PATCH_PX, PATCH_PX],
    );
    add("pretrained.patch_embed.proj.bias".into(), &[c]);
    for i in 0..BLOCKS {
        let p = format!("pretrained.blocks.{i}");
        for norm in ["norm1", "norm2"] {
            add(format!("{p}.{norm}.weight"), &[c]);
            add(format!("{p}.{norm}.bias"), &[c]);
        }
        add(format!("{p}.attn.qkv.weight"), &[3 * c, c]);
        add(format!("{p}.attn.qkv.bias"), &[3 * c]);
        add(format!("{p}.attn.proj.weight"), &[c, c]);
        add(format!("{p}.attn.proj.bias"), &[c]);
        add(format!("{p}.ls1.gamma"), &[c]);
        add(format!("{p}.mlp.fc1.weight"), &[MLP_DIM, c]);
        add(format!("{p}.mlp.fc1.bias"), &[MLP_DIM]);
        add(format!("{p}.mlp.fc2.weight"), &[c, MLP_DIM]);
        add(format!("{p}.mlp.fc2.bias"), &[c]);
        add(format!("{p}.ls2.gamma"), &[c]);
    }
    add("pretrained.norm.weight".into(), &[c]);
    add("pretrained.norm.bias".into(), &[c]);
    let oc = OUT_CHANNELS;
    let f = FEATURES;
    for (i, &o) in oc.iter().enumerate() {
        add(format!("depth_head.projects.{i}.weight"), &[o, c, 1, 1]);
        add(format!("depth_head.projects.{i}.bias"), &[o]);
    }
    add(
        "depth_head.resize_layers.0.weight".into(),
        &[oc[0], oc[0], 4, 4],
    );
    add("depth_head.resize_layers.0.bias".into(), &[oc[0]]);
    add(
        "depth_head.resize_layers.1.weight".into(),
        &[oc[1], oc[1], 2, 2],
    );
    add("depth_head.resize_layers.1.bias".into(), &[oc[1]]);
    add(
        "depth_head.resize_layers.3.weight".into(),
        &[oc[3], oc[3], 3, 3],
    );
    add("depth_head.resize_layers.3.bias".into(), &[oc[3]]);
    for (i, &o) in oc.iter().enumerate() {
        add(
            format!("depth_head.scratch.layer{}_rn.weight", i + 1),
            &[f, o, 3, 3],
        );
    }
    for i in 1..=4 {
        let p = format!("depth_head.scratch.refinenet{i}");
        add(format!("{p}.out_conv.weight"), &[f, f, 1, 1]);
        add(format!("{p}.out_conv.bias"), &[f]);
        // `resConfUnit1` of `refinenet4` is not read (see `is_unused`) and so is not listed.
        for unit in if i == 4 {
            &["resConfUnit2"][..]
        } else {
            &["resConfUnit1", "resConfUnit2"][..]
        } {
            for conv in ["conv1", "conv2"] {
                add(format!("{p}.{unit}.{conv}.weight"), &[f, f, 3, 3]);
                add(format!("{p}.{unit}.{conv}.bias"), &[f]);
            }
        }
    }
    add(
        "depth_head.scratch.output_conv1.weight".into(),
        &[f / 2, f, 3, 3],
    );
    add("depth_head.scratch.output_conv1.bias".into(), &[f / 2]);
    add(
        "depth_head.scratch.output_conv2.0.weight".into(),
        &[32, f / 2, 3, 3],
    );
    add("depth_head.scratch.output_conv2.0.bias".into(), &[32]);
    add(
        "depth_head.scratch.output_conv2.2.weight".into(),
        &[1, 32, 1, 1],
    );
    add("depth_head.scratch.output_conv2.2.bias".into(), &[1]);
    v
}

/// Seeded random weights for the tests and the benchmark: every tensor of [`tensor_specs`], f16 safetensors, scaled so that
/// activations stay finite and of order one through the 12 blocks and the head (layer norm weights 1, LayerScale 0.5,
/// other weights uniform with a variance of `1 / fan_in`, biases small). Not a model: the output means nothing.
///
/// # Panics
///
/// Never in practice (the tensors are built here and serialised in memory).
pub fn synthetic_weights(seed: u64) -> Vec<u8> {
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    let mut next = move || -> f32 {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 40) as f32) / (1u64 << 24) as f32 * 2.0 - 1.0
    };
    let specs = tensor_specs();
    let buffers: Vec<Vec<u8>> = specs
        .iter()
        .map(|(name, shape)| {
            let len: usize = shape.iter().product();
            let out_first = !(name.starts_with("depth_head.resize_layers.0")
                || name.starts_with("depth_head.resize_layers.1"));
            // fan-in: everything but the first dimension (the output) of PyTorch's layout; the transposed convolutions
            // `[in][out][k][k]` read `in` inputs per output position.
            let fan_in = if shape.len() >= 2 {
                if out_first { len / shape[0] } else { shape[0] }
            } else {
                1
            };
            let amp = (3.0 / fan_in as f32).sqrt();
            (0..len)
                .flat_map(|_| {
                    let v = if name.ends_with("norm1.weight")
                        || name.ends_with("norm2.weight")
                        || name == "pretrained.norm.weight"
                    {
                        1.0
                    } else if name.ends_with(".gamma") {
                        0.5
                    } else if shape.len() == 1 {
                        0.05 * next()
                    } else if name.contains("pos_embed") || name.contains("cls_token") {
                        0.3 * next()
                    } else {
                        amp * next()
                    };
                    half::f16::from_f32(v).to_le_bytes()
                })
                .collect()
        })
        .collect();
    let views = specs.iter().zip(&buffers).map(|((name, shape), bytes)| {
        (
            name.clone(),
            TensorView::new(Dtype::F16, shape.clone(), bytes)
                // The buffers above have exactly `product(shape) * 2` bytes.
                .expect("tensor view of a buffer of the right size"),
        )
    });
    safetensors::serialize(views, None).expect("serialising in memory")
}

/// A depth map: relative inverse depth, larger is nearer, `[y][x]`, at the size of the network input.
#[derive(Debug, Clone, PartialEq)]
pub struct DepthMap {
    /// Width in pixels.
    pub width: usize,
    /// Height in pixels.
    pub height: usize,
    /// `width * height` values, row-major, all finite and at least 0.
    pub data: Vec<f32>,
}

/// Every stage of a run, for the parity tests against the oracle. Token streams are `[token][384]`, maps are
/// `[y][x][channel]` (PyTorch's tensors are `[channel][y][x]`).
#[derive(Debug, Default)]
pub struct Trace {
    /// The encoder's stages.
    pub encoder: EncoderTrace,
    /// The four tapped layers: final norm of blocks 2, 5, 8 and 11, class token removed, `[ph * pw][384]`.
    pub taps: Vec<Vec<f32>>,
    /// The head's stages.
    pub head: HeadTrace,
}

/// The network with its weights.
pub struct DepthAnything {
    encoder: Encoder,
    head: Head,
    sha256: Option<String>,
}

impl std::fmt::Debug for DepthAnything {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DepthAnything")
            .field("sha256", &self.sha256)
            .finish()
    }
}

impl DepthAnything {
    /// Loads `dir/depth_anything_v2_vits.f16.safetensors`, checked against its line in `dir/SHA256SUMS` (the model folder
    /// `data/models/depth/`).
    ///
    /// # Errors
    ///
    /// [`MlError::Io`] when a file cannot be read, [`MlError::Checksum`] when the weights do not match `SHA256SUMS`, and the
    /// errors of [`DepthAnything::from_bytes`].
    pub fn load(dir: &Path) -> Result<Self> {
        let bytes = std::fs::read(dir.join(WEIGHTS_FILE))?;
        let sums = std::fs::read_to_string(dir.join("SHA256SUMS"))?;
        let expected = sums
            .lines()
            .filter_map(|l| l.split_once("  "))
            .find(|(_, name)| name.trim() == WEIGHTS_FILE)
            .map(|(hash, _)| hash.trim().to_owned())
            .ok_or_else(|| {
                MlError::Weights(format!("SHA256SUMS has no line for {WEIGHTS_FILE}"))
            })?;
        Self::from_bytes(&bytes, Some(&expected))
    }

    /// Loads the model of a data folder: `<data_root>/models/depth/` ([`DepthAnything::load`]).
    ///
    /// # Errors
    ///
    /// As [`DepthAnything::load`].
    pub fn load_bundled(data_root: &Path) -> Result<Self> {
        Self::load(&data_root.join("models").join("depth"))
    }

    /// Loads a weights file without a checksum (development and tests).
    ///
    /// # Errors
    ///
    /// [`MlError::Io`] when the file cannot be read, and the errors of [`DepthAnything::from_bytes`].
    pub fn from_file(path: &Path) -> Result<Self> {
        Self::from_bytes(&std::fs::read(path)?, None)
    }

    /// Builds the network from the bytes of a safetensors file (f16, f32 or bf16) with the upstream state-dict names
    /// ([`tensor_specs`]), checked against `expected_sha256` when given. The header's metadata is not read: the architecture
    /// is fixed by the tensor names and shapes (ViT-S/14, DPT head with 64 features), so a file of another size fails on the
    /// first shape that differs.
    ///
    /// # Errors
    ///
    /// [`MlError::Checksum`] on a digest mismatch, [`MlError::Weights`] when the file is not safetensors, lacks a tensor,
    /// holds a value that is not finite or holds a tensor the network does not know, [`MlError::TensorShape`] when a tensor
    /// has the wrong shape.
    pub fn from_bytes(bytes: &[u8], expected_sha256: Option<&str>) -> Result<Self> {
        if let Some(want) = expected_sha256 {
            verify_sha256("Depth Anything V2-Small weights", bytes, want)?;
        }
        // Reject a file that is not safetensors before reading it as one.
        SafeTensors::read_metadata(bytes).map_err(|e| MlError::Weights(e.to_string()))?;
        let mut w = Tensors::from_tensors(read_float_tensors(bytes)?);
        let encoder = Encoder::load(&mut w)?;
        let head = Head::load(&mut w)?;
        if let Some(extra) = w.remaining().into_iter().find(|n| !is_unused(n)) {
            return Err(MlError::Weights(format!(
                "tensor `{extra}` is not part of Depth Anything V2-Small (vits)"
            )));
        }
        Ok(Self {
            encoder,
            head,
            sha256: expected_sha256.map(str::to_owned),
        })
    }

    /// The SHA-256 the weights were checked against, when they were.
    pub fn sha256(&self) -> Option<&str> {
        self.sha256.as_deref()
    }

    /// Runs the network on a normalised `[height][width][3]` picture (RGB, `(x / 255 - mean) / std`, see [`prepare_input`]) whose
    /// sides are multiples of [`PATCH`] between [`MIN_SIDE_PX`] and [`MAX_SIDE_PX`].
    ///
    /// # Errors
    ///
    /// [`MlError::Shape`] when the size is not allowed or `rgb` has another length, [`MlError::Input`] when it holds a value
    /// that is not finite.
    pub fn infer(&self, rgb: &[f32], width: usize, height: usize) -> Result<DepthMap> {
        self.infer_with(rgb, width, height, &|| false)
    }

    /// [`DepthAnything::infer`] that stops with [`MlError::Cancelled`] soon after `cancel` returns true (it is asked between
    /// the encoder's blocks and the head's stages, about every 50 ms).
    ///
    /// # Errors
    ///
    /// As [`DepthAnything::infer`], and [`MlError::Cancelled`].
    pub fn infer_with(
        &self,
        rgb: &[f32],
        width: usize,
        height: usize,
        cancel: &dyn Fn() -> bool,
    ) -> Result<DepthMap> {
        self.run(rgb, width, height, cancel, None)
    }

    /// [`DepthAnything::infer`] that also returns every stage (for the parity tests; it copies them, so it is slower).
    ///
    /// # Errors
    ///
    /// As [`DepthAnything::infer`].
    pub fn infer_traced(
        &self,
        rgb: &[f32],
        width: usize,
        height: usize,
    ) -> Result<(DepthMap, Trace)> {
        let mut trace = Trace::default();
        let map = self.run(rgb, width, height, &|| false, Some(&mut trace))?;
        Ok((map, trace))
    }

    fn run(
        &self,
        rgb: &[f32],
        width: usize,
        height: usize,
        cancel: &dyn Fn() -> bool,
        mut trace: Option<&mut Trace>,
    ) -> Result<DepthMap> {
        let allowed =
            |n: usize| n.is_multiple_of(PATCH) && (MIN_SIDE_PX..=MAX_SIDE_PX).contains(&n);
        if !allowed(width) || !allowed(height) {
            return Err(MlError::Shape(format!(
                "a depth input of {width} x {height} px: both sides must be multiples of {PATCH} from {MIN_SIDE_PX} to {MAX_SIDE_PX}"
            )));
        }
        if rgb.len() != width * height * 3 {
            return Err(MlError::Shape(format!(
                "the depth input holds {} values, {} expected ({width} x {height} x 3)",
                rgb.len(),
                width * height * 3
            )));
        }
        if rgb.iter().any(|v| !v.is_finite()) {
            return Err(MlError::Input(
                "the depth input holds a value that is not finite".into(),
            ));
        }
        let (ph, pw) = (height / PATCH, width / PATCH);
        let taps = self.encoder.forward(
            rgb,
            ph,
            pw,
            cancel,
            trace.as_deref_mut().map(|t| &mut t.encoder),
        )?;
        if let Some(t) = trace.as_deref_mut() {
            t.taps = taps.to_vec();
        }
        let data = self
            .head
            .forward(&taps, ph, pw, cancel, trace.map(|t| &mut t.head))?;
        if data.iter().any(|v| !v.is_finite()) {
            return Err(MlError::Input(
                "the depth network produced a value that is not finite".into(),
            ));
        }
        Ok(DepthMap {
            width,
            height,
            data,
        })
    }
}

/// The multiply-adds of one run on a `width` x `height` input (both multiples of [`PATCH`]): the products and convolutions,
/// not the attention softmax, the layer norms, the activations, the resizes or the biases.
pub fn multiply_adds(width: usize, height: usize) -> u64 {
    let (ph, pw) = ((height / PATCH) as u64, (width / PATCH) as u64);
    let n = ph * pw;
    let t = n + 1;
    let c = EMBED_DIM as u64;
    let patch = n * (PATCH_PX * PATCH_PX * 3) as u64 * c;
    let per_block = t * (c * 3 * c + c * c + 2 * c * MLP_DIM as u64) + 2 * t * t * c;
    let encoder = patch + BLOCKS as u64 * per_block;
    let conv3 = |px: u64, cin: u64, cout: u64| px * 9 * cin * cout;
    let oc = OUT_CHANNELS.map(|o| o as u64);
    let f = FEATURES as u64;
    let (h2, w2) = (ph.div_ceil(2), pw.div_ceil(2));
    let rcu = |px: u64| 2 * conv3(px, f, f);
    let mut head = n * c * oc.iter().sum::<u64>();
    head += n * oc[0] * 16 * oc[0] + n * oc[1] * 4 * oc[1];
    head += conv3(h2 * w2, oc[3], oc[3]);
    head += conv3(16 * n, oc[0], f)
        + conv3(4 * n, oc[1], f)
        + conv3(n, oc[2], f)
        + conv3(h2 * w2, oc[3], f);
    // refinenet4 (no skip), 3, 2, 1: residual units at the input's size, then the 1 x 1 convolution after the resize.
    head += rcu(h2 * w2) + n * f * f;
    head += 2 * rcu(n) + 4 * n * f * f;
    head += 2 * rcu(4 * n) + 16 * n * f * f;
    head += 2 * rcu(16 * n) + 64 * n * f * f;
    head += conv3(64 * n, f, f / 2);
    head += (height * width) as u64 * (9 * 32 * 32 + 32);
    encoder + head
}

#[cfg(test)]
mod tests;
