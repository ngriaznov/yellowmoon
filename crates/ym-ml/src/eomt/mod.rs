//! EoMT small (COCO panoptic) in pure Rust on the CPU: the scene segmenter of Reference Grade (`docs/34-reference-grade.md` section
//! 4b.1). It turns a picture into a label map: for every pixel one of the 133 COCO panoptic classes, or "unlabelled", which
//! [`element_of`] maps to the elements the grade tells apart (sky, vegetation, water, ground, architecture, interior, person).
//!
//! EoMT (the Encoder-only Mask Transformer of Kerssies et al., CVPR 2025) is a plain DINOv2 ViT-S with 4 register tokens, **patch
//! 16** on a 640 x 640 input (a 40 x 40 grid: the patch convolution is 16 x 16 and the position table has 1600 rows, whatever the
//! training configuration's `vit_small_patch14_reg4_dinov2` says), 200 learned queries that enter before block 9, two upscaling
//! blocks, a class head and a mask head. The weights are the authors' checkpoint `tue-mps/coco_panoptic_eomt_small_640_2x` (MIT; the
//! DINOv2 backbone is Apache-2.0), converted once to f16 safetensors by `training/eomt/export.py` (`data/models/eomt/`,
//! `MODEL_CARD.md`). This module is our own code, written from a reading of the model code in Hugging Face's `transformers`
//! (Apache-2.0) and checked against it (`tests/eomt.rs`, the parity fixture of `tools/eomt/parity_fixture.py`). It reuses the ViT-S
//! block and the transposed convolution of the depth network (`crate::depth`) and the packed matrix product of SAM's kernels.
//!
//! # Use
//!
//! ```no_run
//! use ym_ml::eomt::{Eomt, Element, element_of};
//!
//! # fn run(rgb: &[u8]) -> ym_ml::model::Result<()> {
//! let model = Eomt::load_bundled(std::path::Path::new("data"))?;
//! // Once per photo, in the background: the 8-bit picture (the Input's, 1024 px on the long edge), rows top to bottom.
//! let labels = model.segment(rgb, 1024, 683)?;
//! // The label map has the model's own grid: the picture fitted to 640 px. Any size by nearest neighbour.
//! let sky = labels.classes.iter().filter(|&&c| element_of(c) == Element::Sky).count();
//! # let _ = sky;
//! # Ok(()) }
//! ```
//!
//! # What runs
//!
//! 1. [`preprocess`]: the picture is resized to fit 640 x 640 keeping its aspect ratio (an antialiased triangle filter, each pass
//!    rounded to 8 bits), padded at the bottom and the right with black, and normalised with the ImageNet mean and standard deviation.
//! 2. The network (`network.rs`): the 16 x 16 patch embedding with the learned positions, the class token and 4 registers, 12 blocks
//!    of global attention and MLP with LayerScale, the 200 queries in front of the stream from block 9 on, a final layer norm, the
//!    class head (134 outputs) and the mask head on the queries, two upscaling blocks on the patches (a 2 x 2 transposed
//!    convolution, GELU, a depthwise 3 x 3 convolution, a layer norm over the channels) and the product of the two: 200 mask logits
//!    of 160 x 160. The checkpoint's attention mask is annealed to zero, so inference makes none (`network.rs`).
//! 3. `post.rs`: the panoptic post-processing of Hugging Face's processor with its defaults: queries whose best class is not "none"
//!    and whose probability is above 0.8 are kept; each pixel goes to the kept query with the largest `score * sigmoid(mask)`; a
//!    query is a segment when it owns more than 0.8 of its mask; a pixel no segment holds is unlabelled.
//!
//! # Determinism
//!
//! Everything is `f32` in a fixed order, with no fused multiply-add, with `libm` for `exp` and `erf`, and with the sums of layer
//! norms and softmaxes in `f64` in index order, so the same picture gives the same bits on every run and on any number of threads
//! (the kernels of `sam2/kernels.rs`); as far as IEEE 754 goes, also on every platform.

use std::path::Path;

use safetensors::{Dtype, SafeTensors, tensor::TensorView};
use serde_json::Value;

use crate::cpu_runtime::read_float_tensors;
use crate::model::verify_sha256;
pub use crate::model::{MlError, Result};
use crate::sam2::Weights as Tensors;

mod classes;
mod network;
mod post;
pub mod preprocess;

pub use classes::{
    CLASS_NAMES, CLASSES, ELEMENTS, Element, FIRST_STUFF, UNLABELLED, class_name, element_of,
};
pub use network::MASK_PX;
use network::{BLOCKS, C, CLASS_OUTPUTS, GRID, HEADS, MLP, Network, PATCH_PX, QUERIES, REGISTERS};
pub use network::{Raw, multiply_adds};
pub use post::{Kept, MASK_THRESHOLD, OVERLAP_THRESHOLD, SCORE_THRESHOLD, kept_queries};
pub use preprocess::{IMAGE_PX, fit_size, input_tensor};

/// `format` of the safetensors header (`training/eomt/export.py`).
pub const FORMAT: &str = "yellowmoon-eomt";
/// `version` of the header this runtime reads.
pub const VERSION: &str = "1";
/// The weights file in the model folder (`data/models/eomt/`).
pub const WEIGHTS_FILE: &str = "eomt_coco_panoptic_small.f16.safetensors";
/// Parameters of the checkpoint (244 tensors), as `MODEL_CARD.md` counts them.
pub const PARAMETERS: usize = 23_976_079;
/// The side of a patch, in pixels (the patch size of the checkpoint: 16, not DINOv2's 14).
pub const PATCH: usize = PATCH_PX;
/// The side of the patch grid.
pub const GRID_PX: usize = GRID;
/// The queries of the network.
pub const NUM_QUERIES: usize = QUERIES;
/// The longest side of a picture [`Eomt::segment`] takes, in pixels.
pub const MAX_PICTURE_PX: usize = 16_384;

/// A label map: one class id per pixel, at the model's grid (the picture fitted to 640 px), rows top to bottom.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LabelMap {
    /// Width in pixels (at most 640).
    pub width: usize,
    /// Height in pixels (at most 640).
    pub height: usize,
    /// `width * height` class ids, 0 to 132, or [`UNLABELLED`].
    pub classes: Vec<u8>,
}

impl LabelMap {
    /// The element of every pixel ([`Element`] as its number).
    pub fn elements(&self) -> Vec<u8> {
        self.classes.iter().map(|&c| element_of(c) as u8).collect()
    }

    /// The map at another size by nearest neighbour: output pixel `(x, y)` takes the source pixel containing the point
    /// `((x + 0.5) / width, (y + 0.5) / height)` of the picture.
    ///
    /// # Panics
    ///
    /// Never for sizes of at least 1 (an empty request returns an empty map).
    pub fn resized_nearest(&self, width: usize, height: usize) -> LabelMap {
        let mut classes = Vec::with_capacity(width * height);
        let sx: Vec<usize> = (0..width)
            .map(|x| ((x * 2 + 1) * self.width / (2 * width)).min(self.width - 1))
            .collect();
        for y in 0..height {
            let sy = ((y * 2 + 1) * self.height / (2 * height)).min(self.height - 1);
            let row = &self.classes[sy * self.width..(sy + 1) * self.width];
            classes.extend(sx.iter().map(|&x| row[x]));
        }
        LabelMap {
            width,
            height,
            classes,
        }
    }
}

/// Every tensor of the checkpoint the network reads, `(name, shape)`, in a fixed order. Convolution weights are PyTorch's
/// `[out][in][ky][kx]` (`ConvTranspose2d`: `[in][out][ky][kx]`), linear weights `[out][in]`.
pub fn tensor_specs() -> Vec<(String, Vec<usize>)> {
    let mut v: Vec<(String, Vec<usize>)> = Vec::new();
    let mut add = |name: String, shape: &[usize]| v.push((name, shape.to_vec()));
    add("embeddings.cls_token".into(), &[1, 1, C]);
    add("embeddings.register_tokens".into(), &[1, REGISTERS, C]);
    add(
        "embeddings.patch_embeddings.projection.weight".into(),
        &[C, 3, PATCH_PX, PATCH_PX],
    );
    add("embeddings.patch_embeddings.projection.bias".into(), &[C]);
    add(
        "embeddings.position_embeddings.weight".into(),
        &[GRID * GRID, C],
    );
    add("layernorm.weight".into(), &[C]);
    add("layernorm.bias".into(), &[C]);
    add("query.weight".into(), &[QUERIES, C]);
    add("class_predictor.weight".into(), &[CLASS_OUTPUTS, C]);
    add("class_predictor.bias".into(), &[CLASS_OUTPUTS]);
    add("attn_mask_probs".into(), &[3]);
    add("criterion.empty_weight".into(), &[CLASS_OUTPUTS]);
    for n in 0..BLOCKS {
        let p = format!("layers.{n}");
        for norm in ["norm1", "norm2"] {
            add(format!("{p}.{norm}.weight"), &[C]);
            add(format!("{p}.{norm}.bias"), &[C]);
        }
        for proj in ["q_proj", "k_proj", "v_proj", "out_proj"] {
            add(format!("{p}.attention.{proj}.weight"), &[C, C]);
            add(format!("{p}.attention.{proj}.bias"), &[C]);
        }
        add(format!("{p}.layer_scale1.lambda1"), &[C]);
        add(format!("{p}.layer_scale2.lambda1"), &[C]);
        add(format!("{p}.mlp.fc1.weight"), &[MLP, C]);
        add(format!("{p}.mlp.fc1.bias"), &[MLP]);
        add(format!("{p}.mlp.fc2.weight"), &[C, MLP]);
        add(format!("{p}.mlp.fc2.bias"), &[C]);
    }
    for fc in ["fc1", "fc2", "fc3"] {
        add(format!("mask_head.{fc}.weight"), &[C, C]);
        add(format!("mask_head.{fc}.bias"), &[C]);
    }
    for m in 0..2 {
        let p = format!("upscale_block.block.{m}");
        add(format!("{p}.conv1.weight"), &[C, C, 2, 2]);
        add(format!("{p}.conv1.bias"), &[C]);
        add(format!("{p}.conv2.weight"), &[C, 1, 3, 3]);
        add(format!("{p}.layernorm2d.weight"), &[C]);
        add(format!("{p}.layernorm2d.bias"), &[C]);
    }
    v
}

/// The tensors of the checkpoint that the network does not read: the loss's class weight (`criterion.empty_weight`).
fn is_unused(name: &str) -> bool {
    name == "criterion.empty_weight"
}

/// The header's `config.id2label` against [`CLASS_NAMES`]: the class table must be the checkpoint's.
fn check_header(bytes: &[u8]) -> Result<()> {
    let (_, meta) =
        SafeTensors::read_metadata(bytes).map_err(|e| MlError::Weights(e.to_string()))?;
    let Some(map) = meta.metadata() else {
        return Err(MlError::Weights(
            "the file has no header metadata (not an EoMT file made by training/eomt/export.py)"
                .into(),
        ));
    };
    let field = |name: &str| map.get(name).map(String::as_str);
    if field("format") != Some(FORMAT) || field("version") != Some(VERSION) {
        return Err(MlError::Unsupported(format!(
            "an EoMT file with format {:?} version {:?}; {FORMAT} version {VERSION} is read",
            field("format"),
            field("version")
        )));
    }
    let config: Value = serde_json::from_str(field("config").unwrap_or("null"))
        .map_err(|e| MlError::Weights(format!("the header's config is not JSON: {e}")))?;
    let names: Vec<&str> = config["id2label"]
        .as_array()
        .map(|a| a.iter().filter_map(Value::as_str).collect())
        .unwrap_or_default();
    if names != CLASS_NAMES {
        return Err(MlError::Weights(
            "the header's id2label is not the class table of this runtime".into(),
        ));
    }
    let check = |key: &str, want: u64| -> Result<()> {
        if config[key].as_u64() == Some(want) {
            Ok(())
        } else {
            Err(MlError::Unsupported(format!(
                "an EoMT file with {key} = {}, {want} expected",
                config[key]
            )))
        }
    };
    check("hidden_size", C as u64)?;
    check("num_hidden_layers", BLOCKS as u64)?;
    check("num_attention_heads", HEADS as u64)?;
    check("patch_size", PATCH_PX as u64)?;
    check("num_positions", (GRID * GRID) as u64)?;
    check("num_register_tokens", REGISTERS as u64)?;
    check("num_queries", QUERIES as u64)?;
    check("num_labels", CLASSES as u64)
}

/// The network with its weights.
pub struct Eomt {
    network: Network,
    sha256: Option<String>,
}

impl std::fmt::Debug for Eomt {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Eomt")
            .field("sha256", &self.sha256)
            .finish()
    }
}

impl Eomt {
    /// Loads `dir/eomt_coco_panoptic_small.f16.safetensors`, checked against its line in `dir/SHA256SUMS` (the model folder
    /// `data/models/eomt/`).
    ///
    /// # Errors
    ///
    /// [`MlError::Io`] when a file cannot be read, [`MlError::Weights`] when `SHA256SUMS` has no line for the weights,
    /// [`MlError::Checksum`] when the weights do not match it, and the errors of [`Eomt::from_bytes`].
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

    /// Loads the model of a data folder: `<data_root>/models/eomt/` ([`Eomt::load`]).
    ///
    /// # Errors
    ///
    /// As [`Eomt::load`].
    pub fn load_bundled(data_root: &Path) -> Result<Self> {
        Self::load(&data_root.join("models").join("eomt"))
    }

    /// Loads a weights file without a checksum (development and tests).
    ///
    /// # Errors
    ///
    /// [`MlError::Io`] when the file cannot be read, and the errors of [`Eomt::from_bytes`].
    pub fn from_file(path: &Path) -> Result<Self> {
        Self::from_bytes(&std::fs::read(path)?, None)
    }

    /// Builds the network from the bytes of a safetensors file (f16, f32 or bf16) with the checkpoint's names
    /// ([`tensor_specs`]), checked against `expected_sha256` when given. The header must say `format` [`FORMAT`], `version`
    /// [`VERSION`] and carry the configuration of `training/eomt/export.py` with the class table of this runtime.
    ///
    /// # Errors
    ///
    /// [`MlError::Checksum`] on a digest mismatch, [`MlError::Weights`] when the file is not safetensors, lacks a tensor, holds
    /// a value that is not finite, holds a tensor the network does not know or has another class table,
    /// [`MlError::TensorShape`] when a tensor has the wrong shape, [`MlError::Unsupported`] when the header describes another
    /// model or the checkpoint's attention mask is not annealed to zero.
    pub fn from_bytes(bytes: &[u8], expected_sha256: Option<&str>) -> Result<Self> {
        if let Some(want) = expected_sha256 {
            verify_sha256("EoMT small weights", bytes, want)?;
        }
        check_header(bytes)?;
        let mut w = Tensors::from_tensors(read_float_tensors(bytes)?);
        let network = Network::load(&mut w)?;
        if let Some(extra) = w.remaining().into_iter().find(|n| !is_unused(n)) {
            return Err(MlError::Weights(format!(
                "tensor `{extra}` is not part of EoMT small"
            )));
        }
        Ok(Self {
            network,
            sha256: expected_sha256.map(str::to_owned),
        })
    }

    /// The SHA-256 the weights were checked against, when they were.
    pub fn sha256(&self) -> Option<&str> {
        self.sha256.as_deref()
    }

    /// Runs the network on a normalised `[640][640][3]` input (see [`preprocess`]) and returns the raw outputs.
    ///
    /// # Errors
    ///
    /// [`MlError::Shape`] when `input` has another length, [`MlError::Input`] when it holds a value that is not finite or the
    /// network produces one.
    pub fn infer(&self, input: &[f32]) -> Result<Raw> {
        self.infer_with(input, &|| false)
    }

    /// [`Eomt::infer`] that stops with [`MlError::Cancelled`] soon after `cancel` returns true (it is asked between the blocks,
    /// about every 100 ms).
    ///
    /// # Errors
    ///
    /// As [`Eomt::infer`], and [`MlError::Cancelled`].
    pub fn infer_with(&self, input: &[f32], cancel: &dyn Fn() -> bool) -> Result<Raw> {
        if input.len() != IMAGE_PX * IMAGE_PX * 3 {
            return Err(MlError::Shape(format!(
                "the EoMT input holds {} values, {} expected (640 x 640 x 3)",
                input.len(),
                IMAGE_PX * IMAGE_PX * 3
            )));
        }
        if input.iter().any(|v| !v.is_finite()) {
            return Err(MlError::Input(
                "the EoMT input holds a value that is not finite".into(),
            ));
        }
        let raw = self.network.forward(input, cancel)?;
        if raw
            .class_logits
            .iter()
            .chain(&raw.mask_logits)
            .any(|v| !v.is_finite())
        {
            return Err(MlError::Input(
                "the EoMT network produced a value that is not finite".into(),
            ));
        }
        Ok(raw)
    }

    /// The label map of an 8-bit RGB picture of `width` x `height` pixels (rows top to bottom, `R G B` per pixel): the whole
    /// path of [`preprocess`], the network and the post-processing. The map is at the model's grid ([`fit_size`]).
    ///
    /// # Errors
    ///
    /// [`MlError::Shape`] when a side is 0 or over [`MAX_PICTURE_PX`] or `rgb` has another length, and the errors of
    /// [`Eomt::infer`].
    pub fn segment(&self, rgb: &[u8], width: usize, height: usize) -> Result<LabelMap> {
        self.segment_with(rgb, width, height, &|| false)
    }

    /// [`Eomt::segment`] that stops with [`MlError::Cancelled`] soon after `cancel` returns true.
    ///
    /// # Errors
    ///
    /// As [`Eomt::segment`], and [`MlError::Cancelled`].
    pub fn segment_with(
        &self,
        rgb: &[u8],
        width: usize,
        height: usize,
        cancel: &dyn Fn() -> bool,
    ) -> Result<LabelMap> {
        let allowed = |n: usize| (1..=MAX_PICTURE_PX).contains(&n);
        if !allowed(width) || !allowed(height) {
            return Err(MlError::Shape(format!(
                "a picture of {width} x {height} px: both sides must be from 1 to {MAX_PICTURE_PX}"
            )));
        }
        if rgb.len() != width * height * 3 {
            return Err(MlError::Shape(format!(
                "the picture holds {} bytes, {} expected ({width} x {height} x 3)",
                rgb.len(),
                width * height * 3
            )));
        }
        let (input, (th, tw)) = input_tensor(rgb, width, height);
        let raw = self.infer_with(&input, cancel)?;
        Ok(self.label_map(&raw, th, tw))
    }

    /// The post-processing of [`Eomt::segment`] on raw outputs, for a picture that was fitted to `fit_height` x `fit_width`
    /// ([`fit_size`]; both at most 640).
    pub fn label_map(&self, raw: &Raw, fit_height: usize, fit_width: usize) -> LabelMap {
        let (h, w) = (fit_height.clamp(1, IMAGE_PX), fit_width.clamp(1, IMAGE_PX));
        LabelMap {
            width: w,
            height: h,
            classes: post::label_map(raw, h, w),
        }
    }
}

/// Seeded random weights for the tests and the benchmark: every tensor of [`tensor_specs`], f16 safetensors with the header the
/// loader reads, scaled so that activations stay finite and of order one through the 12 blocks (layer norm weights 1, LayerScale
/// 0.5, other weights uniform with a variance of `1 / fan_in`, biases small). Not a model: the output means nothing.
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
            // fan-in: everything but the first dimension (the output) of PyTorch's layout; the transposed convolutions
            // `[in][out][k][k]` read `in` inputs per output position.
            let fan_in = if shape.len() >= 2 {
                if name.contains(".conv1.weight") {
                    shape[0]
                } else {
                    len / shape[0]
                }
            } else {
                1
            };
            let amp = (3.0 / fan_in as f32).sqrt();
            (0..len)
                .flat_map(|_| {
                    let v = if name == "attn_mask_probs" {
                        0.0
                    } else if name.ends_with("norm1.weight")
                        || name.ends_with("norm2.weight")
                        || name.ends_with("layernorm.weight")
                        || name.ends_with("layernorm2d.weight")
                    {
                        1.0
                    } else if name.ends_with(".lambda1") {
                        0.5
                    } else if shape.len() == 1 {
                        0.05 * next()
                    } else if name.contains("position_embeddings")
                        || name.contains("cls_token")
                        || name.contains("register_tokens")
                        || name.starts_with("query.")
                    {
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
    let config = serde_json::json!({
        "hidden_size": C,
        "num_hidden_layers": BLOCKS,
        "num_attention_heads": HEADS,
        "patch_size": PATCH_PX,
        "num_positions": GRID * GRID,
        "num_register_tokens": REGISTERS,
        "num_queries": QUERIES,
        "num_labels": CLASSES,
        "id2label": CLASS_NAMES.as_slice(),
    });
    let metadata = std::collections::HashMap::from([
        ("format".to_owned(), FORMAT.to_owned()),
        ("version".to_owned(), VERSION.to_owned()),
        ("config".to_owned(), config.to_string()),
    ]);
    safetensors::serialize(views, Some(metadata)).expect("serialising in memory")
}

#[cfg(test)]
mod tests;
