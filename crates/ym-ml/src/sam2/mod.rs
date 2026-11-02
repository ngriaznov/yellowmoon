//! SAM 2.1 small for images, in pure Rust on the CPU (`docs/algorithms/06-masks.md` section 6, `docs/23-nodes.md` section
//! 4.4): the model behind the Object mask. The person clicks the photo or draws a box and the model answers with a mask of
//! the object there.
//!
//! The weights are Meta's (Apache-2.0), converted once to f16 safetensors by `training/sam2/export.py` and cut down to the
//! image path (`data/models/sam2/`, `MODEL_CARD.md`); this module is our own code, written from a reading of Meta's and
//! checked against it (`tests/sam2.rs`, the oracle of `docs/algorithms/06` section 6.5). Nothing of the video model is here:
//! no memory attention, no memory encoder, no tracking, no mask prompts.
//!
//! # Use
//!
//! ```no_run
//! use ym_ml::sam2::{Decoding, Picture, Point, Prompt, Sam2};
//!
//! # fn run(rgb: &[u8]) -> ym_ml::model::Result<()> {
//! let model = Sam2::load_bundled(std::path::Path::new("data"))?;
//! // Once per photo, in the background; the features (about 16 MB) are kept by the caller.
//! let features = model.encode(&Picture::new(1024, 683, rgb)?)?;
//! // Per click, about a tenth of a second.
//! match model.decode(&features, &Prompt::click(0.4, 0.55))? {
//!     Decoding::Found(found) => {
//!         let s0 = found.soft_mask(1024, 683); // sigma(logit), bilinear, 1024 px on the long edge
//!         assert_eq!(s0.data.len(), s0.width * s0.height);
//!     }
//!     Decoding::NothingFound { .. } => { /* "Nothing found there. ..." */ }
//! }
//! # Ok(()) }
//! ```
//!
//! # What runs
//!
//! 1. [`preprocess`]: the 8-bit picture is resized to 1024 x 1024 (not keeping the aspect ratio) with an antialiased
//!    triangle filter of our own, and normalised with the ImageNet mean and standard deviation.
//! 2. [`Sam2::encode`]: Hiera-S (patch embedding, 16 blocks, windowed and global attention) and the FPN neck give the image
//!    embedding (64 x 64 x 256, `no_mem_embed` added) and the two high-resolution maps of the decoder (256 x 256 x 32 and
//!    128 x 128 x 64).
//! 3. [`Sam2::decode`]: the prompt encoder (points, box), the two-way transformer, the upscaling and the hypernetworks give
//!    the 256 x 256 logits of the mask tokens, their predicted IoUs and the object score; a single positive point is decoded
//!    with the multimask output and the mask of the highest predicted IoU is taken; two or more points, or a box, use the
//!    single-mask token, with Meta's fall-back to the best multimask output when its stability is under 0.98. The logits are
//!    clamped to +-32. A click is refused ([`Decoding::NothingFound`]) when the object score is not above 0 or the mask
//!    covers under 0.05 % of the photo ([`OBJECT_MIN_AREA`]).
//!
//! # Determinism
//!
//! Everything is `f32` in a fixed order, with no fused multiply-add, with `libm` for `exp`, `erf`, `sin` and `cos`, and with the
//! sums of layer norms and softmaxes in `f64` in index order, so the same picture and prompt give the same bits on every run
//! and on any number of threads (the kernels of `sam2/kernels.rs`); as far as IEEE 754 goes, also on every platform.

use std::collections::HashMap;
use std::path::Path;

use rayon::prelude::*;
use safetensors::SafeTensors;
use serde_json::Value;

use crate::cpu_runtime::read_float_tensors;
pub use crate::model::{MlError, Result};
use crate::model::{sha256_hex, verify_sha256};

mod decoder;
mod kernels;
mod preprocess;
mod trunk;

use decoder::{Kind, MaskDecoder, PromptEncoder, check_finite};
use kernels::Packed;
use trunk::{Neck, Trunk};

/// Format string of the safetensors header (`training/sam2/export.py`).
pub const FORMAT: &str = "yellowmoon-sam2";
/// Format version this runtime reads.
pub const VERSION: &str = "1";
/// The weights file in the model folder (`data/models/sam2/`).
pub const WEIGHTS_FILE: &str = "sam2.1_hiera_small.f16.safetensors";
/// Parameters of the image path (359 tensors), as `MODEL_CARD.md` counts them.
pub const PARAMETERS: usize = 38_538_833;
/// The side of the model's input, in pixels.
pub const IMAGE_PX: usize = 1024;
/// The side of the image embedding (stride 16).
pub const EMBED_PX: usize = 64;
/// The width of the image embedding.
pub const EMBED_DIM: usize = 256;
/// The side of the decoder's logits.
pub const LOW_RES_PX: usize = 256;
/// The logits are clamped to this magnitude (Meta's clamp).
pub const LOGIT_CLAMP: f32 = 32.0;
/// A click is refused when the mask at logit > 0 covers less than this fraction of the photo (`docs/23` section 4.4).
pub const OBJECT_MIN_AREA: f64 = 0.0005;
/// `S0` (`docs/algorithms/06` section 6.3) is made at this size on the photo's long edge.
pub const S0_LONG_PX: usize = 1024;
/// The stability fall-back: a single mask whose stability is under this is replaced by the best multimask output.
pub const STABILITY_THRESHOLD: f32 = 0.98;
/// The stability is the IoU of the mask at logit > `delta` and at logit > `-delta`.
pub const STABILITY_DELTA: f32 = 0.05;
/// The most points a prompt may hold.
pub const MAX_POINTS: usize = 256;
/// The longest side of a picture [`Picture::new`] takes, in pixels.
pub const MAX_PICTURE_PX: usize = 16_384;

/// The ImageNet mean and standard deviation of the normalisation.
pub(crate) const PIXEL_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
pub(crate) const PIXEL_STD: [f32; 3] = [0.229, 0.224, 0.225];

// -------------------------------------------------------------------------------------------------
// Weights
// -------------------------------------------------------------------------------------------------

/// The tensors of a weights file by name, each taken out once with its shape checked.
pub(crate) struct Weights {
    tensors: HashMap<String, (Vec<usize>, Vec<f32>)>,
    taken: usize,
}

impl Weights {
    pub(crate) fn take(&mut self, name: &str, shape: &[usize]) -> Result<Vec<f32>> {
        let (found, data) = self
            .tensors
            .remove(name)
            .ok_or_else(|| MlError::Weights(format!("tensor `{name}` missing")))?;
        if found != shape {
            return Err(MlError::TensorShape {
                name: name.to_owned(),
                expected: shape.to_vec(),
                found,
            });
        }
        if data.iter().any(|v| !v.is_finite()) {
            return Err(MlError::Weights(format!(
                "tensor `{name}` holds a value that is not finite"
            )));
        }
        self.taken += data.len();
        Ok(data)
    }

    /// `{prefix}.weight` as `[n][k]` and `{prefix}.bias`, packed.
    pub(crate) fn linear(&mut self, prefix: &str, n: usize, k: usize) -> Result<Packed> {
        let weight = self.take(&format!("{prefix}.weight"), &[n, k])?;
        let bias = self.take(&format!("{prefix}.bias"), &[n])?;
        Ok(Packed::from_nk(&weight, n, k, Some(&bias)))
    }

    /// `{prefix}.weight` as `[n][k][1][1]` and `{prefix}.bias`, packed.
    pub(crate) fn conv1x1(&mut self, prefix: &str, n: usize, k: usize) -> Result<Packed> {
        let weight = self.take(&format!("{prefix}.weight"), &[n, k, 1, 1])?;
        let bias = self.take(&format!("{prefix}.bias"), &[n])?;
        Ok(Packed::from_nk(&weight, n, k, Some(&bias)))
    }
}

fn unsupported(what: impl std::fmt::Display) -> MlError {
    MlError::Unsupported(format!("a {FORMAT} file that {what}"))
}

/// Whether a header value is what this runtime implements (numbers compared to a relative 1e-9).
fn same_value(got: &Value, want: &Value) -> bool {
    match (got, want) {
        (Value::Number(a), Value::Number(b)) => match (a.as_f64(), b.as_f64()) {
            (Some(a), Some(b)) => (a - b).abs() <= 1e-9 * b.abs().max(1.0),
            _ => false,
        },
        (Value::Array(a), Value::Array(b)) => {
            a.len() == b.len() && a.iter().zip(b).all(|(x, y)| same_value(x, y))
        }
        _ => got == want,
    }
}

/// Looks `pointer` (a JSON pointer) up in the header's `config` and compares it with what this runtime implements.
fn expect(config: &Value, pointer: &str, want: &Value) -> Result<()> {
    let got = config
        .pointer(pointer)
        .ok_or_else(|| MlError::Weights(format!("header config has no `{pointer}`")))?;
    if same_value(got, want) {
        Ok(())
    } else {
        Err(unsupported(format!(
            "has `{pointer}` = {got} (this runtime implements {want})"
        )))
    }
}

/// Checks the header: the format and its version, and that the architecture in `config` is the one this runtime implements.
fn check_header(header: &HashMap<String, String>) -> Result<()> {
    let field = |k: &str| header.get(k).map_or("", String::as_str);
    if field("format") != FORMAT {
        return Err(MlError::Weights(format!(
            "not a {FORMAT} file (format = {:?})",
            field("format")
        )));
    }
    if field("version") != VERSION {
        return Err(MlError::Unsupported(format!(
            "{FORMAT} version {:?} (this runtime reads {VERSION})",
            field("version")
        )));
    }
    if field("model") != "sam2.1_hiera_small" {
        return Err(unsupported(format!("is the model {:?}", field("model"))));
    }
    let config: Value = serde_json::from_str(field("config"))
        .map_err(|e| MlError::Weights(format!("header config: {e}")))?;
    use serde_json::json;
    let numbers = [
        ("/image_size", json!(1024)),
        ("/backbone_stride", json!(16)),
        ("/image_embedding_size", json!([64, 64])),
        ("/low_res_mask_size", json!(256)),
        ("/logit_clamp", json!(32.0)),
        ("/mask_threshold", json!(0.0)),
        ("/directly_add_no_mem_embed", json!(true)),
        ("/pixel_mean", json!([0.485, 0.456, 0.406])),
        ("/pixel_std", json!([0.229, 0.224, 0.225])),
        ("/resize/antialias", json!(true)),
        ("/resize/keep_aspect_ratio", json!(false)),
        ("/resize/mode", json!("bilinear")),
        ("/multimask/dynamic_multimask_via_stability", json!(true)),
        ("/multimask/dynamic_multimask_stability_thresh", json!(0.98)),
        ("/multimask/dynamic_multimask_stability_delta", json!(0.05)),
        ("/multimask/multimask_min_pt_num", json!(0)),
        ("/multimask/multimask_max_pt_num", json!(1)),
        ("/multimask/multimask_output_in_sam", json!(true)),
        ("/prompt_encoder/embed_dim", json!(256)),
        ("/prompt_encoder/num_point_embeddings", json!(4)),
        ("/prompt_encoder/input_image_size", json!([1024, 1024])),
        ("/mask_decoder/transformer/depth", json!(2)),
        ("/mask_decoder/transformer/embedding_dim", json!(256)),
        ("/mask_decoder/transformer/mlp_dim", json!(2048)),
        ("/mask_decoder/transformer/num_heads", json!(8)),
        (
            "/mask_decoder/transformer/attention_downsample_rate",
            json!(2),
        ),
        ("/mask_decoder/num_mask_tokens", json!(4)),
        ("/mask_decoder/iou_head_depth", json!(3)),
        ("/mask_decoder/iou_head_hidden_dim", json!(256)),
        ("/mask_decoder/iou_prediction_use_sigmoid", json!(true)),
        ("/mask_decoder/pred_obj_scores", json!(true)),
        ("/mask_decoder/pred_obj_scores_mlp", json!(true)),
        ("/mask_decoder/use_high_res_features", json!(true)),
        ("/mask_decoder/layer_norm_2d_eps", json!(1e-6)),
        (
            "/mask_decoder/upscaling_activation",
            json!("gelu (exact, erf)"),
        ),
        ("/mask_decoder/num_multimask_outputs", json!(3)),
    ];
    for (pointer, want) in &numbers {
        expect(&config, pointer, want)?;
    }
    let trunk = [
        ("/trunk/type", json!("Hiera")),
        ("/trunk/embed_dim", json!(96)),
        ("/trunk/num_heads_first_stage", json!(1)),
        ("/trunk/stages", json!([1, 2, 11, 2])),
        ("/trunk/stage_ends", json!([0, 2, 13, 15])),
        ("/trunk/global_att_blocks", json!([7, 10, 13])),
        ("/trunk/q_pool_blocks", json!([1, 3, 14])),
        ("/trunk/q_stride", json!([2, 2])),
        ("/trunk/window_spec", json!([8, 4, 14, 7])),
        ("/trunk/window_pos_embed_bkg_spatial_size", json!([7, 7])),
        ("/trunk/mlp_ratio", json!(4.0)),
        ("/trunk/layer_norm_eps", json!(1e-6)),
        ("/trunk/mlp_activation", json!("gelu (exact, erf)")),
        ("/trunk/channel_list", json!([768, 384, 192, 96])),
    ];
    for (pointer, want) in &trunk {
        expect(&config, pointer, want)?;
    }
    for (pointer, want) in [
        ("/trunk/patch_embed/kernel", json!(7)),
        ("/trunk/patch_embed/stride", json!(4)),
        ("/trunk/patch_embed/padding", json!(3)),
        ("/trunk/patch_embed/in_chans", json!(3)),
        ("/neck/type", json!("FpnNeck")),
        ("/neck/d_model", json!(256)),
        ("/neck/backbone_channel_list", json!([768, 384, 192, 96])),
        ("/neck/fpn_interp_model", json!("nearest")),
        ("/neck/fpn_top_down_levels", json!([2, 3])),
        ("/neck/fuse_type", json!("sum")),
        ("/neck/scalp", json!(1)),
    ] {
        expect(&config, pointer, &want)?;
    }
    // The blocks, one by one: widths, heads, windows and which of them pool.
    let blocks = config
        .pointer("/trunk/blocks")
        .and_then(Value::as_array)
        .ok_or_else(|| MlError::Weights("header config has no `/trunk/blocks`".into()))?;
    if blocks.len() != trunk::BLOCKS.len() {
        return Err(unsupported(format!(
            "has {} blocks (this runtime implements {})",
            blocks.len(),
            trunk::BLOCKS.len()
        )));
    }
    for (i, (got, want)) in blocks.iter().zip(&trunk::BLOCKS).enumerate() {
        let table = json!({
            "dim_in": want.dim_in, "dim_out": want.dim_out, "num_heads": want.heads,
            "window_size": want.window, "query_pool": want.pool, "index": i,
        });
        for (key, value) in table.as_object().into_iter().flatten() {
            if got.get(key) != Some(value) {
                return Err(unsupported(format!(
                    "has block {i} with {key} = {:?} (this runtime implements {value})",
                    got.get(key)
                )));
            }
        }
    }
    Ok(())
}

// -------------------------------------------------------------------------------------------------
// The API
// -------------------------------------------------------------------------------------------------

/// An 8-bit sRGB picture, rows top to bottom, `R G B` per pixel: the picture the Input renders, at any size.
#[derive(Debug, Clone, Copy)]
pub struct Picture<'a> {
    width: usize,
    height: usize,
    rgb: &'a [u8],
}

impl<'a> Picture<'a> {
    /// Wraps `rgb` (`3 * width * height` bytes).
    ///
    /// # Errors
    ///
    /// [`MlError::Shape`] when the buffer is not `3 * width * height` bytes, [`MlError::Input`] when a side is 0 or
    /// longer than [`MAX_PICTURE_PX`].
    pub fn new(width: usize, height: usize, rgb: &'a [u8]) -> Result<Self> {
        if width == 0 || height == 0 || width > MAX_PICTURE_PX || height > MAX_PICTURE_PX {
            return Err(MlError::Input(format!(
                "a picture of {width} x {height} px (1 to {MAX_PICTURE_PX} on each side)"
            )));
        }
        if rgb.len() != 3 * width * height {
            return Err(MlError::Shape(format!(
                "a {width} x {height} picture needs {} bytes, the buffer holds {}",
                3 * width * height,
                rgb.len()
            )));
        }
        Ok(Self { width, height, rgb })
    }

    /// Width in pixels.
    #[must_use]
    pub fn width(&self) -> usize {
        self.width
    }

    /// Height in pixels.
    #[must_use]
    pub fn height(&self) -> usize {
        self.height
    }
}

/// The model's input for a picture: `[1024][1024][3]` `f32` (rows, columns, channels), the picture resized to 1024 x 1024
/// with the antialiased triangle filter (the aspect ratio is not kept) and normalised with the mean `(0.485, 0.456, 0.406)`
/// and the standard deviation `(0.229, 0.224, 0.225)`.
#[must_use]
pub fn preprocess(picture: &Picture<'_>) -> Vec<f32> {
    preprocess::input_tensor(picture.rgb, picture.width, picture.height)
}

/// The encoder's three maps for one photo, kept by the caller (about 16 MB) and given to every [`Sam2::decode`] of that photo.
///
/// Token-major: element `(y, x, c)` of a map is at `(y * side + x) * channels + c`.
#[derive(Debug, Clone)]
pub struct Features {
    image_embed: Vec<f32>,
    high_res_0: Vec<f32>,
    high_res_1: Vec<f32>,
}

impl Features {
    /// Builds the maps from planar (`[channel][y][x]`, PyTorch's layout) arrays: the image embedding (`256 x 64 x 64`, with
    /// `no_mem_embed` added) and the high-resolution maps (`32 x 256 x 256` and `64 x 128 x 128`). For tests against Meta's
    /// tensors and for tools that keep features on disk.
    ///
    /// # Errors
    ///
    /// [`MlError::Shape`] when a map has the wrong number of values, [`MlError::Input`] when one is not finite.
    pub fn from_chw(image_embed: &[f32], high_res_0: &[f32], high_res_1: &[f32]) -> Result<Self> {
        let to_hwc = |name: &str, src: &[f32], c: usize, side: usize| -> Result<Vec<f32>> {
            if src.len() != c * side * side {
                return Err(MlError::Shape(format!(
                    "{name}: {} values, {} expected ({c} x {side} x {side})",
                    src.len(),
                    c * side * side
                )));
            }
            if src.iter().any(|v| !v.is_finite()) {
                return Err(MlError::Input(format!(
                    "{name} holds a value that is not finite"
                )));
            }
            let mut out = vec![0.0f32; src.len()];
            out.par_chunks_mut(c).enumerate().for_each(|(p, px)| {
                for (ch, v) in px.iter_mut().enumerate() {
                    *v = src[ch * side * side + p];
                }
            });
            Ok(out)
        };
        Ok(Self {
            image_embed: to_hwc("image embedding", image_embed, EMBED_DIM, EMBED_PX)?,
            high_res_0: to_hwc(
                "high-resolution map 0",
                high_res_0,
                EMBED_DIM / 8,
                LOW_RES_PX,
            )?,
            high_res_1: to_hwc(
                "high-resolution map 1",
                high_res_1,
                EMBED_DIM / 4,
                LOW_RES_PX / 2,
            )?,
        })
    }

    /// The image embedding, `[64 * 64][256]`.
    #[must_use]
    pub fn image_embed(&self) -> &[f32] {
        &self.image_embed
    }

    /// The stride-4 map after `conv_s0`, `[256 * 256][32]`.
    #[must_use]
    pub fn high_res_0(&self) -> &[f32] {
        &self.high_res_0
    }

    /// The stride-8 map after `conv_s1`, `[128 * 128][64]`.
    #[must_use]
    pub fn high_res_1(&self) -> &[f32] {
        &self.high_res_1
    }

    /// The memory the maps take, in bytes.
    #[must_use]
    pub fn bytes(&self) -> usize {
        4 * (self.image_embed.len() + self.high_res_0.len() + self.high_res_1.len())
    }
}

/// A click on the photo: `x` and `y` are fractions of the photo's width and height (0 at the left and the top).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Point {
    /// Fraction of the width.
    pub x: f32,
    /// Fraction of the height.
    pub y: f32,
    /// `true` on the object, `false` for a part to take away.
    pub positive: bool,
}

/// What the person asked for: points, or a box, or both. A mask is a function of the points and the box alone.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Prompt {
    /// The clicks, in order.
    pub points: Vec<Point>,
    /// A box as `[x0, y0, x1, y1]`, fractions of the photo (top-left corner first).
    pub bbox: Option<[f32; 4]>,
}

impl Prompt {
    /// One positive click.
    #[must_use]
    pub fn click(x: f32, y: f32) -> Self {
        Self {
            points: vec![Point {
                x,
                y,
                positive: true,
            }],
            bbox: None,
        }
    }

    /// A box.
    #[must_use]
    pub fn rect(x0: f32, y0: f32, x1: f32, y1: f32) -> Self {
        Self {
            points: Vec::new(),
            bbox: Some([x0, y0, x1, y1]),
        }
    }

    /// The same prompt with one more point.
    #[must_use]
    pub fn with_point(mut self, x: f32, y: f32, positive: bool) -> Self {
        self.points.push(Point { x, y, positive });
        self
    }

    /// Whether the decoder runs with the multimask output: a single positive point and no box (`docs/algorithms/06` 6.1).
    #[must_use]
    pub fn multimask(&self) -> bool {
        self.bbox.is_none() && self.points.len() == 1 && self.points[0].positive
    }

    fn validate(&self) -> Result<()> {
        if self.points.is_empty() && self.bbox.is_none() {
            return Err(MlError::Input("a prompt needs a point or a box".into()));
        }
        if self.points.len() > MAX_POINTS {
            return Err(MlError::Input(format!(
                "a prompt of {} points (at most {MAX_POINTS})",
                self.points.len()
            )));
        }
        let finite = self
            .points
            .iter()
            .all(|p| p.x.is_finite() && p.y.is_finite())
            && self.bbox.is_none_or(|b| b.iter().all(|v| v.is_finite()));
        if !finite {
            return Err(MlError::Input(
                "a point or box coordinate is not a finite number".into(),
            ));
        }
        Ok(())
    }
}

/// What the decoder answered for a prompt.
#[derive(Debug, Clone)]
pub struct Decoded {
    /// The candidate masks' logits, `candidates x 256 x 256`, clamped to +-32: three for the multimask output, one otherwise.
    pub logits: Vec<f32>,
    /// The predicted IoU of each candidate (after the sigmoid).
    pub iou: Vec<f32>,
    /// The object score logit: the click is on an object when it is above 0.
    pub object_score: f32,
    /// The candidate taken: the one with the highest predicted IoU (always 0 for a single mask).
    pub chosen: usize,
    /// Whether the multimask output ran.
    pub multimask: bool,
    /// The mask token the chosen mask came from: 0 for the single-mask token, 1 to 3 for the multimask ones (a single mask
    /// that was not stable comes from the best multimask token).
    pub token: usize,
}

impl Decoded {
    /// The number of candidate masks.
    #[must_use]
    pub fn candidates(&self) -> usize {
        self.iou.len()
    }

    /// The logits of the chosen mask, `256 x 256` row-major, clamped to +-32.
    #[must_use]
    pub fn low_res_logits(&self) -> &[f32] {
        let n = LOW_RES_PX * LOW_RES_PX;
        &self.logits[self.chosen * n..(self.chosen + 1) * n]
    }

    /// The logits of candidate `i`.
    #[must_use]
    pub fn candidate_logits(&self, i: usize) -> &[f32] {
        let n = LOW_RES_PX * LOW_RES_PX;
        &self.logits[i * n..(i + 1) * n]
    }

    /// The predicted IoU of the chosen mask.
    #[must_use]
    pub fn chosen_iou(&self) -> f32 {
        self.iou[self.chosen]
    }

    /// The fraction of the photo the chosen mask covers at logit > 0 (the 256 x 256 square is the whole photo).
    #[must_use]
    pub fn area(&self) -> f64 {
        let inside = self.low_res_logits().iter().filter(|&&v| v > 0.0).count();
        inside as f64 / (LOW_RES_PX * LOW_RES_PX) as f64
    }

    /// `S0` (`docs/algorithms/06` section 6.3) for a photo of `photo_width` x `photo_height` pixels: `sigma(logit)` upsampled
    /// bilinearly to [`S0_LONG_PX`] on the long edge, the 256 x 256 square stretched back to the photo's aspect.
    #[must_use]
    pub fn soft_mask(&self, photo_width: usize, photo_height: usize) -> SoftMask {
        soft_mask(self.low_res_logits(), photo_width, photo_height)
    }
}

/// Why a click found nothing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NothingReason {
    /// The object score is not above 0: the model says the click is not on an object.
    NotAnObject,
    /// The mask at logit > 0 covers less than [`OBJECT_MIN_AREA`] of the photo.
    TooSmall,
}

/// The answer to a click, with the rule of `docs/23` section 4.4 applied.
#[derive(Debug, Clone)]
pub enum Decoding {
    /// A mask.
    Found(Decoded),
    /// "Nothing found there": no point is added and no mask is kept.
    NothingFound {
        /// Which of the two rules refused it.
        reason: NothingReason,
        /// The object score logit.
        object_score: f32,
        /// The fraction of the photo the chosen mask covers at logit > 0.
        area: f64,
    },
}

/// A soft mask at the size of `S0`: values in 0 to 1, row-major.
#[derive(Debug, Clone, PartialEq)]
pub struct SoftMask {
    /// Width in pixels.
    pub width: usize,
    /// Height in pixels.
    pub height: usize,
    /// `width * height` values.
    pub data: Vec<f32>,
}

/// The size of `S0` for a photo: [`S0_LONG_PX`] on the long edge, the other side rounded to keep the aspect.
#[must_use]
pub fn s0_size(photo_width: usize, photo_height: usize) -> (usize, usize) {
    let (w, h) = (photo_width.max(1), photo_height.max(1));
    if w >= h {
        (S0_LONG_PX, ((S0_LONG_PX * h + w / 2) / w).max(1))
    } else {
        (((S0_LONG_PX * w + h / 2) / h).max(1), S0_LONG_PX)
    }
}

/// `sigma(logit)` of 256 x 256 logits upsampled bilinearly (half-pixel centres, edges repeated) to [`s0_size`].
#[must_use]
pub fn soft_mask(logits: &[f32], photo_width: usize, photo_height: usize) -> SoftMask {
    assert_eq!(logits.len(), LOW_RES_PX * LOW_RES_PX, "256 x 256 logits");
    let (width, height) = s0_size(photo_width, photo_height);
    let p: Vec<f32> = logits.iter().map(|&v| kernels::sigmoid(v)).collect();
    // Per output index: the two source indices and the weight of the second.
    let axis = |n_out: usize| -> Vec<(usize, usize, f32)> {
        let scale = LOW_RES_PX as f64 / n_out as f64;
        (0..n_out)
            .map(|o| {
                let real = (scale * (o as f64 + 0.5) - 0.5).max(0.0);
                let i0 = (real.floor() as usize).min(LOW_RES_PX - 1);
                let i1 = (i0 + 1).min(LOW_RES_PX - 1);
                (i0, i1, (real - i0 as f64) as f32)
            })
            .collect()
    };
    let (ax, ay) = (axis(width), axis(height));
    let mut data = vec![0.0f32; width * height];
    data.par_chunks_mut(width).enumerate().for_each(|(y, row)| {
        let (y0, y1, wy) = ay[y];
        for (x, v) in row.iter_mut().enumerate() {
            let (x0, x1, wx) = ax[x];
            let top = p[y0 * LOW_RES_PX + x0] * (1.0 - wx) + p[y0 * LOW_RES_PX + x1] * wx;
            let bottom = p[y1 * LOW_RES_PX + x0] * (1.0 - wx) + p[y1 * LOW_RES_PX + x1] * wx;
            *v = top * (1.0 - wy) + bottom * wy;
        }
    });
    SoftMask {
        width,
        height,
        data,
    }
}

/// How long the stages of an [`Sam2::encode_timed`] took, in milliseconds.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct EncodeTimings {
    /// The resize and the normalisation.
    pub preprocess_ms: f64,
    /// Hiera-S.
    pub trunk_ms: f64,
    /// The neck and the decoder's two 1 x 1 convolutions.
    pub neck_ms: f64,
}

/// SAM 2.1 small, loaded from its weights file. See the module documentation.
pub struct Sam2 {
    trunk: Trunk,
    neck: Neck,
    prompt: PromptEncoder,
    decoder: MaskDecoder,
    sha256: String,
    parameters: usize,
}

impl std::fmt::Debug for Sam2 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Sam2")
            .field("parameters", &self.parameters)
            .field("sha256", &self.sha256)
            .finish()
    }
}

impl Sam2 {
    /// Loads `dir/sam2.1_hiera_small.f16.safetensors`, checked against its line in `dir/SHA256SUMS` (the model folder
    /// `data/models/sam2/`).
    ///
    /// # Errors
    ///
    /// [`MlError::Io`] when a file cannot be read, [`MlError::Checksum`] when the weights do not match `SHA256SUMS`, and the
    /// errors of [`Sam2::from_bytes`].
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

    /// Loads the model of a data folder: `<data_root>/models/sam2/` ([`Sam2::load`]).
    ///
    /// # Errors
    ///
    /// As [`Sam2::load`].
    pub fn load_bundled(data_root: &Path) -> Result<Self> {
        Self::load(&data_root.join("models").join("sam2"))
    }

    /// Loads a weights file without a checksum (development and tests; the shipped model goes through [`Sam2::load`]).
    ///
    /// # Errors
    ///
    /// [`MlError::Io`] when the file cannot be read, and the errors of [`Sam2::from_bytes`].
    pub fn from_file(path: &Path) -> Result<Self> {
        Self::from_bytes(&std::fs::read(path)?, None)
    }

    /// Builds the model from the bytes of a weights file, checked against `expected_sha256` when given.
    ///
    /// # Errors
    ///
    /// [`MlError::Checksum`] on a digest mismatch, [`MlError::Weights`] when the file is not a `yellowmoon-sam2` file, lacks a
    /// tensor or holds a value that is not finite, [`MlError::Unsupported`] for another format version or an architecture the
    /// runtime does not implement, [`MlError::TensorShape`] when a tensor has the wrong shape.
    pub fn from_bytes(bytes: &[u8], expected_sha256: Option<&str>) -> Result<Self> {
        if let Some(want) = expected_sha256 {
            verify_sha256("SAM 2.1 small weights", bytes, want)?;
        }
        let (_, meta) =
            SafeTensors::read_metadata(bytes).map_err(|e| MlError::Weights(e.to_string()))?;
        let header = meta.metadata().clone().unwrap_or_default();
        check_header(&header)?;
        let mut w = Weights {
            tensors: read_float_tensors(bytes)?,
            taken: 0,
        };
        let trunk = Trunk::load(&mut w)?;
        let neck = Neck::load(&mut w)?;
        let prompt = PromptEncoder::load(&mut w)?;
        let decoder = MaskDecoder::load(&mut w)?;
        if let Some(extra) = w.tensors.keys().min() {
            return Err(MlError::Weights(format!("unexpected tensor `{extra}`")));
        }
        if w.taken != PARAMETERS {
            return Err(MlError::Weights(format!(
                "{} parameters read, {PARAMETERS} expected",
                w.taken
            )));
        }
        Ok(Self {
            trunk,
            neck,
            prompt,
            decoder,
            sha256: sha256_hex(bytes),
            parameters: w.taken,
        })
    }

    /// SHA-256 of the weights file (hex).
    #[must_use]
    pub fn sha256(&self) -> &str {
        &self.sha256
    }

    /// Number of parameters of the image path.
    #[must_use]
    pub fn parameters(&self) -> usize {
        self.parameters
    }

    /// The encoder for a picture: [`preprocess`], then Hiera-S and the neck. Run once per photo.
    ///
    /// # Errors
    ///
    /// Never for a valid [`Picture`]; see [`Sam2::encode_input`].
    pub fn encode(&self, picture: &Picture<'_>) -> Result<Features> {
        self.encode_timed(picture).map(|(f, _)| f)
    }

    /// [`Sam2::encode`] that stops when `cancel` says so: it is asked before each of the trunk's 16 blocks (about a fifth of
    /// a second apart on four cores), and the encoder then returns [`MlError::Cancelled`] having kept nothing. The features of a
    /// run that finishes are those of [`Sam2::encode`], bit for bit: the hook only reads.
    ///
    /// # Errors
    ///
    /// [`MlError::Cancelled`] when `cancel` returned `true`; otherwise as [`Sam2::encode`].
    pub fn encode_cancellable(
        &self,
        picture: &Picture<'_>,
        cancel: &dyn Fn() -> bool,
    ) -> Result<Features> {
        let input = preprocess(picture);
        self.encode_input_with(&input, cancel).map(|(f, _)| f)
    }

    /// [`Sam2::encode`] with the time of each stage.
    ///
    /// # Errors
    ///
    /// As [`Sam2::encode`].
    pub fn encode_timed(&self, picture: &Picture<'_>) -> Result<(Features, EncodeTimings)> {
        let t0 = std::time::Instant::now();
        let input = preprocess(picture);
        let t1 = std::time::Instant::now();
        let (features, mut timings) = self.encode_input_timed(&input)?;
        timings.preprocess_ms = (t1 - t0).as_secs_f64() * 1e3;
        Ok((features, timings))
    }

    /// The encoder for an input tensor made by [`preprocess`] (or by Meta's code, converted to `[y][x][channel]`).
    ///
    /// # Errors
    ///
    /// [`MlError::Shape`] when the tensor is not `1024 * 1024 * 3` values, [`MlError::Input`] when one is not finite.
    pub fn encode_input(&self, input: &[f32]) -> Result<Features> {
        self.encode_input_timed(input).map(|(f, _)| f)
    }

    fn encode_input_timed(&self, input: &[f32]) -> Result<(Features, EncodeTimings)> {
        self.encode_input_with(input, &|| false)
    }

    fn encode_input_with(
        &self,
        input: &[f32],
        cancel: &dyn Fn() -> bool,
    ) -> Result<(Features, EncodeTimings)> {
        if input.iter().any(|v| !v.is_finite()) {
            return Err(MlError::Input(
                "the input tensor holds a value that is not finite".into(),
            ));
        }
        let t0 = std::time::Instant::now();
        let stages = self.trunk.forward(input, cancel)?;
        let t1 = std::time::Instant::now();
        let maps = self.neck.forward(&stages);
        let t2 = std::time::Instant::now();
        Ok((
            Features {
                image_embed: maps.image_embed,
                high_res_0: maps.high_res_0,
                high_res_1: maps.high_res_1,
            },
            EncodeTimings {
                preprocess_ms: 0.0,
                trunk_ms: (t1 - t0).as_secs_f64() * 1e3,
                neck_ms: (t2 - t1).as_secs_f64() * 1e3,
            },
        ))
    }

    /// The decoder's answer to a prompt, whatever it says: the candidate masks, their IoUs, the object score and the one
    /// chosen. [`Sam2::decode`] adds the rule that refuses a click.
    ///
    /// # Errors
    ///
    /// [`MlError::Input`] for an empty prompt, too many points or a coordinate that is not finite, or when the decoder
    /// produces a value that is not finite.
    pub fn decode_raw(&self, features: &Features, prompt: &Prompt) -> Result<Decoded> {
        prompt.validate()?;
        // The tokens, in Meta's order: the box corners first, then the points, then the padding token.
        let to_frame = |x: f32| x * IMAGE_PX as f32;
        let mut tokens: Vec<((f32, f32), Kind)> = Vec::new();
        if let Some([x0, y0, x1, y1]) = prompt.bbox {
            tokens.push(((to_frame(x0), to_frame(y0)), Kind::TopLeft));
            tokens.push(((to_frame(x1), to_frame(y1)), Kind::BottomRight));
        }
        for p in &prompt.points {
            tokens.push((
                (to_frame(p.x), to_frame(p.y)),
                if p.positive {
                    Kind::Positive
                } else {
                    Kind::Negative
                },
            ));
        }
        tokens.push(((0.0, 0.0), Kind::Padding));
        let sparse = self.prompt.sparse(&tokens);
        let raw = self.decoder.run(
            &features.image_embed,
            &features.high_res_0,
            &features.high_res_1,
            &sparse,
            &self.prompt,
        );
        check_finite(&raw)?;
        let plane = LOW_RES_PX * LOW_RES_PX;
        let mask = |t: usize| &raw.logits[t * plane..(t + 1) * plane];
        let multimask = prompt.multimask();
        // The token whose mask is taken, and the candidates.
        let (tokens_out, chosen): (Vec<usize>, usize) = if multimask {
            // Three candidates (tokens 1 to 3); the highest predicted IoU.
            (vec![1, 2, 3], argmax(&raw.iou[1..]))
        } else {
            // The single-mask token, unless it is not stable.
            let best = 1 + argmax(&raw.iou[1..]);
            if stability(mask(0)) >= STABILITY_THRESHOLD {
                (vec![0], 0)
            } else {
                (vec![best], 0)
            }
        };
        let mut logits = Vec::with_capacity(tokens_out.len() * plane);
        for &t in &tokens_out {
            logits.extend(mask(t).iter().map(|&v| v.clamp(-LOGIT_CLAMP, LOGIT_CLAMP)));
        }
        Ok(Decoded {
            iou: tokens_out.iter().map(|&t| raw.iou[t]).collect(),
            logits,
            object_score: raw.object_score,
            chosen,
            multimask,
            token: tokens_out[chosen],
        })
    }

    /// The decoder's answer to a prompt with the rule of `docs/23` section 4.4 applied: [`Decoding::NothingFound`] when the
    /// object score is not above 0 or when the mask at logit > 0 covers under [`OBJECT_MIN_AREA`] of the photo.
    ///
    /// # Errors
    ///
    /// As [`Sam2::decode_raw`].
    pub fn decode(&self, features: &Features, prompt: &Prompt) -> Result<Decoding> {
        Ok(judge(self.decode_raw(features, prompt)?))
    }
}

/// The rule of `docs/23` section 4.4: a click is refused when the object score is not above 0 (a NaN is not above 0) or when
/// the chosen mask at logit > 0 covers less than [`OBJECT_MIN_AREA`] of the photo.
fn judge(decoded: Decoded) -> Decoding {
    let area = decoded.area();
    let reason = if decoded.object_score.partial_cmp(&0.0) != Some(std::cmp::Ordering::Greater) {
        Some(NothingReason::NotAnObject)
    } else if area < OBJECT_MIN_AREA {
        Some(NothingReason::TooSmall)
    } else {
        None
    };
    match reason {
        Some(reason) => Decoding::NothingFound {
            reason,
            object_score: decoded.object_score,
            area,
        },
        None => Decoding::Found(decoded),
    }
}

/// The index of the largest value (the first one on a tie, as `torch.argmax`).
fn argmax(v: &[f32]) -> usize {
    let mut best = 0;
    for (i, &x) in v.iter().enumerate() {
        if x > v[best] {
            best = i;
        }
    }
    best
}

/// The stability of a mask (Meta's `_get_stability_scores`): the area at logit > delta over the area at logit > -delta,
/// 1 for an empty mask.
fn stability(logits: &[f32]) -> f32 {
    let inner = logits.iter().filter(|&&v| v > STABILITY_DELTA).count();
    let outer = logits.iter().filter(|&&v| v > -STABILITY_DELTA).count();
    if outer > 0 {
        inner as f32 / outer as f32
    } else {
        1.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn s0_keeps_the_aspect_on_a_long_edge_of_1024() {
        assert_eq!(s0_size(6000, 4000), (1024, 683));
        assert_eq!(s0_size(4000, 6000), (683, 1024));
        assert_eq!(s0_size(1024, 1024), (1024, 1024));
        assert_eq!(s0_size(100, 100_000), (1, 1024));
        assert_eq!(s0_size(1, 1), (1024, 1024));
    }

    #[test]
    fn the_soft_mask_of_a_flat_logit_is_flat_and_the_sigmoid_of_it() {
        let logits = vec![2.0f32; LOW_RES_PX * LOW_RES_PX];
        let m = soft_mask(&logits, 300, 200);
        assert_eq!((m.width, m.height), (1024, 683));
        let want = 1.0 / (1.0 + (-2.0f64).exp());
        assert!(m.data.iter().all(|&v| (f64::from(v) - want).abs() < 1e-6));
    }

    #[test]
    fn the_soft_mask_is_bilinear_between_cell_centres() {
        // Left half in, right half out: the middle of the picture is the middle of the two cells' values.
        let mut logits = vec![-10.0f32; LOW_RES_PX * LOW_RES_PX];
        for y in 0..LOW_RES_PX {
            for x in 0..LOW_RES_PX / 2 {
                logits[y * LOW_RES_PX + x] = 10.0;
            }
        }
        let m = soft_mask(&logits, 1024, 1024);
        let at = |x: usize| m.data[500 * 1024 + x];
        assert!(at(0) > 0.99 && at(1023) < 0.01);
        // The step is between source columns 127 and 128, i.e. output columns 510..514 of the 4-fold enlargement.
        assert!(at(505) > 0.99 && at(518) < 0.01);
        let mid = 0.5 * (kernels::sigmoid(10.0) + kernels::sigmoid(-10.0));
        assert!(
            (at(511) + at(512) - 2.0 * mid).abs() < 1e-5,
            "{} {}",
            at(511),
            at(512)
        );
        // Monotone across the edge.
        for x in 1..1024 {
            assert!(at(x) <= at(x - 1) + 1e-7);
        }
    }

    #[test]
    fn stability_is_one_for_a_decisive_mask() {
        let mut v = vec![-5.0f32; 100];
        for x in &mut v[..30] {
            *x = 5.0;
        }
        assert_eq!(stability(&v), 1.0);
        // Ten of 40 inside the band (-0.05, 0.05]: 30 above 0.05, 40 above -0.05.
        for x in &mut v[30..40] {
            *x = 0.0;
        }
        assert_eq!(stability(&v), 30.0 / 40.0);
        assert_eq!(stability(&[-1.0, -2.0]), 1.0);
    }

    /// A decoder answer whose chosen mask covers `inside` of the 65536 cells at logit > 0.
    fn answer(inside: usize, object_score: f32) -> Decoded {
        let mut logits = vec![-3.0f32; LOW_RES_PX * LOW_RES_PX];
        logits[..inside].fill(3.0);
        Decoded {
            logits,
            iou: vec![0.9],
            object_score,
            chosen: 0,
            multimask: false,
            token: 0,
        }
    }

    #[test]
    fn a_click_is_refused_for_a_score_not_above_zero_or_a_mask_under_a_twentieth_of_a_percent() {
        // 0.05 % of 65536 cells is 32.768: 33 cells are enough, 32 are not.
        let found = |d: Decoded| matches!(judge(d), Decoding::Found(_));
        assert!(found(answer(33, 0.01)));
        assert!(found(answer(30_000, 12.0)));
        for (inside, score, why) in [
            (32, 5.0, NothingReason::TooSmall),
            (0, 5.0, NothingReason::TooSmall),
            (30_000, 0.0, NothingReason::NotAnObject),
            (30_000, -4.0, NothingReason::NotAnObject),
            (30_000, f32::NAN, NothingReason::NotAnObject),
            (0, -1.0, NothingReason::NotAnObject),
        ] {
            match judge(answer(inside, score)) {
                Decoding::NothingFound { reason, .. } => {
                    assert_eq!(reason, why, "{inside} cells, score {score}")
                }
                Decoding::Found(_) => panic!("{inside} cells, score {score} should find nothing"),
            }
        }
        let d = answer(16_384, 3.0);
        assert_eq!(d.area(), 0.25);
        assert_eq!(d.low_res_logits().len(), LOW_RES_PX * LOW_RES_PX);
    }

    #[test]
    fn features_from_planar_arrays_are_checked_and_made_token_major() {
        let embed: Vec<f32> = (0..EMBED_DIM * EMBED_PX * EMBED_PX)
            .map(|i| i as f32)
            .collect();
        let hr0 = vec![0.5f32; 32 * 256 * 256];
        let hr1 = vec![0.25f32; 64 * 128 * 128];
        let f = Features::from_chw(&embed, &hr0, &hr1).expect("features");
        // Channel c of cell p was at c * 4096 + p, and is at p * 256 + c now.
        assert_eq!(f.image_embed()[3 * 256 + 7], embed[7 * 4096 + 3]);
        assert_eq!(f.bytes(), 4 * (embed.len() + hr0.len() + hr1.len()));
        assert!(matches!(
            Features::from_chw(&embed[1..], &hr0, &hr1),
            Err(MlError::Shape(_))
        ));
        let mut bad = hr1.clone();
        bad[5] = f32::NAN;
        assert!(matches!(
            Features::from_chw(&embed, &hr0, &bad),
            Err(MlError::Input(_))
        ));
    }

    #[test]
    fn the_model_can_be_shared_between_threads() {
        fn shared<T: Send + Sync>() {}
        shared::<Sam2>();
        shared::<Features>();
    }

    #[test]
    fn argmax_takes_the_first_of_equals() {
        assert_eq!(argmax(&[0.1, 0.9, 0.9]), 1);
        assert_eq!(argmax(&[3.0]), 0);
    }

    #[test]
    fn the_multimask_rule_is_one_positive_point() {
        assert!(Prompt::click(0.5, 0.5).multimask());
        assert!(
            !Prompt::click(0.5, 0.5)
                .with_point(0.2, 0.2, false)
                .multimask()
        );
        assert!(!Prompt::rect(0.1, 0.1, 0.9, 0.9).multimask());
        assert!(
            !Prompt {
                points: vec![Point {
                    x: 0.5,
                    y: 0.5,
                    positive: false
                }],
                bbox: None
            }
            .multimask()
        );
    }

    #[test]
    fn bad_prompts_and_pictures_are_errors_not_panics() {
        assert!(matches!(Picture::new(0, 10, &[]), Err(MlError::Input(_))));
        assert!(matches!(
            Picture::new(2, 2, &[0; 5]),
            Err(MlError::Shape(_))
        ));
        assert!(Prompt::default().validate().is_err());
        assert!(Prompt::click(f32::NAN, 0.5).validate().is_err());
        assert!(
            Prompt::rect(0.0, 0.0, f32::INFINITY, 1.0)
                .validate()
                .is_err()
        );
        let many = Prompt {
            points: vec![
                Point {
                    x: 0.5,
                    y: 0.5,
                    positive: true
                };
                MAX_POINTS + 1
            ],
            bbox: None,
        };
        assert!(many.validate().is_err());
    }
}
