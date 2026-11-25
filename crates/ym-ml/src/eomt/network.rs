//! The EoMT network: patch embedding, the 12 ViT-S blocks with the 200 queries entering before block 9, the layer norm, the class and
//! mask heads and the two upscaling blocks.
//!
//! Written from a reading of `modeling_eomt.py` of Hugging Face's `transformers` (Apache-2.0; a port of github.com/tue-mps/eomt,
//! MIT): `EomtEmbeddings`, `EomtLayer`, `EomtScaleLayer`, `EomtMaskHead` and `EomtForUniversalSegmentation.forward` and `.predict`.
//! The block is the ViT-S block of `crate::depth::encoder` (the same arithmetic, in a fixed order), loaded from this model's names.
//!
//! **No attention mask.** The forward pass builds a "masked attention" for the three blocks that carry the queries only when
//! `attn_mask_probs[block] > 0`; this checkpoint's three values are exactly 0 (the mask was annealed away in training), so inference
//! attends to everything and makes no intermediate prediction. The loader refuses a file where one of them is not 0.

use rayon::prelude::*;

use crate::depth::encoder::{Block, LN_EPS, Norm};
use crate::depth::ops::TConv;
use crate::model::MlError;
use crate::sam2::Weights as Tensors;
use crate::sam2::kernels::{Act, Packed, Residual, gelu, gemm, gemm_par, layer_norm_par};

use super::{CLASSES, Result};

/// The width of a token.
pub(crate) const C: usize = 384;
/// Heads of the attention.
pub(crate) const HEADS: usize = 6;
/// The hidden width of an MLP.
pub(crate) const MLP: usize = 4 * C;
/// The blocks of the ViT.
pub(crate) const BLOCKS: usize = 12;
/// The side of a patch, in pixels (`patch_size` of the configuration; the kernel of the patch convolution is 16 x 16).
pub(crate) const PATCH_PX: usize = 16;
/// The side of the patch grid: 640 / 16 (the position table has 40 x 40 rows).
pub(crate) const GRID: usize = 40;
/// The side of the input, in pixels.
pub(crate) const IMAGE_PX: usize = PATCH_PX * GRID;
/// Register tokens after the class token.
pub(crate) const REGISTERS: usize = 4;
/// Learned queries.
pub(crate) const QUERIES: usize = 200;
/// The queries enter before this block (`num_hidden_layers - num_blocks`).
pub(crate) const QUERY_FROM: usize = 9;
/// The outputs of the class head: the classes and "none".
pub(crate) const CLASS_OUTPUTS: usize = CLASSES + 1;
/// The side of the mask logits: the grid doubled by each of the two upscaling blocks (160).
pub const MASK_PX: usize = 4 * GRID;
/// Tokens before the patches in the stream: the class token and the registers.
const PREFIX: usize = 1 + REGISTERS;
/// `nn.LayerNorm(..., eps=config.layer_norm_eps)` and `EomtLayerNorm2d`'s default.
const EPS: f64 = LN_EPS;

/// One upscaling block (`EomtScaleLayer`): a 2 x 2 stride 2 transposed convolution, GELU, a depthwise 3 x 3 convolution without
/// bias and a layer norm over the channels.
#[derive(Debug)]
struct Upscale {
    conv1: TConv,
    /// The depthwise kernel as `[ky * 3 + kx][channel]`.
    taps: Vec<f32>,
    ln: Norm,
}

impl Upscale {
    fn load(w: &mut Tensors, index: usize) -> Result<Self> {
        let p = format!("upscale_block.block.{index}");
        let depthwise = w.take(&format!("{p}.conv2.weight"), &[C, 1, 3, 3])?;
        let mut taps = vec![0.0f32; 9 * C];
        for c in 0..C {
            for t in 0..9 {
                taps[t * C + c] = depthwise[c * 9 + t];
            }
        }
        Ok(Self {
            conv1: w.tconv(&format!("{p}.conv1"), C, 2)?,
            taps,
            ln: norm(w, &format!("{p}.layernorm2d"))?,
        })
    }

    /// The map `[2h][2w][384]` of an `h` x `w` map.
    fn forward(&self, x: &[f32], h: usize, w: usize) -> Vec<f32> {
        let mut y = self.conv1.forward(x, h, w);
        y.par_chunks_mut(256 * C).for_each(|chunk| {
            for v in chunk.iter_mut() {
                *v = gelu(*v);
            }
        });
        let (oh, ow) = (2 * h, 2 * w);
        let mut conv = vec![0.0f32; oh * ow * C];
        conv.par_chunks_mut(ow * C)
            .enumerate()
            .for_each(|(oy, row)| {
                for (ox, acc) in row.chunks_exact_mut(C).enumerate() {
                    // The taps in the order of the kernel (ky, kx) from the zero start; padding 1 skips the taps outside.
                    for ky in 0..3 {
                        let Some(sy) = (oy + ky).checked_sub(1).filter(|&v| v < oh) else {
                            continue;
                        };
                        for kx in 0..3 {
                            let Some(sx) = (ox + kx).checked_sub(1).filter(|&v| v < ow) else {
                                continue;
                            };
                            let src = &y[(sy * ow + sx) * C..][..C];
                            let tap = &self.taps[(ky * 3 + kx) * C..][..C];
                            for ((a, &s), &t) in acc.iter_mut().zip(src).zip(tap) {
                                *a += s * t;
                            }
                        }
                    }
                }
            });
        drop(y);
        let mut out = vec![0.0f32; oh * ow * C];
        layer_norm_par(&conv, C, &self.ln.0, &self.ln.1, EPS, &mut out);
        out
    }
}

/// `{prefix}.weight` and `{prefix}.bias` of a layer norm.
fn norm(w: &mut Tensors, prefix: &str) -> Result<Norm> {
    Ok(Norm(
        w.take(&format!("{prefix}.weight"), &[C])?,
        w.take(&format!("{prefix}.bias"), &[C])?,
    ))
}

/// One transformer block from the names of `EomtLayer`; the queries, keys and values are stacked into one product.
fn load_block(w: &mut Tensors, index: usize) -> Result<Block> {
    let p = format!("layers.{index}");
    let mut qkv_w = Vec::with_capacity(3 * C * C);
    let mut qkv_b = Vec::with_capacity(3 * C);
    for proj in ["q_proj", "k_proj", "v_proj"] {
        qkv_w.extend(w.take(&format!("{p}.attention.{proj}.weight"), &[C, C])?);
        qkv_b.extend(w.take(&format!("{p}.attention.{proj}.bias"), &[C])?);
    }
    Ok(Block::from_parts(
        norm(w, &format!("{p}.norm1"))?,
        Packed::from_nk(&qkv_w, 3 * C, C, Some(&qkv_b)),
        w.linear(&format!("{p}.attention.out_proj"), C, C)?,
        w.take(&format!("{p}.layer_scale1.lambda1"), &[C])?,
        norm(w, &format!("{p}.norm2"))?,
        w.linear(&format!("{p}.mlp.fc1"), MLP, C)?,
        w.linear(&format!("{p}.mlp.fc2"), C, MLP)?,
        w.take(&format!("{p}.layer_scale2.lambda1"), &[C])?,
    ))
}

/// What the network returns, before any post-processing.
#[derive(Debug, Clone, PartialEq)]
pub struct Raw {
    /// `[200][134]`: the class logits of each query (133 classes and "none").
    pub class_logits: Vec<f32>,
    /// `[200][160][160]`: the mask logits of each query, a quarter of the input's resolution.
    pub mask_logits: Vec<f32>,
}

/// The network's weights.
#[derive(Debug)]
pub(crate) struct Network {
    /// The 16 x 16 stride-16 convolution as a product of 768 inputs (ky, kx, channel) and 384 outputs.
    patch: Packed,
    cls: Vec<f32>,
    registers: Vec<f32>,
    /// `[1600][384]`: the trained 40 x 40 grid (the class and register tokens get none).
    pos: Vec<f32>,
    query: Vec<f32>,
    blocks: Vec<Block>,
    norm: Norm,
    class_head: Packed,
    mask_head: [Packed; 3],
    up: [Upscale; 2],
}

impl Network {
    pub(crate) fn load(w: &mut Tensors) -> Result<Self> {
        let k = PATCH_PX * PATCH_PX * 3;
        let conv = w.take(
            "embeddings.patch_embeddings.projection.weight",
            &[C, 3, PATCH_PX, PATCH_PX],
        )?;
        // PyTorch's [384][3][16][16] to the order (ky, kx, channel) of the gathered inputs.
        let mut kn = vec![0.0f32; k * C];
        for o in 0..C {
            for c in 0..3 {
                for ky in 0..PATCH_PX {
                    for kx in 0..PATCH_PX {
                        kn[((ky * PATCH_PX + kx) * 3 + c) * C + o] =
                            conv[((o * 3 + c) * PATCH_PX + ky) * PATCH_PX + kx];
                    }
                }
            }
        }
        let bias = w.take("embeddings.patch_embeddings.projection.bias", &[C])?;
        let mask_probs = w.take("attn_mask_probs", &[3])?;
        if mask_probs.iter().any(|&p| p != 0.0) {
            return Err(MlError::Unsupported(
                "an EoMT file whose attn_mask_probs are not 0 (masked attention at inference is not implemented)".into(),
            ));
        }
        Ok(Self {
            patch: Packed::from_kn(&kn, k, C, Some(&bias)),
            cls: w.take("embeddings.cls_token", &[1, 1, C])?,
            registers: w.take("embeddings.register_tokens", &[1, REGISTERS, C])?,
            pos: w.take("embeddings.position_embeddings.weight", &[GRID * GRID, C])?,
            query: w.take("query.weight", &[QUERIES, C])?,
            blocks: (0..BLOCKS)
                .map(|i| load_block(w, i))
                .collect::<Result<Vec<_>>>()?,
            norm: norm(w, "layernorm")?,
            class_head: w.linear("class_predictor", CLASS_OUTPUTS, C)?,
            mask_head: [
                w.linear("mask_head.fc1", C, C)?,
                w.linear("mask_head.fc2", C, C)?,
                w.linear("mask_head.fc3", C, C)?,
            ],
            up: [Upscale::load(w, 0)?, Upscale::load(w, 1)?],
        })
    }

    /// Runs the network on the normalised `[640][640][3]` input.
    pub(crate) fn forward(&self, input: &[f32], cancel: &dyn Fn() -> bool) -> Result<Raw> {
        debug_assert_eq!(input.len(), IMAGE_PX * IMAGE_PX * 3);
        let n = GRID * GRID;
        // Tokens: the class token, the registers and then the patches with their positions (`EomtEmbeddings.forward`).
        let mut x = vec![0.0f32; (PREFIX + n) * C];
        x[..C].copy_from_slice(&self.cls);
        x[C..PREFIX * C].copy_from_slice(&self.registers);
        const K: usize = PATCH_PX * PATCH_PX * 3;
        x[PREFIX * C..]
            .par_chunks_mut(GRID * C)
            .enumerate()
            .for_each_init(Vec::new, |cols, (py, row)| {
                // The gathered inputs of one row of patches: `[40][16 * 16 * 3]`, rows of a patch (ky) of 16 * 3 values.
                cols.clear();
                cols.resize(GRID * K, 0.0);
                for px in 0..GRID {
                    for ky in 0..PATCH_PX {
                        let src = &input[((py * PATCH_PX + ky) * IMAGE_PX + px * PATCH_PX) * 3..]
                            [..PATCH_PX * 3];
                        cols[px * K + ky * PATCH_PX * 3..][..PATCH_PX * 3].copy_from_slice(src);
                    }
                }
                let res = Residual {
                    data: &self.pos[py * GRID * C..(py + 1) * GRID * C],
                    ld: C,
                };
                gemm(cols, K, GRID, &self.patch, Act::None, Some(res), row, C);
            });
        for block in &self.blocks[..QUERY_FROM] {
            if cancel() {
                return Err(MlError::Cancelled);
            }
            block.forward(&mut x, PREFIX + n);
        }
        // The queries are put in front of the stream (`torch.cat((query, hidden_states), dim=1)`).
        let tokens = QUERIES + PREFIX + n;
        let mut y = Vec::with_capacity(tokens * C);
        y.extend_from_slice(&self.query);
        y.append(&mut x);
        for block in &self.blocks[QUERY_FROM..] {
            if cancel() {
                return Err(MlError::Cancelled);
            }
            block.forward(&mut y, tokens);
        }
        let mut seq = vec![0.0f32; tokens * C];
        layer_norm_par(&y, C, &self.norm.0, &self.norm.1, EPS, &mut seq);
        drop(y);
        if cancel() {
            return Err(MlError::Cancelled);
        }
        // `predict`: the class head and the mask head on the queries, the upscaling blocks on the patches.
        let queries = &seq[..QUERIES * C];
        let mut class_logits = vec![0.0f32; QUERIES * CLASS_OUTPUTS];
        gemm_par(
            queries,
            QUERIES,
            &self.class_head,
            Act::None,
            None,
            &mut class_logits,
        );
        let mut h1 = vec![0.0f32; QUERIES * C];
        let mut h2 = vec![0.0f32; QUERIES * C];
        gemm_par(
            queries,
            QUERIES,
            &self.mask_head[0],
            Act::Gelu,
            None,
            &mut h1,
        );
        gemm_par(&h1, QUERIES, &self.mask_head[1], Act::Gelu, None, &mut h2);
        gemm_par(&h2, QUERIES, &self.mask_head[2], Act::None, None, &mut h1);
        let query_masks = Packed::from_nk(&h1, QUERIES, C, None);
        let mut map = self.up[0].forward(&seq[(QUERIES + PREFIX) * C..], GRID, GRID);
        map = self.up[1].forward(&map, 2 * GRID, 2 * GRID);
        drop(seq);
        if cancel() {
            return Err(MlError::Cancelled);
        }
        // `einsum("bqc, bchw -> bqhw")`: position by position, then one plane per query.
        let positions = MASK_PX * MASK_PX;
        let mut product = vec![0.0f32; positions * QUERIES];
        gemm_par(&map, positions, &query_masks, Act::None, None, &mut product);
        drop(map);
        let mut mask_logits = vec![0.0f32; QUERIES * positions];
        mask_logits
            .par_chunks_mut(positions)
            .enumerate()
            .for_each(|(q, plane)| {
                for (p, v) in plane.iter_mut().enumerate() {
                    *v = product[p * QUERIES + q];
                }
            });
        Ok(Raw {
            class_logits,
            mask_logits,
        })
    }
}

/// The multiply-adds of one run: the products and convolutions, not the attention softmax, the layer norms, the activations or
/// the biases.
pub fn multiply_adds() -> u64 {
    let c = C as u64;
    let n = (GRID * GRID) as u64;
    let block = |t: u64| t * (c * 3 * c + c * c + 2 * c * MLP as u64) + 2 * t * t * c;
    let early = (PREFIX as u64 + n, QUERY_FROM as u64);
    let late = (
        QUERIES as u64 + PREFIX as u64 + n,
        (BLOCKS - QUERY_FROM) as u64,
    );
    let patch = n * (PATCH_PX * PATCH_PX * 3) as u64 * c;
    let transformer = early.1 * block(early.0) + late.1 * block(late.0);
    let heads = QUERIES as u64 * c * CLASS_OUTPUTS as u64 + 3 * QUERIES as u64 * c * c;
    // Transposed convolutions: every input position makes a 2 x 2 x 384 block; the depthwise 3 x 3 on the doubled maps.
    let up = n * c * 4 * c + 4 * n * c * 4 * c + 9 * c * (4 * n + 16 * n);
    let einsum = QUERIES as u64 * c * (MASK_PX * MASK_PX) as u64;
    patch + transformer + heads + up + einsum
}
