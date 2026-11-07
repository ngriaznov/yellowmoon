//! The DINOv2 ViT-S/14 encoder of Depth Anything V2-Small: patch embedding, class token, positional embedding, 12 blocks and
//! the four tapped layers.
//!
//! Ported from `depth_anything_v2/dinov2.py` and `dinov2_layers/` of github.com/DepthAnything/Depth-Anything-V2 (Apache-2.0,
//! the code of Meta's DINOv2 as modified there). The line numbers quoted in the comments are those of that repository's
//! `main` branch read on 2026-11-07.

use rayon::prelude::*;

use super::ops::{add_layer_scaled, resize_bicubic_scaled};
use super::{Result, Tensors};
use crate::model::MlError;
use crate::sam2::kernels::{
    Act, HeadKv, Packed, Residual, attend, gemm, gemm_par, layer_norm, layer_norm_par,
};

/// `DINOv2("vits")` (`dinov2.py` line 339, `vit_small`): the width of a token.
pub(crate) const EMBED_DIM: usize = 384;
/// `depth=12` of `vit_small`.
pub(crate) const BLOCKS: usize = 12;
/// `num_heads=6` of `vit_small`.
pub(crate) const HEADS: usize = 6;
/// `mlp_ratio=4` (line 345): the hidden width of the MLP is `int(dim * mlp_ratio)`.
pub(crate) const MLP_DIM: usize = 4 * EMBED_DIM;
/// `patch_size=14` (`dinov2.py` line 408).
pub(crate) const PATCH_PX: usize = 14;
/// `img_size=518` (line 407): the positional embedding is trained on a 37 x 37 grid.
pub(crate) const POS_GRID: usize = 518 / PATCH_PX;
/// `norm_layer = partial(nn.LayerNorm, eps=1e-6)` (line 95).
const LN_EPS: f64 = 1e-6;
/// `interpolate_offset=0.1` (line 414).
const INTERPOLATE_OFFSET: f64 = 0.1;
/// `intermediate_layer_idx['vits']` (`dpt.py` line 165).
pub(crate) const TAPS: [usize; 4] = [2, 5, 8, 11];

/// A layer norm's weight and bias.
#[derive(Debug)]
struct Norm(Vec<f32>, Vec<f32>);

impl Norm {
    fn load(w: &mut Tensors, prefix: &str) -> Result<Self> {
        Ok(Self(
            w.take(&format!("{prefix}.weight"), &[EMBED_DIM])?,
            w.take(&format!("{prefix}.bias"), &[EMBED_DIM])?,
        ))
    }
}

/// One transformer block (`dinov2_layers/block.py` lines 36 to 107): `x = x + ls1(attn(norm1(x)))`, then
/// `x = x + ls2(mlp(norm2(x)))`.
#[derive(Debug)]
struct Block {
    norm1: Norm,
    qkv: Packed,
    proj: Packed,
    ls1: Vec<f32>,
    norm2: Norm,
    fc1: Packed,
    fc2: Packed,
    ls2: Vec<f32>,
}

impl Block {
    fn load(w: &mut Tensors, index: usize) -> Result<Self> {
        let p = format!("pretrained.blocks.{index}");
        Ok(Self {
            norm1: Norm::load(w, &format!("{p}.norm1"))?,
            qkv: w.linear(&format!("{p}.attn.qkv"), 3 * EMBED_DIM, EMBED_DIM)?,
            proj: w.linear(&format!("{p}.attn.proj"), EMBED_DIM, EMBED_DIM)?,
            ls1: w.take(&format!("{p}.ls1.gamma"), &[EMBED_DIM])?,
            norm2: Norm::load(w, &format!("{p}.norm2"))?,
            fc1: w.linear(&format!("{p}.mlp.fc1"), MLP_DIM, EMBED_DIM)?,
            fc2: w.linear(&format!("{p}.mlp.fc2"), EMBED_DIM, MLP_DIM)?,
            ls2: w.take(&format!("{p}.ls2.gamma"), &[EMBED_DIM])?,
        })
    }

    /// `x` holds `n` tokens of [`EMBED_DIM`] values; the block updates it in place.
    fn forward(&self, x: &mut [f32], n: usize) {
        let c = EMBED_DIM;
        // Attention (`attention.py` lines 49 to 62; the xFormers path of `MemEffAttention` is not taken on CPU).
        let mut n1 = vec![0.0f32; n * c];
        layer_norm_par(x, c, &self.norm1.0, &self.norm1.1, LN_EPS, &mut n1);
        let att = self.attention(&n1, n);
        drop(n1);
        let mut branch = vec![0.0f32; n * c];
        gemm_par(&att, n, &self.proj, Act::None, None, &mut branch);
        drop(att);
        add_layer_scaled(x, &branch, &self.ls1);
        drop(branch);
        // MLP (`mlp.py` lines 35 to 41), then the second LayerScale and residual, by tiles of rows.
        const ROWS: usize = 64;
        x.par_chunks_mut(ROWS * c).for_each_init(
            || (Vec::new(), Vec::new(), Vec::new()),
            |(norm, hidden, branch), chunk| {
                let rows = chunk.len() / c;
                norm.resize(rows * c, 0.0);
                layer_norm(chunk, c, &self.norm2.0, &self.norm2.1, LN_EPS, norm);
                hidden.resize(rows * MLP_DIM, 0.0);
                gemm(norm, c, rows, &self.fc1, Act::Gelu, None, hidden, MLP_DIM);
                branch.resize(rows * c, 0.0);
                gemm(hidden, MLP_DIM, rows, &self.fc2, Act::None, None, branch, c);
                for (xr, br) in chunk.chunks_exact_mut(c).zip(branch.chunks_exact(c)) {
                    for ((xv, &bv), &g) in xr.iter_mut().zip(br).zip(&self.ls2) {
                        *xv += bv * g;
                    }
                }
            },
        );
    }

    /// Global multi-head attention over all `n` tokens: `softmax(q k^T * head_dim^-0.5) v` per head, the heads
    /// concatenated (`qkv` is laid out `[token][3][head][head_dim]`, `attention.py` line 51).
    fn attention(&self, n1: &[f32], n: usize) -> Vec<f32> {
        let (c, heads) = (EMBED_DIM, HEADS);
        let hd = c / heads;
        // `self.scale = head_dim**-0.5` (line 42): 64 ** -0.5 = 0.125, a power of two, so scaling the queries before the
        // product (PyTorch) or the scores after it (our kernel) gives the same numbers.
        let scale = 1.0 / (hd as f32).sqrt();
        let mut qkv = vec![0.0f32; n * 3 * c];
        gemm_par(n1, n, &self.qkv, Act::None, None, &mut qkv);
        let kvs: Vec<HeadKv> = (0..heads)
            .into_par_iter()
            .map(|head| {
                let mut kv = HeadKv::default();
                kv.pack(
                    &qkv[c + head * hd..],
                    3 * c,
                    &qkv[2 * c + head * hd..],
                    3 * c,
                    n,
                    hd,
                );
                kv
            })
            .collect();
        const QUERIES_PER_TASK: usize = 16;
        let mut att = vec![0.0f32; n * c];
        att.par_chunks_mut(QUERIES_PER_TASK * c)
            .enumerate()
            .for_each_init(Vec::new, |scores, (i, chunk)| {
                let q0 = i * QUERIES_PER_TASK;
                let nq = chunk.len() / c;
                for (head, kv) in kvs.iter().enumerate() {
                    attend(
                        &qkv[q0 * 3 * c + head * hd..],
                        3 * c,
                        nq,
                        kv,
                        scale,
                        scores,
                        &mut chunk[head * hd..],
                        c,
                    );
                }
            });
        att
    }
}

/// What a traced run of the encoder keeps (for the parity tests): the stream is `[1 + ph * pw][384]`, the class token first.
#[derive(Debug, Default)]
pub struct EncoderTrace {
    /// The tokens after the patch embedding, the class token and the positional embedding.
    pub embedded: Vec<f32>,
    /// The stream after each of the 12 blocks.
    pub blocks: Vec<Vec<f32>>,
}

/// The encoder's weights.
#[derive(Debug)]
pub(crate) struct Encoder {
    /// The 14 x 14 stride-14 convolution as a product of 588 inputs (ky, kx, channel) and 384 outputs.
    patch: Packed,
    cls: Vec<f32>,
    /// `[1 + 37 * 37][384]`: the class token's embedding and the trained 37 x 37 grid.
    pos: Vec<f32>,
    blocks: Vec<Block>,
    norm: Norm,
}

impl Encoder {
    pub(crate) fn load(w: &mut Tensors) -> Result<Self> {
        let p = "pretrained";
        let k = PATCH_PX * PATCH_PX * 3;
        let conv = w.take(
            &format!("{p}.patch_embed.proj.weight"),
            &[EMBED_DIM, 3, PATCH_PX, PATCH_PX],
        )?;
        // PyTorch's [384][3][14][14] to the order (ky, kx, channel) of the gathered inputs.
        let mut kn = vec![0.0f32; k * EMBED_DIM];
        for o in 0..EMBED_DIM {
            for c in 0..3 {
                for ky in 0..PATCH_PX {
                    for kx in 0..PATCH_PX {
                        kn[((ky * PATCH_PX + kx) * 3 + c) * EMBED_DIM + o] =
                            conv[((o * 3 + c) * PATCH_PX + ky) * PATCH_PX + kx];
                    }
                }
            }
        }
        let bias = w.take(&format!("{p}.patch_embed.proj.bias"), &[EMBED_DIM])?;
        let patch = Packed::from_kn(&kn, k, EMBED_DIM, Some(&bias));
        let cls = w.take(&format!("{p}.cls_token"), &[1, 1, EMBED_DIM])?;
        let pos = w.take(
            &format!("{p}.pos_embed"),
            &[1, 1 + POS_GRID * POS_GRID, EMBED_DIM],
        )?;
        let blocks = (0..BLOCKS)
            .map(|i| Block::load(w, i))
            .collect::<Result<Vec<_>>>()?;
        let norm = Norm::load(w, &format!("{p}.norm"))?;
        Ok(Self {
            patch,
            cls,
            pos,
            blocks,
            norm,
        })
    }

    /// The positional embedding of a `ph` x `pw` patch grid, `[ph * pw][384]` (without the class token's), by
    /// `interpolate_pos_encoding` (`dinov2.py` lines 179 to 210).
    ///
    /// Line 183: `if npatch == N and w == h: return self.pos_embed`, so a 518 x 518 input (37 x 37 patches) uses the trained
    /// grid as it is. Otherwise lines 189 to 205: `w0 = w // patch_size + 0.1`, `h0 = h // patch_size + 0.1`
    /// (`interpolate_offset=0.1`), `sx, sy = float(w0) / sqrt_N, float(h0) / sqrt_N` with `sqrt_N = 37`, and
    /// `F.interpolate(grid, scale_factor=(sx, sy), mode="bicubic", antialias=False)`. Note that `w` and `h` of that code are
    /// `x.shape[2]` and `x.shape[3]` (line 213, `B, nc, w, h = x.shape`): the height and the width of the picture, so `sx`
    /// scales the rows and `sy` the columns of the grid. The output has `floor(37 * s)` = `ph` and `pw` positions (the
    /// asserts of lines 207 and 208).
    pub(crate) fn position_embedding(&self, ph: usize, pw: usize) -> Result<Vec<f32>> {
        let grid = &self.pos[EMBED_DIM..];
        if ph == POS_GRID && pw == POS_GRID {
            return Ok(grid.to_vec());
        }
        let scale = |patches: usize| (patches as f64 + INTERPOLATE_OFFSET) / POS_GRID as f64;
        let (out, oh, ow) = resize_bicubic_scaled(grid, POS_GRID, EMBED_DIM, scale(ph), scale(pw));
        if (oh, ow) != (ph, pw) {
            return Err(MlError::Shape(format!(
                "the positional embedding resizes to {oh} x {ow}, {ph} x {pw} expected"
            )));
        }
        Ok(out)
    }

    /// Runs the encoder on the normalised `[ph * 14][pw * 14][3]` picture and returns the four tapped layers, each the
    /// final norm of the stream after block 2, 5, 8 and 11 without the class token (`get_intermediate_layers`,
    /// `dinov2.py` lines 271 to 321 with `norm=True`, `return_class_token=True`; the DPT head ignores the class token
    /// when `use_clstoken` is false, `dpt.py` line 125). Each tap is `[ph * pw][384]`.
    pub(crate) fn forward(
        &self,
        input: &[f32],
        ph: usize,
        pw: usize,
        cancel: &dyn Fn() -> bool,
        mut trace: Option<&mut EncoderTrace>,
    ) -> Result<[Vec<f32>; 4]> {
        let c = EMBED_DIM;
        let width = pw * PATCH_PX;
        let n = ph * pw;
        let pos = self.position_embedding(ph, pw)?;
        // Tokens: the class token and then the patches (line 218), plus the positional embedding (line 219).
        let mut x = vec![0.0f32; (1 + n) * c];
        for ((d, &t), &p) in x[..c].iter_mut().zip(&self.cls).zip(&self.pos[..c]) {
            *d = t + p;
        }
        const K: usize = PATCH_PX * PATCH_PX * 3;
        x[c..]
            .par_chunks_mut(pw * c)
            .enumerate()
            .for_each_init(Vec::new, |cols, (py, row)| {
                // The gathered inputs of one row of patches: `[pw][14 * 14 * 3]`, rows of a patch (ky) of 14 * 3 values.
                cols.clear();
                cols.resize(pw * K, 0.0);
                for px in 0..pw {
                    for ky in 0..PATCH_PX {
                        let src = &input[((py * PATCH_PX + ky) * width + px * PATCH_PX) * 3..]
                            [..PATCH_PX * 3];
                        cols[px * K + ky * PATCH_PX * 3..][..PATCH_PX * 3].copy_from_slice(src);
                    }
                }
                let res = Residual {
                    data: &pos[py * pw * c..(py + 1) * pw * c],
                    ld: c,
                };
                gemm(cols, K, pw, &self.patch, Act::None, Some(res), row, c);
            });
        if let Some(t) = trace.as_deref_mut() {
            t.embedded = x.clone();
        }
        let mut taps: Vec<Vec<f32>> = Vec::with_capacity(4);
        for (i, block) in self.blocks.iter().enumerate() {
            if cancel() {
                return Err(MlError::Cancelled);
            }
            block.forward(&mut x, 1 + n);
            if let Some(t) = trace.as_deref_mut() {
                t.blocks.push(x.clone());
            }
            if TAPS.contains(&i) {
                // `self.norm(out)` then `out[:, 1:]`.
                let mut normed = vec![0.0f32; n * c];
                layer_norm_par(&x[c..], c, &self.norm.0, &self.norm.1, LN_EPS, &mut normed);
                taps.push(normed);
            }
        }
        <[Vec<f32>; 4]>::try_from(taps)
            .map_err(|_| MlError::Unsupported("an encoder without four taps".into()))
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;

    struct Lcg(u64);

    impl Lcg {
        fn next(&mut self) -> f32 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((self.0 >> 40) as f32) / (1u64 << 24) as f32 * 2.0 - 1.0
        }

        fn vec(&mut self, n: usize, amp: f32) -> Vec<f32> {
            (0..n).map(|_| self.next() * amp).collect()
        }
    }

    /// A block in `f64`, straight from the definition in `block.py` and `attention.py`.
    struct Reference {
        tensors: HashMap<String, (Vec<usize>, Vec<f32>)>,
    }

    impl Reference {
        fn t(&self, name: &str) -> Vec<f64> {
            self.tensors[&format!("pretrained.blocks.0.{name}")]
                .1
                .iter()
                .map(|&v| f64::from(v))
                .collect()
        }

        fn layer_norm(&self, x: &[f64], n: usize, which: &str) -> Vec<f64> {
            let (w, b) = (
                self.t(&format!("{which}.weight")),
                self.t(&format!("{which}.bias")),
            );
            let c = EMBED_DIM;
            let mut out = vec![0.0; n * c];
            for r in 0..n {
                let row = &x[r * c..(r + 1) * c];
                let mean = row.iter().sum::<f64>() / c as f64;
                let var = row.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / c as f64;
                for i in 0..c {
                    out[r * c + i] = (row[i] - mean) / (var + 1e-6).sqrt() * w[i] + b[i];
                }
            }
            out
        }

        fn linear(&self, x: &[f64], n: usize, name: &str, k: usize, m: usize) -> Vec<f64> {
            let (w, b) = (
                self.t(&format!("{name}.weight")),
                self.t(&format!("{name}.bias")),
            );
            let mut out = vec![0.0; n * m];
            for r in 0..n {
                for o in 0..m {
                    out[r * m + o] =
                        b[o] + (0..k).map(|i| x[r * k + i] * w[o * k + i]).sum::<f64>();
                }
            }
            out
        }

        fn forward(&self, x: &[f64], n: usize) -> Vec<f64> {
            let c = EMBED_DIM;
            let hd = c / HEADS;
            let n1 = self.layer_norm(x, n, "norm1");
            let qkv = self.linear(&n1, n, "attn.qkv", c, 3 * c);
            let mut att = vec![0.0; n * c];
            for head in 0..HEADS {
                for i in 0..n {
                    let q = &qkv[i * 3 * c + head * hd..][..hd];
                    let mut s: Vec<f64> = (0..n)
                        .map(|j| {
                            let k = &qkv[j * 3 * c + c + head * hd..][..hd];
                            (0..hd).map(|d| q[d] * 0.125 * k[d]).sum::<f64>()
                        })
                        .collect();
                    let max = s.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                    let sum: f64 = s.iter().map(|v| (v - max).exp()).sum();
                    for v in &mut s {
                        *v = (*v - max).exp() / sum;
                    }
                    for d in 0..hd {
                        att[i * c + head * hd + d] = (0..n)
                            .map(|j| s[j] * qkv[j * 3 * c + 2 * c + head * hd + d])
                            .sum();
                    }
                }
            }
            let proj = self.linear(&att, n, "attn.proj", c, c);
            let g1 = self.t("ls1.gamma");
            let x2: Vec<f64> = (0..n * c).map(|i| x[i] + proj[i] * g1[i % c]).collect();
            let n2 = self.layer_norm(&x2, n, "norm2");
            let h = self.linear(&n2, n, "mlp.fc1", c, MLP_DIM);
            let h: Vec<f64> = h
                .iter()
                .map(|&v| v * 0.5 * (1.0 + libm::erf(v / std::f64::consts::SQRT_2)))
                .collect();
            let m = self.linear(&h, n, "mlp.fc2", MLP_DIM, c);
            let g2 = self.t("ls2.gamma");
            (0..n * c).map(|i| x2[i] + m[i] * g2[i % c]).collect()
        }
    }

    #[test]
    fn a_block_matches_a_direct_evaluation_of_its_definition() {
        let mut rng = Lcg(77);
        let c = EMBED_DIM;
        let mut tensors: HashMap<String, (Vec<usize>, Vec<f32>)> = HashMap::new();
        let mut put = |name: &str, shape: &[usize], data: Vec<f32>| {
            tensors.insert(
                format!("pretrained.blocks.0.{name}"),
                (shape.to_vec(), data),
            );
        };
        for norm in ["norm1", "norm2"] {
            put(
                &format!("{norm}.weight"),
                &[c],
                rng.vec(c, 0.3).iter().map(|v| v + 1.0).collect(),
            );
            put(&format!("{norm}.bias"), &[c], rng.vec(c, 0.1));
        }
        put("attn.qkv.weight", &[3 * c, c], rng.vec(3 * c * c, 0.15));
        put("attn.qkv.bias", &[3 * c], rng.vec(3 * c, 0.1));
        put("attn.proj.weight", &[c, c], rng.vec(c * c, 0.1));
        put("attn.proj.bias", &[c], rng.vec(c, 0.1));
        put("ls1.gamma", &[c], rng.vec(c, 1.0));
        put("mlp.fc1.weight", &[MLP_DIM, c], rng.vec(MLP_DIM * c, 0.1));
        put("mlp.fc1.bias", &[MLP_DIM], rng.vec(MLP_DIM, 0.1));
        put("mlp.fc2.weight", &[c, MLP_DIM], rng.vec(c * MLP_DIM, 0.05));
        put("mlp.fc2.bias", &[c], rng.vec(c, 0.1));
        put("ls2.gamma", &[c], rng.vec(c, 1.0));
        let reference = Reference {
            tensors: tensors.clone(),
        };
        let mut w = Tensors::from_tensors(tensors);
        let block = Block::load(&mut w, 0).expect("loads");
        assert!(
            w.remaining().is_empty(),
            "every tensor of the block is read"
        );
        // 70 tokens: more than one tile of the attention queries and of the MLP rows is not needed to see an error, but the
        // tail tiles (70 = 4 * 16 + 6, 64 + 6) are exercised.
        for n in [1usize, 7, 70] {
            let x = rng.vec(n * c, 2.0);
            let want = reference.forward(&x.iter().map(|&v| f64::from(v)).collect::<Vec<_>>(), n);
            let mut got = x.clone();
            block.forward(&mut got, n);
            let worst = got
                .iter()
                .zip(&want)
                .map(|(&g, &w)| (f64::from(g) - w).abs())
                .fold(0.0, f64::max);
            let scale = want.iter().map(|v| v.abs()).fold(0.0, f64::max);
            assert!(
                worst < 2e-4 * scale.max(1.0),
                "n {n}: largest difference {worst:.2e} (values up to {scale:.1})"
            );
        }
    }
}
