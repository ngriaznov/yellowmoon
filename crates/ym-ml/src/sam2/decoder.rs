//! The prompt encoder and the mask decoder (`docs/algorithms/06-masks.md` section 6.1), written from the description of
//! Meta's `prompt_encoder.py`, `transformer.py` and `mask_decoder.py` (Apache-2.0) and checked against their output.
//!
//! A prompt becomes tokens: a random-Fourier positional encoding of each point (`(x + 0.5) / 1024` of the 1024-pixel
//! frame) plus a learned embedding of its kind (negative, positive, box corner), and always one padding token. The
//! decoder's own tokens (object score, IoU, four masks) and the prompt's go through a two-layer two-way transformer
//! against the 64 x 64 image embedding (plus the dense "no mask" embedding), whose output is upscaled twice with the two
//! high-resolution maps into 256 x 256 x 32 features; each mask token's hypernetwork gives 32 weights that turn the
//! features into that token's mask logits.

use rayon::prelude::*;

use super::kernels::{
    Act, HeadKv, Packed, add_into, attend, gelu, gemm, gemm_par, layer_norm, layer_norm_par,
    sigmoid,
};
use super::{EMBED_DIM, EMBED_PX, IMAGE_PX, LOW_RES_PX, MlError, Result, Weights};

/// Tokens that are not the prompt's: object score, IoU, and the four mask tokens.
const OWN_TOKENS: usize = 6;
/// Mask tokens (one for a single mask and three for the multimask output).
pub(crate) const MASK_TOKENS: usize = 4;
/// Layer norm epsilons: the transformer's `nn.LayerNorm` default, and `LayerNorm2d`'s.
const EPS_TRANSFORMER: f64 = 1e-5;
const EPS_2D: f64 = 1e-6;
/// `2 * pi` rounded to `f32`, the scale PyTorch applies to the projected coordinates.
const TWO_PI: f32 = std::f32::consts::TAU;

/// The kinds of a prompt token, `point_labels` of Meta's code.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Kind {
    /// Not a point: the padding token (`-1`).
    Padding,
    /// A point on what is not the object (`0`).
    Negative,
    /// A point on the object (`1`).
    Positive,
    /// A box's top-left corner (`2`).
    TopLeft,
    /// A box's bottom-right corner (`3`).
    BottomRight,
}

/// The prompt encoder.
#[derive(Debug)]
pub(crate) struct PromptEncoder {
    /// `[2][128]`.
    gaussian: Vec<f32>,
    /// Negative, positive, top-left and bottom-right embeddings.
    point_embed: [Vec<f32>; 4],
    not_a_point: Vec<f32>,
    no_mask: Vec<f32>,
    /// The positional encoding of the 64 x 64 grid as `[4096][256]`.
    dense_pe: Vec<f32>,
}

impl PromptEncoder {
    pub(crate) fn load(w: &mut Weights) -> Result<Self> {
        let p = "sam_prompt_encoder";
        let gaussian = w.take(
            &format!("{p}.pe_layer.positional_encoding_gaussian_matrix"),
            &[2, EMBED_DIM / 2],
        )?;
        let point = |w: &mut Weights, i: usize| {
            w.take(&format!("{p}.point_embeddings.{i}.weight"), &[1, EMBED_DIM])
        };
        let point_embed = [point(w, 0)?, point(w, 1)?, point(w, 2)?, point(w, 3)?];
        let not_a_point = w.take(&format!("{p}.not_a_point_embed.weight"), &[1, EMBED_DIM])?;
        let no_mask = w.take(&format!("{p}.no_mask_embed.weight"), &[1, EMBED_DIM])?;
        // The mask-input path is not used (a mask is a function of the points and the box alone): its tensors are
        // checked to be there and dropped.
        for (name, shape) in [
            ("mask_downscaling.0.weight", &[4usize, 1, 2, 2][..]),
            ("mask_downscaling.0.bias", &[4]),
            ("mask_downscaling.1.weight", &[4]),
            ("mask_downscaling.1.bias", &[4]),
            ("mask_downscaling.3.weight", &[16, 4, 2, 2]),
            ("mask_downscaling.3.bias", &[16]),
            ("mask_downscaling.4.weight", &[16]),
            ("mask_downscaling.4.bias", &[16]),
            ("mask_downscaling.6.weight", &[256, 16, 1, 1]),
            ("mask_downscaling.6.bias", &[256]),
        ] {
            w.take(&format!("{p}.{name}"), shape)?;
        }
        let mut enc = Self {
            gaussian,
            point_embed,
            not_a_point,
            no_mask,
            dense_pe: Vec::new(),
        };
        // `PositionEmbeddingRandom.forward((64, 64))`: the cell centres of the grid, `(i + 0.5) / 64`.
        let mut pe = vec![0.0f32; EMBED_PX * EMBED_PX * EMBED_DIM];
        for y in 0..EMBED_PX {
            for x in 0..EMBED_PX {
                let (fx, fy) = (
                    (x as f32 + 0.5) / EMBED_PX as f32,
                    (y as f32 + 0.5) / EMBED_PX as f32,
                );
                enc.fourier(
                    fx,
                    fy,
                    &mut pe[(y * EMBED_PX + x) * EMBED_DIM..][..EMBED_DIM],
                );
            }
        }
        enc.dense_pe = pe;
        Ok(enc)
    }

    /// `PositionEmbeddingRandom._pe_encoding` of a point given as fractions of the frame: `2c - 1`, the projection on the
    /// Gaussian matrix, times `2 pi`, then the sines and the cosines.
    fn fourier(&self, fx: f32, fy: f32, out: &mut [f32]) {
        let half = EMBED_DIM / 2;
        let (cx, cy) = (2.0 * fx - 1.0, 2.0 * fy - 1.0);
        for j in 0..half {
            let v = (cx * self.gaussian[j] + cy * self.gaussian[half + j]) * TWO_PI;
            out[j] = libm::sinf(v);
            out[half + j] = libm::cosf(v);
        }
    }

    /// The sparse embedding of a prompt: one token per entry of `tokens` (a position in the 1024-pixel frame and a kind),
    /// `[n][256]`.
    pub(crate) fn sparse(&self, tokens: &[((f32, f32), Kind)]) -> Vec<f32> {
        let mut out = vec![0.0f32; tokens.len() * EMBED_DIM];
        for (i, &((x, y), kind)) in tokens.iter().enumerate() {
            let row = &mut out[i * EMBED_DIM..(i + 1) * EMBED_DIM];
            if kind == Kind::Padding {
                row.copy_from_slice(&self.not_a_point);
                continue;
            }
            // `points + 0.5` (the centre of the pixel), over the frame's size.
            self.fourier(
                (x + 0.5) / IMAGE_PX as f32,
                (y + 0.5) / IMAGE_PX as f32,
                row,
            );
            let embed = match kind {
                Kind::Negative => &self.point_embed[0],
                Kind::Positive => &self.point_embed[1],
                Kind::TopLeft => &self.point_embed[2],
                _ => &self.point_embed[3],
            };
            for (r, &e) in row.iter_mut().zip(embed) {
                *r += e;
            }
        }
        out
    }
}

/// One attention layer of the decoder: projections to `internal` values, `heads` heads, and back to 256.
#[derive(Debug)]
struct Attn {
    q: Packed,
    k: Packed,
    v: Packed,
    out: Packed,
    heads: usize,
}

impl Attn {
    fn load(w: &mut Weights, prefix: &str, internal: usize) -> Result<Self> {
        Ok(Self {
            q: w.linear(&format!("{prefix}.q_proj"), internal, EMBED_DIM)?,
            k: w.linear(&format!("{prefix}.k_proj"), internal, EMBED_DIM)?,
            v: w.linear(&format!("{prefix}.v_proj"), internal, EMBED_DIM)?,
            out: w.linear(&format!("{prefix}.out_proj"), EMBED_DIM, internal)?,
            heads: 8,
        })
    }

    fn internal(&self) -> usize {
        self.q.n
    }

    fn scale(&self) -> f32 {
        1.0 / ((self.internal() / self.heads) as f32).sqrt()
    }

    /// Attention among the few tokens: `n_q` queries against `n_k` keys and values, all on the calling thread.
    fn tokens(&self, q_in: &[f32], n_q: usize, k_in: &[f32], v_in: &[f32], n_k: usize) -> Vec<f32> {
        let d = self.internal();
        let (q, k, v) = (
            linear(q_in, n_q, &self.q),
            linear(k_in, n_k, &self.k),
            linear(v_in, n_k, &self.v),
        );
        let hd = d / self.heads;
        let mut att = vec![0.0f32; n_q * d];
        let (mut kv, mut scores) = (HeadKv::default(), Vec::new());
        for head in 0..self.heads {
            kv.pack(&k[head * hd..], d, &v[head * hd..], d, n_k, hd);
            attend(
                &q[head * hd..],
                d,
                n_q,
                &kv,
                self.scale(),
                &mut scores,
                &mut att[head * hd..],
                d,
            );
        }
        linear(&att, n_q, &self.out)
    }

    /// Tokens attending to the image: `n_q` queries, the image as `k_in` (keys, plus the positional encoding) and `v_in`.
    fn token_to_image(
        &self,
        q_in: &[f32],
        n_q: usize,
        k_in: &[f32],
        v_in: &[f32],
        n_k: usize,
    ) -> Vec<f32> {
        let d = self.internal();
        let hd = d / self.heads;
        let q = linear(q_in, n_q, &self.q);
        let (k, v) = (linear(k_in, n_k, &self.k), linear(v_in, n_k, &self.v));
        let per_head: Vec<Vec<f32>> = (0..self.heads)
            .into_par_iter()
            .map(|head| {
                let mut kv = HeadKv::default();
                kv.pack(&k[head * hd..], d, &v[head * hd..], d, n_k, hd);
                let mut out = vec![0.0f32; n_q * hd];
                attend(
                    &q[head * hd..],
                    d,
                    n_q,
                    &kv,
                    self.scale(),
                    &mut Vec::new(),
                    &mut out,
                    hd,
                );
                out
            })
            .collect();
        let mut att = vec![0.0f32; n_q * d];
        for (head, o) in per_head.iter().enumerate() {
            for r in 0..n_q {
                att[r * d + head * hd..][..hd].copy_from_slice(&o[r * hd..(r + 1) * hd]);
            }
        }
        linear(&att, n_q, &self.out)
    }

    /// The image attending to the tokens: `n_q` image queries (`q_in`), `n_k` token keys and values.
    fn image_to_token(
        &self,
        q_in: &[f32],
        n_q: usize,
        k_in: &[f32],
        v_in: &[f32],
        n_k: usize,
    ) -> Vec<f32> {
        let d = self.internal();
        let hd = d / self.heads;
        let q = linear(q_in, n_q, &self.q);
        let (k, v) = (linear(k_in, n_k, &self.k), linear(v_in, n_k, &self.v));
        let kvs: Vec<HeadKv> = (0..self.heads)
            .map(|head| {
                let mut kv = HeadKv::default();
                kv.pack(&k[head * hd..], d, &v[head * hd..], d, n_k, hd);
                kv
            })
            .collect();
        const ROWS: usize = 64;
        let mut att = vec![0.0f32; n_q * d];
        att.par_chunks_mut(ROWS * d)
            .enumerate()
            .for_each_init(Vec::new, |scores, (i, chunk)| {
                let rows = chunk.len() / d;
                for (head, kv) in kvs.iter().enumerate() {
                    attend(
                        &q[i * ROWS * d + head * hd..],
                        d,
                        rows,
                        kv,
                        self.scale(),
                        scores,
                        &mut chunk[head * hd..],
                        d,
                    );
                }
            });
        drop(q);
        linear(&att, n_q, &self.out)
    }
}

/// `x . w + bias` for `m` rows, over the pool when there are many.
fn linear(x: &[f32], m: usize, w: &Packed) -> Vec<f32> {
    let mut out = vec![0.0f32; m * w.n];
    if m >= 256 {
        gemm_par(x, m, w, Act::None, None, &mut out);
    } else {
        gemm(x, w.k, m, w, Act::None, None, &mut out, w.n);
    }
    out
}

/// A small MLP of `Packed` layers with ReLU between them (the last layer plain).
#[derive(Debug)]
struct Mlp(Vec<Packed>);

impl Mlp {
    fn load(w: &mut Weights, prefix: &str, dims: &[usize]) -> Result<Self> {
        let layers = dims
            .windows(2)
            .enumerate()
            .map(|(i, d)| w.linear(&format!("{prefix}.layers.{i}"), d[1], d[0]))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self(layers))
    }

    fn forward(&self, x: &[f32], m: usize) -> Vec<f32> {
        let mut cur = x.to_vec();
        for (i, layer) in self.0.iter().enumerate() {
            let mut out = vec![0.0f32; m * layer.n];
            let act = if i + 1 < self.0.len() {
                Act::Relu
            } else {
                Act::None
            };
            gemm(&cur, layer.k, m, layer, act, None, &mut out, layer.n);
            cur = out;
        }
        cur
    }
}

#[derive(Debug)]
struct Norm {
    weight: Vec<f32>,
    bias: Vec<f32>,
}

impl Norm {
    fn load(w: &mut Weights, prefix: &str, c: usize) -> Result<Self> {
        Ok(Self {
            weight: w.take(&format!("{prefix}.weight"), &[c])?,
            bias: w.take(&format!("{prefix}.bias"), &[c])?,
        })
    }

    fn apply(&self, x: &[f32], eps: f64) -> Vec<f32> {
        let mut out = vec![0.0f32; x.len()];
        if x.len() >= 256 * EMBED_DIM {
            layer_norm_par(
                x,
                self.weight.len(),
                &self.weight,
                &self.bias,
                eps,
                &mut out,
            );
        } else {
            layer_norm(
                x,
                self.weight.len(),
                &self.weight,
                &self.bias,
                eps,
                &mut out,
            );
        }
        out
    }
}

#[derive(Debug)]
struct Layer {
    self_attn: Attn,
    norm1: Norm,
    token_to_image: Attn,
    norm2: Norm,
    mlp: Mlp,
    norm3: Norm,
    norm4: Norm,
    image_to_token: Attn,
}

/// The decoder's output before the choice of a mask: the logits of the four mask tokens, their IoU predictions and the
/// object score.
pub(crate) struct Raw {
    /// `[4][256 * 256]`, not clamped.
    pub logits: Vec<f32>,
    /// The four predicted IoUs (sigmoid).
    pub iou: [f32; MASK_TOKENS],
    /// The object score logit.
    pub object_score: f32,
}

/// The two-way transformer, the upscaling and the heads.
#[derive(Debug)]
pub(crate) struct MaskDecoder {
    obj_token: Vec<f32>,
    iou_token: Vec<f32>,
    mask_tokens: Vec<f32>,
    layers: Vec<Layer>,
    final_attn: Attn,
    norm_final: Norm,
    /// `ConvTranspose2d(256, 64, 2, 2)` as a product: 256 inputs, outputs `(dy, dx, channel)`.
    dc1: Packed,
    ln1: Norm,
    /// `ConvTranspose2d(64, 32, 2, 2)`: 64 inputs, outputs `(dy, dx, channel)`.
    dc2: Packed,
    hypernetworks: Vec<Mlp>,
    iou_head: Mlp,
    object_head: Mlp,
}

impl MaskDecoder {
    pub(crate) fn load(w: &mut Weights) -> Result<Self> {
        let p = "sam_mask_decoder";
        let mut layers = Vec::new();
        for i in 0..2 {
            let l = format!("{p}.transformer.layers.{i}");
            layers.push(Layer {
                self_attn: Attn::load(w, &format!("{l}.self_attn"), EMBED_DIM)?,
                norm1: Norm::load(w, &format!("{l}.norm1"), EMBED_DIM)?,
                token_to_image: Attn::load(
                    w,
                    &format!("{l}.cross_attn_token_to_image"),
                    EMBED_DIM / 2,
                )?,
                norm2: Norm::load(w, &format!("{l}.norm2"), EMBED_DIM)?,
                mlp: Mlp::load(w, &format!("{l}.mlp"), &[EMBED_DIM, 2048, EMBED_DIM])?,
                norm3: Norm::load(w, &format!("{l}.norm3"), EMBED_DIM)?,
                norm4: Norm::load(w, &format!("{l}.norm4"), EMBED_DIM)?,
                image_to_token: Attn::load(
                    w,
                    &format!("{l}.cross_attn_image_to_token"),
                    EMBED_DIM / 2,
                )?,
            });
        }
        // ConvTranspose2d weights are [in][out][2][2]; as a product the outputs are (dy, dx, out) and the bias repeats.
        let transposed =
            |w: &mut Weights, prefix: &str, cin: usize, cout: usize| -> Result<Packed> {
                let weight = w.take(&format!("{prefix}.weight"), &[cin, cout, 2, 2])?;
                let bias = w.take(&format!("{prefix}.bias"), &[cout])?;
                let mut kn = vec![0.0f32; cin * 4 * cout];
                for i in 0..cin {
                    for o in 0..cout {
                        for dy in 0..2 {
                            for dx in 0..2 {
                                kn[i * 4 * cout + (dy * 2 + dx) * cout + o] =
                                    weight[((i * cout + o) * 2 + dy) * 2 + dx];
                            }
                        }
                    }
                }
                let bias4: Vec<f32> = (0..4 * cout).map(|n| bias[n % cout]).collect();
                Ok(Packed::from_kn(&kn, cin, 4 * cout, Some(&bias4)))
            };
        let dc1 = transposed(
            w,
            &format!("{p}.output_upscaling.0"),
            EMBED_DIM,
            EMBED_DIM / 4,
        )?;
        let ln1 = Norm::load(w, &format!("{p}.output_upscaling.1"), EMBED_DIM / 4)?;
        let dc2 = transposed(
            w,
            &format!("{p}.output_upscaling.3"),
            EMBED_DIM / 4,
            EMBED_DIM / 8,
        )?;
        let mut hypernetworks = Vec::new();
        for i in 0..MASK_TOKENS {
            hypernetworks.push(Mlp::load(
                w,
                &format!("{p}.output_hypernetworks_mlps.{i}"),
                &[EMBED_DIM, EMBED_DIM, EMBED_DIM, EMBED_DIM / 8],
            )?);
        }
        Ok(Self {
            obj_token: w.take(&format!("{p}.obj_score_token.weight"), &[1, EMBED_DIM])?,
            iou_token: w.take(&format!("{p}.iou_token.weight"), &[1, EMBED_DIM])?,
            mask_tokens: w.take(
                &format!("{p}.mask_tokens.weight"),
                &[MASK_TOKENS, EMBED_DIM],
            )?,
            layers,
            final_attn: Attn::load(
                w,
                &format!("{p}.transformer.final_attn_token_to_image"),
                EMBED_DIM / 2,
            )?,
            norm_final: Norm::load(w, &format!("{p}.transformer.norm_final_attn"), EMBED_DIM)?,
            dc1,
            ln1,
            dc2,
            hypernetworks,
            iou_head: Mlp::load(
                w,
                &format!("{p}.iou_prediction_head"),
                &[EMBED_DIM, EMBED_DIM, EMBED_DIM, MASK_TOKENS],
            )?,
            object_head: Mlp::load(
                w,
                &format!("{p}.pred_obj_score_head"),
                &[EMBED_DIM, EMBED_DIM, EMBED_DIM, 1],
            )?,
        })
    }

    /// Runs the decoder: `sparse` is the prompt encoder's `[n][256]`, `prompt` holds the image embedding, and the two
    /// high-resolution maps. `dense_pe` is the encoder's positional grid and `no_mask` its dense embedding.
    pub(crate) fn run(
        &self,
        image_embed: &[f32],
        high_res_0: &[f32],
        high_res_1: &[f32],
        sparse: &[f32],
        encoder: &PromptEncoder,
    ) -> Raw {
        let n_sparse = sparse.len() / EMBED_DIM;
        let n = OWN_TOKENS + n_sparse;
        let n_img = EMBED_PX * EMBED_PX;
        // The tokens: object score, IoU, four masks, then the prompt's.
        let mut tokens = Vec::with_capacity(n * EMBED_DIM);
        tokens.extend_from_slice(&self.obj_token);
        tokens.extend_from_slice(&self.iou_token);
        tokens.extend_from_slice(&self.mask_tokens);
        tokens.extend_from_slice(sparse);
        // The image plus the dense embedding of "no mask".
        let mut keys = vec![0.0f32; n_img * EMBED_DIM];
        keys.par_chunks_mut(EMBED_DIM)
            .zip(image_embed.par_chunks(EMBED_DIM))
            .for_each(|(k, i)| add_into(i, &encoder.no_mask, k));
        let pe = &encoder.dense_pe;
        let add_pe = |a: &[f32]| -> Vec<f32> {
            let mut out = vec![0.0f32; a.len()];
            out.par_chunks_mut(EMBED_PX * EMBED_DIM)
                .zip(a.par_chunks(EMBED_PX * EMBED_DIM))
                .zip(pe.par_chunks(EMBED_PX * EMBED_DIM))
                .for_each(|((o, a), p)| add_into(a, p, o));
            out
        };
        let sum = |a: &[f32], b: &[f32]| -> Vec<f32> {
            let mut out = vec![0.0f32; a.len()];
            add_into(a, b, &mut out);
            out
        };

        let mut queries = tokens.clone();
        for (li, layer) in self.layers.iter().enumerate() {
            // Self-attention of the tokens: the first layer without the positional step and without the skip.
            if li == 0 {
                queries = layer.self_attn.tokens(&queries, n, &queries, &queries, n);
            } else {
                let q = sum(&queries, &tokens);
                let a = layer.self_attn.tokens(&q, n, &q, &queries, n);
                queries = sum(&queries, &a);
            }
            queries = layer.norm1.apply(&queries, EPS_TRANSFORMER);
            // Tokens attend to the image.
            let q_in = sum(&queries, &tokens);
            let k_in = add_pe(&keys);
            let a = layer
                .token_to_image
                .token_to_image(&q_in, n, &k_in, &keys, n_img);
            queries = layer.norm2.apply(&sum(&queries, &a), EPS_TRANSFORMER);
            // The MLP of the tokens.
            let m = layer.mlp.forward(&queries, n);
            queries = layer.norm3.apply(&sum(&queries, &m), EPS_TRANSFORMER);
            // The image attends to the tokens.
            let q_in = sum(&queries, &tokens);
            let a = layer
                .image_to_token
                .image_to_token(&k_in, n_img, &q_in, &queries, n);
            let mut next = vec![0.0f32; keys.len()];
            next.par_chunks_mut(EMBED_DIM)
                .zip(keys.par_chunks(EMBED_DIM))
                .zip(a.par_chunks(EMBED_DIM))
                .for_each(|((o, k), a)| add_into(k, a, o));
            keys = layer.norm4.apply(&next, EPS_TRANSFORMER);
        }
        // The last attention of the tokens to the image.
        let q_in = sum(&queries, &tokens);
        let k_in = add_pe(&keys);
        let a = self
            .final_attn
            .token_to_image(&q_in, n, &k_in, &keys, n_img);
        queries = self.norm_final.apply(&sum(&queries, &a), EPS_TRANSFORMER);

        let features = self.upscale(&keys, high_res_0, high_res_1);
        // One hypernetwork per mask token.
        let mut hyper = vec![0.0f32; MASK_TOKENS * (EMBED_DIM / 8)];
        for (m, net) in self.hypernetworks.iter().enumerate() {
            let token = &queries[(2 + m) * EMBED_DIM..(3 + m) * EMBED_DIM];
            hyper[m * (EMBED_DIM / 8)..(m + 1) * (EMBED_DIM / 8)]
                .copy_from_slice(&net.forward(token, 1));
        }
        // masks[m][pixel] = sum over the 32 features, in order.
        let weights = Packed::from_nk(&hyper, MASK_TOKENS, EMBED_DIM / 8, None);
        let mut per_pixel = vec![0.0f32; LOW_RES_PX * LOW_RES_PX * MASK_TOKENS];
        gemm_par(
            &features,
            LOW_RES_PX * LOW_RES_PX,
            &weights,
            Act::None,
            None,
            &mut per_pixel,
        );
        let mut logits = vec![0.0f32; MASK_TOKENS * LOW_RES_PX * LOW_RES_PX];
        logits
            .par_chunks_mut(LOW_RES_PX * LOW_RES_PX)
            .enumerate()
            .for_each(|(m, plane)| {
                for (p, v) in plane.iter_mut().enumerate() {
                    *v = per_pixel[p * MASK_TOKENS + m];
                }
            });
        let iou_logits = self.iou_head.forward(&queries[EMBED_DIM..2 * EMBED_DIM], 1);
        let mut iou = [0.0f32; MASK_TOKENS];
        for (o, &l) in iou.iter_mut().zip(&iou_logits) {
            *o = sigmoid(l);
        }
        let object_score = self.object_head.forward(&queries[..EMBED_DIM], 1)[0];
        Raw {
            logits,
            iou,
            object_score,
        }
    }

    /// The two transposed convolutions with the high-resolution maps: `[256 * 256][32]`.
    fn upscale(&self, src: &[f32], high_res_0: &[f32], high_res_1: &[f32]) -> Vec<f32> {
        let (c1, c2) = (EMBED_DIM / 4, EMBED_DIM / 8);
        let side1 = EMBED_PX * 2;
        // 64 x 64 x 256 -> 128 x 128 x 64: + the stride-8 map, layer norm over the channels, GELU.
        let mut act1 = vec![0.0f32; side1 * side1 * c1];
        act1.par_chunks_mut(2 * side1 * c1)
            .enumerate()
            .for_each_init(Vec::new, |tmp, (y, rows)| {
                tmp.resize(EMBED_PX * 4 * c1, 0.0);
                gemm(
                    &src[y * EMBED_PX * EMBED_DIM..],
                    EMBED_DIM,
                    EMBED_PX,
                    &self.dc1,
                    Act::None,
                    None,
                    tmp,
                    4 * c1,
                );
                let mut px = vec![0.0f32; c1];
                for x in 0..EMBED_PX {
                    for dy in 0..2 {
                        for dx in 0..2 {
                            let from = &tmp[x * 4 * c1 + (dy * 2 + dx) * c1..][..c1];
                            let skip =
                                &high_res_1[((2 * y + dy) * side1 + 2 * x + dx) * c1..][..c1];
                            add_into(from, skip, &mut px);
                            let dst = &mut rows[(dy * side1 + 2 * x + dx) * c1..][..c1];
                            layer_norm(&px, c1, &self.ln1.weight, &self.ln1.bias, EPS_2D, dst);
                            for v in dst.iter_mut() {
                                *v = gelu(*v);
                            }
                        }
                    }
                }
            });
        // 128 x 128 x 64 -> 256 x 256 x 32: + the stride-4 map, GELU.
        let side2 = LOW_RES_PX;
        let mut out = vec![0.0f32; side2 * side2 * c2];
        out.par_chunks_mut(2 * side2 * c2)
            .enumerate()
            .for_each_init(Vec::new, |tmp, (y, rows)| {
                tmp.resize(side1 * 4 * c2, 0.0);
                gemm(
                    &act1[y * side1 * c1..],
                    c1,
                    side1,
                    &self.dc2,
                    Act::None,
                    None,
                    tmp,
                    4 * c2,
                );
                for x in 0..side1 {
                    for dy in 0..2 {
                        for dx in 0..2 {
                            let from = &tmp[x * 4 * c2 + (dy * 2 + dx) * c2..][..c2];
                            let skip =
                                &high_res_0[((2 * y + dy) * side2 + 2 * x + dx) * c2..][..c2];
                            let dst = &mut rows[(dy * side2 + 2 * x + dx) * c2..][..c2];
                            for ((d, &f), &s) in dst.iter_mut().zip(from).zip(skip) {
                                *d = gelu(f + s);
                            }
                        }
                    }
                }
            });
        out
    }
}

/// Checks a decoder output for values that cannot be turned into a mask.
pub(crate) fn check_finite(raw: &Raw) -> Result<()> {
    if raw.object_score.is_finite()
        && raw.iou.iter().all(|v| v.is_finite())
        && raw.logits.iter().all(|v| v.is_finite())
    {
        Ok(())
    } else {
        Err(MlError::Input("the decoder produced a value that is not finite (the features or the prompt are damaged)".into()))
    }
}
