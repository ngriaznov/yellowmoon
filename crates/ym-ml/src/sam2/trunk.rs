//! The image encoder: Hiera-S and the FPN neck (`docs/algorithms/06-masks.md` section 6.1), written from the
//! description of Meta's `hieradet.py` and `image_encoder.py` (Apache-2.0) and checked against their output.
//!
//! The trunk reads the `[1024][1024][3]` input through a 7 x 7 stride-4 patch embedding (+ the interpolated
//! positional embedding), then 16 blocks. Block `i` is: layer norm; a skip that is the input itself or, where the
//! width grows, a linear map of the normalised input (max-pooled 2 x 2 where the block pools); attention (global, or
//! within windows of the picture padded with zeros, the queries max-pooled 2 x 2 where the block pools); the output
//! projection and the skip; layer norm, a GELU MLP of ratio 4 and its skip. The maps after blocks 0, 2, 13 and 15
//! (strides 4, 8, 16, 32) feed the neck, whose 1 x 1 convolutions bring them to 256 channels and add the
//! upsampled coarser map on the two coarsest levels.

use rayon::prelude::*;

use super::kernels::{
    Act, HeadKv, Packed, Residual, add_into, attend, gemm, gemm_par, layer_norm, layer_norm_par,
};
use super::{IMAGE_PX, MlError, Result, Weights};

/// Hiera-S: the embedding width of the first stage.
pub(crate) const EMBED_DIM0: usize = 96;
/// Patch embedding kernel, stride and padding.
const PATCH_K: usize = 7;
const PATCH_STRIDE: usize = 4;
const PATCH_PAD: usize = 3;
/// Positional embedding: the background grid and the window grid.
const POS_BKG_PX: usize = 7;
const POS_WINDOW_PX: usize = 8;
/// Layer norm epsilon of the trunk.
const EPS: f64 = 1e-6;
/// The blocks after which the trunk hands a map to the neck.
const STAGE_ENDS: [usize; 4] = [0, 2, 13, 15];

/// What one of the 16 blocks does (the table Meta's configuration gives for Hiera-S).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct BlockSpec {
    /// Input width.
    pub dim_in: usize,
    /// Output width.
    pub dim_out: usize,
    /// Attention heads (of `dim_out / heads = 96` values each).
    pub heads: usize,
    /// Window of the attention in tokens, 0 for global attention.
    pub window: usize,
    /// Whether the queries and the skip are max-pooled 2 x 2 (the block halves the map).
    pub pool: bool,
}

const fn spec(dim_in: usize, dim_out: usize, heads: usize, window: usize, pool: bool) -> BlockSpec {
    BlockSpec {
        dim_in,
        dim_out,
        heads,
        window,
        pool,
    }
}

/// The 16 blocks of Hiera-S: stages of 1, 2, 11 and 2 blocks, widths 96, 192, 384, 768, windows 8, 4, 14 and 7
/// (a stage's first block keeps the window of the stage before), global attention in blocks 7, 10 and 13.
pub(crate) const BLOCKS: [BlockSpec; 16] = [
    spec(96, 96, 1, 8, false),
    spec(96, 192, 2, 8, true),
    spec(192, 192, 2, 4, false),
    spec(192, 384, 4, 4, true),
    spec(384, 384, 4, 14, false),
    spec(384, 384, 4, 14, false),
    spec(384, 384, 4, 14, false),
    spec(384, 384, 4, 0, false),
    spec(384, 384, 4, 14, false),
    spec(384, 384, 4, 14, false),
    spec(384, 384, 4, 0, false),
    spec(384, 384, 4, 14, false),
    spec(384, 384, 4, 14, false),
    spec(384, 384, 4, 0, false),
    spec(384, 768, 8, 14, true),
    spec(768, 768, 8, 7, false),
];

/// A map of `h` x `w` tokens of `c` channels, row-major with the channels of a token together.
#[derive(Debug, Clone)]
pub(crate) struct Grid {
    pub h: usize,
    pub w: usize,
    pub c: usize,
    pub data: Vec<f32>,
}

impl Grid {
    fn zeros(h: usize, w: usize, c: usize) -> Self {
        Self {
            h,
            w,
            c,
            data: vec![0.0; h * w * c],
        }
    }
}

/// One block's weights.
#[derive(Debug)]
pub(crate) struct Block {
    spec: BlockSpec,
    norm1: (Vec<f32>, Vec<f32>),
    norm2: (Vec<f32>, Vec<f32>),
    /// `dim_in` -> `3 * dim_out`; the output channels are `[q | k | v]`, each `[head][96]`.
    qkv: Packed,
    attn_proj: Packed,
    /// `dim_in` -> `dim_out` where the width grows.
    skip: Option<Packed>,
    fc1: Packed,
    fc2: Packed,
}

/// Scratch buffers of a window task.
#[derive(Default)]
struct WindowScratch {
    xin: Vec<f32>,
    qkv: Vec<f32>,
    att: Vec<f32>,
    qpool: Vec<f32>,
    scores: Vec<f32>,
    kv: HeadKv,
}

impl Block {
    fn load(w: &mut Weights, index: usize, spec: BlockSpec) -> Result<Self> {
        let p = format!("image_encoder.trunk.blocks.{index}");
        let (d_in, d_out) = (spec.dim_in, spec.dim_out);
        Ok(Self {
            spec,
            norm1: (
                w.take(&format!("{p}.norm1.weight"), &[d_in])?,
                w.take(&format!("{p}.norm1.bias"), &[d_in])?,
            ),
            norm2: (
                w.take(&format!("{p}.norm2.weight"), &[d_out])?,
                w.take(&format!("{p}.norm2.bias"), &[d_out])?,
            ),
            qkv: w.linear(&format!("{p}.attn.qkv"), 3 * d_out, d_in)?,
            attn_proj: w.linear(&format!("{p}.attn.proj"), d_out, d_out)?,
            skip: if d_in == d_out {
                None
            } else {
                Some(w.linear(&format!("{p}.proj"), d_out, d_in)?)
            },
            fc1: w.linear(&format!("{p}.mlp.layers.0"), 4 * d_out, d_out)?,
            fc2: w.linear(&format!("{p}.mlp.layers.1"), d_out, 4 * d_out)?,
        })
    }

    /// The block's output for the map `x` (`dim_in` channels): the map halved in each direction where the block pools.
    fn forward(&self, x: &Grid) -> Grid {
        let s = &self.spec;
        debug_assert_eq!(x.c, s.dim_in);
        let mut n1 = Grid::zeros(x.h, x.w, x.c);
        layer_norm_par(
            &x.data,
            x.c,
            &self.norm1.0,
            &self.norm1.1,
            EPS,
            &mut n1.data,
        );
        let skipped = self.skip.as_ref().map(|skip| self.skip_map(skip, &n1));
        let shortcut = skipped.as_ref().unwrap_or(x);
        let x2 = if s.window > 0 {
            self.window_branch(&n1, shortcut)
        } else {
            self.global_branch(&n1, shortcut)
        };
        drop(n1);
        self.mlp(&x2)
    }

    /// The skip of a block that widens: the linear map of the normalised input, max-pooled 2 x 2 where the block pools.
    fn skip_map(&self, skip: &Packed, n1: &Grid) -> Grid {
        let c = self.spec.dim_out;
        if !self.spec.pool {
            let mut out = Grid::zeros(n1.h, n1.w, c);
            gemm_par(&n1.data, n1.h * n1.w, skip, Act::None, None, &mut out.data);
            return out;
        }
        let (ho, wo) = (n1.h / 2, n1.w / 2);
        let mut out = Grid::zeros(ho, wo, c);
        out.data
            .par_chunks_mut(wo * c)
            .enumerate()
            .for_each_init(Vec::new, |tmp, (y, row)| {
                // Two input rows at once, then the maximum of each 2 x 2 group.
                tmp.resize(2 * n1.w * c, 0.0);
                gemm(
                    &n1.data[2 * y * n1.w * n1.c..],
                    n1.c,
                    2 * n1.w,
                    skip,
                    Act::None,
                    None,
                    tmp,
                    c,
                );
                for x in 0..wo {
                    for ch in 0..c {
                        let at = |dy: usize, dx: usize| tmp[(dy * n1.w + 2 * x + dx) * c + ch];
                        row[x * c + ch] = max4(at(0, 0), at(0, 1), at(1, 0), at(1, 1));
                    }
                }
            });
        out
    }

    /// Attention within windows, the output projection and the skip. The picture is padded with zeros to a whole
    /// number of windows (the padding takes part in the attention as keys, as in Meta's code), the queries are pooled
    /// where the block pools, and the result is cut back to the skip's size.
    fn window_branch(&self, n1: &Grid, shortcut: &Grid) -> Grid {
        let s = &self.spec;
        let (ws, c, c_in, heads) = (s.window, s.dim_out, s.dim_in, s.heads);
        let hd = c / heads;
        let (h, w) = (n1.h, n1.w);
        let (nwy, nwx) = (h.div_ceil(ws), w.div_ceil(ws));
        let tok_in = ws * ws;
        let wo = if s.pool { ws / 2 } else { ws };
        let tok_out = wo * wo;
        let scale = 1.0 / (hd as f32).sqrt();
        // Windows per task: a few hundred tokens for the products.
        let group = (128 / tok_in).max(1);
        let mut windowed = vec![0.0f32; nwy * nwx * tok_out * c];
        windowed
            .par_chunks_mut(group * tok_out * c)
            .enumerate()
            .for_each_init(WindowScratch::default, |sc, (gi, chunk)| {
                let w0 = gi * group;
                let g = chunk.len() / (tok_out * c);
                // The windows' tokens, zeros beyond the picture.
                sc.xin.clear();
                sc.xin.resize(g * tok_in * c_in, 0.0);
                for j in 0..g {
                    let (wy, wx) = ((w0 + j) / nwx, (w0 + j) % nwx);
                    let cols = ws.min(w - wx * ws);
                    for iy in 0..ws {
                        let y = wy * ws + iy;
                        if y >= h {
                            break;
                        }
                        let src = &n1.data[(y * w + wx * ws) * c_in..][..cols * c_in];
                        sc.xin[(j * tok_in + iy * ws) * c_in..][..cols * c_in].copy_from_slice(src);
                    }
                }
                sc.qkv.resize(g * tok_in * 3 * c, 0.0);
                gemm(
                    &sc.xin,
                    c_in,
                    g * tok_in,
                    &self.qkv,
                    Act::None,
                    None,
                    &mut sc.qkv,
                    3 * c,
                );
                sc.att.resize(g * tok_out * c, 0.0);
                let WindowScratch {
                    qkv,
                    att,
                    qpool,
                    scores,
                    kv,
                    ..
                } = sc;
                for j in 0..g {
                    let base = j * tok_in * 3 * c;
                    for head in 0..heads {
                        let (qo, ko, vo) = (
                            base + head * hd,
                            base + c + head * hd,
                            base + 2 * c + head * hd,
                        );
                        kv.pack(&qkv[ko..], 3 * c, &qkv[vo..], 3 * c, tok_in, hd);
                        let out = &mut att[j * tok_out * c + head * hd..];
                        if s.pool {
                            // Queries max-pooled 2 x 2 within the window, per channel.
                            qpool.resize(tok_out * hd, 0.0);
                            for qy in 0..wo {
                                for qx in 0..wo {
                                    for d in 0..hd {
                                        let at = |dy: usize, dx: usize| {
                                            qkv[qo + ((2 * qy + dy) * ws + 2 * qx + dx) * 3 * c + d]
                                        };
                                        qpool[(qy * wo + qx) * hd + d] =
                                            max4(at(0, 0), at(0, 1), at(1, 0), at(1, 1));
                                    }
                                }
                            }
                            attend(qpool, hd, tok_out, kv, scale, scores, out, c);
                        } else {
                            attend(&qkv[qo..], 3 * c, tok_in, kv, scale, scores, out, c);
                        }
                    }
                }
                gemm(
                    att,
                    c,
                    g * tok_out,
                    &self.attn_proj,
                    Act::None,
                    None,
                    chunk,
                    c,
                );
            });
        // Back from windows to the map, cropped to the skip's size, plus the skip.
        let mut x2 = Grid::zeros(shortcut.h, shortcut.w, c);
        let width = shortcut.w;
        x2.data
            .par_chunks_mut(width * c)
            .enumerate()
            .for_each(|(y, row)| {
                let (wy, iy) = (y / wo, y % wo);
                for x in 0..width {
                    let (wx, ix) = (x / wo, x % wo);
                    let src = &windowed[((wy * nwx + wx) * tok_out + iy * wo + ix) * c..][..c];
                    let skip = &shortcut.data[(y * width + x) * c..][..c];
                    add_into(skip, src, &mut row[x * c..(x + 1) * c]);
                }
            });
        x2
    }

    /// Global attention over the whole map, the output projection and the skip.
    fn global_branch(&self, n1: &Grid, shortcut: &Grid) -> Grid {
        let s = &self.spec;
        debug_assert!(!s.pool && s.dim_in == s.dim_out);
        let (c, heads) = (s.dim_out, s.heads);
        let hd = c / heads;
        let n = n1.h * n1.w;
        let scale = 1.0 / (hd as f32).sqrt();
        let mut qkv = vec![0.0f32; n * 3 * c];
        gemm_par(&n1.data, n, &self.qkv, Act::None, None, &mut qkv);
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
        drop(qkv);
        let mut x2 = Grid::zeros(shortcut.h, shortcut.w, c);
        gemm_par(
            &att,
            n,
            &self.attn_proj,
            Act::None,
            Some(Residual {
                data: &shortcut.data,
                ld: c,
            }),
            &mut x2.data,
        );
        x2
    }

    /// `x + fc2(gelu(fc1(norm2(x))))`.
    fn mlp(&self, x2: &Grid) -> Grid {
        let c = self.spec.dim_out;
        const ROWS: usize = 64;
        let mut out = Grid::zeros(x2.h, x2.w, c);
        out.data.par_chunks_mut(ROWS * c).enumerate().for_each_init(
            || (Vec::new(), Vec::new()),
            |(norm, hidden), (i, chunk)| {
                let rows = chunk.len() / c;
                let src = &x2.data[i * ROWS * c..][..rows * c];
                norm.resize(rows * c, 0.0);
                layer_norm(src, c, &self.norm2.0, &self.norm2.1, EPS, norm);
                hidden.resize(rows * 4 * c, 0.0);
                gemm(norm, c, rows, &self.fc1, Act::Gelu, None, hidden, 4 * c);
                gemm(
                    hidden,
                    4 * c,
                    rows,
                    &self.fc2,
                    Act::None,
                    Some(Residual { data: src, ld: c }),
                    chunk,
                    c,
                );
            },
        );
        out
    }
}

/// The maximum of four values (`MaxPool2d`'s answer for finite values).
#[inline(always)]
fn max4(a: f32, b: f32, c: f32, d: f32) -> f32 {
    let mut m = a;
    for v in [b, c, d] {
        if v > m {
            m = v;
        }
    }
    m
}

/// The trunk: patch embedding, positional embedding and the 16 blocks.
#[derive(Debug)]
pub(crate) struct Trunk {
    /// 7 x 7 x 3 -> 96, the inputs ordered `(ky, kx, channel)`.
    patch: Packed,
    /// `[256 * 256][96]`: the bicubic interpolation of the 7 x 7 background embedding plus the tiled 8 x 8 window embedding.
    pos: Vec<f32>,
    blocks: Vec<Block>,
}

/// The feature maps the trunk hands to the neck, finest first (strides 4, 8, 16 and 32).
pub(crate) struct Stages(pub [Grid; 4]);

impl Trunk {
    pub(crate) fn load(w: &mut Weights) -> Result<Self> {
        let p = "image_encoder.trunk";
        // PyTorch's [96][3][7][7] to the order (ky, kx, channel) of the gathered inputs.
        let conv = w.take(
            &format!("{p}.patch_embed.proj.weight"),
            &[EMBED_DIM0, 3, PATCH_K, PATCH_K],
        )?;
        let mut kn = vec![0.0f32; PATCH_K * PATCH_K * 3 * EMBED_DIM0];
        for o in 0..EMBED_DIM0 {
            for c in 0..3 {
                for ky in 0..PATCH_K {
                    for kx in 0..PATCH_K {
                        kn[((ky * PATCH_K + kx) * 3 + c) * EMBED_DIM0 + o] =
                            conv[((o * 3 + c) * PATCH_K + ky) * PATCH_K + kx];
                    }
                }
            }
        }
        let bias = w.take(&format!("{p}.patch_embed.proj.bias"), &[EMBED_DIM0])?;
        let patch = Packed::from_kn(&kn, PATCH_K * PATCH_K * 3, EMBED_DIM0, Some(&bias));
        let bkg = w.take(
            &format!("{p}.pos_embed"),
            &[1, EMBED_DIM0, POS_BKG_PX, POS_BKG_PX],
        )?;
        let window = w.take(
            &format!("{p}.pos_embed_window"),
            &[1, EMBED_DIM0, POS_WINDOW_PX, POS_WINDOW_PX],
        )?;
        let side = IMAGE_PX / PATCH_STRIDE;
        let pos = position_embedding(&bkg, &window, side);
        let blocks = BLOCKS
            .iter()
            .enumerate()
            .map(|(i, &s)| Block::load(w, i, s))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self { patch, pos, blocks })
    }

    /// Runs the trunk on the `[1024][1024][3]` input and returns the four stage maps.
    pub(crate) fn forward(&self, input: &[f32], cancel: &dyn Fn() -> bool) -> Result<Stages> {
        if input.len() != IMAGE_PX * IMAGE_PX * 3 {
            return Err(MlError::Shape(format!(
                "the encoder input holds {} values, {} expected (1024 x 1024 x 3)",
                input.len(),
                IMAGE_PX * IMAGE_PX * 3
            )));
        }
        let side = IMAGE_PX / PATCH_STRIDE;
        let mut x = Grid::zeros(side, side, EMBED_DIM0);
        x.data
            .par_chunks_mut(side * EMBED_DIM0)
            .enumerate()
            .for_each_init(Vec::new, |cols, (y, row)| {
                // The gathered inputs of one output row: `[side][7 * 7 * 3]`, zeros where the kernel is off the picture.
                const K: usize = PATCH_K * PATCH_K * 3;
                cols.clear();
                cols.resize(side * K, 0.0);
                for x in 0..side {
                    for ky in 0..PATCH_K {
                        let iy = (y * PATCH_STRIDE + ky) as isize - PATCH_PAD as isize;
                        if iy < 0 || iy >= IMAGE_PX as isize {
                            continue;
                        }
                        for kx in 0..PATCH_K {
                            let ix = (x * PATCH_STRIDE + kx) as isize - PATCH_PAD as isize;
                            if ix < 0 || ix >= IMAGE_PX as isize {
                                continue;
                            }
                            let src = &input[(iy as usize * IMAGE_PX + ix as usize) * 3..][..3];
                            cols[x * K + (ky * PATCH_K + kx) * 3..][..3].copy_from_slice(src);
                        }
                    }
                }
                let res = Residual {
                    data: &self.pos[y * side * EMBED_DIM0..(y + 1) * side * EMBED_DIM0],
                    ld: EMBED_DIM0,
                };
                gemm(
                    cols,
                    K,
                    side,
                    &self.patch,
                    Act::None,
                    Some(res),
                    row,
                    EMBED_DIM0,
                );
            });
        let mut maps: Vec<Grid> = Vec::with_capacity(4);
        for (i, block) in self.blocks.iter().enumerate() {
            // A block takes about a fifth of a second: the caller's stop is noticed between them.
            if cancel() {
                return Err(MlError::Cancelled);
            }
            x = block.forward(&x);
            if STAGE_ENDS.contains(&i) {
                maps.push(x.clone());
            }
        }
        let Ok(maps) = <[Grid; 4]>::try_from(maps) else {
            return Err(MlError::Unsupported(
                "a trunk with other than four stages".into(),
            ));
        };
        Ok(Stages(maps))
    }
}

/// PyTorch's cubic convolution coefficients for the fraction `t`, with `A = -0.75` (`get_cubic_upsample_coefficients`).
fn cubic_coefficients(t: f64) -> [f64; 4] {
    const A: f64 = -0.75;
    let conv1 = |x: f64| ((A + 2.0) * x - (A + 3.0)) * x * x + 1.0;
    let conv2 = |x: f64| ((A * x - 5.0 * A) * x + 8.0 * A) * x - 4.0 * A;
    [conv2(t + 1.0), conv1(t), conv1(1.0 - t), conv2(2.0 - t)]
}

/// The positional embedding of the trunk for a `side` x `side` map as `[side * side][96]`: the `bkg` embedding
/// (`[96][7][7]`) interpolated bicubically (`F.interpolate(..., mode="bicubic")`, `align_corners = False`, edge
/// indices clamped) plus the `window` embedding (`[96][8][8]`) tiled over the map.
pub(crate) fn position_embedding(bkg: &[f32], window: &[f32], side: usize) -> Vec<f32> {
    let n = POS_BKG_PX;
    let scale = n as f64 / side as f64;
    // Per output index: the four source indices and their weights.
    let axis: Vec<([usize; 4], [f64; 4])> = (0..side)
        .map(|o| {
            let real = scale * (o as f64 + 0.5) - 0.5;
            let base = real.floor();
            let coeff = cubic_coefficients(real - base);
            let idx = std::array::from_fn(|k| {
                (base as isize - 1 + k as isize).clamp(0, n as isize - 1) as usize
            });
            (idx, coeff)
        })
        .collect();
    let c = EMBED_DIM0;
    let mut out = vec![0.0f32; side * side * c];
    out.par_chunks_mut(side * c)
        .enumerate()
        .for_each(|(y, row)| {
            let (iy, wy) = &axis[y];
            for x in 0..side {
                let (ix, wx) = &axis[x];
                for ch in 0..c {
                    let mut acc = 0.0f64;
                    for (ky, &yy) in iy.iter().enumerate() {
                        let mut line = 0.0f64;
                        for (kx, &xx) in ix.iter().enumerate() {
                            line += wx[kx] * f64::from(bkg[(ch * n + yy) * n + xx]);
                        }
                        acc += wy[ky] * line;
                    }
                    let win = window[(ch * POS_WINDOW_PX + y % POS_WINDOW_PX) * POS_WINDOW_PX
                        + x % POS_WINDOW_PX];
                    row[x * c + ch] = acc as f32 + win;
                }
            }
        });
    out
}

/// The neck: four 1 x 1 convolutions to 256 channels with the top-down sum on the two coarsest levels, then the
/// decoder's `conv_s0` and `conv_s1` on the two finest, and `no_mem_embed` on the coarsest of the three kept.
#[derive(Debug)]
pub(crate) struct Neck {
    /// For the maps of 768, 384, 192 and 96 channels (coarsest first).
    convs: [Packed; 4],
    conv_s0: Packed,
    conv_s1: Packed,
    no_mem_embed: Vec<f32>,
}

/// What the encoder hands to the decoder (token-major, `[y * w + x][channel]`).
pub(crate) struct Maps {
    /// `[64 * 64][256]`, `no_mem_embed` added.
    pub image_embed: Vec<f32>,
    /// `[256 * 256][32]` after `conv_s0`.
    pub high_res_0: Vec<f32>,
    /// `[128 * 128][64]` after `conv_s1`.
    pub high_res_1: Vec<f32>,
}

impl Neck {
    pub(crate) fn load(w: &mut Weights) -> Result<Self> {
        let chans = [768usize, 384, 192, 96];
        let mut convs = Vec::with_capacity(4);
        for (j, &c) in chans.iter().enumerate() {
            convs.push(w.conv1x1(&format!("image_encoder.neck.convs.{j}.conv"), 256, c)?);
        }
        let Ok(convs) = <[Packed; 4]>::try_from(convs) else {
            return Err(MlError::Unsupported(
                "a neck with other than four levels".into(),
            ));
        };
        Ok(Self {
            convs,
            conv_s0: w.conv1x1("sam_mask_decoder.conv_s0", 32, 256)?,
            conv_s1: w.conv1x1("sam_mask_decoder.conv_s1", 64, 256)?,
            no_mem_embed: w.take("no_mem_embed", &[1, 1, 256])?,
        })
    }

    pub(crate) fn forward(&self, stages: &Stages) -> Maps {
        let [s0, s1, s2, s3] = &stages.0;
        // The coarsest level: 32 x 32.
        let mut l3 = vec![0.0f32; s3.h * s3.w * 256];
        gemm_par(
            &s3.data,
            s3.h * s3.w,
            &self.convs[0],
            Act::None,
            None,
            &mut l3,
        );
        // Stride 16: the lateral map plus the nearest-neighbour doubling of the coarser one, then `no_mem_embed`.
        let mut embed = vec![0.0f32; s2.h * s2.w * 256];
        gemm_par(
            &s2.data,
            s2.h * s2.w,
            &self.convs[1],
            Act::None,
            None,
            &mut embed,
        );
        let nm = &self.no_mem_embed;
        embed
            .par_chunks_mut(s2.w * 256)
            .enumerate()
            .for_each(|(y, row)| {
                for x in 0..s2.w {
                    let up = &l3[((y / 2) * s3.w + x / 2) * 256..][..256];
                    let px = &mut row[x * 256..(x + 1) * 256];
                    for ch in 0..256 {
                        px[ch] = (px[ch] + up[ch]) + nm[ch];
                    }
                }
            });
        // Strides 8 and 4: the lateral map only, straight on into the decoder's 1 x 1 convolution.
        let high_res_1 = self.lateral_then(&s1.data, s1.h * s1.w, &self.convs[2], &self.conv_s1);
        let high_res_0 = self.lateral_then(&s0.data, s0.h * s0.w, &self.convs[3], &self.conv_s0);
        Maps {
            image_embed: embed,
            high_res_0,
            high_res_1,
        }
    }

    /// `second(first(x))` for `m` tokens, a task of 64 tokens at a time so that the 256-channel map is never whole.
    fn lateral_then(&self, x: &[f32], m: usize, first: &Packed, second: &Packed) -> Vec<f32> {
        const ROWS: usize = 64;
        let mut out = vec![0.0f32; m * second.n];
        out.par_chunks_mut(ROWS * second.n)
            .enumerate()
            .for_each_init(Vec::new, |mid, (i, chunk)| {
                let rows = chunk.len() / second.n;
                mid.resize(rows * 256, 0.0);
                gemm(
                    &x[i * ROWS * first.k..],
                    first.k,
                    rows,
                    first,
                    Act::None,
                    None,
                    mid,
                    256,
                );
                gemm(mid, 256, rows, second, Act::None, None, chunk, second.n);
            });
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_block_table_is_hiera_s() {
        // 1, 2, 11 and 2 blocks; the width doubles and the map halves at blocks 1, 3 and 14 and only there.
        for (i, b) in BLOCKS.iter().enumerate() {
            assert_eq!(b.pool, [1, 3, 14].contains(&i), "block {i}");
            assert_eq!(b.dim_in != b.dim_out, b.pool, "block {i}");
            assert_eq!(b.window == 0, [7, 10, 13].contains(&i), "block {i}");
            assert_eq!(b.dim_out / b.heads, 96, "block {i}");
            assert_eq!(b.dim_out % b.heads, 0);
        }
        assert_eq!(BLOCKS[STAGE_ENDS[0]].dim_out, 96);
        assert_eq!(BLOCKS[STAGE_ENDS[1]].dim_out, 192);
        assert_eq!(BLOCKS[STAGE_ENDS[2]].dim_out, 384);
        assert_eq!(BLOCKS[STAGE_ENDS[3]].dim_out, 768);
    }

    #[test]
    fn the_cubic_kernel_is_pytorch_s() {
        // At t = 0 the kernel passes through the sample: weights (0, 1, 0, 0); they always sum to 1.
        let c = cubic_coefficients(0.0);
        assert!(
            c[0].abs() < 1e-12
                && (c[1] - 1.0).abs() < 1e-12
                && c[2].abs() < 1e-12
                && c[3].abs() < 1e-12,
            "{c:?}"
        );
        for t in [0.1, 0.25, 0.5, 0.9] {
            let c = cubic_coefficients(t);
            assert!((c.iter().sum::<f64>() - 1.0).abs() < 1e-12, "{t}");
        }
        // t = 0.5 with A = -0.75: the outer taps are -0.09375, the inner ones 0.59375.
        let c = cubic_coefficients(0.5);
        assert!(
            (c[0] + 0.09375).abs() < 1e-12 && (c[1] - 0.59375).abs() < 1e-12,
            "{c:?}"
        );
    }

    #[test]
    fn a_constant_background_embedding_stays_constant_and_the_window_embedding_tiles() {
        let bkg = vec![2.0f32; EMBED_DIM0 * POS_BKG_PX * POS_BKG_PX];
        let window: Vec<f32> = (0..EMBED_DIM0 * POS_WINDOW_PX * POS_WINDOW_PX)
            .map(|i| (i % 17) as f32 * 0.25)
            .collect();
        let pos = position_embedding(&bkg, &window, 32);
        for (y, x, ch) in [
            (0usize, 0usize, 0usize),
            (9, 17, 5),
            (31, 31, 95),
            (8, 8, 3),
        ] {
            let want = 2.0 + window[(ch * POS_WINDOW_PX + y % 8) * POS_WINDOW_PX + x % 8];
            assert!(
                (pos[(y * 32 + x) * EMBED_DIM0 + ch] - want).abs() < 1e-5,
                "{y} {x} {ch}"
            );
        }
    }
}
