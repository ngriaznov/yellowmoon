//! Pure-Rust CPU implementation of the `jdd` network (`docs/07-ml-models.md` sections Runtime and CPU
//! fallback): for machines with no usable GPU, for the headless CLI on CI runners, and as the reference
//! the optional burn and ort backends are compared with.
//!
//! The architecture is implemented exactly as in `training/jdd/model.py`: 3x3 convolutions with stride 1
//! or 2 (zero padding), SiLU, pixel shuffle, channel-concat skips and a bilinear-demosaic residual; the
//! weights come from a safetensors file whose header (`config`, `halo_half_px`, ...) describes the
//! architecture. Nothing here needs a GPU, a C library or a build script. Convolutions are parallelised
//! over output rows with rayon; on x86-64 an AVX2+FMA version of the inner loop is selected at run time.
//!
//! # Layout
//!
//! Internally tensors are `[row][channel][column]` (each row holds all channels contiguously), so a row
//! of the output is a contiguous chunk that one rayon task can own, and channel concatenation is a
//! per-row copy. The public API stays planar (`[channel][row][column]`, [`crate::model`]).
//!
//! # Tile independence
//!
//! Every output pixel is computed by the same sequence of floating-point operations whatever the tile
//! it belongs to, so tiled and whole-frame results differ only where a tile's zero padding reaches a
//! pixel, i.e. inside the halo that tiling crops away.

use std::path::Path;

use rayon::prelude::*;
use safetensors::{Dtype, SafeTensors};

use crate::model::{
    ExecContext, IN_CHANNELS, JddConfig, MlError, ModelId, OUT_CHANNELS, OUT_SCALE, Result,
    TILE_MULTIPLE, TileInputs, TileModel, TileOutput, TileShape, sha256_hex, verify_sha256,
};

/// Format string of the safetensors header written by `training/jdd/weights_io.py`.
pub const ARCH_NAME: &str = "ym-jdd";
/// Architecture version this runtime implements.
pub const ARCH_VERSION: &str = "1";
/// Default tile core (half-resolution pixels) used for the tile shape of a freshly loaded model.
pub const DEFAULT_CORE_HALF_PX: usize = 512;
/// Largest tile halo (half-resolution pixels) a weights header may declare (the starting layout uses 32).
pub const MAX_HALO_HALF_PX: usize = 1024;

/// Output channels handled together by the stride-2 (axpy) convolution kernel.
const OC_BLOCK: usize = 8;
/// Register tile of the stride-1 kernel: 4 output channels x 16 output columns of accumulators.
const TILE_OC: usize = 4;
const TILE_X: usize = 16;

/// Row-interleaved tensor: element `(y, c, x)` is at `(y * c_total + c) * w + x`.
#[derive(Clone, Debug)]
struct Tensor {
    c: usize,
    h: usize,
    w: usize,
    data: Vec<f32>,
}

impl Tensor {
    fn zeros(c: usize, h: usize, w: usize) -> Self {
        Self {
            c,
            h,
            w,
            data: vec![0.0; c * h * w],
        }
    }

    fn row_len(&self) -> usize {
        self.c * self.w
    }
}

/// One convolution layer with weights packed for the blocked kernel.
#[derive(Clone, Debug)]
struct Conv {
    oc: usize,
    ic: usize,
    k: usize,
    stride: usize,
    pad: usize,
    /// Stride 2: `[block][ky][kx][ic][8]` (output channels of a block innermost, zero padded to 8).
    packed: Vec<f32>,
    /// Stride 1: `[block][ky][kx][ic][4]`, zero padded to a multiple of 4 output channels.
    packed4: Vec<f32>,
    bias: Vec<f32>,
}

impl Conv {
    fn new(
        weight: &[f32],
        bias: Option<&[f32]>,
        oc: usize,
        ic: usize,
        k: usize,
        stride: usize,
    ) -> Self {
        let pack = |block: usize| {
            let mut out = vec![0.0f32; oc.div_ceil(block) * k * k * ic * block];
            for o in 0..oc {
                let (ob, oi) = (o / block, o % block);
                for i in 0..ic {
                    for ky in 0..k {
                        for kx in 0..k {
                            let src = ((o * ic + i) * k + ky) * k + kx;
                            out[((((ob * k + ky) * k + kx) * ic) + i) * block + oi] = weight[src];
                        }
                    }
                }
            }
            out
        };
        let (packed, packed4) = if stride == 1 {
            (Vec::new(), pack(TILE_OC))
        } else {
            (pack(OC_BLOCK), Vec::new())
        };
        Self {
            oc,
            ic,
            k,
            stride,
            pad: k / 2,
            packed,
            packed4,
            bias: bias.map_or_else(|| vec![0.0; oc], <[f32]>::to_vec),
        }
    }

    fn out_dim(&self, n: usize) -> usize {
        (n + 2 * self.pad - self.k) / self.stride + 1
    }

    /// Convolve; with `silu` the activation is applied to the result.
    fn forward(&self, x: &Tensor, silu: bool) -> Tensor {
        debug_assert_eq!(x.c, self.ic);
        let (ho, wo) = (self.out_dim(x.h), self.out_dim(x.w));
        let mut out = Tensor::zeros(self.oc, ho, wo);
        let row_len = self.oc * wo;
        if self.stride == 1 {
            let rl = round_up(wo, TILE_X) + 2 * self.pad;
            out.data.par_chunks_mut(row_len).enumerate().for_each_init(
                || vec![0.0f32; self.k * self.ic * rl],
                |scratch, (y, row)| conv_row_s1_dispatch(self, x, y, wo, silu, scratch, row),
            );
        } else {
            out.data.par_chunks_mut(row_len).enumerate().for_each_init(
                || vec![0.0f32; OC_BLOCK * wo],
                |acc, (y, row)| conv_row_dispatch(self, x, y, wo, silu, acc, row),
            );
        }
        out
    }
}

#[inline(always)]
fn silu_scalar(v: f32) -> f32 {
    v / (1.0 + (-v).exp())
}

/// One output row of a convolution. `FMA` selects fused multiply-add (only valid when compiled with the
/// hardware feature, see [`conv_row_avx2`]; elsewhere `mul_add` would be a slow software routine).
#[inline(always)]
fn conv_row_impl<const FMA: bool>(
    conv: &Conv,
    x: &Tensor,
    y: usize,
    wo: usize,
    silu: bool,
    acc: &mut [f32],
    out_row: &mut [f32],
) {
    let (k, stride, pad, ic_n) = (conv.k, conv.stride, conv.pad as isize, conv.ic);
    let xrow_len = x.row_len();
    let blocks = conv.oc.div_ceil(OC_BLOCK);
    for ob in 0..blocks {
        let valid = (conv.oc - ob * OC_BLOCK).min(OC_BLOCK);
        for o in 0..valid {
            acc[o * wo..(o + 1) * wo].fill(conv.bias[ob * OC_BLOCK + o]);
        }
        for ky in 0..k {
            let iy = (y * stride) as isize + ky as isize - pad;
            if iy < 0 || iy >= x.h as isize {
                continue;
            }
            let in_base = iy as usize * xrow_len;
            for kx in 0..k {
                // Output columns whose input column ox * stride + kx - pad lies inside [0, w).
                let lo = (pad - kx as isize).max(0);
                let ox_lo = (lo as usize).div_ceil(stride);
                let hi = x.w as isize - 1 + pad - kx as isize;
                if hi < 0 {
                    continue;
                }
                let ox_hi = ((hi as usize) / stride + 1).min(wo);
                if ox_lo >= ox_hi {
                    continue;
                }
                let n = ox_hi - ox_lo;
                let ix0 = (ox_lo * stride) as isize + kx as isize - pad;
                let wbase = ((ob * k + ky) * k + kx) * ic_n * OC_BLOCK;
                for i in 0..ic_n {
                    let in_row = &x.data[in_base + i * x.w..in_base + (i + 1) * x.w];
                    let w8 = &conv.packed[wbase + i * OC_BLOCK..wbase + (i + 1) * OC_BLOCK];
                    let src = &in_row[ix0 as usize..];
                    for (o, &wv) in w8.iter().enumerate().take(valid) {
                        let dst = &mut acc[o * wo + ox_lo..o * wo + ox_hi];
                        if stride == 1 {
                            for (a, &v) in dst.iter_mut().zip(&src[..n]) {
                                *a = if FMA { wv.mul_add(v, *a) } else { *a + wv * v };
                            }
                        } else {
                            for (a, &v) in dst.iter_mut().zip(src.iter().step_by(stride)) {
                                *a = if FMA { wv.mul_add(v, *a) } else { *a + wv * v };
                            }
                        }
                    }
                }
            }
        }
        for o in 0..valid {
            let dst = &mut out_row[(ob * OC_BLOCK + o) * wo..(ob * OC_BLOCK + o + 1) * wo];
            let src = &acc[o * wo..(o + 1) * wo];
            if silu {
                for (d, &s) in dst.iter_mut().zip(src) {
                    *d = silu_scalar(s);
                }
            } else {
                dst.copy_from_slice(src);
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
fn conv_row_avx2(
    conv: &Conv,
    x: &Tensor,
    y: usize,
    wo: usize,
    silu: bool,
    acc: &mut [f32],
    out_row: &mut [f32],
) {
    conv_row_impl::<true>(conv, x, y, wo, silu, acc, out_row);
}

fn conv_row_dispatch(
    conv: &Conv,
    x: &Tensor,
    y: usize,
    wo: usize,
    silu: bool,
    acc: &mut [f32],
    out_row: &mut [f32],
) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: the CPU supports AVX2 and FMA (checked just above), the only requirement of
            // `conv_row_avx2`.
            unsafe { conv_row_avx2(conv, x, y, wo, silu, acc, out_row) };
            return;
        }
    }
    // Fused multiply-add is part of the AArch64 base instruction set (Apple silicon is the target of the
    // CPU fallback): `mul_add` is a single `fmla`/`fmadd` there, not a software routine.
    #[cfg(target_arch = "aarch64")]
    {
        conv_row_impl::<true>(conv, x, y, wo, silu, acc, out_row);
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        conv_row_impl::<false>(conv, x, y, wo, silu, acc, out_row);
    }
}

fn round_up(n: usize, m: usize) -> usize {
    n.div_ceil(m) * m
}

/// One output row of a stride-1 convolution with a register tile of `TILE_OC x TILE_X` accumulators that
/// stays in registers across the whole `ky, kx, ic` reduction (output written once). The `k` input rows
/// the output row needs are first copied into `scratch` with zero borders, so the inner loops have no
/// bounds handling. `FMA` as in [`conv_row_impl`].
#[inline(always)]
fn conv_row_s1_impl<const FMA: bool>(
    conv: &Conv,
    x: &Tensor,
    y: usize,
    wo: usize,
    silu: bool,
    scratch: &mut [f32],
    out_row: &mut [f32],
) {
    let (k, ic_n, pad) = (conv.k, conv.ic, conv.pad);
    let rl = round_up(wo, TILE_X) + 2 * pad;
    let xrow_len = x.row_len();
    for ky in 0..k {
        let iy = y as isize + ky as isize - pad as isize;
        for i in 0..ic_n {
            let dst = &mut scratch[(ky * ic_n + i) * rl..(ky * ic_n + i + 1) * rl];
            if iy < 0 || iy >= x.h as isize {
                dst.fill(0.0);
            } else {
                let src = &x.data
                    [iy as usize * xrow_len + i * x.w..iy as usize * xrow_len + (i + 1) * x.w];
                dst[..pad].fill(0.0);
                dst[pad..pad + x.w].copy_from_slice(src);
                dst[pad + x.w..].fill(0.0);
            }
        }
    }
    let blocks = conv.oc.div_ceil(TILE_OC);
    let mut ox0 = 0;
    while ox0 < wo {
        let nvalid = (wo - ox0).min(TILE_X);
        for ob in 0..blocks {
            let valid = (conv.oc - ob * TILE_OC).min(TILE_OC);
            let mut acc = [[0.0f32; TILE_X]; TILE_OC];
            for (o, a) in acc.iter_mut().enumerate().take(valid) {
                a.fill(conv.bias[ob * TILE_OC + o]);
            }
            for ky in 0..k {
                for kx in 0..k {
                    let wbase = ((ob * k + ky) * k + kx) * ic_n * TILE_OC;
                    for i in 0..ic_n {
                        let start = (ky * ic_n + i) * rl + ox0 + kx;
                        let src = &scratch[start..start + TILE_X];
                        let w4 = &conv.packed4[wbase + i * TILE_OC..wbase + (i + 1) * TILE_OC];
                        for (a, &wv) in acc.iter_mut().zip(w4) {
                            for (acc_j, &v) in a.iter_mut().zip(src) {
                                *acc_j = if FMA {
                                    wv.mul_add(v, *acc_j)
                                } else {
                                    *acc_j + wv * v
                                };
                            }
                        }
                    }
                }
            }
            for (o, a) in acc.iter().enumerate().take(valid) {
                let dst = &mut out_row
                    [(ob * TILE_OC + o) * wo + ox0..(ob * TILE_OC + o) * wo + ox0 + nvalid];
                if silu {
                    for (d, &v) in dst.iter_mut().zip(a) {
                        *d = silu_scalar(v);
                    }
                } else {
                    dst.copy_from_slice(&a[..nvalid]);
                }
            }
        }
        ox0 += TILE_X;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
fn conv_row_s1_avx2(
    conv: &Conv,
    x: &Tensor,
    y: usize,
    wo: usize,
    silu: bool,
    scratch: &mut [f32],
    out_row: &mut [f32],
) {
    conv_row_s1_impl::<true>(conv, x, y, wo, silu, scratch, out_row);
}

fn conv_row_s1_dispatch(
    conv: &Conv,
    x: &Tensor,
    y: usize,
    wo: usize,
    silu: bool,
    scratch: &mut [f32],
    out_row: &mut [f32],
) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: the CPU supports AVX2 and FMA (checked just above), the only requirement of
            // `conv_row_s1_avx2`.
            unsafe { conv_row_s1_avx2(conv, x, y, wo, silu, scratch, out_row) };
            return;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        // FMA is baseline on AArch64, see `conv_row_dispatch`.
        conv_row_s1_impl::<true>(conv, x, y, wo, silu, scratch, out_row);
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        conv_row_s1_impl::<false>(conv, x, y, wo, silu, scratch, out_row);
    }
}

/// `pixel_shuffle(x, 2)` (PyTorch channel order: input channel `c*4 + a*2 + b` -> output `c`, pixel `(2y+a, 2x+b)`).
fn pixel_shuffle(x: &Tensor) -> Tensor {
    debug_assert_eq!(x.c % 4, 0);
    let c = x.c / 4;
    let mut out = Tensor::zeros(c, x.h * 2, x.w * 2);
    let out_row_len = out.row_len();
    out.data
        .par_chunks_mut(out_row_len)
        .enumerate()
        .for_each(|(big_y, row)| {
            let (y, a) = (big_y / 2, big_y % 2);
            let src = &x.data[y * x.row_len()..(y + 1) * x.row_len()];
            for ch in 0..c {
                let dst = &mut row[ch * 2 * x.w..(ch + 1) * 2 * x.w];
                for b in 0..2 {
                    let s = &src[(ch * 4 + a * 2 + b) * x.w..(ch * 4 + a * 2 + b + 1) * x.w];
                    for (xi, &v) in s.iter().enumerate() {
                        dst[2 * xi + b] = v;
                    }
                }
            }
        });
    out
}

/// Channel concatenation `[a, b]`.
fn concat(a: &Tensor, b: &Tensor) -> Tensor {
    debug_assert!(a.h == b.h && a.w == b.w);
    let mut out = Tensor::zeros(a.c + b.c, a.h, a.w);
    let (ra, rb) = (a.row_len(), b.row_len());
    out.data
        .par_chunks_mut(ra + rb)
        .enumerate()
        .for_each(|(y, row)| {
            row[..ra].copy_from_slice(&a.data[y * ra..(y + 1) * ra]);
            row[ra..].copy_from_slice(&b.data[y * rb..(y + 1) * rb]);
        });
    out
}

/// The fixed bilinear-demosaic kernel `[12, 4, 3, 3]` (`training/jdd/model.py::bilinear_weights`).
pub fn bilinear_weights() -> Vec<f32> {
    let mut w = vec![0.0f32; 12 * 4 * 9];
    let idx = |oc: usize, plane: usize, dy: i32, dx: i32| {
        ((oc * 4 + plane) * 3 + (dy + 1) as usize) * 3 + (dx + 1) as usize
    };
    let mut put = |c: usize, a: usize, b: usize, plane: usize, taps: &[(i32, i32)], scale: f32| {
        for &(dy, dx) in taps {
            w[idx(4 * c + 2 * a + b, plane, dy, dx)] += scale / taps.len() as f32;
        }
    };
    const R: usize = 0;
    const G1: usize = 1;
    const G2: usize = 2;
    const B: usize = 3;
    // (0,0) red site
    put(0, 0, 0, R, &[(0, 0)], 1.0);
    put(1, 0, 0, G1, &[(0, -1), (0, 0)], 0.5);
    put(1, 0, 0, G2, &[(-1, 0), (0, 0)], 0.5);
    put(2, 0, 0, B, &[(-1, -1), (-1, 0), (0, -1), (0, 0)], 1.0);
    // (0,1) green site in a red row
    put(1, 0, 1, G1, &[(0, 0)], 1.0);
    put(0, 0, 1, R, &[(0, 0), (0, 1)], 1.0);
    put(2, 0, 1, B, &[(-1, 0), (0, 0)], 1.0);
    // (1,0) green site in a blue row
    put(1, 1, 0, G2, &[(0, 0)], 1.0);
    put(0, 1, 0, R, &[(0, 0), (1, 0)], 1.0);
    put(2, 1, 0, B, &[(0, -1), (0, 0)], 1.0);
    // (1,1) blue site
    put(2, 1, 1, B, &[(0, 0)], 1.0);
    put(1, 1, 1, G2, &[(0, 0), (0, 1)], 0.5);
    put(1, 1, 1, G1, &[(0, 0), (1, 0)], 0.5);
    put(0, 1, 1, R, &[(0, 0), (0, 1), (1, 0), (1, 1)], 1.0);
    w
}

/// Several 3x3 convolutions in a row, each followed by SiLU (`Block` in `model.py`).
#[derive(Clone, Debug)]
struct Block(Vec<Conv>);

impl Block {
    fn forward(&self, x: &Tensor) -> Tensor {
        let mut cur = self.0[0].forward(x, true);
        for conv in &self.0[1..] {
            cur = conv.forward(&cur, true);
        }
        cur
    }
}

/// The `jdd` network on the CPU. See the module documentation.
#[derive(Clone, Debug)]
pub struct CpuJdd {
    cfg: JddConfig,
    id: ModelId,
    halo_half_px: usize,
    tile: TileShape,
    enc0: Block,
    down1: Conv,
    enc1: Block,
    down2: Conv,
    bott: Block,
    up1: Conv,
    dec1: Block,
    up0: Conv,
    dec0: Block,
    head: Conv,
    bilinear: Conv,
}

/// A parsed weights file: float tensors by name.
struct Weights<'a> {
    st: SafeTensors<'a>,
}

impl Weights<'_> {
    fn get(&self, name: &str, expected: &[usize]) -> Result<Vec<f32>> {
        let t = self
            .st
            .tensor(name)
            .map_err(|e| MlError::Weights(format!("tensor `{name}`: {e}")))?;
        if t.shape() != expected {
            return Err(MlError::TensorShape {
                name: name.to_string(),
                expected: expected.to_vec(),
                found: t.shape().to_vec(),
            });
        }
        let bytes = t.data();
        match t.dtype() {
            Dtype::F32 => Ok(bytes
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect()),
            Dtype::F16 => Ok(bytes
                .chunks_exact(2)
                .map(|b| half::f16::from_le_bytes([b[0], b[1]]).to_f32())
                .collect()),
            Dtype::BF16 => Ok(bytes
                .chunks_exact(2)
                .map(|b| half::bf16::from_le_bytes([b[0], b[1]]).to_f32())
                .collect()),
            other => Err(MlError::Weights(format!(
                "tensor `{name}` has unsupported dtype {other:?}"
            ))),
        }
    }

    fn has(&self, name: &str) -> bool {
        self.st.tensor(name).is_ok()
    }

    fn conv(
        &self,
        prefix: &str,
        oc: usize,
        ic: usize,
        k: usize,
        stride: usize,
        bias: bool,
    ) -> Result<Conv> {
        let w = self.get(&format!("{prefix}.weight"), &[oc, ic, k, k])?;
        let b = if bias {
            Some(self.get(&format!("{prefix}.bias"), &[oc])?)
        } else {
            None
        };
        Ok(Conv::new(&w, b.as_deref(), oc, ic, k, stride))
    }

    fn block(&self, prefix: &str, cin: usize, cout: usize, n: usize) -> Result<Block> {
        (0..n)
            .map(|i| {
                self.conv(
                    &format!("{prefix}.conv{}", i + 1),
                    cout,
                    if i == 0 { cin } else { cout },
                    3,
                    1,
                    true,
                )
            })
            .collect::<Result<Vec<_>>>()
            .map(Block)
    }
}

/// What the weights header says about the network (shared by all backends).
#[derive(Clone, Debug, PartialEq)]
pub struct Header {
    /// Architecture config.
    pub cfg: JddConfig,
    /// Tile halo in half-resolution pixels.
    pub halo_half_px: usize,
    /// Name, version and SHA-256 of the file.
    pub id: ModelId,
}

/// Parse and validate the safetensors header of a `jdd` weights file.
pub fn read_header(bytes: &[u8]) -> Result<Header> {
    let (_, meta) =
        SafeTensors::read_metadata(bytes).map_err(|e| MlError::Weights(e.to_string()))?;
    let header = meta.metadata().clone().unwrap_or_default();
    let field = |k: &str| {
        header
            .get(k)
            .ok_or_else(|| MlError::Weights(format!("header field `{k}` missing")))
    };
    if field("format")? != ARCH_NAME {
        return Err(MlError::Weights(format!(
            "not a {ARCH_NAME} file (format = {})",
            field("format")?
        )));
    }
    if field("arch_version")? != ARCH_VERSION {
        return Err(MlError::Unsupported(format!(
            "architecture version {} (this runtime implements {ARCH_VERSION})",
            field("arch_version")?
        )));
    }
    let cfg: JddConfig = serde_json::from_str(field("config")?)
        .map_err(|e| MlError::Weights(format!("config: {e}")))?;
    cfg.validate()?;
    let halo_half_px: usize = field("halo_half_px")?
        .parse()
        .map_err(|_| MlError::Weights("halo_half_px is not a number".into()))?;
    // Tiles keep the stride phase of a whole-frame run only if the halo is a multiple of the tile multiple
    // (`tiles::core_size`); catching it here gives the error at load time instead of at the first tile.
    if halo_half_px == 0
        || halo_half_px > MAX_HALO_HALF_PX
        || !halo_half_px.is_multiple_of(TILE_MULTIPLE)
    {
        return Err(MlError::Weights(format!(
            "halo_half_px {halo_half_px} must be a multiple of {TILE_MULTIPLE} between {TILE_MULTIPLE} and {MAX_HALO_HALF_PX}"
        )));
    }
    Ok(Header {
        cfg,
        halo_half_px,
        id: ModelId {
            name: "jdd".to_string(),
            version: header
                .get("model_version")
                .cloned()
                .unwrap_or_else(|| "0".to_string()),
            sha256: sha256_hex(bytes),
        },
    })
}

/// One float tensor of a weights file: shape and row-major data.
pub type FloatTensor = (Vec<usize>, Vec<f32>);

/// Read every tensor of a weights file as `f32` (F16 and BF16 files are widened).
pub fn read_float_tensors(bytes: &[u8]) -> Result<std::collections::HashMap<String, FloatTensor>> {
    let st = SafeTensors::deserialize(bytes).map_err(|e| MlError::Weights(e.to_string()))?;
    let w = Weights { st };
    let names: Vec<String> = w.st.names().into_iter().map(str::to_string).collect();
    names
        .into_iter()
        .map(|n| {
            let shape =
                w.st.tensor(&n)
                    .map_err(|e| MlError::Weights(e.to_string()))?
                    .shape()
                    .to_vec();
            let data = w.get(&n, &shape)?;
            Ok((n, (shape, data)))
        })
        .collect()
}

impl CpuJdd {
    /// Load from the bytes of a safetensors file (the header describes the architecture). If
    /// `expected_sha256` is given the bytes are checked against it first.
    pub fn from_bytes(bytes: &[u8], expected_sha256: Option<&str>) -> Result<Self> {
        if let Some(want) = expected_sha256 {
            verify_sha256("jdd weights", bytes, want)?;
        }
        let Header {
            cfg,
            halo_half_px,
            id,
        } = read_header(bytes)?;
        let st = SafeTensors::deserialize(bytes).map_err(|e| MlError::Weights(e.to_string()))?;
        let w = Weights { st };
        let [c0, c1, c2] = cfg.channels;
        let n = cfg.convs_per_level;
        let (uk, hk) = (cfg.up_kernel, cfg.head_kernel);
        let bil = bilinear_weights();
        let bilinear = if w.has("bilinear.weight") {
            let from_file = w.get("bilinear.weight", &[12, 4, 3, 3])?;
            if from_file
                .iter()
                .zip(&bil)
                .any(|(a, b)| (a - b).abs() > 1e-6)
            {
                return Err(MlError::Weights(
                    "bilinear.weight differs from the fixed bilinear demosaic kernel".into(),
                ));
            }
            Conv::new(&from_file, None, 12, 4, 3, 1)
        } else {
            Conv::new(&bil, None, 12, 4, 3, 1)
        };
        let me = Self {
            id,
            halo_half_px,
            tile: TileShape::square(DEFAULT_CORE_HALF_PX + 2 * halo_half_px),
            enc0: w.block("enc0", IN_CHANNELS, c0, n)?,
            down1: w.conv("down1", c1, c0, 3, 2, true)?,
            enc1: w.block("enc1", c1, c1, n)?,
            down2: w.conv("down2", c2, c1, 3, 2, true)?,
            bott: w.block("bott", c2, c2, n)?,
            up1: w.conv("up1", 4 * c1, c2, uk, 1, true)?,
            dec1: w.block("dec1", 2 * c1, c1, n)?,
            up0: w.conv("up0", 4 * c0, c1, uk, 1, true)?,
            dec0: w.block("dec0", 2 * c0, c0, n)?,
            head: w.conv("head", 12, c0, hk, 1, true)?,
            bilinear,
            cfg,
        };
        Ok(me)
    }

    /// Load from a safetensors file.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        Self::from_bytes(&std::fs::read(path)?, None)
    }

    /// Set the tile core (half-resolution pixels); the input shape becomes `core + 2 * halo`.
    #[must_use]
    pub fn with_core_half_px(mut self, core: usize) -> Self {
        self.tile = TileShape::square(core + 2 * self.halo_half_px);
        self
    }

    /// Set the exact tile input shape reported by [`TileModel::input_shape`].
    #[must_use]
    pub fn with_tile_shape(mut self, shape: TileShape) -> Self {
        self.tile = shape;
        self
    }

    /// Identity (name, version, SHA-256) of the loaded weights.
    pub fn id(&self) -> &ModelId {
        &self.id
    }

    /// The architecture config from the weights header.
    pub fn config(&self) -> &JddConfig {
        &self.cfg
    }

    /// Run on a tile of any valid shape (multiples of 4); the static-shape contract of
    /// [`TileModel::run`] does not apply.
    pub fn run_dynamic(&self, inputs: &TileInputs) -> Result<TileOutput> {
        let shape = inputs.shape();
        let data = self.forward_planar(inputs.as_slice(), shape.height, shape.width)?;
        TileOutput::new(shape, data)
    }

    /// Forward pass on planar `7 x h x w` data; returns planar `3 x 2h x 2w`. `h` and `w` must be
    /// positive multiples of 4 and `planar` must hold exactly `7 * h * w` values, otherwise an error is
    /// returned (nothing here panics on a bad shape).
    pub fn forward_planar(&self, planar: &[f32], h: usize, w: usize) -> Result<Vec<f32>> {
        TileShape {
            height: h,
            width: w,
        }
        .validate()?;
        if h.checked_mul(w).and_then(|n| n.checked_mul(IN_CHANNELS)) != Some(planar.len()) {
            return Err(MlError::Shape(format!(
                "planar input of {h}x{w} needs {IN_CHANNELS} x {h} x {w} values, got {}",
                planar.len()
            )));
        }
        let scale = self.cfg.input_scale();
        // Planar -> row-interleaved, applying the fixed input scaling.
        let mut x = Tensor::zeros(IN_CHANNELS, h, w);
        let row_len = x.row_len();
        x.data
            .par_chunks_mut(row_len)
            .enumerate()
            .for_each(|(y, row)| {
                for c in 0..IN_CHANNELS {
                    let src = &planar[(c * h + y) * w..(c * h + y + 1) * w];
                    for (d, &s) in row[c * w..(c + 1) * w].iter_mut().zip(src) {
                        *d = s * scale[c];
                    }
                }
            });
        // The bilinear base sees only the four scaled planes: copy them into their own tensor.
        let mut planes = Tensor::zeros(4, h, w);
        planes
            .data
            .par_chunks_mut(4 * w)
            .enumerate()
            .for_each(|(y, row)| {
                row.copy_from_slice(&x.data[y * row_len..y * row_len + 4 * w]);
            });

        let e0 = self.enc0.forward(&x);
        let e1 = self.enc1.forward(&self.down1.forward(&e0, true));
        let b = self.bott.forward(&self.down2.forward(&e1, true));
        let u1 = pixel_shuffle(&self.up1.forward(&b, true)); // SiLU commutes with the shuffle (pointwise)
        let d1 = self.dec1.forward(&concat(&u1, &e1));
        let u0 = pixel_shuffle(&self.up0.forward(&d1, true));
        let d0 = self.dec0.forward(&concat(&u0, &e0));
        let mut y12 = self.head.forward(&d0, false);
        let base = self.bilinear.forward(&planes, false);
        y12.data
            .par_iter_mut()
            .zip(base.data.par_iter())
            .for_each(|(a, &b)| *a += b);

        // Pixel shuffle to planar full-resolution RGB, undoing the plane scale.
        let inv = 1.0 / self.cfg.plane_scale;
        let (ow, oh) = (OUT_SCALE * w, OUT_SCALE * h);
        let mut out = vec![0.0f32; OUT_CHANNELS * oh * ow];
        out.par_chunks_mut(2 * ow)
            .enumerate()
            .for_each(|(i, rows)| {
                let (c, y) = (i / h, i % h);
                let src = &y12.data[y * y12.row_len()..(y + 1) * y12.row_len()];
                for a in 0..2 {
                    let dst = &mut rows[a * ow..(a + 1) * ow];
                    for b in 0..2 {
                        let s = &src[(c * 4 + a * 2 + b) * w..(c * 4 + a * 2 + b + 1) * w];
                        for (xi, &v) in s.iter().enumerate() {
                            dst[2 * xi + b] = v * inv;
                        }
                    }
                }
            });
        Ok(out)
    }
}

impl TileModel for CpuJdd {
    fn input_shape(&self) -> TileShape {
        self.tile
    }

    fn halo_half_px(&self) -> usize {
        self.halo_half_px
    }

    fn run(&self, inputs: &TileInputs, out: &mut TileOutput, _ctx: &ExecContext<'_>) -> Result<()> {
        if inputs.shape() != self.tile {
            return Err(MlError::Shape(format!(
                "model expects tiles of {}, got {}",
                self.tile,
                inputs.shape()
            )));
        }
        if out.shape() != self.tile {
            return Err(MlError::Shape(format!(
                "output buffer is for {}, model tile is {}",
                out.shape(),
                self.tile
            )));
        }
        let data = self.forward_planar(inputs.as_slice(), self.tile.height, self.tile.width)?;
        out.as_mut_slice().copy_from_slice(&data);
        Ok(())
    }

    fn name(&self) -> &str {
        "cpu"
    }
}

/// Smallest valid tile side that covers `n` pixels (a multiple of [`TILE_MULTIPLE`]).
pub fn round_up_tile(n: usize) -> usize {
    n.div_ceil(TILE_MULTIPLE) * TILE_MULTIPLE
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bilinear_kernel_rows_sum_to_one_per_color() {
        // Every output channel is a convex combination of same-coloured samples: weights sum to 1.
        let w = bilinear_weights();
        for oc in 0..12 {
            let s: f32 = w[oc * 36..(oc + 1) * 36].iter().sum();
            assert!((s - 1.0).abs() < 1e-6, "output channel {oc} sums to {s}");
        }
    }

    #[test]
    fn pixel_shuffle_matches_pytorch_channel_order() {
        // 4 channels, 1x2 image: channel c*4 + a*2 + b goes to pixel (2y + a, 2x + b).
        let mut t = Tensor::zeros(4, 1, 2);
        for x in 0..2 {
            for ch in 0..4 {
                t.data[ch * 2 + x] = (ch * 10 + x) as f32;
            }
        }
        let o = pixel_shuffle(&t);
        assert_eq!((o.c, o.h, o.w), (1, 2, 4));
        // row 0: a = 0 -> channels 0 (b=0) and 1 (b=1) interleaved
        assert_eq!(&o.data[0..4], &[0.0, 10.0, 1.0, 11.0]);
        assert_eq!(&o.data[4..8], &[20.0, 30.0, 21.0, 31.0]);
    }

    #[test]
    fn convolution_against_a_naive_reference() {
        // 5 -> 11 channels (not a multiple of the block size), 3x3, stride 1 and 2, odd sizes.
        let (ic, oc, k) = (5usize, 11usize, 3usize);
        let mut seed = 12345u32;
        let mut rnd = move || {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            (seed >> 8) as f32 / (1u32 << 24) as f32 - 0.5
        };
        let w: Vec<f32> = (0..oc * ic * k * k).map(|_| rnd()).collect();
        let b: Vec<f32> = (0..oc).map(|_| rnd()).collect();
        // (stride, height, width): widths below, at and above the 16-column register tile, odd and even
        for (stride, h, wd) in [
            (1usize, 9usize, 14usize),
            (1, 4, 37),
            (1, 3, 16),
            (1, 2, 1),
            (2, 9, 14),
            (2, 5, 37),
        ] {
            let mut x = Tensor::zeros(ic, h, wd);
            for v in &mut x.data {
                *v = rnd();
            }
            let conv = Conv::new(&w, Some(&b), oc, ic, k, stride);
            let got = conv.forward(&x, false);
            let (ho, wo) = (conv.out_dim(h), conv.out_dim(wd));
            assert_eq!((got.h, got.w, got.c), (ho, wo, oc));
            for o in 0..oc {
                for y in 0..ho {
                    for xx in 0..wo {
                        let mut s = b[o] as f64;
                        for i in 0..ic {
                            for ky in 0..k {
                                for kx in 0..k {
                                    let iy = (y * stride + ky) as isize - 1;
                                    let ix = (xx * stride + kx) as isize - 1;
                                    if iy >= 0 && iy < h as isize && ix >= 0 && ix < wd as isize {
                                        s += w[((o * ic + i) * k + ky) * k + kx] as f64
                                            * x.data[(iy as usize * ic + i) * wd + ix as usize]
                                                as f64;
                                    }
                                }
                            }
                        }
                        let g = got.data[(y * oc + o) * wo + xx] as f64;
                        assert!(
                            (g - s).abs() < 1e-5,
                            "stride {stride} o {o} y {y} x {xx}: {g} vs {s}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn one_by_one_convolution_and_silu() {
        let conv = Conv::new(&[2.0, -1.0], Some(&[0.5]), 1, 2, 1, 1);
        let mut x = Tensor::zeros(2, 1, 3);
        x.data.copy_from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]); // channel 0 = 1 2 3, channel 1 = 4 5 6
        let y = conv.forward(&x, false);
        assert_eq!(y.data, vec![-1.5, -0.5, 0.5]);
        let s = conv.forward(&x, true);
        assert!((s.data[2] - 0.5 / (1.0 + (-0.5f32).exp())).abs() < 1e-6);
    }
}
