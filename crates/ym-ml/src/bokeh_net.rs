//! The bokeh map network on the CPU (`docs/21-lens-character.md` §7, `training/bokeh/`): a small
//! convolutional network that reads the lens plane and answers, for every 2 × 2 block of it, the
//! radius `b` of the disc blur the Sony lens already left in the photo (in half-diagonal units) and
//! how sure that is.
//!
//! The network is the one `training/bokeh/export.py` writes to `data/models/bokeh/bokehnet.safetensors`
//! (batch norms folded into the convolutions): a stride-2 3 × 3 stem, a MobileNetV2-style stack of
//! inverted-residual blocks (1 × 1 expansion, 3 × 3 depthwise of stride 1 or 2, 1 × 1 projection,
//! ReLU6) down to stride 32, a global-context block (the stride-32 features averaged over the whole
//! plane, a 1 × 1 convolution and ReLU6, added to every cell), a U-Net decoder (bilinear ×2,
//! concatenation with a 1 × 1-projected skip, then 1 × 1, depthwise 3 × 3 and 1 × 1 convolutions)
//! and a 1 × 1 head at half resolution. The architecture is read from the file's header, so a
//! retrained model of the same family loads without a change here; anything the runtime does not
//! implement is refused at load.
//!
//! # From a lens plane to a map
//!
//! [`BokehNet::map`] does what `training/bokeh/encode.py` and `README.md` define:
//!
//! 1. **Reduce** the plane by an exact area-weighted box filter to a long edge of
//!    [`FRAME_LONG_PX`] (never enlarged; a plane of 1024 px or less is used as it is). The
//!    network works in pixels of that frame.
//! 2. **Encode** it ([`encode_linear`]): luminance, its lower median `m` (at least `1e-3` times the
//!    mean), `v = RGB · 0.18 / m`, `e = log2(1 + v / K) / 16` per channel. A plane twice as bright
//!    gives the same input.
//! 3. **Pad** with replicated edge pixels to a multiple of [`PAD_MULTIPLE`] ([`pad_encoded`]).
//! 4. **Run** the network ([`BokehNet::forward_padded`]): `(μ, s)` at half resolution, `μ = ln r`
//!    with `r` the disc radius in pixels of the reduced plane and `s = ln β` with `β` the Laplace
//!    scale of the error of `μ`.
//! 5. **Convert** the output (cropped back to `⌈w/2⌉ × ⌈h/2⌉`): `b = max(r, 1 px) / hd` with `hd`
//!    the half-diagonal of the reduced plane in pixels (before padding), and
//!    `c = exp(−β / 0.25)` with `β = exp(clamp(s, −6, 3))` ([`BokehMap`]).
//!
//! `b` is a ratio, so it is the same for the reduced plane and for the lens plane it came from:
//! the lens plane covers the whole area whose half-diagonal is the unit of the lens data
//! (`docs/21` §3). The map is *raw*: the caution rules of `docs/21` §7 step 5 (`b = 0` where `c`
//! is low, below `b_min`, near a sharp edge or inside the subject's anchor) belong to the caller.
//!
//! # Determinism
//!
//! Everything is `f32` (the reduction and the encoding `f64`) in a fixed order, with no fused
//! multiply-add and with `libm` for `exp` and `log2`, so the same plane gives the same bits on
//! every run, on any number of threads (every output row is computed by one task, whatever the
//! pool) and, as far as IEEE 754 goes, on every platform. Convolutions accumulate the bias first,
//! then the input channels (or taps) in index order; the global mean sums in `f64` in row-major
//! order.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use rayon::prelude::*;
use safetensors::SafeTensors;
use serde::Deserialize;

use crate::cpu_runtime::{FloatTensor, read_float_tensors};
use crate::model::{MlError, Result, sha256_hex, verify_sha256};

/// Format string of the safetensors header (`training/bokeh/export.py`).
pub const FORMAT: &str = "ym-bokehnet";
/// Format version this runtime reads.
pub const VERSION: &str = "1";
/// The weights file in the model folder (`data/models/bokeh/`).
pub const WEIGHTS_FILE: &str = "bokehnet.safetensors";
/// The long edge the network is fed, in pixels (`encode.FRAME_LONG`).
pub const FRAME_LONG_PX: usize = 1024;
/// The input's width and height are padded to a multiple of this (`encode.PAD_MULTIPLE`).
pub const PAD_MULTIPLE: usize = 32;
/// Input pixels per output cell along each axis.
pub const OUTPUT_STRIDE: usize = 2;
/// The smallest radius the network is trained to tell apart, in pixels (`encode.R_FLOOR_PX`).
pub const R_FLOOR_PX: f64 = 1.0;
/// `c = exp(-β / CONF_BETA0)` (`encode.CONF_BETA0`).
pub const CONF_BETA0: f64 = 0.25;

/// Luma weights of the encoding (`encode.LUMA`, Rec. 709).
const LUMA: [f64; 3] = [0.2126, 0.7152, 0.0722];
/// The median luminance is moved to this (`encode.MID`).
const MID: f64 = 0.18;
/// Knee of the encoding's curve (`encode.K`).
const K: f64 = MID / 32.0;
/// Stops the encoding spans (`encode.STOPS`).
const STOPS: f64 = 16.0;
/// The gain's median is at least this times the mean (`encode.MEDIAN_FLOOR_OF_MEAN`).
const MEDIAN_FLOOR_OF_MEAN: f64 = 1e-3;
/// The input is `(e - INPUT_MEAN) / INPUT_STD` (`encode.INPUT_MEAN`, `INPUT_STD`).
const INPUT_MEAN: f32 = 0.30;
const INPUT_STD: f32 = 0.15;
/// `s` is clamped to this range before `exp` (`encode.radius_and_confidence`).
const LOG_BETA_RANGE: (f64, f64) = (-6.0, 3.0);

// -------------------------------------------------------------------------------------------------
// Tensors and kernels
// -------------------------------------------------------------------------------------------------

/// Output channels computed together by the 1 × 1 kernel.
const TO: usize = 4;
/// Output columns computed together by the 1 × 1 kernel.
const TX: usize = 16;

/// The activation after a layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Act {
    None,
    Relu6,
}

/// `clamp(v, 0, 6)` with the same answer for every input on every platform (`-0.0` and NaN give `+0.0`).
#[inline(always)]
fn relu6(v: f32) -> f32 {
    if v > 0.0 {
        if v < 6.0 { v } else { 6.0 }
    } else {
        0.0
    }
}

#[inline(always)]
fn activate(v: f32, act: Act) -> f32 {
    match act {
        Act::None => v,
        Act::Relu6 => relu6(v),
    }
}

/// A feature map, row-interleaved: element `(y, c, x)` is at `(y * c_total + c) * w + x`, so a row
/// of all channels is one contiguous chunk that one task can own.
#[derive(Debug, Clone)]
struct Feat {
    c: usize,
    h: usize,
    w: usize,
    data: Vec<f32>,
}

impl Feat {
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

    fn row(&self, y: usize) -> &[f32] {
        let n = self.row_len();
        &self.data[y * n..(y + 1) * n]
    }
}

/// A 1 × 1 convolution with its bias.
#[derive(Debug, Clone)]
struct Pointwise {
    cin: usize,
    cout: usize,
    /// `[output block of TO][input channel][TO]`, zero padded to a multiple of `TO` outputs.
    packed: Vec<f32>,
    bias: Vec<f32>,
}

impl Pointwise {
    /// `weight` is `[cout][cin]`.
    fn new(weight: &[f32], bias: Vec<f32>, cin: usize, cout: usize) -> Self {
        let mut packed = vec![0.0; cout.div_ceil(TO) * cin * TO];
        for o in 0..cout {
            for i in 0..cin {
                packed[((o / TO) * cin + i) * TO + o % TO] = weight[o * cin + i];
            }
        }
        Self {
            cin,
            cout,
            packed,
            bias,
        }
    }

    /// One row: `src` holds `cin` channels of `w` columns, `out` `cout` channels; `res` (the same
    /// shape as `out`) is added after the convolution and before the activation.
    fn row(&self, src: &[f32], w: usize, out: &mut [f32], act: Act, res: Option<&[f32]>) {
        debug_assert_eq!(src.len(), self.cin * w);
        debug_assert_eq!(out.len(), self.cout * w);
        for ob in 0..self.cout.div_ceil(TO) {
            let valid = (self.cout - ob * TO).min(TO);
            let weights = &self.packed[ob * self.cin * TO..(ob + 1) * self.cin * TO];
            let mut x0 = 0;
            while x0 < w {
                let n = (w - x0).min(TX);
                let mut acc = [[0.0f32; TX]; TO];
                for (o, a) in acc.iter_mut().enumerate().take(valid) {
                    a.fill(self.bias[ob * TO + o]);
                }
                if n == TX {
                    self.tile(&mut acc, weights, src, w, x0, TX);
                } else {
                    self.tile(&mut acc, weights, src, w, x0, n);
                }
                for (o, a) in acc.iter().enumerate().take(valid) {
                    let at = (ob * TO + o) * w + x0;
                    let dst = &mut out[at..at + n];
                    match res {
                        Some(r) => {
                            for ((d, &v), &r) in dst.iter_mut().zip(a).zip(&r[at..at + n]) {
                                *d = activate(v + r, act);
                            }
                        }
                        None => {
                            for (d, &v) in dst.iter_mut().zip(a) {
                                *d = activate(v, act);
                            }
                        }
                    }
                }
                x0 += n;
            }
        }
    }

    /// The reduction over the input channels of one tile of `TO` outputs by `n` columns, in
    /// channel order. `n` is the constant `TX` for full tiles, so those loops have a fixed length.
    #[inline(always)]
    fn tile(
        &self,
        acc: &mut [[f32; TX]; TO],
        weights: &[f32],
        src: &[f32],
        w: usize,
        x0: usize,
        n: usize,
    ) {
        for i in 0..self.cin {
            let s = &src[i * w + x0..i * w + x0 + n];
            let w4 = &weights[i * TO..(i + 1) * TO];
            for (a, &wv) in acc.iter_mut().zip(w4) {
                for (a, &v) in a[..n].iter_mut().zip(s) {
                    *a += wv * v;
                }
            }
        }
    }

    /// The whole map, one task per row. `res` is a map of the output's shape added before the
    /// activation (a block's skip connection).
    fn forward(&self, x: &Feat, act: Act, res: Option<&Feat>) -> Feat {
        debug_assert_eq!(x.c, self.cin);
        let mut out = Feat::zeros(self.cout, x.h, x.w);
        let (n_in, n_out) = (x.row_len(), out.row_len());
        out.data
            .par_chunks_mut(n_out)
            .enumerate()
            .for_each(|(y, row)| {
                self.row(
                    &x.data[y * n_in..(y + 1) * n_in],
                    x.w,
                    row,
                    act,
                    res.map(|r| r.row(y)),
                );
            });
        out
    }

    /// The layer on one vector (a map of 1 × 1 cells): the bias, then the inputs in order.
    fn vector(&self, x: &[f32], act: Act) -> Vec<f32> {
        (0..self.cout)
            .map(|o| {
                let mut acc = self.bias[o];
                for (i, &v) in x.iter().enumerate() {
                    acc += self.packed[((o / TO) * self.cin + i) * TO + o % TO] * v;
                }
                activate(acc, act)
            })
            .collect()
    }
}

/// A 3 × 3 depthwise convolution of stride 1 or 2 with zero padding 1.
#[derive(Debug, Clone)]
struct Depthwise {
    c: usize,
    stride: usize,
    /// `[c][3][3]`.
    weight: Vec<f32>,
    bias: Vec<f32>,
}

/// The output columns `lo..hi` whose input column `stride * xo + kx - 1` lies inside `0..w`.
fn tap_range(w: usize, wo: usize, stride: usize, kx: usize) -> (usize, usize) {
    let lo = usize::from(kx == 0);
    let hi = if w >= kx {
        ((w - kx) / stride + 1).min(wo)
    } else {
        0
    };
    (lo.min(hi), hi)
}

impl Depthwise {
    /// Output row `yo` (all channels) into `out`: per output pixel the bias, then the taps in
    /// `(ky, kx)` order, the taps that fall on the padding left out.
    fn row(&self, x: &Feat, yo: usize, out: &mut [f32], act: Act) {
        let (w, s) = (x.w, self.stride);
        let wo = out.len() / self.c;
        for ch in 0..self.c {
            let k = &self.weight[ch * 9..ch * 9 + 9];
            let dst = &mut out[ch * wo..(ch + 1) * wo];
            dst.fill(self.bias[ch]);
            for ky in 0..3 {
                let Some(iy) = (s * yo + ky).checked_sub(1).filter(|&v| v < x.h) else {
                    continue;
                };
                let src = &x.data[(iy * x.c + ch) * w..(iy * x.c + ch + 1) * w];
                for kx in 0..3 {
                    let wk = k[ky * 3 + kx];
                    let (lo, hi) = tap_range(w, wo, s, kx);
                    if lo >= hi {
                        continue;
                    }
                    let first = s * lo + kx - 1;
                    if s == 1 {
                        for (d, &v) in dst[lo..hi].iter_mut().zip(&src[first..]) {
                            *d += wk * v;
                        }
                    } else {
                        for (d, &v) in dst[lo..hi].iter_mut().zip(src[first..].iter().step_by(s)) {
                            *d += wk * v;
                        }
                    }
                }
            }
            if act != Act::None {
                for d in dst.iter_mut() {
                    *d = activate(*d, act);
                }
            }
        }
    }

    fn forward(&self, x: &Feat, act: Act) -> Feat {
        debug_assert_eq!(x.c, self.c);
        let (ho, wo) = ((x.h - 1) / self.stride + 1, (x.w - 1) / self.stride + 1);
        let mut out = Feat::zeros(self.c, ho, wo);
        let n_out = out.row_len();
        out.data
            .par_chunks_mut(n_out)
            .enumerate()
            .for_each(|(y, row)| self.row(x, y, row, act));
        out
    }
}

/// The stem: a dense 3 × 3 convolution of stride 2 on the three input channels, then ReLU6.
#[derive(Debug, Clone)]
struct Stem {
    cout: usize,
    /// `[cout][3][3][3]`.
    weight: Vec<f32>,
    bias: Vec<f32>,
}

impl Stem {
    fn row(&self, x: &Feat, yo: usize, out: &mut [f32]) {
        let w = x.w;
        let wo = out.len() / self.cout;
        for o in 0..self.cout {
            let dst = &mut out[o * wo..(o + 1) * wo];
            dst.fill(self.bias[o]);
            for i in 0..3 {
                for ky in 0..3 {
                    let Some(iy) = (2 * yo + ky).checked_sub(1).filter(|&v| v < x.h) else {
                        continue;
                    };
                    let src = &x.data[(iy * 3 + i) * w..(iy * 3 + i + 1) * w];
                    for kx in 0..3 {
                        let wk = self.weight[((o * 3 + i) * 3 + ky) * 3 + kx];
                        let (lo, hi) = tap_range(w, wo, 2, kx);
                        if lo >= hi {
                            continue;
                        }
                        let first = 2 * lo + kx - 1;
                        for (d, &v) in dst[lo..hi].iter_mut().zip(src[first..].iter().step_by(2)) {
                            *d += wk * v;
                        }
                    }
                }
            }
            for d in dst.iter_mut() {
                *d = relu6(*d);
            }
        }
    }

    fn forward(&self, x: &Feat) -> Feat {
        debug_assert_eq!(x.c, 3);
        let mut out = Feat::zeros(self.cout, (x.h - 1) / 2 + 1, (x.w - 1) / 2 + 1);
        let n_out = out.row_len();
        out.data
            .par_chunks_mut(n_out)
            .enumerate()
            .for_each(|(y, row)| self.row(x, y, row));
        out
    }
}

/// One inverted-residual block.
#[derive(Debug, Clone)]
struct Block {
    expand: Option<Pointwise>,
    depthwise: Depthwise,
    project: Pointwise,
    residual: bool,
}

impl Block {
    fn forward(&self, x: &Feat) -> Feat {
        let expanded = self.expand.as_ref().map(|e| e.forward(x, Act::Relu6, None));
        let d = self
            .depthwise
            .forward(expanded.as_ref().unwrap_or(x), Act::Relu6);
        self.project
            .forward(&d, Act::None, self.residual.then_some(x))
    }
}

/// How an output index reads the coarse axis in a bilinear ×2 upsample with `align_corners=False`:
/// the source index `max((i + 0.5) / 2 - 0.5, 0)` split into its two neighbours and the weight of
/// the second (exact in `f32`: 0, 0.25 or 0.75).
fn upsample_axis(n_in: usize) -> Vec<(usize, usize, f32)> {
    (0..2 * n_in)
        .map(|i| {
            let src = ((i as f32 + 0.5) * 0.5 - 0.5).max(0.0);
            let i0 = (src as usize).min(n_in - 1);
            let i1 = (i0 + 1).min(n_in - 1);
            (i0, i1, src - i0 as f32)
        })
        .collect()
}

/// One decoder step: upsample, concatenate the projected skip, then 1 × 1, depthwise 3 × 3, 1 × 1.
#[derive(Debug, Clone)]
struct Step {
    skip: Pointwise,
    mix: Pointwise,
    dw: Depthwise,
    pw: Pointwise,
}

impl Step {
    fn forward(&self, x: &Feat, skip: &Feat) -> Result<Feat> {
        let (h, w) = (x.h * 2, x.w * 2);
        if skip.h != h || skip.w != w || skip.c != self.skip.cin || x.c + self.skip.cout != self.mix.cin
        {
            return Err(MlError::Shape(format!(
                "decoder step: {}x{}x{} coarse and {}x{}x{} skip do not fit a {}-input mix",
                x.c, x.h, x.w, skip.c, skip.h, skip.w, self.mix.cin
            )));
        }
        let cols = upsample_axis(x.w);
        let rows = upsample_axis(x.h);
        let mut cat = Feat::zeros(self.mix.cin, h, w);
        let n_cat = cat.row_len();
        cat.data
            .par_chunks_mut(n_cat)
            .enumerate()
            .for_each(|(y, row)| {
                let (up, rest) = row.split_at_mut(x.c * w);
                let (y0, y1, fy) = rows[y];
                let (r0, r1) = (x.row(y0), x.row(y1));
                let (h0, h1) = (1.0 - fy, fy);
                for ch in 0..x.c {
                    let a = &r0[ch * x.w..(ch + 1) * x.w];
                    let b = &r1[ch * x.w..(ch + 1) * x.w];
                    for (d, &(i0, i1, fx)) in up[ch * w..(ch + 1) * w].iter_mut().zip(&cols) {
                        let (w0, w1) = (1.0 - fx, fx);
                        *d = h0 * (w0 * a[i0] + w1 * a[i1]) + h1 * (w0 * b[i0] + w1 * b[i1]);
                    }
                }
                self.skip.row(skip.row(y), w, rest, Act::Relu6, None);
            });
        let mixed = self.mix.forward(&cat, Act::Relu6, None);
        drop(cat);
        let filtered = self.dw.forward(&mixed, Act::Relu6);
        drop(mixed);
        Ok(self.pw.forward(&filtered, Act::Relu6, None))
    }
}

/// The mean of each channel over the whole map, summed in `f64` in row-major order.
fn channel_means(x: &Feat) -> Vec<f32> {
    let cells = (x.h * x.w) as f64;
    (0..x.c)
        .into_par_iter()
        .map(|ch| {
            let mut sum = 0.0f64;
            for y in 0..x.h {
                let row = &x.data[(y * x.c + ch) * x.w..(y * x.c + ch + 1) * x.w];
                for &v in row {
                    sum += f64::from(v);
                }
            }
            (sum / cells) as f32
        })
        .collect()
}

// -------------------------------------------------------------------------------------------------
// The weights file
// -------------------------------------------------------------------------------------------------

/// The `arch` field of the weights header (`export.arch_json`).
#[derive(Debug, Deserialize)]
struct ArchJson {
    stem: usize,
    blocks: Vec<BlockJson>,
    taps: Vec<usize>,
    decoder: Vec<usize>,
    skip_channels: Vec<usize>,
    out_channels: usize,
}

#[derive(Debug, Deserialize)]
struct BlockJson {
    stage: usize,
    expand: bool,
    cin: usize,
    mid: usize,
    cout: usize,
    stride: usize,
    residual: bool,
}

/// The tensors of a weights file by name, each taken out once with its shape checked.
struct Tensors(HashMap<String, FloatTensor>);

impl Tensors {
    fn take(&mut self, name: &str, shape: &[usize]) -> Result<Vec<f32>> {
        let (found, data) = self
            .0
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
        Ok(data)
    }

    fn pointwise(&mut self, prefix: &str, cin: usize, cout: usize) -> Result<Pointwise> {
        let weight = self.take(&format!("{prefix}.weight"), &[cout, cin, 1, 1])?;
        let bias = self.take(&format!("{prefix}.bias"), &[cout])?;
        Ok(Pointwise::new(&weight, bias, cin, cout))
    }

    fn depthwise(&mut self, prefix: &str, c: usize, stride: usize) -> Result<Depthwise> {
        Ok(Depthwise {
            c,
            stride,
            weight: self.take(&format!("{prefix}.weight"), &[c, 1, 3, 3])?,
            bias: self.take(&format!("{prefix}.bias"), &[c])?,
        })
    }
}

fn unsupported(what: impl std::fmt::Display) -> MlError {
    MlError::Unsupported(format!("a {FORMAT} file that {what}"))
}

/// Checks the header fields that decide how the runtime must treat the file.
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
    // The constants are `f64` in `encode.py`; the two of the network's input are `f32` here (the
    // training code's tensor arithmetic), so they are compared to single precision.
    let number = |key: &str, want: f64, tolerance: f64| -> Result<()> {
        let got: f64 = field(key)
            .parse()
            .map_err(|_| MlError::Weights(format!("header field `{key}` is not a number")))?;
        if (got - want).abs() > tolerance * want.abs().max(1.0) {
            return Err(unsupported(format!("has {key} = {got} (this runtime: {want})")));
        }
        Ok(())
    };
    number("mid", MID, 1e-12)?;
    number("k", K, 1e-12)?;
    number("stops", STOPS, 1e-12)?;
    number("input_mean", f64::from(INPUT_MEAN), 1e-6)?;
    number("input_std", f64::from(INPUT_STD), 1e-6)?;
    number("conf_beta0", CONF_BETA0, 1e-12)?;
    let luma: Vec<f64> = serde_json::from_str(field("luma"))
        .map_err(|e| MlError::Weights(format!("luma: {e}")))?;
    if luma.len() != 3 || luma.iter().zip(LUMA).any(|(a, b)| (a - b).abs() > 1e-12) {
        return Err(unsupported(format!("has luma weights {luma:?}")));
    }
    let same = |key: &str, want: &str| -> Result<()> {
        if field(key) == want {
            Ok(())
        } else {
            Err(unsupported(format!(
                "has {key} = {:?} (this runtime: {want:?})",
                field(key)
            )))
        }
    };
    same("activation", "relu6")?;
    same("pad_multiple", &PAD_MULTIPLE.to_string())?;
    same("output_stride", &OUTPUT_STRIDE.to_string())?;
    same("frame_long_px", &FRAME_LONG_PX.to_string())?;
    if !field("upsample").contains("align_corners=false") {
        return Err(unsupported(format!("upsamples as {:?}", field("upsample"))));
    }
    Ok(())
}

/// The bokeh map network, loaded from its weights file. See the module documentation.
#[derive(Clone)]
pub struct BokehNet {
    stem: Stem,
    blocks: Vec<Block>,
    /// Indices of the blocks whose output is a skip (the last block of each tapped stage).
    taps: Vec<usize>,
    context: Pointwise,
    steps: Vec<Step>,
    head: Pointwise,
    sha256: String,
    parameters: usize,
}

impl std::fmt::Debug for BokehNet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BokehNet")
            .field("blocks", &self.blocks.len())
            .field("steps", &self.steps.len())
            .field("parameters", &self.parameters)
            .field("sha256", &self.sha256)
            .finish()
    }
}

impl BokehNet {
    /// Loads `dir/bokehnet.safetensors`, checked against its line in `dir/SHA256SUMS` (the model
    /// folder `data/models/bokeh/`).
    ///
    /// # Errors
    ///
    /// [`MlError::Io`] when a file cannot be read, [`MlError::Checksum`] when the weights do not
    /// match `SHA256SUMS`, and the errors of [`BokehNet::from_bytes`].
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

    /// Loads the model of a data folder: `<data_root>/models/bokeh/` ([`BokehNet::load`]).
    ///
    /// # Errors
    ///
    /// As [`BokehNet::load`].
    pub fn load_bundled(data_root: &Path) -> Result<Self> {
        Self::load(&data_root.join("models").join("bokeh"))
    }

    /// Loads a weights file without a checksum (development and tests; the shipped model goes
    /// through [`BokehNet::load`]).
    ///
    /// # Errors
    ///
    /// [`MlError::Io`] when the file cannot be read, and the errors of [`BokehNet::from_bytes`].
    pub fn from_file(path: &Path) -> Result<Self> {
        Self::from_bytes(&std::fs::read(path)?, None)
    }

    /// Builds the network from the bytes of a weights file, checked against `expected_sha256` when
    /// given.
    ///
    /// # Errors
    ///
    /// [`MlError::Checksum`] on a digest mismatch, [`MlError::Weights`] when the file is not a
    /// `ym-bokehnet` file, lacks a tensor or holds a value that is not finite,
    /// [`MlError::Unsupported`] for another format version or a network the runtime does not
    /// implement, [`MlError::TensorShape`] when a tensor has the wrong shape.
    pub fn from_bytes(bytes: &[u8], expected_sha256: Option<&str>) -> Result<Self> {
        if let Some(want) = expected_sha256 {
            verify_sha256("bokeh network weights", bytes, want)?;
        }
        let (_, meta) =
            SafeTensors::read_metadata(bytes).map_err(|e| MlError::Weights(e.to_string()))?;
        let header = meta.metadata().clone().unwrap_or_default();
        check_header(&header)?;
        let arch: ArchJson = serde_json::from_str(header.get("arch").map_or("", String::as_str))
            .map_err(|e| MlError::Weights(format!("arch: {e}")))?;
        let taps = Self::check_arch(&arch)?;

        let mut t = Tensors(read_float_tensors(bytes)?);
        let stem = Stem {
            cout: arch.stem,
            weight: t.take("stem.weight", &[arch.stem, 3, 3, 3])?,
            bias: t.take("stem.bias", &[arch.stem])?,
        };
        let mut blocks = Vec::with_capacity(arch.blocks.len());
        for (i, b) in arch.blocks.iter().enumerate() {
            blocks.push(Block {
                expand: if b.expand {
                    Some(t.pointwise(&format!("blocks.{i}.expand"), b.cin, b.mid)?)
                } else {
                    None
                },
                depthwise: t.depthwise(&format!("blocks.{i}.depthwise"), b.mid, b.stride)?,
                project: t.pointwise(&format!("blocks.{i}.project"), b.mid, b.cout)?,
                residual: b.residual,
            });
        }
        let bottleneck = arch.blocks.last().map_or(arch.stem, |b| b.cout);
        let context = t.pointwise("context", bottleneck, bottleneck)?;
        let skips = &arch.skip_channels[..arch.skip_channels.len() - 1];
        let mut steps = Vec::with_capacity(arch.decoder.len());
        let mut prev = bottleneck;
        for (j, (&width, &skip)) in arch.decoder.iter().zip(skips.iter().rev()).enumerate() {
            steps.push(Step {
                skip: t.pointwise(&format!("decoder.{j}.skip"), skip, width)?,
                mix: t.pointwise(&format!("decoder.{j}.mix"), prev + width, width)?,
                dw: t.depthwise(&format!("decoder.{j}.dw"), width, 1)?,
                pw: t.pointwise(&format!("decoder.{j}.pw"), width, width)?,
            });
            prev = width;
        }
        let head = t.pointwise("head", prev, arch.out_channels)?;
        if let Some(extra) = t.0.keys().min() {
            return Err(MlError::Weights(format!("unexpected tensor `{extra}`")));
        }
        let parameters = Self::count(&stem, &blocks, &context, &steps, &head);
        Ok(Self {
            stem,
            blocks,
            taps,
            context,
            steps,
            head,
            sha256: sha256_hex(bytes),
            parameters,
        })
    }

    /// Checks that the header's architecture is one this runtime implements and returns the
    /// indices of the blocks whose output is a skip.
    fn check_arch(arch: &ArchJson) -> Result<Vec<usize>> {
        if arch.stem == 0 || arch.blocks.is_empty() || arch.out_channels != 2 {
            return Err(unsupported("has no stem or no blocks, or does not answer (μ, s)"));
        }
        let mut cin = arch.stem;
        let mut stride = 2usize;
        // Strides of the skips: the stem's, then those of the tapped stages' last blocks.
        let mut skip_channels = vec![arch.stem];
        let mut skip_strides = vec![stride];
        let mut taps = Vec::new();
        for (i, b) in arch.blocks.iter().enumerate() {
            let ok = b.cin == cin
                && b.mid >= 1
                && b.cout >= 1
                && (b.stride == 1 || b.stride == 2)
                && b.residual == (b.stride == 1 && b.cin == b.cout)
                && (b.expand || b.mid == b.cin);
            if !ok {
                return Err(unsupported(format!("has an inconsistent block {i}")));
            }
            cin = b.cout;
            stride *= b.stride;
            let ends_stage = arch.blocks.get(i + 1).is_none_or(|n| n.stage != b.stage);
            if ends_stage && arch.taps.contains(&b.stage) {
                taps.push(i);
                skip_channels.push(b.cout);
                skip_strides.push(stride);
            }
        }
        if skip_channels != arch.skip_channels
            || arch.decoder.len() + 1 != skip_strides.len()
            || arch.decoder.contains(&0)
            || skip_strides.windows(2).any(|p| p[1] != 2 * p[0])
            || stride != skip_strides[skip_strides.len() - 1]
            || stride != PAD_MULTIPLE
        {
            return Err(unsupported(format!(
                "has skips {skip_channels:?} at strides {skip_strides:?} and a bottleneck of stride {stride}"
            )));
        }
        Ok(taps)
    }

    fn count(stem: &Stem, blocks: &[Block], ctx: &Pointwise, steps: &[Step], head: &Pointwise) -> usize {
        let pw = |p: &Pointwise| p.cin * p.cout + p.cout;
        let dw = |d: &Depthwise| d.c * 9 + d.c;
        let mut n = stem.cout * 27 + stem.cout;
        for b in blocks {
            n += b.expand.as_ref().map_or(0, pw) + dw(&b.depthwise) + pw(&b.project);
        }
        n += pw(ctx) + pw(head);
        for s in steps {
            n += pw(&s.skip) + pw(&s.mix) + dw(&s.dw) + pw(&s.pw);
        }
        n
    }

    /// SHA-256 of the weights file (hex).
    #[must_use]
    pub fn sha256(&self) -> &str {
        &self.sha256
    }

    /// Number of parameters, batch norms folded.
    #[must_use]
    pub fn parameters(&self) -> usize {
        self.parameters
    }

    /// The network on an encoded plane: `e` is planar `3 × height × width` (the output of
    /// [`encode_linear`] after [`pad_encoded`]), `width` and `height` multiples of
    /// [`PAD_MULTIPLE`]. The input is normalised here (`(e - 0.30) / 0.15`), as the training
    /// code's `forward` does. The answer is `(μ, s)` at half resolution.
    ///
    /// # Errors
    ///
    /// [`MlError::Shape`] when the size is not a positive multiple of 32 or `e` has the wrong
    /// length.
    pub fn forward_padded(&self, e: &[f32], width: usize, height: usize) -> Result<RawOutput> {
        if width == 0
            || height == 0
            || !width.is_multiple_of(PAD_MULTIPLE)
            || !height.is_multiple_of(PAD_MULTIPLE)
        {
            return Err(MlError::Shape(format!(
                "the input is {width} x {height}, not a positive multiple of {PAD_MULTIPLE}"
            )));
        }
        if e.len() != 3 * width * height {
            return Err(MlError::Shape(format!(
                "a {width} x {height} plane needs {} values, got {}",
                3 * width * height,
                e.len()
            )));
        }
        let mut input = Feat::zeros(3, height, width);
        input
            .data
            .par_chunks_mut(3 * width)
            .enumerate()
            .for_each(|(y, row)| {
                for c in 0..3 {
                    let src = &e[(c * height + y) * width..(c * height + y + 1) * width];
                    for (d, &v) in row[c * width..(c + 1) * width].iter_mut().zip(src) {
                        *d = (v - INPUT_MEAN) / INPUT_STD;
                    }
                }
            });

        let mut cur = Arc::new(self.stem.forward(&input));
        drop(input);
        let mut skips = vec![Arc::clone(&cur)];
        for (i, block) in self.blocks.iter().enumerate() {
            cur = Arc::new(block.forward(&cur));
            if self.taps.contains(&i) {
                skips.push(Arc::clone(&cur));
            }
        }
        let mut x = (*cur).clone();
        drop(cur);
        let context = self.context.vector(&channel_means(&x), Act::Relu6);
        let (n, cells) = (x.row_len(), x.w);
        x.data.par_chunks_mut(n).for_each(|row| {
            for (plane, &add) in row.chunks_exact_mut(cells).zip(&context) {
                for v in plane {
                    *v += add;
                }
            }
        });
        let coarse = skips.len() - 1;
        for (j, step) in self.steps.iter().enumerate() {
            x = step.forward(&x, &skips[coarse - 1 - j])?;
        }
        let out = self.head.forward(&x, Act::None, None);
        let (w2, h2) = (out.w, out.h);
        let mut mu = vec![0.0f32; w2 * h2];
        let mut s = vec![0.0f32; w2 * h2];
        for y in 0..h2 {
            let row = out.row(y);
            mu[y * w2..(y + 1) * w2].copy_from_slice(&row[..w2]);
            s[y * w2..(y + 1) * w2].copy_from_slice(&row[w2..2 * w2]);
        }
        Ok(RawOutput {
            width: w2,
            height: h2,
            mu,
            s,
        })
    }

    /// The bokeh map of a lens plane (module documentation).
    ///
    /// # Errors
    ///
    /// [`MlError::Shape`] when the plane's size does not match its data, [`MlError::Input`] when it
    /// holds a value that is not finite.
    pub fn map(&self, plane: &PlanarRgbRef<'_>) -> Result<BokehMap> {
        Ok(self.map_timed(plane)?.0)
    }

    /// [`BokehNet::map`] with the time each part took.
    ///
    /// # Errors
    ///
    /// As [`BokehNet::map`].
    pub fn map_timed(&self, plane: &PlanarRgbRef<'_>) -> Result<(BokehMap, MapTimings)> {
        plane.check()?;
        let t0 = Instant::now();
        let (rw, rh) = reduced_size(plane.width, plane.height);
        let reduced = box_reduce(plane, rw, rh)?;
        let t1 = Instant::now();
        let encoded = encode_linear(&reduced, rw, rh)?;
        drop(reduced);
        let (padded, pw, ph) = pad_encoded(&encoded, rw, rh);
        drop(encoded);
        let t2 = Instant::now();
        let raw = self.forward_padded(&padded, pw, ph)?;
        let t3 = Instant::now();
        let map = raw.into_map(rw, rh);
        let t4 = Instant::now();
        let ms = |a: Instant, b: Instant| b.duration_since(a).as_secs_f64() * 1e3;
        Ok((
            map,
            MapTimings {
                reduce_ms: ms(t0, t1),
                encode_ms: ms(t1, t2),
                forward_ms: ms(t2, t3),
                convert_ms: ms(t3, t4),
            },
        ))
    }
}

/// Where the time of one [`BokehNet::map_timed`] went, in milliseconds.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MapTimings {
    /// The box reduction to a long edge of [`FRAME_LONG_PX`].
    pub reduce_ms: f64,
    /// Encoding and padding.
    pub encode_ms: f64,
    /// The network.
    pub forward_ms: f64,
    /// The conversion of the output to `b` and `c`.
    pub convert_ms: f64,
}

// -------------------------------------------------------------------------------------------------
// From a lens plane to the network's input, and back to a map
// -------------------------------------------------------------------------------------------------

/// A borrowed lens plane: planar scene-linear camera RGB (all of R, then all of G, then all of B,
/// row-major), any overall scale.
#[derive(Debug, Clone, Copy)]
pub struct PlanarRgbRef<'a> {
    /// Width in pixels.
    pub width: usize,
    /// Height in pixels.
    pub height: usize,
    /// `3 × height × width` values.
    pub data: &'a [f32],
}

impl<'a> PlanarRgbRef<'a> {
    /// A plane over `data`, checked.
    ///
    /// # Errors
    ///
    /// [`MlError::Shape`] when a side is 0 or `data` is not `3 × height × width` values.
    pub fn new(width: usize, height: usize, data: &'a [f32]) -> Result<Self> {
        let plane = Self {
            width,
            height,
            data,
        };
        plane.check()?;
        Ok(plane)
    }

    fn check(&self) -> Result<()> {
        if self.width == 0
            || self.height == 0
            || self.width.checked_mul(self.height).and_then(|n| n.checked_mul(3))
                != Some(self.data.len())
        {
            return Err(MlError::Shape(format!(
                "the plane is {} x {} pixels but holds {} values (3 x width x height expected)",
                self.width,
                self.height,
                self.data.len()
            )));
        }
        Ok(())
    }
}

/// The size the network sees for a plane of `width × height`: the long edge reduced to
/// [`FRAME_LONG_PX`], the short edge in proportion (rounded, at least 1); a plane that is not
/// longer is used as it is (it is never enlarged).
#[must_use]
pub fn reduced_size(width: usize, height: usize) -> (usize, usize) {
    let long = width.max(height);
    if long <= FRAME_LONG_PX {
        return (width, height);
    }
    let scale = FRAME_LONG_PX as f64 / long as f64;
    let fit = |side: usize| {
        if side == long {
            FRAME_LONG_PX
        } else {
            ((side as f64 * scale).round() as usize).clamp(1, FRAME_LONG_PX)
        }
    };
    (fit(width), fit(height))
}

/// Per output sample of a box reduction from `n` to `m` samples: the first input sample and the
/// weights, each input sample's overlap with the output's footprint (summing to 1).
fn box_weights(n: usize, m: usize) -> Vec<(usize, Vec<f64>)> {
    let scale = n as f64 / m as f64;
    (0..m)
        .map(|o| {
            let (a, b) = (o as f64 * scale, ((o + 1) as f64 * scale).min(n as f64));
            let first = (a.floor() as usize).min(n - 1);
            let last = (b.ceil() as usize).clamp(first + 1, n);
            let mut w: Vec<f64> = (first..last)
                .map(|i| ((i + 1) as f64).min(b) - (i as f64).max(a))
                .map(|v| v.max(0.0))
                .collect();
            let total: f64 = w.iter().sum();
            if total > 0.0 {
                for v in &mut w {
                    *v /= total;
                }
            } else {
                w = vec![1.0];
            }
            (first, w)
        })
        .collect()
}

/// Reduces a plane to `out_w × out_h` with an exact area-weighted box filter (a horizontal pass,
/// then a vertical one), in `f64`, planar.
///
/// # Errors
///
/// [`MlError::Shape`] when the plane does not match its size, [`MlError::Input`] when it holds a
/// value that is not finite.
pub fn box_reduce(plane: &PlanarRgbRef<'_>, out_w: usize, out_h: usize) -> Result<Vec<f64>> {
    plane.check()?;
    if out_w == 0 || out_h == 0 || out_w > plane.width || out_h > plane.height {
        return Err(MlError::Shape(format!(
            "cannot reduce {} x {} to {out_w} x {out_h}",
            plane.width, plane.height
        )));
    }
    if plane.data.iter().any(|v| !v.is_finite()) {
        return Err(MlError::Input(
            "the plane holds a value that is not finite".into(),
        ));
    }
    let (w, h) = (plane.width, plane.height);
    let columns = box_weights(w, out_w);
    let rows = box_weights(h, out_h);
    // Horizontal pass: every input row of every channel.
    let mut mid = vec![0.0f64; 3 * h * out_w];
    mid.par_chunks_mut(out_w).enumerate().for_each(|(r, dst)| {
        let src = &plane.data[r * w..(r + 1) * w];
        for (d, (first, weights)) in dst.iter_mut().zip(&columns) {
            let mut acc = 0.0f64;
            for (k, &wk) in weights.iter().enumerate() {
                acc += wk * f64::from(src[first + k]);
            }
            *d = acc;
        }
    });
    // Vertical pass: every output row of every channel.
    let mut out = vec![0.0f64; 3 * out_h * out_w];
    out.par_chunks_mut(out_w).enumerate().for_each(|(r, dst)| {
        let (c, y) = (r / out_h, r % out_h);
        let (first, weights) = &rows[y];
        for (k, &wk) in weights.iter().enumerate() {
            let src = &mid[(c * h + first + k) * out_w..(c * h + first + k + 1) * out_w];
            for (d, &v) in dst.iter_mut().zip(src) {
                *d += wk * v;
            }
        }
    });
    Ok(out)
}

/// The fixed encoding of a linear plane (`training/bokeh/encode.py`, `encode_np`): `rgb` is planar
/// `3 × height × width`; the answer is planar too, in `f64`.
///
/// # Errors
///
/// [`MlError::Shape`] when `rgb` is not `3 × height × width` values, [`MlError::Input`] when one is
/// not finite.
pub fn encode_linear(rgb: &[f64], width: usize, height: usize) -> Result<Vec<f64>> {
    let n = width * height;
    if n == 0 || rgb.len() != 3 * n {
        return Err(MlError::Shape(format!(
            "a {width} x {height} plane needs {} values, got {}",
            3 * n,
            rgb.len()
        )));
    }
    if rgb.iter().any(|v| !v.is_finite()) {
        return Err(MlError::Input(
            "the plane holds a value that is not finite".into(),
        ));
    }
    let luma: Vec<f64> = (0..n)
        .map(|i| LUMA[0] * rgb[i] + LUMA[1] * rgb[n + i] + LUMA[2] * rgb[2 * n + i])
        .collect();
    let mean = luma.iter().sum::<f64>() / n as f64;
    // The lower median: the element at index (n - 1) / 2 of the sorted values.
    let mut sorted = luma;
    let (_, median, _) = sorted.select_nth_unstable_by((n - 1) / 2, f64::total_cmp);
    let m = median.max(mean * MEDIAN_FLOOR_OF_MEAN).max(1e-12);
    let gain = MID / m;
    let mut out = vec![0.0f64; 3 * n];
    out.par_chunks_mut(width)
        .zip(rgb.par_chunks(width))
        .for_each(|(dst, src)| {
            for (d, &v) in dst.iter_mut().zip(src) {
                let v = (v * gain).max(0.0);
                *d = libm::log2(1.0 + v / K) / STOPS;
            }
        });
    Ok(out)
}

/// Pads an encoded plane (planar `3 × height × width`) to a multiple of [`PAD_MULTIPLE`] with
/// replicated edge pixels and narrows it to `f32`. Returns the plane and its new width and height.
#[must_use]
pub fn pad_encoded(e: &[f64], width: usize, height: usize) -> (Vec<f32>, usize, usize) {
    let (pw, ph) = (
        width.next_multiple_of(PAD_MULTIPLE),
        height.next_multiple_of(PAD_MULTIPLE),
    );
    let mut out = vec![0.0f32; 3 * pw * ph];
    out.par_chunks_mut(pw).enumerate().for_each(|(r, dst)| {
        let (c, y) = (r / ph, (r % ph).min(height - 1));
        let src = &e[(c * height + y) * width..(c * height + y + 1) * width];
        for (x, d) in dst.iter_mut().enumerate() {
            *d = src[x.min(width - 1)] as f32;
        }
    });
    (out, pw, ph)
}

/// The network's raw answer at half resolution, padded size: `μ = ln r` (`r` the disc radius in
/// pixels of the reduced plane) and `s = ln β` (`β` the Laplace scale of the error of `μ`),
/// row-major.
#[derive(Debug, Clone, PartialEq)]
pub struct RawOutput {
    /// Columns.
    pub width: usize,
    /// Rows.
    pub height: usize,
    /// `μ`, `width × height` values.
    pub mu: Vec<f32>,
    /// `s`, `width × height` values.
    pub s: Vec<f32>,
}

impl RawOutput {
    /// Crops the padding away and converts to the spec's units (module documentation), for a
    /// reduced plane of `plane_w × plane_h` pixels (before padding).
    #[must_use]
    pub fn into_map(self, plane_w: usize, plane_h: usize) -> BokehMap {
        let (width, height) = (plane_w.div_ceil(OUTPUT_STRIDE), plane_h.div_ceil(OUTPUT_STRIDE));
        let half_diagonal_px = 0.5 * (plane_w as f64).hypot(plane_h as f64);
        let mut b = vec![0.0f32; width * height];
        let mut c = vec![0.0f32; width * height];
        for y in 0..height.min(self.height) {
            for x in 0..width.min(self.width) {
                let (mu, s) = (
                    f64::from(self.mu[y * self.width + x]),
                    f64::from(self.s[y * self.width + x]),
                );
                if mu.is_finite() && s.is_finite() {
                    let r = libm::exp(mu.max(R_FLOOR_PX.ln()));
                    let beta = libm::exp(s.clamp(LOG_BETA_RANGE.0, LOG_BETA_RANGE.1));
                    b[y * width + x] = (r / half_diagonal_px) as f32;
                    c[y * width + x] = libm::exp(-beta / CONF_BETA0) as f32;
                }
            }
        }
        BokehMap {
            width,
            height,
            plane_width: plane_w,
            plane_height: plane_h,
            half_diagonal_px,
            b,
            c,
        }
    }
}

/// The bokeh map of a lens plane: one cell per 2 × 2 pixels of the plane the network saw.
///
/// `b` is the blur radius in half-diagonals of the frame and `c` the confidence in `[0, 1]`
/// (`docs/21` §7); both are raw (module documentation). Cells are row-major.
#[derive(Debug, Clone, PartialEq)]
pub struct BokehMap {
    /// Columns of the map.
    pub width: usize,
    /// Rows of the map.
    pub height: usize,
    /// Width of the reduced plane the network saw, in pixels.
    pub plane_width: usize,
    /// Height of the reduced plane the network saw, in pixels.
    pub plane_height: usize,
    /// Half-diagonal of that plane in its pixels: `b = r / half_diagonal_px`.
    pub half_diagonal_px: f64,
    /// Blur radius per cell, in half-diagonals (at least `1 / half_diagonal_px`).
    pub b: Vec<f32>,
    /// Confidence per cell, `exp(-β / 0.25)`.
    pub c: Vec<f32>,
}

impl BokehMap {
    /// `(b, c)` at a position given as fractions of the plane, `(0, 0)` its top-left corner and
    /// `(1, 1)` its bottom-right (pixel `x` of a plane `W` wide is at `(x + 0.5) / W`), by bilinear
    /// interpolation between the cells' centres, held at the edge beyond the outer centres.
    #[must_use]
    pub fn sample(&self, u: f64, v: f64) -> (f32, f32) {
        let cell = |t: f64, plane: usize, cells: usize| {
            let p = (t * plane as f64 / OUTPUT_STRIDE as f64 - 0.5).clamp(0.0, (cells - 1) as f64);
            let p = if p.is_nan() { 0.0 } else { p };
            let i0 = (p.floor() as usize).min(cells - 1);
            (i0, (i0 + 1).min(cells - 1), (p - i0 as f64) as f32)
        };
        let (x0, x1, fx) = cell(u, self.plane_width, self.width);
        let (y0, y1, fy) = cell(v, self.plane_height, self.height);
        let lerp = |m: &[f32]| {
            let top = (1.0 - fx) * m[y0 * self.width + x0] + fx * m[y0 * self.width + x1];
            let bottom = (1.0 - fx) * m[y1 * self.width + x0] + fx * m[y1 * self.width + x1];
            (1.0 - fy) * top + fy * bottom
        };
        (lerp(&self.b), lerp(&self.c))
    }

    /// `(b, c)` at pixel `(x, y)` (fractions allowed; the pixel's centre is at `+0.5`) of a plane
    /// `width × height` pixels, which may be any size: the lens plane the map was made from, or
    /// any reduction of it.
    #[must_use]
    pub fn sample_px(&self, x: f64, y: f64, width: usize, height: usize) -> (f32, f32) {
        self.sample((x + 0.5) / width as f64, (y + 0.5) / height as f64)
    }

    /// The map resampled to `width × height` cells (bilinear, [`BokehMap::sample_px`]): `(b, c)`,
    /// row-major.
    #[must_use]
    pub fn resample(&self, width: usize, height: usize) -> (Vec<f32>, Vec<f32>) {
        let mut b = vec![0.0f32; width * height];
        let mut c = vec![0.0f32; width * height];
        for y in 0..height {
            for x in 0..width {
                let (bv, cv) = self.sample_px(x as f64, y as f64, width, height);
                b[y * width + x] = bv;
                c[y * width + x] = cv;
            }
        }
        (b, c)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A tiny deterministic generator for test data in `[-0.5, 0.5)`.
    struct Lcg(u32);

    impl Lcg {
        fn next(&mut self) -> f32 {
            self.0 = self.0.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (self.0 >> 8) as f32 / (1u32 << 24) as f32 - 0.5
        }

        fn vec(&mut self, n: usize) -> Vec<f32> {
            (0..n).map(|_| self.next()).collect()
        }
    }

    fn feat(rng: &mut Lcg, c: usize, h: usize, w: usize) -> Feat {
        Feat {
            c,
            h,
            w,
            data: rng.vec(c * h * w),
        }
    }

    #[test]
    fn pointwise_agrees_with_a_naive_loop_at_every_width() {
        let mut rng = Lcg(7);
        // 11 outputs (not a multiple of the block), widths below, at and above the tile.
        let (cin, cout) = (5, 11);
        let weight = rng.vec(cout * cin);
        let bias = rng.vec(cout);
        let layer = Pointwise::new(&weight, bias.clone(), cin, cout);
        for w in [1, 3, 15, 16, 17, 33] {
            let x = feat(&mut rng, cin, 2, w);
            let res = feat(&mut rng, cout, 2, w);
            for act in [Act::None, Act::Relu6] {
                let got = layer.forward(&x, act, Some(&res));
                for y in 0..2 {
                    for o in 0..cout {
                        for xx in 0..w {
                            let mut s = f64::from(bias[o]);
                            for i in 0..cin {
                                s += f64::from(weight[o * cin + i])
                                    * f64::from(x.data[(y * cin + i) * w + xx]);
                            }
                            s += f64::from(res.data[(y * cout + o) * w + xx]);
                            let want = if act == Act::Relu6 { s.clamp(0.0, 6.0) } else { s };
                            let g = f64::from(got.data[(y * cout + o) * w + xx]);
                            assert!((g - want).abs() < 1e-5, "w {w} o {o}: {g} vs {want}");
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn a_pixels_value_does_not_depend_on_its_tile() {
        // The same column gets the same bits whatever the width of the map it is in.
        let mut rng = Lcg(11);
        let (cin, cout) = (9, 6);
        let layer = Pointwise::new(&rng.vec(cout * cin), rng.vec(cout), cin, cout);
        let wide = feat(&mut rng, cin, 1, 40);
        let a = layer.forward(&wide, Act::Relu6, None);
        for start in [0, 5, 16, 23] {
            let mut narrow = Feat::zeros(cin, 1, 40 - start);
            for i in 0..cin {
                narrow.data[i * (40 - start)..(i + 1) * (40 - start)]
                    .copy_from_slice(&wide.data[i * 40 + start..(i + 1) * 40]);
            }
            let b = layer.forward(&narrow, Act::Relu6, None);
            for o in 0..cout {
                for x in 0..40 - start {
                    assert_eq!(
                        a.data[o * 40 + start + x].to_bits(),
                        b.data[o * (40 - start) + x].to_bits()
                    );
                }
            }
        }
    }

    #[test]
    fn depthwise_agrees_with_a_naive_loop_for_both_strides() {
        let mut rng = Lcg(3);
        let c = 5;
        for stride in [1, 2] {
            let layer = Depthwise {
                c,
                stride,
                weight: rng.vec(c * 9),
                bias: rng.vec(c),
            };
            for (h, w) in [(1, 1), (2, 3), (5, 8), (6, 7), (9, 33)] {
                let x = feat(&mut rng, c, h, w);
                let got = layer.forward(&x, Act::None);
                let (ho, wo) = ((h - 1) / stride + 1, (w - 1) / stride + 1);
                assert_eq!((got.h, got.w), (ho, wo));
                for ch in 0..c {
                    for yo in 0..ho {
                        for xo in 0..wo {
                            let mut s = f64::from(layer.bias[ch]);
                            for ky in 0..3 {
                                for kx in 0..3 {
                                    let iy = (stride * yo + ky) as isize - 1;
                                    let ix = (stride * xo + kx) as isize - 1;
                                    if iy >= 0 && iy < h as isize && ix >= 0 && ix < w as isize {
                                        s += f64::from(layer.weight[ch * 9 + ky * 3 + kx])
                                            * f64::from(
                                                x.data[(iy as usize * c + ch) * w + ix as usize],
                                            );
                                    }
                                }
                            }
                            let g = f64::from(got.data[(yo * c + ch) * wo + xo]);
                            assert!(
                                (g - s).abs() < 1e-5,
                                "stride {stride} {h}x{w} ch {ch} ({yo},{xo}): {g} vs {s}"
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn the_stem_agrees_with_a_naive_loop() {
        let mut rng = Lcg(5);
        let cout = 7;
        let stem = Stem {
            cout,
            weight: rng.vec(cout * 27),
            bias: rng.vec(cout),
        };
        for (h, w) in [(2, 2), (6, 10), (7, 9)] {
            let x = feat(&mut rng, 3, h, w);
            let got = stem.forward(&x);
            let (ho, wo) = ((h - 1) / 2 + 1, (w - 1) / 2 + 1);
            assert_eq!((got.h, got.w), (ho, wo));
            for o in 0..cout {
                for yo in 0..ho {
                    for xo in 0..wo {
                        let mut s = f64::from(stem.bias[o]);
                        for i in 0..3 {
                            for ky in 0..3 {
                                for kx in 0..3 {
                                    let iy = (2 * yo + ky) as isize - 1;
                                    let ix = (2 * xo + kx) as isize - 1;
                                    if iy >= 0 && iy < h as isize && ix >= 0 && ix < w as isize {
                                        s += f64::from(stem.weight[((o * 3 + i) * 3 + ky) * 3 + kx])
                                            * f64::from(
                                                x.data[(iy as usize * 3 + i) * w + ix as usize],
                                            );
                                    }
                                }
                            }
                        }
                        let g = f64::from(got.data[(yo * cout + o) * wo + xo]);
                        assert!((g - s.clamp(0.0, 6.0)).abs() < 1e-5, "{g} vs {s}");
                    }
                }
            }
        }
    }

    #[test]
    fn the_upsample_is_pytorch_s_half_pixel_bilinear() {
        // Output i reads input (i + 0.5) / 2 - 0.5, held at the first sample: for n = 3 the
        // weights are [0, 0.75, 0.25, ...] on (i0, i1) as below.
        let axis = upsample_axis(3);
        let want = [
            (0, 1, 0.0),
            (0, 1, 0.25),
            (0, 1, 0.75),
            (1, 2, 0.25),
            (1, 2, 0.75),
            (2, 2, 0.25),
        ];
        assert_eq!(axis.len(), 6);
        for (got, want) in axis.iter().zip(want) {
            assert_eq!((got.0, got.1), (want.0, want.1));
            // 0.25 is exact; the last sample's i1 is clamped to itself, so its weight is moot.
            assert!((got.2 - want.2).abs() < 1e-7, "{got:?} vs {want:?}");
        }
    }

    #[test]
    fn tap_ranges_cover_exactly_the_inputs_inside() {
        for (w, stride) in [(1usize, 1usize), (2, 1), (5, 1), (1, 2), (2, 2), (7, 2), (8, 2)] {
            let wo = (w - 1) / stride + 1;
            for kx in 0..3 {
                let (lo, hi) = tap_range(w, wo, stride, kx);
                for xo in 0..wo {
                    let ix = (stride * xo + kx) as isize - 1;
                    let inside = ix >= 0 && ix < w as isize;
                    assert_eq!((lo..hi).contains(&xo), inside, "w {w} s {stride} kx {kx} xo {xo}");
                }
            }
        }
    }

    #[test]
    fn the_reduced_size_keeps_the_aspect_and_never_enlarges() {
        assert_eq!(reduced_size(1512, 1006), (1024, 681));
        assert_eq!(reduced_size(1006, 1512), (681, 1024));
        assert_eq!(reduced_size(2047, 1365), (1024, 683));
        assert_eq!(reduced_size(1024, 683), (1024, 683));
        assert_eq!(reduced_size(300, 200), (300, 200));
        assert_eq!(reduced_size(4000, 1), (1024, 1));
    }

    #[test]
    fn the_box_reduction_keeps_flat_planes_flat_and_the_mean() {
        let (w, h) = (37, 23);
        let flat = vec![0.75f32; 3 * w * h];
        let r = box_reduce(&PlanarRgbRef::new(w, h, &flat).unwrap(), 16, 10).unwrap();
        assert!(r.iter().all(|v| (v - 0.75).abs() < 1e-12));
        let mut rng = Lcg(9);
        let noisy: Vec<f32> = rng.vec(3 * w * h).iter().map(|v| v + 1.0).collect();
        let r = box_reduce(&PlanarRgbRef::new(w, h, &noisy).unwrap(), 16, 10).unwrap();
        for c in 0..3 {
            let before: f64 = noisy[c * w * h..(c + 1) * w * h].iter().map(|&v| f64::from(v)).sum();
            let after: f64 = r[c * 160..(c + 1) * 160].iter().sum();
            // Every output cell is a weighted mean over a footprint of area (w/16)(h/10); the mean
            // of the plane is the mean of the cells.
            assert!(
                (before / (w * h) as f64 - after / 160.0).abs() < 1e-6,
                "{before} {after}"
            );
        }
        // Halving by two exact pairs is the mean of the pair.
        let ramp: Vec<f32> = (0..3 * 4 * 2).map(|i| (i % 4) as f32).collect();
        let r = box_reduce(&PlanarRgbRef::new(4, 2, &ramp).unwrap(), 2, 1).unwrap();
        assert_eq!(&r[..2], &[0.5, 2.5]);
    }

    #[test]
    fn the_encoding_is_scale_free_and_matches_the_formula() {
        let (w, h) = (9, 7);
        let mut rng = Lcg(21);
        let plane: Vec<f64> = rng.vec(3 * w * h).iter().map(|&v| f64::from(v) + 0.6).collect();
        let e = encode_linear(&plane, w, h).unwrap();
        let doubled: Vec<f64> = plane.iter().map(|v| v * 2.0).collect();
        let e2 = encode_linear(&doubled, w, h).unwrap();
        for (a, b) in e.iter().zip(&e2) {
            assert!((a - b).abs() < 1e-12);
        }
        // The pixel at the median luminance maps to log2(1 + 0.18 / K) / 16 on a grey plane.
        let grey = vec![0.4f64; 3 * 4 * 4];
        let e = encode_linear(&grey, 4, 4).unwrap();
        let want = (1.0 + MID / K).log2() / STOPS;
        assert!(e.iter().all(|v| (v - want).abs() < 1e-12));
        assert!(encode_linear(&[f64::NAN; 12], 2, 2).is_err());
        assert!(encode_linear(&[1.0; 11], 2, 2).is_err());
    }

    #[test]
    fn the_median_is_the_lower_one() {
        // Luminance values 1, 2, 3, 4 (n = 4): the lower median is 2, so a pixel of 2 maps to 0.18.
        let n = 4;
        let mut rgb = vec![0.0f64; 3 * n];
        for i in 0..n {
            for c in 0..3 {
                rgb[c * n + i] = (i + 1) as f64;
            }
        }
        let e = encode_linear(&rgb, 2, 2).unwrap();
        let want = (1.0 + MID / K).log2() / STOPS;
        assert!((e[1] - want).abs() < 1e-12, "{} vs {want}", e[1]);
    }

    #[test]
    fn padding_replicates_the_edges() {
        let (w, h) = (3, 2);
        let e: Vec<f64> = (0..3 * w * h).map(|i| i as f64 / 100.0).collect();
        let (p, pw, ph) = pad_encoded(&e, w, h);
        assert_eq!((pw, ph), (32, 32));
        for c in 0..3 {
            for y in 0..ph {
                for x in 0..pw {
                    let want = e[(c * h + y.min(h - 1)) * w + x.min(w - 1)] as f32;
                    assert_eq!(p[(c * ph + y) * pw + x], want);
                }
            }
        }
    }

    #[test]
    fn the_output_conversion_follows_the_training_code() {
        // mu = ln 10 px, s = ln 0.25 on a 600 x 400 plane: hd = hypot(600, 400) / 2.
        let raw = RawOutput {
            width: 2,
            height: 2,
            mu: vec![10f32.ln(), 0.5f32.ln(), 1000f32.ln(), f32::NAN],
            s: vec![0.25f32.ln(), -20.0, 20.0, 0.0],
        };
        let map = raw.into_map(3, 3); // 3 x 3 pixels: 2 x 2 cells
        assert_eq!((map.width, map.height), (2, 2));
        let hd = 0.5 * 18f64.sqrt();
        assert!((f64::from(map.b[0]) - 10.0 / hd).abs() < 1e-5);
        assert!((f64::from(map.c[0]) - (-1.0f64).exp()).abs() < 1e-6);
        // Below one pixel the radius is held at one pixel; beta is clamped to e^-6 and e^3.
        assert!((f64::from(map.b[1]) - 1.0 / hd).abs() < 1e-6);
        assert!((f64::from(map.c[1]) - (-(-6f64).exp() / 0.25).exp()).abs() < 1e-6);
        assert!((f64::from(map.c[2]) - (-(3f64).exp() / 0.25).exp()).abs() < 1e-9);
        // A value that is not finite is "not confident".
        assert_eq!((map.b[3], map.c[3]), (0.0, 0.0));
    }

    #[test]
    fn sampling_is_bilinear_between_cell_centres() {
        // A 4 x 2 plane gives 2 x 1 cells with centres at u = 0.25 and 0.75.
        let map = BokehMap {
            width: 2,
            height: 1,
            plane_width: 4,
            plane_height: 2,
            half_diagonal_px: 2.0,
            b: vec![1.0, 3.0],
            c: vec![0.0, 1.0],
        };
        assert_eq!(map.sample(0.25, 0.5), (1.0, 0.0));
        assert_eq!(map.sample(0.75, 0.5), (3.0, 1.0));
        assert_eq!(map.sample(0.5, 0.5), (2.0, 0.5));
        assert_eq!(map.sample(0.0, 0.0), (1.0, 0.0));
        assert_eq!(map.sample(1.0, 1.0), (3.0, 1.0));
        // Pixel 1 of 4 is at u = 0.375: a quarter of the way from the first centre.
        let (b, _) = map.sample_px(1.0, 0.0, 4, 2);
        assert!((b - 1.5).abs() < 1e-6);
        let (b, c) = map.resample(8, 4);
        assert_eq!((b.len(), c.len()), (32, 32));
        assert!(b.iter().all(|&v| (1.0..=3.0).contains(&v)));
    }
}
