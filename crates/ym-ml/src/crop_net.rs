//! The crop student on the CPU (`docs/20-auto-crop.md` §6, `training/crop/model.py`): a small
//! convolutional network that looks at a photo's picture and answers one free crop, as fractions of
//! the photo.
//!
//! The network is the one `training/crop/export.py` writes to `data/models/crop/cropnet.safetensors`,
//! batch norms folded into the convolutions: a 3×3 stem of stride 2, seventeen inverted-residual
//! blocks (1×1 expansion, 3×3 depthwise of stride 1 or 2, 1×1 projection, a residual when the shape
//! is kept), a 1×1 neck to 32 channels at 8×8, and two linear layers. SiLU after every layer but the
//! projections and the last. The last layer's four outputs, through a sigmoid, are `p` in `[0, 1]`
//! with `x0 = p0`, `x1 = x0 + (1 − x0) p2` (and the same for y), so every answer is a box inside the
//! photo.
//!
//! The input is the picture letterboxed into a 256 × 256 square ([`letterbox`]): scaled to fit with
//! the bilinear filter of Pillow's `Image.resize` (the training data's, reproduced to the bit, so the
//! network sees here exactly what it saw in training), centred on black. Everything is `f32` on one
//! thread in a fixed order: the same picture gives the same bits on every run.

use std::path::Path;

use safetensors::SafeTensors;

use crate::cpu_runtime::read_float_tensors;
use crate::model::{MlError, Result, sha256_hex, verify_sha256};

/// Format string of the safetensors header (`training/crop/export.py`).
pub const FORMAT: &str = "ym-cropnet";
/// Format version this runtime reads.
pub const VERSION: &str = "1";
/// Side of the square the network looks at, in pixels.
pub const INPUT_PX: usize = 256;
/// The weights file in the model folder (`data/models/crop/`).
pub const WEIGHTS_FILE: &str = "cropnet.safetensors";

/// `(expansion, output channels, blocks, stride)` of each stage after the stem (`model.py`'s
/// `STAGES`).
const STAGES: [(usize, usize, usize, usize); 5] = [
    (1, 24, 1, 1),
    (4, 32, 2, 2),
    (4, 48, 3, 2),
    (4, 96, 3, 2),
    (4, 160, 2, 2),
];
/// Channels out of the stem.
const STEM_CHANNELS: usize = 24;
/// Channels out of the neck.
const HEAD_CHANNELS: usize = 32;
/// Units of the hidden linear layer.
const HIDDEN: usize = 256;
/// Side of the neck's grid (the input over the network's stride of 32).
const GRID: usize = INPUT_PX / 32;
/// The input normalisation: `(v / 255 − MEAN) / STD`.
const MEAN: f32 = 0.5;
const STD: f32 = 0.25;
/// Fixed-point bits of Pillow's 8-bit resampling (`32 − 8 − 2`).
const PRECISION_BITS: u32 = 22;

/// A planar `f32` tensor: `c` channels of `h × w`.
#[derive(Debug, Clone)]
struct Planes {
    c: usize,
    h: usize,
    w: usize,
    data: Vec<f32>,
}

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// A 3×3 convolution of stride 2 with zero padding 1 (the stem), then SiLU.
#[derive(Debug, Clone)]
struct Stem {
    cin: usize,
    cout: usize,
    /// `[cout][cin][3][3]`.
    weight: Vec<f32>,
    bias: Vec<f32>,
}

impl Stem {
    fn apply(&self, x: &Planes) -> Planes {
        let (h, w) = ((x.h - 1) / 2 + 1, (x.w - 1) / 2 + 1);
        let mut out = vec![0.0; self.cout * h * w];
        for o in 0..self.cout {
            for oy in 0..h {
                for ox in 0..w {
                    let mut acc = self.bias[o];
                    for i in 0..self.cin {
                        for ky in 0..3 {
                            let Some(iy) = (2 * oy + ky).checked_sub(1).filter(|&v| v < x.h) else {
                                continue;
                            };
                            for kx in 0..3 {
                                let Some(ix) = (2 * ox + kx).checked_sub(1).filter(|&v| v < x.w)
                                else {
                                    continue;
                                };
                                acc += self.weight[((o * self.cin + i) * 3 + ky) * 3 + kx]
                                    * x.data[(i * x.h + iy) * x.w + ix];
                            }
                        }
                    }
                    out[(o * h + oy) * w + ox] = silu(acc);
                }
            }
        }
        Planes {
            c: self.cout,
            h,
            w,
            data: out,
        }
    }
}

/// A 1×1 convolution, optionally followed by SiLU.
#[derive(Debug, Clone)]
struct Pointwise {
    cin: usize,
    cout: usize,
    /// `[cout][cin]`.
    weight: Vec<f32>,
    bias: Vec<f32>,
}

impl Pointwise {
    fn apply(&self, x: &Planes, act: bool) -> Planes {
        let n = x.h * x.w;
        let mut out = vec![0.0; self.cout * n];
        for (o, row) in out.chunks_exact_mut(n).enumerate() {
            row.fill(self.bias[o]);
            for i in 0..self.cin {
                let k = self.weight[o * self.cin + i];
                let src = &x.data[i * n..(i + 1) * n];
                for (r, &s) in row.iter_mut().zip(src) {
                    *r += k * s;
                }
            }
            if act {
                for r in row.iter_mut() {
                    *r = silu(*r);
                }
            }
        }
        Planes {
            c: self.cout,
            h: x.h,
            w: x.w,
            data: out,
        }
    }
}

/// A 3×3 depthwise convolution of stride 1 or 2 with zero padding 1, then SiLU.
#[derive(Debug, Clone)]
struct Depthwise {
    c: usize,
    stride: usize,
    /// `[c][3][3]`.
    weight: Vec<f32>,
    bias: Vec<f32>,
}

impl Depthwise {
    fn apply(&self, x: &Planes) -> Planes {
        let s = self.stride;
        let (h, w) = ((x.h - 1) / s + 1, (x.w - 1) / s + 1);
        let mut out = vec![0.0; self.c * h * w];
        for c in 0..self.c {
            let src = &x.data[c * x.h * x.w..(c + 1) * x.h * x.w];
            let k = &self.weight[c * 9..(c + 1) * 9];
            for oy in 0..h {
                for ox in 0..w {
                    let mut acc = self.bias[c];
                    for ky in 0..3 {
                        let Some(iy) = (s * oy + ky).checked_sub(1).filter(|&v| v < x.h) else {
                            continue;
                        };
                        for kx in 0..3 {
                            let Some(ix) = (s * ox + kx).checked_sub(1).filter(|&v| v < x.w) else {
                                continue;
                            };
                            acc += k[ky * 3 + kx] * src[iy * x.w + ix];
                        }
                    }
                    out[(c * h + oy) * w + ox] = silu(acc);
                }
            }
        }
        Planes {
            c: self.c,
            h,
            w,
            data: out,
        }
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
    fn apply(&self, x: &Planes) -> Planes {
        let expanded;
        let mid = match &self.expand {
            Some(e) => {
                expanded = e.apply(x, true);
                &expanded
            }
            None => x,
        };
        let mut y = self.project.apply(&self.depthwise.apply(mid), false);
        if self.residual {
            for (a, b) in y.data.iter_mut().zip(&x.data) {
                *a += b;
            }
        }
        y
    }
}

/// A linear layer.
#[derive(Debug, Clone)]
struct Linear {
    cin: usize,
    cout: usize,
    /// `[cout][cin]`.
    weight: Vec<f32>,
    bias: Vec<f32>,
}

impl Linear {
    fn apply(&self, x: &[f32]) -> Vec<f32> {
        (0..self.cout)
            .map(|o| {
                let row = &self.weight[o * self.cin..(o + 1) * self.cin];
                row.iter()
                    .zip(x)
                    .fold(self.bias[o], |acc, (w, v)| acc + w * v)
            })
            .collect()
    }
}

/// The crop student, loaded from its weights file. See the module documentation.
#[derive(Debug, Clone)]
pub struct CropNet {
    stem: Stem,
    blocks: Vec<Block>,
    neck: Pointwise,
    fc1: Linear,
    fc2: Linear,
    sha256: String,
}

/// The tensors of a weights file by name, each taken out once with its shape checked.
struct Tensors(std::collections::HashMap<String, crate::cpu_runtime::FloatTensor>);

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
        Ok(data)
    }

    fn pointwise(&mut self, prefix: &str, cin: usize, cout: usize) -> Result<Pointwise> {
        Ok(Pointwise {
            cin,
            cout,
            weight: self.take(&format!("{prefix}.weight"), &[cout, cin, 1, 1])?,
            bias: self.take(&format!("{prefix}.bias"), &[cout])?,
        })
    }
}

impl CropNet {
    /// Loads `dir/cropnet.safetensors`, checked against its line in `dir/SHA256SUMS`.
    ///
    /// # Errors
    ///
    /// [`MlError::Io`] when a file cannot be read, [`MlError::Checksum`] when the weights do not
    /// match `SHA256SUMS`, and the errors of [`CropNet::from_bytes`].
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

    /// Builds the network from the bytes of a weights file, checked against `expected_sha256` when
    /// given.
    ///
    /// # Errors
    ///
    /// [`MlError::Checksum`] on a digest mismatch, [`MlError::Weights`] when the file is not a
    /// `ym-cropnet` file or lacks a tensor, [`MlError::Unsupported`] for another format version,
    /// [`MlError::TensorShape`] when a tensor has the wrong shape.
    pub fn from_bytes(bytes: &[u8], expected_sha256: Option<&str>) -> Result<Self> {
        if let Some(want) = expected_sha256 {
            verify_sha256("crop network weights", bytes, want)?;
        }
        let (_, meta) =
            SafeTensors::read_metadata(bytes).map_err(|e| MlError::Weights(e.to_string()))?;
        let header = meta.metadata().clone().unwrap_or_default();
        let field = |k: &str| header.get(k).map(String::as_str).unwrap_or("");
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
        let stages: Vec<[usize; 4]> = serde_json::from_str(field("stages"))
            .map_err(|e| MlError::Weights(format!("stages: {e}")))?;
        let ours: Vec<[usize; 4]> = STAGES.iter().map(|&(a, b, c, d)| [a, b, c, d]).collect();
        if stages != ours
            || field("input_px") != INPUT_PX.to_string()
            || field("head_channels") != HEAD_CHANNELS.to_string()
            || field("hidden") != HIDDEN.to_string()
            || field("head") != "box"
        {
            return Err(MlError::Unsupported(format!(
                "a {FORMAT} file of another shape (stages {stages:?}, head {:?})",
                field("head")
            )));
        }

        let mut t = Tensors(read_float_tensors(bytes)?);
        let stem = Stem {
            cin: 3,
            cout: STEM_CHANNELS,
            weight: t.take("stem.weight", &[STEM_CHANNELS, 3, 3, 3])?,
            bias: t.take("stem.bias", &[STEM_CHANNELS])?,
        };
        let mut blocks = Vec::new();
        let mut cin = STEM_CHANNELS;
        for &(expand, cout, count, stride) in &STAGES {
            for b in 0..count {
                let i = blocks.len();
                let stride = if b == 0 { stride } else { 1 };
                let mid = cin * expand;
                let expand = if expand == 1 {
                    None
                } else {
                    Some(t.pointwise(&format!("blocks.{i}.expand"), cin, mid)?)
                };
                let depthwise = Depthwise {
                    c: mid,
                    stride,
                    weight: t.take(&format!("blocks.{i}.depthwise.weight"), &[mid, 1, 3, 3])?,
                    bias: t.take(&format!("blocks.{i}.depthwise.bias"), &[mid])?,
                };
                let project = t.pointwise(&format!("blocks.{i}.project"), mid, cout)?;
                blocks.push(Block {
                    expand,
                    depthwise,
                    project,
                    residual: stride == 1 && cin == cout,
                });
                cin = cout;
            }
        }
        let neck = t.pointwise("neck", cin, HEAD_CHANNELS)?;
        let flat = HEAD_CHANNELS * GRID * GRID;
        let fc1 = Linear {
            cin: flat,
            cout: HIDDEN,
            weight: t.take("fc1.weight", &[HIDDEN, flat])?,
            bias: t.take("fc1.bias", &[HIDDEN])?,
        };
        let fc2 = Linear {
            cin: HIDDEN,
            cout: 4,
            weight: t.take("fc2.weight", &[4, HIDDEN])?,
            bias: t.take("fc2.bias", &[4])?,
        };
        if let Some(extra) = t.0.keys().min() {
            return Err(MlError::Weights(format!("unexpected tensor `{extra}`")));
        }
        Ok(Self {
            stem,
            blocks,
            neck,
            fc1,
            fc2,
            sha256: sha256_hex(bytes),
        })
    }

    /// SHA-256 of the weights file (hex).
    #[must_use]
    pub fn sha256(&self) -> &str {
        &self.sha256
    }

    /// The network's four numbers `p` in `[0, 1]` (module documentation) for a letterboxed square of
    /// [`INPUT_PX`] × [`INPUT_PX`] 8-bit RGB pixels ([`letterbox`]).
    ///
    /// # Errors
    ///
    /// [`MlError::Shape`] when `square` is not `INPUT_PX × INPUT_PX × 3` bytes.
    pub fn numbers(&self, square: &[u8]) -> Result<[f64; 4]> {
        let n = INPUT_PX * INPUT_PX;
        if square.len() != 3 * n {
            return Err(MlError::Shape(format!(
                "the letterboxed picture holds {} bytes, not {}",
                square.len(),
                3 * n
            )));
        }
        let mut data = vec![0.0; 3 * n];
        for (p, rgb) in square.chunks_exact(3).enumerate() {
            for c in 0..3 {
                data[c * n + p] = (f32::from(rgb[c]) / 255.0 - MEAN) / STD;
            }
        }
        let mut x = self.stem.apply(&Planes {
            c: 3,
            h: INPUT_PX,
            w: INPUT_PX,
            data,
        });
        for block in &self.blocks {
            x = block.apply(&x);
        }
        let x = self.neck.apply(&x, true);
        debug_assert_eq!((x.c, x.h, x.w), (HEAD_CHANNELS, GRID, GRID));
        let hidden: Vec<f32> = self.fc1.apply(&x.data).into_iter().map(silu).collect();
        let out = self.fc2.apply(&hidden);
        Ok([0, 1, 2, 3].map(|i| f64::from(sigmoid(out[i]))))
    }

    /// The network's crop of a picture (8-bit RGB, rows top to bottom): `[x0, y0, x1, y1]` as
    /// fractions of the picture's width and height, inside `[0, 1]`.
    ///
    /// # Errors
    ///
    /// [`MlError::Shape`] when `rgb` is not `width × height × 3` bytes or a side is 0.
    pub fn crop(&self, rgb: &[u8], width: usize, height: usize) -> Result<[f64; 4]> {
        let (square, _) = letterbox(rgb, width, height)?;
        Ok(box_from_numbers(self.numbers(&square)?))
    }
}

/// `[x0, y0, x1, y1]` from the network's four numbers (module documentation).
#[must_use]
pub fn box_from_numbers(p: [f64; 4]) -> [f64; 4] {
    let (x0, y0) = (p[0], p[1]);
    [x0, y0, x0 + (1.0 - x0) * p[2], y0 + (1.0 - y0) * p[3]]
}

/// Pillow's bilinear resampling coefficients from `in_size` to `out_size` samples: per output sample
/// the first input sample and the fixed-point weights (`Resample.c`'s `precompute_coeffs` and
/// `normalize_coeffs_8bpc`, written from their description of the filter).
fn coefficients(in_size: usize, out_size: usize) -> Vec<(usize, Vec<i32>)> {
    let scale = in_size as f64 / out_size as f64;
    let filterscale = scale.max(1.0);
    let support = filterscale; // the bilinear filter's support is 1
    let ss = 1.0 / filterscale;
    (0..out_size)
        .map(|xx| {
            let center = (xx as f64 + 0.5) * scale;
            // C's `(int)` truncates toward zero, as `as` does.
            let xmin = ((center - support + 0.5) as i64).max(0);
            let xmax = ((center + support + 0.5) as i64).min(in_size as i64) - xmin;
            let weights: Vec<f64> = (0..xmax.max(0))
                .map(|x| {
                    let t = (((x + xmin) as f64 - center) + 0.5) * ss;
                    let t = t.abs();
                    if t < 1.0 { 1.0 - t } else { 0.0 }
                })
                .collect();
            let total: f64 = weights.iter().sum();
            let fixed = weights
                .iter()
                .map(|&w| {
                    let w = if total == 0.0 { w } else { w / total };
                    let scaled = w * f64::from(1u32 << PRECISION_BITS);
                    if w < 0.0 {
                        (-0.5 + scaled) as i32
                    } else {
                        (0.5 + scaled) as i32
                    }
                })
                .collect();
            (xmin as usize, fixed)
        })
        .collect()
}

fn clip8(v: i64) -> u8 {
    (v >> PRECISION_BITS).clamp(0, 255) as u8
}

/// Resizes 8-bit RGB `width × height` to `out_w × out_h` as Pillow's `Image.resize(size,
/// Image.BILINEAR, reducing_gap=None)` does: a horizontal pass, then a vertical one, each rounding
/// to 8 bits (a pass is skipped when its side does not change).
fn resize(rgb: &[u8], width: usize, height: usize, out_w: usize, out_h: usize) -> Vec<u8> {
    let half = 1i64 << (PRECISION_BITS - 1);
    let vertical = coefficients(height, out_h);
    // Only the input rows the vertical pass reads go through the horizontal one.
    let first = vertical.first().map_or(0, |(y, _)| *y);
    let last = vertical.last().map_or(0, |(y, k)| y + k.len());
    let (mid, mid_w, row0) = if out_w != width {
        let horizontal = coefficients(width, out_w);
        let rows = last - first;
        let mut mid = vec![0u8; rows * out_w * 3];
        for y in 0..rows {
            let src = &rgb[(first + y) * width * 3..(first + y + 1) * width * 3];
            for (xx, (xmin, k)) in horizontal.iter().enumerate() {
                for c in 0..3 {
                    let mut acc = half;
                    for (x, &w) in k.iter().enumerate() {
                        acc += i64::from(src[(xmin + x) * 3 + c]) * i64::from(w);
                    }
                    mid[(y * out_w + xx) * 3 + c] = clip8(acc);
                }
            }
        }
        (mid, out_w, first)
    } else {
        (rgb.to_vec(), width, 0)
    };
    if out_h == height {
        return mid;
    }
    let mut out = vec![0u8; out_h * mid_w * 3];
    for (yy, (ymin, k)) in vertical.iter().enumerate() {
        for x in 0..mid_w {
            for c in 0..3 {
                let mut acc = half;
                for (y, &w) in k.iter().enumerate() {
                    acc += i64::from(mid[((ymin + y - row0) * mid_w + x) * 3 + c]) * i64::from(w);
                }
                out[(yy * mid_w + x) * 3 + c] = clip8(acc);
            }
        }
    }
    out
}

/// The picture scaled to fit an [`INPUT_PX`] square (long side `INPUT_PX`, the short side rounded
/// half to even as Python rounds), centred on black, as 8-bit RGB; and the picture's box in the
/// square as fractions of it. The same bytes as `training/crop/data.py`'s `letterbox`.
///
/// # Errors
///
/// [`MlError::Shape`] when `rgb` is not `width × height × 3` bytes or a side is 0.
pub fn letterbox(rgb: &[u8], width: usize, height: usize) -> Result<(Vec<u8>, [f64; 4])> {
    if width == 0 || height == 0 || rgb.len() != width * height * 3 {
        return Err(MlError::Shape(format!(
            "the picture is {width} x {height} pixels but holds {} bytes",
            rgb.len()
        )));
    }
    let size = INPUT_PX;
    let scale = size as f64 / width.max(height) as f64;
    let fit = |side: usize| ((side as f64 * scale).round_ties_even() as usize).clamp(1, size);
    let (cw, ch) = (fit(width), fit(height));
    let (x0, y0) = ((size - cw) / 2, (size - ch) / 2);
    let scaled = if (cw, ch) == (width, height) {
        rgb.to_vec()
    } else {
        resize(rgb, width, height, cw, ch)
    };
    let mut square = vec![0u8; size * size * 3];
    for y in 0..ch {
        let dst = ((y0 + y) * size + x0) * 3;
        square[dst..dst + cw * 3].copy_from_slice(&scaled[y * cw * 3..(y + 1) * cw * 3]);
    }
    let s = size as f64;
    Ok((
        square,
        [
            x0 as f64 / s,
            y0 as f64 / s,
            (x0 + cw) as f64 / s,
            (y0 + ch) as f64 / s,
        ],
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn box_from_numbers_stays_inside_and_inverts() {
        assert_eq!(box_from_numbers([0.0, 0.0, 1.0, 1.0]), [0.0, 0.0, 1.0, 1.0]);
        let b = box_from_numbers([0.2, 0.1, 0.5, 0.25]);
        assert!((b[2] - 0.6).abs() < 1e-12 && (b[3] - 0.325).abs() < 1e-12);
    }

    #[test]
    fn coefficients_sum_to_one_in_fixed_point() {
        for (i, o) in [(768, 256), (511, 170), (97, 256), (256, 256)] {
            for (_, k) in coefficients(i, o) {
                let sum: i64 = k.iter().map(|&w| i64::from(w)).sum();
                assert!(
                    (sum - (1 << PRECISION_BITS)).abs() <= k.len() as i64,
                    "{i}->{o}: {sum}"
                );
            }
        }
    }

    #[test]
    fn a_flat_picture_stays_flat_and_centred() {
        let rgb = vec![200u8; 300 * 150 * 3];
        let (sq, content) = letterbox(&rgb, 300, 150).unwrap();
        assert_eq!(content, [0.0, 64.0 / 256.0, 1.0, 192.0 / 256.0]);
        assert_eq!(&sq[..3], &[0, 0, 0]);
        let mid = (128 * INPUT_PX + 128) * 3;
        assert_eq!(&sq[mid..mid + 3], &[200, 200, 200]);
        assert!(letterbox(&rgb, 300, 149).is_err());
    }
}
