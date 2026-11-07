//! The operations of the depth network that the SAM 2 kernels do not have: the 3 x 3 convolution (stride 1 or 2), the
//! non-overlapping transposed convolution, bilinear resizing with `align_corners=True`, the bicubic resize of the positional
//! embedding with PyTorch's `scale_factor` semantics, and the LayerScale step.
//!
//! Every one is plain `f32` in a fixed order with no fused multiply-add and no reassociation: the convolutions are the
//! packed product of `sam2/kernels.rs` (one chain per output element, the bias and then the inputs in index order), so they
//! give the same bits on any number of threads and any instruction set. Maps are `[y][x][channel]`.

use rayon::prelude::*;

use crate::sam2::kernels::{Act, Packed, Residual, gemm};

/// A 3 x 3 convolution with padding 1 and stride 1 or 2, as a product over the 9 taps (`ky`, `kx`, input channel) of each
/// output position.
#[derive(Debug)]
pub(crate) struct Conv3 {
    w: Packed,
    cin: usize,
}

impl Conv3 {
    /// Packs a PyTorch `Conv2d` weight `[cout][cin][3][3]` with its optional bias.
    pub(crate) fn from_oihw(weight: &[f32], cout: usize, cin: usize, bias: Option<&[f32]>) -> Self {
        debug_assert_eq!(weight.len(), cout * cin * 9);
        let mut kn = vec![0.0f32; 9 * cin * cout];
        for o in 0..cout {
            for c in 0..cin {
                for ky in 0..3 {
                    for kx in 0..3 {
                        kn[((ky * 3 + kx) * cin + c) * cout + o] =
                            weight[((o * cin + c) * 3 + ky) * 3 + kx];
                    }
                }
            }
        }
        Self {
            w: Packed::from_kn(&kn, 9 * cin, cout, bias),
            cin,
        }
    }

    /// The size of the output for an input of `h` x `w` (`floor((n + 2 - 3) / stride) + 1`).
    pub(crate) fn out_size(h: usize, w: usize, stride: usize) -> (usize, usize) {
        ((h - 1) / stride + 1, (w - 1) / stride + 1)
    }

    /// One output row. `rows[ky]` is the input row of tap `ky` (`None` where the padding is), `in_w` its width. With
    /// `pre_relu` the input is passed through `max(x, 0)` first (zero padding is unchanged by it). `res` is added after
    /// `act`; `scratch` is reused between calls.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn row(
        &self,
        rows: [Option<&[f32]>; 3],
        in_w: usize,
        stride: usize,
        pre_relu: bool,
        act: Act,
        res: Option<Residual<'_>>,
        scratch: &mut Vec<f32>,
        out: &mut [f32],
    ) {
        let cin = self.cin;
        let k = 9 * cin;
        let ow = (in_w - 1) / stride + 1;
        scratch.clear();
        scratch.resize(ow * k, 0.0);
        for (ky, row) in rows.iter().enumerate() {
            let Some(row) = row else { continue };
            for ox in 0..ow {
                for kx in 0..3 {
                    let ix = (ox * stride + kx) as isize - 1;
                    if ix < 0 || ix >= in_w as isize {
                        continue;
                    }
                    let src = &row[ix as usize * cin..][..cin];
                    let dst = &mut scratch[ox * k + (ky * 3 + kx) * cin..][..cin];
                    if pre_relu {
                        for (d, &v) in dst.iter_mut().zip(src) {
                            *d = if v > 0.0 { v } else { 0.0 };
                        }
                    } else {
                        dst.copy_from_slice(src);
                    }
                }
            }
        }
        gemm(scratch, k, ow, &self.w, act, res, out, self.w.n);
    }

    /// The convolution of a whole `h` x `w` map over all the threads of the pool. `res` (the size of the output) is added
    /// after `act`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn forward(
        &self,
        x: &[f32],
        h: usize,
        w: usize,
        stride: usize,
        pre_relu: bool,
        act: Act,
        res: Option<&[f32]>,
    ) -> Vec<f32> {
        let cin = self.cin;
        let cout = self.w.n;
        debug_assert_eq!(x.len(), h * w * cin);
        let (oh, ow) = Self::out_size(h, w, stride);
        let mut out = vec![0.0f32; oh * ow * cout];
        out.par_chunks_mut(ow * cout)
            .enumerate()
            .for_each_init(Vec::new, |scratch, (oy, dst)| {
                let rows: [Option<&[f32]>; 3] = std::array::from_fn(|ky| {
                    let iy = (oy * stride + ky) as isize - 1;
                    (iy >= 0 && iy < h as isize).then(|| &x[iy as usize * w * cin..][..w * cin])
                });
                let res = res.map(|r| Residual {
                    data: &r[oy * ow * cout..],
                    ld: cout,
                });
                self.row(rows, w, stride, pre_relu, act, res, scratch, dst);
            });
        out
    }
}

/// A `ConvTranspose2d` whose kernel size equals its stride (4 x 4 stride 4, 2 x 2 stride 2): every input position writes
/// its own `k` x `k` block of the output, so it is one product of `cin` inputs and `k * k * cout` outputs.
#[derive(Debug)]
pub(crate) struct TConv {
    w: Packed,
    k: usize,
    cout: usize,
}

impl TConv {
    /// Packs a PyTorch `ConvTranspose2d` weight `[cin][cout][k][k]` with its bias.
    pub(crate) fn from_iokk(
        weight: &[f32],
        cin: usize,
        cout: usize,
        k: usize,
        bias: &[f32],
    ) -> Self {
        debug_assert_eq!(weight.len(), cin * cout * k * k);
        let mut kn = vec![0.0f32; cin * k * k * cout];
        for i in 0..cin {
            for o in 0..cout {
                for dy in 0..k {
                    for dx in 0..k {
                        kn[i * k * k * cout + (dy * k + dx) * cout + o] =
                            weight[((i * cout + o) * k + dy) * k + dx];
                    }
                }
            }
        }
        let bias_k: Vec<f32> = (0..k * k * cout).map(|n| bias[n % cout]).collect();
        Self {
            w: Packed::from_kn(&kn, cin, k * k * cout, Some(&bias_k)),
            k,
            cout,
        }
    }

    /// The map `[h * k][w * k][cout]` of an `h` x `w` input.
    pub(crate) fn forward(&self, x: &[f32], h: usize, w: usize) -> Vec<f32> {
        let (k, cout, cin) = (self.k, self.cout, self.w.k);
        debug_assert_eq!(x.len(), h * w * cin);
        let mut out = vec![0.0f32; h * k * w * k * cout];
        out.par_chunks_mut(k * w * k * cout)
            .enumerate()
            .for_each_init(Vec::new, |tmp, (y, chunk)| {
                tmp.clear();
                tmp.resize(w * k * k * cout, 0.0);
                gemm(
                    &x[y * w * cin..],
                    cin,
                    w,
                    &self.w,
                    Act::None,
                    None,
                    tmp,
                    k * k * cout,
                );
                for xx in 0..w {
                    for dy in 0..k {
                        for dx in 0..k {
                            let src = &tmp[(xx * k * k + dy * k + dx) * cout..][..cout];
                            let at = (dy * w * k + xx * k + dx) * cout;
                            chunk[at..at + cout].copy_from_slice(src);
                        }
                    }
                }
            });
        out
    }
}

/// The source indices and weights of one axis of `F.interpolate(..., mode="bilinear", align_corners=True)`.
#[derive(Debug)]
pub(crate) struct Axis {
    i0: Vec<usize>,
    i1: Vec<usize>,
    l1: Vec<f32>,
}

impl Axis {
    /// PyTorch's `area_pixel_compute_scale` with `align_corners`: `(in - 1) / (out - 1)` in `f32` (0 for one output),
    /// the source coordinate `scale * dst`, its floor and the fraction clamped to [0, 1].
    pub(crate) fn align_corners(n_in: usize, n_out: usize) -> Self {
        let scale = if n_out > 1 {
            (n_in - 1) as f32 / (n_out - 1) as f32
        } else {
            0.0
        };
        let mut axis = Self {
            i0: Vec::with_capacity(n_out),
            i1: Vec::with_capacity(n_out),
            l1: Vec::with_capacity(n_out),
        };
        for o in 0..n_out {
            let src = scale * o as f32;
            let i0 = (src as usize).min(n_in - 1);
            let i1 = i0 + usize::from(i0 < n_in - 1);
            axis.i0.push(i0);
            axis.i1.push(i1);
            axis.l1.push((src - i0 as f32).clamp(0.0, 1.0));
        }
        axis
    }
}

/// Output row `oy` of the bilinear resize of a `[sh][sw][c]` map (`out` holds `ax.len() * c` values):
/// `ly0 * (lx0 * a + lx1 * b) + ly1 * (lx0 * c + lx1 * d)`, in `f32`, along x and then along y.
pub(crate) fn bilinear_row(
    src: &[f32],
    sw: usize,
    c: usize,
    ax: &Axis,
    ay: &Axis,
    oy: usize,
    out: &mut [f32],
) {
    let (y0, y1) = (ay.i0[oy], ay.i1[oy]);
    let ly1 = ay.l1[oy];
    let ly0 = 1.0 - ly1;
    let (r0, r1) = (&src[y0 * sw * c..][..sw * c], &src[y1 * sw * c..][..sw * c]);
    for (ox, dst) in out.chunks_exact_mut(c).enumerate() {
        let (x0, x1) = (ax.i0[ox] * c, ax.i1[ox] * c);
        let lx1 = ax.l1[ox];
        let lx0 = 1.0 - lx1;
        for (ch, d) in dst.iter_mut().enumerate() {
            let top = lx0 * r0[x0 + ch] + lx1 * r0[x1 + ch];
            let bottom = lx0 * r1[x0 + ch] + lx1 * r1[x1 + ch];
            *d = ly0 * top + ly1 * bottom;
        }
    }
}

/// `F.interpolate(x, (oh, ow), mode="bilinear", align_corners=True)` of a `[sh][sw][c]` map.
pub(crate) fn bilinear(
    src: &[f32],
    sh: usize,
    sw: usize,
    c: usize,
    oh: usize,
    ow: usize,
) -> Vec<f32> {
    debug_assert_eq!(src.len(), sh * sw * c);
    let (ax, ay) = (Axis::align_corners(sw, ow), Axis::align_corners(sh, oh));
    let mut out = vec![0.0f32; oh * ow * c];
    out.par_chunks_mut(ow * c)
        .enumerate()
        .for_each(|(oy, row)| bilinear_row(src, sw, c, &ax, &ay, oy, row));
    out
}

/// `x += t * gamma` (`gamma` per channel, `c` channels per row): the LayerScale of a residual branch and its sum with the
/// stream, each a separate `f32` multiply and add as in PyTorch's `x + ls(branch)`.
pub(crate) fn add_layer_scaled(x: &mut [f32], t: &[f32], gamma: &[f32]) {
    let c = gamma.len();
    const ROWS: usize = 256;
    x.par_chunks_mut(ROWS * c)
        .zip(t.par_chunks(ROWS * c))
        .for_each(|(xs, ts)| {
            for (xr, tr) in xs.chunks_exact_mut(c).zip(ts.chunks_exact(c)) {
                for ((xv, &tv), &g) in xr.iter_mut().zip(tr).zip(gamma) {
                    *xv += tv * g;
                }
            }
        });
}

/// PyTorch's cubic convolution weights (`A = -0.75`) for the fraction `t`, in `f32` as its CPU kernel computes them.
pub(crate) fn cubic_weights(t: f32) -> [f32; 4] {
    const A: f32 = -0.75;
    let conv1 = |x: f32| ((A + 2.0) * x - (A + 3.0)) * x * x + 1.0;
    let conv2 = |x: f32| ((A * x - 5.0 * A) * x + 8.0 * A) * x - 4.0 * A;
    [conv2(t + 1.0), conv1(t), conv1(1.0 - t), conv2(2.0 - t)]
}

/// One axis of the bicubic resize of `F.interpolate(x, scale_factor=s, mode="bicubic", align_corners=False)`: the source
/// coordinate is `(1 / s) * (dst + 0.5) - 0.5` with `1 / s` rounded to `f32` (PyTorch uses the given scale factor, not the
/// ratio of the sizes, whenever one is given), the four taps are `floor - 1 ..= floor + 2` clamped into the grid.
fn cubic_axis(n_in: usize, n_out: usize, scale: f64) -> Vec<([usize; 4], [f32; 4])> {
    let inv = (1.0 / scale) as f32;
    (0..n_out)
        .map(|o| {
            let real = inv * (o as f32 + 0.5) - 0.5;
            let base = real.floor();
            let weights = cubic_weights(real - base);
            let idx = std::array::from_fn(|k| {
                (base as isize - 1 + k as isize).clamp(0, n_in as isize - 1) as usize
            });
            (idx, weights)
        })
        .collect()
}

/// Resizes a `[n][n][dim]` grid to `[out_h][out_w][dim]` bicubically with the given scale factors (`out = floor(n * s)`),
/// `antialias=False`, taps along x and then along y in `f32`.
pub(crate) fn resize_bicubic_scaled(
    grid: &[f32],
    n: usize,
    dim: usize,
    scale_h: f64,
    scale_w: f64,
) -> (Vec<f32>, usize, usize) {
    let out_h = (n as f64 * scale_h).floor() as usize;
    let out_w = (n as f64 * scale_w).floor() as usize;
    let (ay, ax) = (cubic_axis(n, out_h, scale_h), cubic_axis(n, out_w, scale_w));
    let mut out = vec![0.0f32; out_h * out_w * dim];
    out.par_chunks_mut(out_w * dim)
        .enumerate()
        .for_each(|(y, row)| {
            let (iy, wy) = &ay[y];
            for (x, dst) in row.chunks_exact_mut(dim).enumerate() {
                let (ix, wx) = &ax[x];
                for (ch, d) in dst.iter_mut().enumerate() {
                    let line = |yy: usize| {
                        let base = yy * n;
                        wx[0] * grid[(base + ix[0]) * dim + ch]
                            + wx[1] * grid[(base + ix[1]) * dim + ch]
                            + wx[2] * grid[(base + ix[2]) * dim + ch]
                            + wx[3] * grid[(base + ix[3]) * dim + ch]
                    };
                    *d = wy[0] * line(iy[0])
                        + wy[1] * line(iy[1])
                        + wy[2] * line(iy[2])
                        + wy[3] * line(iy[3]);
                }
            }
        });
    (out, out_h, out_w)
}

#[cfg(test)]
mod tests {
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

        fn vec(&mut self, n: usize) -> Vec<f32> {
            (0..n).map(|_| self.next()).collect()
        }
    }

    fn bits(v: &[f32]) -> Vec<u32> {
        v.iter().map(|x| x.to_bits()).collect()
    }

    /// The definition of a 3 x 3 convolution with padding 1: the bias and then the taps in (ky, kx, channel) order.
    #[allow(clippy::too_many_arguments)]
    fn naive_conv3(
        x: &[f32],
        h: usize,
        w: usize,
        cin: usize,
        weight: &[f32],
        cout: usize,
        bias: Option<&[f32]>,
        stride: usize,
        pre_relu: bool,
    ) -> Vec<f32> {
        let (oh, ow) = Conv3::out_size(h, w, stride);
        let mut out = vec![0.0f32; oh * ow * cout];
        for oy in 0..oh {
            for ox in 0..ow {
                for o in 0..cout {
                    let mut acc = bias.map_or(0.0, |b| b[o]);
                    for ky in 0..3 {
                        for kx in 0..3 {
                            let (iy, ix) = (
                                (oy * stride + ky) as isize - 1,
                                (ox * stride + kx) as isize - 1,
                            );
                            if iy < 0 || ix < 0 || iy >= h as isize || ix >= w as isize {
                                continue;
                            }
                            for c in 0..cin {
                                let mut v = x[(iy as usize * w + ix as usize) * cin + c];
                                if pre_relu {
                                    v = v.max(0.0);
                                }
                                acc += v * weight[((o * cin + c) * 3 + ky) * 3 + kx];
                            }
                        }
                    }
                    out[(oy * ow + ox) * cout + o] = acc;
                }
            }
        }
        out
    }

    #[test]
    fn the_3x3_convolution_is_the_definition_bit_for_bit_at_stride_1_and_2() {
        let mut rng = Lcg(4);
        // Odd and even sizes, so that the stride-2 output is ceil(n / 2), and a channel count off the panel width.
        for &(h, w, cin, cout, stride) in &[
            (5, 7, 3, 20, 1),
            (6, 9, 8, 17, 2),
            (7, 6, 5, 16, 2),
            (1, 1, 2, 3, 1),
            (1, 3, 2, 3, 2),
            (37, 56, 4, 6, 2),
        ] {
            let x = rng.vec(h * w * cin);
            let weight = rng.vec(cout * cin * 9);
            let bias = rng.vec(cout);
            let conv = Conv3::from_oihw(&weight, cout, cin, Some(&bias));
            for pre_relu in [false, true] {
                let want = naive_conv3(&x, h, w, cin, &weight, cout, Some(&bias), stride, pre_relu);
                let got = conv.forward(&x, h, w, stride, pre_relu, Act::None, None);
                assert_eq!(
                    bits(&got),
                    bits(&want),
                    "h {h} w {w} cin {cin} cout {cout} stride {stride} relu {pre_relu}"
                );
            }
            let (oh, ow) = Conv3::out_size(h, w, stride);
            assert_eq!((oh, ow), (h.div_ceil(stride), w.div_ceil(stride)));
        }
    }

    #[test]
    fn the_3x3_convolution_without_bias_and_with_a_residual_and_a_relu_after() {
        let mut rng = Lcg(5);
        let (h, w, cin, cout) = (4, 5, 6, 6);
        let x = rng.vec(h * w * cin);
        let weight = rng.vec(cout * cin * 9);
        let res = rng.vec(h * w * cout);
        let conv = Conv3::from_oihw(&weight, cout, cin, None);
        let plain = naive_conv3(&x, h, w, cin, &weight, cout, None, 1, false);
        let got = conv.forward(&x, h, w, 1, false, Act::Relu, Some(&res));
        for i in 0..plain.len() {
            assert_eq!(got[i].to_bits(), (plain[i].max(0.0) + res[i]).to_bits());
        }
    }

    /// The definition of a transposed convolution with kernel = stride: `out[y * k + dy][x * k + dx][o] = bias[o] + sum_i
    /// in[y][x][i] * w[i][o][dy][dx]`.
    fn naive_tconv(
        x: &[f32],
        h: usize,
        w: usize,
        cin: usize,
        weight: &[f32],
        cout: usize,
        k: usize,
        bias: &[f32],
    ) -> Vec<f32> {
        let mut out = vec![0.0f32; h * k * w * k * cout];
        for y in 0..h {
            for xx in 0..w {
                for dy in 0..k {
                    for dx in 0..k {
                        for o in 0..cout {
                            let mut acc = bias[o];
                            for i in 0..cin {
                                acc += x[(y * w + xx) * cin + i]
                                    * weight[((i * cout + o) * k + dy) * k + dx];
                            }
                            out[((y * k + dy) * w * k + xx * k + dx) * cout + o] = acc;
                        }
                    }
                }
            }
        }
        out
    }

    #[test]
    fn the_transposed_convolutions_are_the_definition_bit_for_bit() {
        let mut rng = Lcg(6);
        for &(h, w, cin, cout, k) in &[
            (3, 5, 7, 9, 4),
            (4, 3, 12, 20, 2),
            (1, 1, 2, 2, 4),
            (37, 56, 3, 3, 2),
        ] {
            let x = rng.vec(h * w * cin);
            let weight = rng.vec(cin * cout * k * k);
            let bias = rng.vec(cout);
            let tconv = TConv::from_iokk(&weight, cin, cout, k, &bias);
            let got = tconv.forward(&x, h, w);
            let want = naive_tconv(&x, h, w, cin, &weight, cout, k, &bias);
            assert_eq!(
                bits(&got),
                bits(&want),
                "h {h} w {w} cin {cin} cout {cout} k {k}"
            );
        }
    }

    /// The textbook bilinear resize with `align_corners=True`, in `f64`.
    fn reference_bilinear(
        src: &[f32],
        sh: usize,
        sw: usize,
        c: usize,
        oh: usize,
        ow: usize,
    ) -> Vec<f64> {
        let ratio = |n_in: usize, n_out: usize| {
            if n_out > 1 {
                (n_in - 1) as f64 / (n_out - 1) as f64
            } else {
                0.0
            }
        };
        let (ry, rx) = (ratio(sh, oh), ratio(sw, ow));
        let mut out = vec![0.0f64; oh * ow * c];
        for oy in 0..oh {
            for ox in 0..ow {
                let (fy, fx) = (ry * oy as f64, rx * ox as f64);
                let (y0, x0) = (fy.floor() as usize, fx.floor() as usize);
                let (y1, x1) = ((y0 + 1).min(sh - 1), (x0 + 1).min(sw - 1));
                let (ty, tx) = (fy - y0 as f64, fx - x0 as f64);
                for ch in 0..c {
                    let at = |y: usize, x: usize| f64::from(src[(y * sw + x) * c + ch]);
                    out[(oy * ow + ox) * c + ch] = (1.0 - ty)
                        * ((1.0 - tx) * at(y0, x0) + tx * at(y0, x1))
                        + ty * ((1.0 - tx) * at(y1, x0) + tx * at(y1, x1));
                }
            }
        }
        out
    }

    #[test]
    fn bilinear_with_align_corners_matches_the_textbook_formula() {
        let mut rng = Lcg(8);
        for &(sh, sw, oh, ow) in &[
            (3, 4, 6, 8),
            (19, 28, 37, 56),
            (37, 56, 74, 112),
            (296, 7, 31, 5),
            (1, 1, 3, 4),
            (5, 5, 1, 1),
            (4, 6, 4, 6),
        ] {
            let c = 3;
            let src = rng.vec(sh * sw * c);
            let got = bilinear(&src, sh, sw, c, oh, ow);
            let want = reference_bilinear(&src, sh, sw, c, oh, ow);
            let worst = got
                .iter()
                .zip(&want)
                .map(|(&g, &w)| (f64::from(g) - w).abs())
                .fold(0.0, f64::max);
            assert!(
                worst < 2e-6,
                "{sh}x{sw} -> {oh}x{ow}: largest difference {worst:.2e}"
            );
        }
        // The corners of the output are the corners of the input (that is what align_corners means).
        let src = rng.vec(5 * 6 * 2);
        let out = bilinear(&src, 5, 6, 2, 11, 13);
        for (oy, ox, sy, sx) in [(0, 0, 0, 0), (0, 12, 0, 5), (10, 0, 4, 0), (10, 12, 4, 5)] {
            for ch in 0..2 {
                assert_eq!(out[(oy * 13 + ox) * 2 + ch], src[(sy * 6 + sx) * 2 + ch]);
            }
        }
        // Same size: the identity.
        let same = bilinear(&src, 5, 6, 2, 5, 6);
        assert_eq!(bits(&same), bits(&src));
    }

    #[test]
    fn layer_scale_multiplies_per_channel_and_adds_to_the_stream() {
        let mut rng = Lcg(9);
        let c = 12;
        let x0 = rng.vec(700 * c);
        let t = rng.vec(700 * c);
        let gamma = rng.vec(c);
        let mut x = x0.clone();
        add_layer_scaled(&mut x, &t, &gamma);
        for i in 0..x.len() {
            assert_eq!(x[i].to_bits(), (x0[i] + t[i] * gamma[i % c]).to_bits());
        }
        // gamma = 1 is the plain residual sum.
        let mut y = x0.clone();
        add_layer_scaled(&mut y, &t, &vec![1.0; c]);
        for i in 0..y.len() {
            assert_eq!(y[i].to_bits(), (x0[i] + t[i]).to_bits());
        }
    }

    /// The textbook bicubic resize with a given scale factor, in `f64`: source coordinate `(dst + 0.5) / s - 0.5`, 16 taps
    /// with the clamped indices and Keys' kernel with `A = -0.75`.
    fn reference_bicubic(
        grid: &[f32],
        n: usize,
        dim: usize,
        scale_h: f64,
        scale_w: f64,
    ) -> (Vec<f64>, usize, usize) {
        let kernel = |x: f64| {
            const A: f64 = -0.75;
            let x = x.abs();
            if x <= 1.0 {
                ((A + 2.0) * x - (A + 3.0)) * x * x + 1.0
            } else if x < 2.0 {
                ((A * x - 5.0 * A) * x + 8.0 * A) * x - 4.0 * A
            } else {
                0.0
            }
        };
        let out_h = (n as f64 * scale_h).floor() as usize;
        let out_w = (n as f64 * scale_w).floor() as usize;
        let mut out = vec![0.0f64; out_h * out_w * dim];
        for y in 0..out_h {
            let sy = (y as f64 + 0.5) / scale_h - 0.5;
            for x in 0..out_w {
                let sx = (x as f64 + 0.5) / scale_w - 0.5;
                for ch in 0..dim {
                    let mut acc = 0.0;
                    for j in -1..=2 {
                        for i in -1..=2 {
                            let (yy, xx) = (sy.floor() as isize + j, sx.floor() as isize + i);
                            let (cy, cx) = (
                                yy.clamp(0, n as isize - 1) as usize,
                                xx.clamp(0, n as isize - 1) as usize,
                            );
                            acc += kernel(sy - yy as f64)
                                * kernel(sx - xx as f64)
                                * f64::from(grid[(cy * n + cx) * dim + ch]);
                        }
                    }
                    out[(y * out_w + x) * dim + ch] = acc;
                }
            }
        }
        (out, out_h, out_w)
    }

    #[test]
    fn the_positional_resize_matches_the_textbook_bicubic_with_the_given_scale_factor() {
        let mut rng = Lcg(10);
        let (n, dim) = (37, 5);
        let grid = rng.vec(n * n * dim);
        // The scale factors DINOv2 computes for 518 x 784 (37.1 / 37 and 56.1 / 37), for a square 28 x 28 grid and for 40 x 21.
        for (ph, pw) in [(37usize, 56usize), (28, 28), (40, 21), (37, 37), (60, 14)] {
            let sh = (ph as f64 + 0.1) / 37.0;
            let sw = (pw as f64 + 0.1) / 37.0;
            let (got, oh, ow) = resize_bicubic_scaled(&grid, n, dim, sh, sw);
            assert_eq!(
                (oh, ow),
                (ph, pw),
                "the output size is floor(37 * s) = the patch count"
            );
            let (want, _, _) = reference_bicubic(&grid, n, dim, sh, sw);
            let worst = got
                .iter()
                .zip(&want)
                .map(|(&g, &w)| (f64::from(g) - w).abs())
                .fold(0.0, f64::max);
            assert!(worst < 5e-6, "{ph} x {pw}: largest difference {worst:.2e}");
        }
    }

    #[test]
    fn the_positional_resize_uses_the_scale_factor_and_not_the_size_ratio() {
        // With the +0.1 offset the coordinate map is (dst + 0.5) * 37 / (n + 0.1) - 0.5, not (dst + 0.5) * 37 / n - 0.5.
        // A linear ramp along x is reproduced exactly by the kernel inside the grid, so the output shows the map directly.
        let (n, dim) = (37, 1);
        let ramp: Vec<f32> = (0..n * n).map(|i| (i % n) as f32).collect();
        let (ph, pw) = (37usize, 56usize);
        let s = (pw as f64 + 0.1) / 37.0;
        let (out, _, ow) = resize_bicubic_scaled(&ramp, n, dim, (ph as f64 + 0.1) / 37.0, s);
        for x in 4..ow - 4 {
            let want = (x as f64 + 0.5) / s - 0.5;
            let got = f64::from(out[10 * ow + x]);
            assert!((got - want).abs() < 2e-5, "x {x}: {got} vs {want}");
            // And it is not what a plain size ratio would give.
            let ratio = (x as f64 + 0.5) * 37.0 / pw as f64 - 0.5;
            assert!((want - ratio).abs() > 1e-3 || x == 0);
        }
    }

    #[test]
    fn cubic_weights_sum_to_one_and_interpolate_at_the_nodes() {
        for i in 0..=20 {
            let t = i as f32 / 20.0;
            let w = cubic_weights(t);
            assert!((w.iter().sum::<f32>() - 1.0).abs() < 1e-6, "t {t}");
        }
        assert_eq!(cubic_weights(0.0), [0.0, 1.0, 0.0, 0.0]);
    }
}
