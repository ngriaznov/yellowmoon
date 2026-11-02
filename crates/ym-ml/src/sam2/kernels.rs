//! The numerical kernels of the SAM 2 runtime: the packed matrix product, layer norm, softmax, the
//! activations and one attention head (`docs/algorithms/06-masks.md` section 6.4).
//!
//! # Determinism
//!
//! Every output element of every kernel is one chain of `f32` operations in a fixed order that does
//! not depend on how the work is cut up, on the number of threads or on the vector width of the
//! machine: the product `out[m][n] = bias[n] + a[m][0] * w[0][n] + a[m][1] * w[1][n] + ...` adds the
//! inputs one after the other in index order (a multiply and a separate add, never a fused one), the
//! blocking only decides which chains run side by side. The x86-64 build is compiled twice from the
//! same source, once for the base instruction set and once with AVX2 (chosen at run time); the two
//! give the same bits because no operation is reassociated. Sums that are not a matrix product (the
//! mean and variance of a layer norm, the denominator of a softmax) are summed in `f64` in index
//! order.

use rayon::prelude::*;

/// Output columns of one panel of a packed weight matrix (two AVX2 registers, four NEON ones).
pub(crate) const NR: usize = 16;
/// Rows of the register tile of the product.
const MR: usize = 4;
/// Rows a parallel task of [`gemm_par`] takes at a time.
const ROWS_PER_TASK: usize = 64;

/// `1 / sqrt(2)` as `f32`, the argument scale of the exact GELU.
const FRAC_1_SQRT_2: f32 = std::f32::consts::FRAC_1_SQRT_2;

/// A weight matrix of `k` inputs and `n` outputs packed for [`gemm`]: panels of [`NR`] output columns,
/// `[panel][k][NR]` (the columns of a panel are contiguous), the last panel padded with zeros, and the
/// bias padded the same way.
#[derive(Debug, Clone, Default)]
pub(crate) struct Packed {
    /// Inputs.
    pub k: usize,
    /// Outputs.
    pub n: usize,
    /// The panels, `n.div_ceil(NR) * k * NR` values.
    pub data: Vec<f32>,
    /// The bias, `n.div_ceil(NR) * NR` values (zeros where there is none).
    pub bias: Vec<f32>,
}

impl Packed {
    /// An all-zero matrix of the given size, to be filled by the `fill_*` functions or [`Packed::pack_from_nk`].
    pub(crate) fn zeros(k: usize, n: usize) -> Self {
        let panels = n.div_ceil(NR);
        Self {
            k,
            n,
            data: vec![0.0; panels * k * NR],
            bias: vec![0.0; panels * NR],
        }
    }

    /// Packs a PyTorch weight `[n][k]` (a `Linear` or a 1 x 1 convolution) with its optional bias.
    pub(crate) fn from_nk(weight: &[f32], n: usize, k: usize, bias: Option<&[f32]>) -> Self {
        debug_assert_eq!(weight.len(), n * k);
        let mut p = Self::zeros(k, n);
        for o in 0..n {
            let (panel, lane) = (o / NR, o % NR);
            for i in 0..k {
                p.data[(panel * k + i) * NR + lane] = weight[o * k + i];
            }
        }
        if let Some(b) = bias {
            p.bias[..n].copy_from_slice(b);
        }
        p
    }

    /// Packs a matrix stored `[k][n]` (inputs first) with its optional bias.
    pub(crate) fn from_kn(weight: &[f32], k: usize, n: usize, bias: Option<&[f32]>) -> Self {
        debug_assert_eq!(weight.len(), n * k);
        let mut p = Self::zeros(k, n);
        for i in 0..k {
            for o in 0..n {
                p.data[((o / NR) * k + i) * NR + o % NR] = weight[i * n + o];
            }
        }
        if let Some(b) = bias {
            p.bias[..n].copy_from_slice(b);
        }
        p
    }

    /// Resizes to `k` x `n` and clears the padding, keeping the allocation (scratch matrices of attention).
    fn reset(&mut self, k: usize, n: usize) {
        self.k = k;
        self.n = n;
        let panels = n.div_ceil(NR);
        self.data.clear();
        self.data.resize(panels * k * NR, 0.0);
        self.bias.clear();
        self.bias.resize(panels * NR, 0.0);
    }
}

/// What is done to a product's output before it is stored.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Act {
    /// Nothing.
    None,
    /// `max(x, 0)`.
    Relu,
    /// The exact GELU, `x * (1 + erf(x / sqrt 2)) / 2`.
    Gelu,
}

/// The residual a product's output is added to: `out = act(acc) + res`, `res` rows of `ld` values.
#[derive(Clone, Copy)]
pub(crate) struct Residual<'a> {
    /// The values, row-major.
    pub data: &'a [f32],
    /// The distance between two rows.
    pub ld: usize,
}

/// `max(x, 0)` that gives `+0.0` for `-0.0` and NaN, whatever the platform.
#[inline(always)]
pub(crate) fn relu(v: f32) -> f32 {
    if v > 0.0 { v } else { 0.0 }
}

/// The exact GELU of PyTorch's `nn.GELU()`: `x * 0.5 * (1 + erf(x / sqrt 2))`, with `libm`'s `erff`.
#[inline(always)]
pub(crate) fn gelu(v: f32) -> f32 {
    v * 0.5 * (1.0 + libm::erff(v * FRAC_1_SQRT_2))
}

/// `1 / (1 + e^-x)` with `libm`'s `expf`.
#[inline(always)]
pub(crate) fn sigmoid(v: f32) -> f32 {
    1.0 / (1.0 + libm::expf(-v))
}

#[inline(always)]
fn activate(v: f32, act: Act) -> f32 {
    match act {
        Act::None => v,
        Act::Relu => relu(v),
        Act::Gelu => gelu(v),
    }
}

/// One register tile: `R` rows of `a` against one panel, the chain of every element being the bias and then the
/// `k` inputs in order. Returns the accumulators.
#[inline(always)]
fn micro<const R: usize>(
    a: &[f32],
    lda: usize,
    k: usize,
    panel: &[f32],
    bias: &[f32],
) -> [[f32; NR]; R] {
    let mut acc = [[0.0f32; NR]; R];
    for row in &mut acc {
        row.copy_from_slice(&bias[..NR]);
    }
    let rows: [&[f32]; R] = std::array::from_fn(|r| &a[r * lda..r * lda + k]);
    let panel = &panel[..k * NR];
    for (kk, w) in panel.chunks_exact(NR).enumerate() {
        for r in 0..R {
            let av = rows[r][kk];
            for j in 0..NR {
                acc[r][j] += av * w[j];
            }
        }
    }
    acc
}

/// Stores a register tile (`rows` x `cols` valid) after the activation and the residual.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn store<const R: usize>(
    acc: &[[f32; NR]; R],
    rows: usize,
    cols: usize,
    act: Act,
    res: Option<Residual<'_>>,
    row0: usize,
    col0: usize,
    out: &mut [f32],
    ldc: usize,
) {
    for (r, a) in acc.iter().enumerate().take(rows) {
        let dst = &mut out[(row0 + r) * ldc + col0..(row0 + r) * ldc + col0 + cols];
        match res {
            None => {
                for (d, &v) in dst.iter_mut().zip(a) {
                    *d = activate(v, act);
                }
            }
            Some(res) => {
                let src = &res.data[(row0 + r) * res.ld + col0..(row0 + r) * res.ld + col0 + cols];
                for ((d, &v), &s) in dst.iter_mut().zip(a).zip(src) {
                    *d = activate(v, act) + s;
                }
            }
        }
    }
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn gemm_impl(
    a: &[f32],
    lda: usize,
    m: usize,
    w: &Packed,
    act: Act,
    res: Option<Residual<'_>>,
    out: &mut [f32],
    ldc: usize,
) {
    let (k, n) = (w.k, w.n);
    for p in 0..n.div_ceil(NR) {
        let panel = &w.data[p * k * NR..(p + 1) * k * NR];
        let bias = &w.bias[p * NR..(p + 1) * NR];
        let cols = (n - p * NR).min(NR);
        let mut r0 = 0;
        while r0 + MR <= m {
            let acc = micro::<MR>(&a[r0 * lda..], lda, k, panel, bias);
            store::<MR>(&acc, MR, cols, act, res, r0, p * NR, out, ldc);
            r0 += MR;
        }
        while r0 < m {
            let acc = micro::<1>(&a[r0 * lda..], lda, k, panel, bias);
            store::<1>(&acc, 1, cols, act, res, r0, p * NR, out, ldc);
            r0 += 1;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(clippy::too_many_arguments)]
fn gemm_avx2(
    a: &[f32],
    lda: usize,
    m: usize,
    w: &Packed,
    act: Act,
    res: Option<Residual<'_>>,
    out: &mut [f32],
    ldc: usize,
) {
    gemm_impl(a, lda, m, w, act, res, out, ldc);
}

/// `out[r][c] = act(bias[c] + sum_i a[r][i] * w[i][c]) + res[r][c]` for `m` rows, on the calling thread.
///
/// `a` holds the rows `lda` apart (`k` values each), `out` the result rows `ldc` apart (`n` values each).
/// The bits of every element are those of the chain described in the module documentation, whatever `m`,
/// the strides or the instruction set.
#[allow(clippy::too_many_arguments)]
pub(crate) fn gemm(
    a: &[f32],
    lda: usize,
    m: usize,
    w: &Packed,
    act: Act,
    res: Option<Residual<'_>>,
    out: &mut [f32],
    ldc: usize,
) {
    if m == 0 || w.n == 0 {
        return;
    }
    debug_assert!(a.len() >= (m - 1) * lda + w.k);
    debug_assert!(out.len() >= (m - 1) * ldc + w.n);
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") {
            // SAFETY: the CPU supports AVX2 (checked just above), the only requirement of `gemm_avx2`.
            unsafe { gemm_avx2(a, lda, m, w, act, res, out, ldc) };
            return;
        }
    }
    gemm_impl(a, lda, m, w, act, res, out, ldc);
}

/// [`gemm`] over all the threads of the pool: `out` has `m` contiguous rows of `w.n` values, `a` has `m`
/// rows of `w.k`. The rows are cut into tasks of [`ROWS_PER_TASK`]; the cut changes no bit.
pub(crate) fn gemm_par(
    a: &[f32],
    m: usize,
    w: &Packed,
    act: Act,
    res: Option<Residual<'_>>,
    out: &mut [f32],
) {
    let (k, n) = (w.k, w.n);
    debug_assert_eq!(a.len(), m * k);
    debug_assert_eq!(out.len(), m * n);
    if m == 0 || n == 0 {
        return;
    }
    out.par_chunks_mut(ROWS_PER_TASK * n)
        .enumerate()
        .for_each(|(i, chunk)| {
            let rows = chunk.len() / n;
            let r0 = i * ROWS_PER_TASK;
            let res = res.map(|r| Residual {
                data: &r.data[r0 * r.ld..],
                ld: r.ld,
            });
            gemm(&a[r0 * k..], k, rows, w, act, res, chunk, n);
        });
}

/// Layer norm of every row of `x` (`c` values each) into `out`: the mean and the variance are summed in
/// `f64` in index order, `y = (x - mean) * (1 / sqrt(var + eps)) * weight + bias` in `f32`.
pub(crate) fn layer_norm(
    x: &[f32],
    c: usize,
    weight: &[f32],
    bias: &[f32],
    eps: f64,
    out: &mut [f32],
) {
    debug_assert_eq!(x.len(), out.len());
    for (src, dst) in x.chunks_exact(c).zip(out.chunks_exact_mut(c)) {
        let mean = src.iter().map(|&v| f64::from(v)).sum::<f64>() / c as f64;
        let var = src
            .iter()
            .map(|&v| {
                let d = f64::from(v) - mean;
                d * d
            })
            .sum::<f64>()
            / c as f64;
        let mean = mean as f32;
        let inv = (1.0 / (var + eps).sqrt()) as f32;
        for (((d, &v), &w), &b) in dst.iter_mut().zip(src).zip(weight).zip(bias) {
            *d = (v - mean) * inv * w + b;
        }
    }
}

/// [`layer_norm`] over all the threads of the pool.
pub(crate) fn layer_norm_par(
    x: &[f32],
    c: usize,
    weight: &[f32],
    bias: &[f32],
    eps: f64,
    out: &mut [f32],
) {
    const ROWS: usize = 256;
    out.par_chunks_mut(ROWS * c)
        .zip(x.par_chunks(ROWS * c))
        .for_each(|(dst, src)| layer_norm(src, c, weight, bias, eps, dst));
}

/// `s = s * scale`, then the softmax of `s` in place: the maximum, `exp(s - max)` with `libm`, the sum in
/// `f64` in index order, and `e * (1 / sum)` in `f32`.
pub(crate) fn softmax_scaled(s: &mut [f32], scale: f32) {
    let mut max = f32::NEG_INFINITY;
    for v in s.iter_mut() {
        *v *= scale;
        if *v > max {
            max = *v;
        }
    }
    let mut sum = 0.0f64;
    for v in s.iter_mut() {
        *v = libm::expf(*v - max);
        sum += f64::from(*v);
    }
    let inv = (1.0 / sum) as f32;
    for v in s.iter_mut() {
        *v *= inv;
    }
}

/// The keys and values of one attention head, packed for [`attend`]: `kt` is the keys transposed (the product
/// `q . kt` gives the scores), `vp` the values (`probabilities . vp` gives the output).
#[derive(Debug, Default)]
pub(crate) struct HeadKv {
    kt: Packed,
    vp: Packed,
    nk: usize,
    hd: usize,
}

impl HeadKv {
    /// Packs `nk` keys and values of `hd` values each; row `j` of `k` starts at `j * ldk` (the same for `v`).
    pub(crate) fn pack(
        &mut self,
        k: &[f32],
        ldk: usize,
        v: &[f32],
        ldv: usize,
        nk: usize,
        hd: usize,
    ) {
        self.nk = nk;
        self.hd = hd;
        // Scores: k = hd inputs, n = nk outputs; panel p, input d, lane l = key p * NR + l, channel d.
        self.kt.reset(hd, nk);
        for j in 0..nk {
            let (panel, lane) = (j / NR, j % NR);
            let row = &k[j * ldk..j * ldk + hd];
            for (d, &x) in row.iter().enumerate() {
                self.kt.data[(panel * hd + d) * NR + lane] = x;
            }
        }
        // Output: k = nk inputs (the keys), n = hd outputs; panel p, input j, lane l = channel p * NR + l.
        self.vp.reset(nk, hd);
        for j in 0..nk {
            let row = &v[j * ldv..j * ldv + hd];
            for (d, &x) in row.iter().enumerate() {
                self.vp.data[((d / NR) * nk + j) * NR + d % NR] = x;
            }
        }
    }
}

/// One attention head for `nq` queries (`hd` values each, rows `ldq` apart): scores against the packed keys scaled by
/// `scale`, the softmax, then the weighted values into `out` (rows `ldo` apart). `scores` is scratch.
#[allow(clippy::too_many_arguments)]
pub(crate) fn attend(
    q: &[f32],
    ldq: usize,
    nq: usize,
    kv: &HeadKv,
    scale: f32,
    scores: &mut Vec<f32>,
    out: &mut [f32],
    ldo: usize,
) {
    let nk_pad = kv.nk.div_ceil(NR) * NR;
    scores.resize(nq * nk_pad, 0.0);
    gemm(q, ldq, nq, &kv.kt, Act::None, None, scores, nk_pad);
    for row in scores.chunks_exact_mut(nk_pad) {
        softmax_scaled(&mut row[..kv.nk], scale);
    }
    gemm(scores, nk_pad, nq, &kv.vp, Act::None, None, out, ldo);
}

/// Sets `out[i] = a[i] + b[i]`.
pub(crate) fn add_into(a: &[f32], b: &[f32], out: &mut [f32]) {
    for ((o, &x), &y) in out.iter_mut().zip(a).zip(b) {
        *o = x + y;
    }
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

    /// The definition of the product: the bias and then the inputs in order, one chain per element.
    fn naive(
        a: &[f32],
        lda: usize,
        m: usize,
        w: &[f32],
        n: usize,
        k: usize,
        bias: &[f32],
    ) -> Vec<f32> {
        let mut out = vec![0.0; m * n];
        for r in 0..m {
            for c in 0..n {
                let mut acc = bias[c];
                for i in 0..k {
                    acc += a[r * lda + i] * w[c * k + i];
                }
                out[r * n + c] = acc;
            }
        }
        out
    }

    #[test]
    fn the_product_is_the_chain_of_the_definition_bit_for_bit() {
        let mut rng = Lcg(7);
        // Row counts around the register tile, column counts around the panel, a strided input.
        for &(m, n, k, lda) in &[
            (1, 1, 1, 1),
            (4, 16, 8, 8),
            (5, 17, 33, 40),
            (7, 31, 96, 96),
            (64, 100, 150, 150),
            (3, 16, 2, 5),
            (130, 48, 20, 21),
        ] {
            let a = rng.vec(m * lda);
            let wt = rng.vec(n * k);
            let bias = rng.vec(n);
            let want = naive(&a, lda, m, &wt, n, k, &bias);
            let packed = Packed::from_nk(&wt, n, k, Some(&bias));
            let mut got = vec![f32::NAN; m * n];
            gemm(&a, lda, m, &packed, Act::None, None, &mut got, n);
            assert_eq!(
                bits(&got),
                bits(&want),
                "m {m} n {n} k {k} (generic or AVX2, whichever this machine runs)"
            );
            // Cut into rows of different sizes in parallel tasks: the same bits.
            if lda == k {
                let mut par = vec![f32::NAN; m * n];
                gemm_par(&a, m, &packed, Act::None, None, &mut par);
                assert_eq!(bits(&par), bits(&want), "gemm_par m {m} n {n} k {k}");
            }
            // The matrix stored [k][n] packs to the same product.
            let kn: Vec<f32> = (0..k * n).map(|i| wt[(i % n) * k + i / n]).collect();
            let packed_kn = Packed::from_kn(&kn, k, n, Some(&bias));
            let mut got2 = vec![f32::NAN; m * n];
            gemm(&a, lda, m, &packed_kn, Act::None, None, &mut got2, n);
            assert_eq!(bits(&got2), bits(&want), "from_kn m {m} n {n} k {k}");
        }
    }

    #[test]
    fn the_generic_and_the_avx2_builds_agree() {
        let mut rng = Lcg(11);
        let (m, n, k) = (37, 70, 123);
        let a = rng.vec(m * k);
        let wt = rng.vec(n * k);
        let bias = rng.vec(n);
        let packed = Packed::from_nk(&wt, n, k, Some(&bias));
        let mut generic = vec![0.0; m * n];
        gemm_impl(&a, k, m, &packed, Act::Gelu, None, &mut generic, n);
        let mut dispatched = vec![0.0; m * n];
        gemm(&a, k, m, &packed, Act::Gelu, None, &mut dispatched, n);
        assert_eq!(bits(&generic), bits(&dispatched));
    }

    #[test]
    fn the_activation_and_the_residual_follow_the_product() {
        let mut rng = Lcg(3);
        let (m, n, k) = (9, 20, 12);
        let a = rng.vec(m * k);
        let wt = rng.vec(n * k);
        let bias = rng.vec(n);
        let res = rng.vec(m * n);
        let plain = naive(&a, k, m, &wt, n, k, &bias);
        let packed = Packed::from_nk(&wt, n, k, Some(&bias));
        for act in [Act::None, Act::Relu, Act::Gelu] {
            let mut got = vec![0.0; m * n];
            gemm(
                &a,
                k,
                m,
                &packed,
                act,
                Some(Residual { data: &res, ld: n }),
                &mut got,
                n,
            );
            for i in 0..m * n {
                assert_eq!(
                    got[i].to_bits(),
                    (activate(plain[i], act) + res[i]).to_bits()
                );
            }
        }
    }

    #[test]
    fn gelu_is_the_exact_one() {
        // erf-based GELU values (computed with Python's math.erf): gelu(1) = 0.8413447460685429, gelu(-1) = -0.15865525393145707.
        assert!((gelu(1.0) - 0.841_344_8).abs() < 1e-7);
        assert!((gelu(-1.0) + 0.158_655_26).abs() < 1e-7);
        assert_eq!(gelu(0.0), 0.0);
        assert!((gelu(5.0) - 5.0).abs() < 1e-5);
        assert!(gelu(-8.0).abs() < 1e-6);
    }

    #[test]
    fn layer_norm_has_zero_mean_and_unit_variance_before_the_affine_step() {
        let mut rng = Lcg(5);
        let c = 96;
        let x: Vec<f32> = rng.vec(4 * c).into_iter().map(|v| v * 3.0 + 1.0).collect();
        let (w, b) = (vec![1.0; c], vec![0.0; c]);
        let mut y = vec![0.0; x.len()];
        layer_norm(&x, c, &w, &b, 1e-6, &mut y);
        for row in y.chunks_exact(c) {
            let mean: f64 = row.iter().map(|&v| f64::from(v)).sum::<f64>() / c as f64;
            let var: f64 = row
                .iter()
                .map(|&v| (f64::from(v) - mean).powi(2))
                .sum::<f64>()
                / c as f64;
            assert!(mean.abs() < 1e-6, "{mean}");
            assert!((var - 1.0).abs() < 1e-4, "{var}");
        }
        let mut z = vec![0.0; x.len()];
        layer_norm_par(&x, c, &w, &b, 1e-6, &mut z);
        assert_eq!(bits(&y), bits(&z));
    }

    #[test]
    fn softmax_sums_to_one_and_ignores_a_shift() {
        let mut a = vec![1.0f32, 2.0, 3.0, -1.0, 0.5];
        let mut b: Vec<f32> = a.iter().map(|v| v + 100.0).collect();
        softmax_scaled(&mut a, 1.0);
        softmax_scaled(&mut b, 1.0);
        let sum: f64 = a.iter().map(|&v| f64::from(v)).sum();
        assert!((sum - 1.0).abs() < 1e-6);
        for (x, y) in a.iter().zip(&b) {
            assert!((x - y).abs() < 1e-5);
        }
    }

    #[test]
    fn attention_matches_a_direct_evaluation() {
        let mut rng = Lcg(21);
        let (nq, nk, hd, ldq, ldk) = (5, 21, 32, 40, 35);
        let q = rng.vec(nq * ldq);
        let k = rng.vec(nk * ldk);
        let v = rng.vec(nk * ldk);
        let scale = 1.0 / (hd as f32).sqrt();
        let mut kv = HeadKv::default();
        kv.pack(&k, ldk, &v, ldk, nk, hd);
        let mut out = vec![0.0; nq * hd];
        attend(&q, ldq, nq, &kv, scale, &mut Vec::new(), &mut out, hd);
        for i in 0..nq {
            let mut s: Vec<f64> = (0..nk)
                .map(|j| {
                    (0..hd)
                        .map(|d| f64::from(q[i * ldq + d]) * f64::from(k[j * ldk + d]))
                        .sum::<f64>()
                        * f64::from(scale)
                })
                .collect();
            let max = s.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let sum: f64 = s.iter().map(|x| (x - max).exp()).sum();
            for x in &mut s {
                *x = (*x - max).exp() / sum;
            }
            for d in 0..hd {
                let want: f64 = (0..nk).map(|j| s[j] * f64::from(v[j * ldk + d])).sum();
                assert!(
                    (f64::from(out[i * hd + d]) - want).abs() < 1e-5,
                    "query {i} channel {d}"
                );
            }
        }
    }

    fn bits(v: &[f32]) -> Vec<u32> {
        v.iter().map(|x| x.to_bits()).collect()
    }
}
