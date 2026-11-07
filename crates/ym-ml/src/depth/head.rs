//! The DPT head of Depth Anything V2-Small: the projections and resize layers that turn the four tapped token layers into
//! feature maps of four scales, the four refinenet fusion blocks and the output convolutions.
//!
//! Ported from `depth_anything_v2/dpt.py` (class `DPTHead`, `DepthAnythingV2.forward`) and `util/blocks.py`
//! (`FeatureFusionBlock`, `ResidualConvUnit`, `_make_scratch`) of github.com/DepthAnything/Depth-Anything-V2 (Apache-2.0).
//! The line numbers quoted are those of that repository's `main` branch read on 2026-11-07. The configuration of the
//! `vits` model (`features=64`, `out_channels=[48, 96, 192, 384]`, `use_bn=False`, `use_clstoken=False`) is the README's.
//!
//! Maps are `[y][x][channel]`; the PyTorch tensors are `[1][channel][y][x]`.

use rayon::prelude::*;

use super::encoder::EMBED_DIM;
use super::ops::{Axis, Conv3, TConv, bilinear, bilinear_row};
use super::{Result, Tensors};
use crate::model::MlError;
use crate::sam2::kernels::{Act, Packed, Residual, add_into, gemm, gemm_par};

/// `features=64`: the width of the fusion path.
pub(crate) const FEATURES: usize = 64;
/// `out_channels=[48, 96, 192, 384]`: the widths of the four projected layers.
pub(crate) const OUT_CHANNELS: [usize; 4] = [48, 96, 192, 384];
/// `head_features_2 = 32` (`dpt.py` line 106).
const HEAD_FEATURES_2: usize = 32;

/// `ResidualConvUnit` (`blocks.py` lines 29 to 80) without batch norm: `conv2(relu(conv1(relu(x)))) + x`.
#[derive(Debug)]
struct ResidualConvUnit {
    conv1: Conv3,
    conv2: Conv3,
}

impl ResidualConvUnit {
    fn load(w: &mut Tensors, prefix: &str) -> Result<Self> {
        Ok(Self {
            conv1: w.conv3(&format!("{prefix}.conv1"), FEATURES, FEATURES, true)?,
            conv2: w.conv3(&format!("{prefix}.conv2"), FEATURES, FEATURES, true)?,
        })
    }

    /// `activation = nn.ReLU(False)` is not in place (`dpt.py` line 15), so `x` is still the input of the skip addition
    /// (`skip_add.add(out, x)`, line 80). The second ReLU follows the first convolution's output.
    fn forward(&self, x: &[f32], h: usize, w: usize) -> Vec<f32> {
        let t = self.conv1.forward(x, h, w, 1, true, Act::Relu, None);
        self.conv2.forward(&t, h, w, 1, false, Act::None, Some(x))
    }
}

/// `FeatureFusionBlock` (`blocks.py` lines 83 to 148) with `deconv=False`, `bn=False`, `expand=False`,
/// `align_corners=True` (`dpt.py` lines 12 to 21).
#[derive(Debug)]
struct Fusion {
    /// `resConfUnit1`, applied to the skip input; the coarsest block (`refinenet4`) never uses its own.
    rcu1: Option<ResidualConvUnit>,
    rcu2: ResidualConvUnit,
    out_conv: Packed,
}

impl Fusion {
    fn load(w: &mut Tensors, index: usize) -> Result<Self> {
        let p = format!("depth_head.scratch.refinenet{index}");
        Ok(Self {
            rcu1: if index == 4 {
                None
            } else {
                Some(ResidualConvUnit::load(w, &format!("{p}.resConfUnit1"))?)
            },
            rcu2: ResidualConvUnit::load(w, &format!("{p}.resConfUnit2"))?,
            out_conv: w.conv1x1(&format!("{p}.out_conv"), FEATURES, FEATURES)?,
        })
    }

    /// `forward(*xs, size=...)` (lines 123 to 148): `output = xs[0]`; with a second input,
    /// `output = output + resConfUnit1(xs[1])` (lines 131 to 133); `output = resConfUnit2(output)`; bilinear resize with
    /// `align_corners=True` to `size`; the 1 x 1 `out_conv`. `x` and `skip` are `h` x `w`; the output is `out_h` x `out_w`.
    fn forward(
        &self,
        x: &[f32],
        skip: Option<&[f32]>,
        (h, w): (usize, usize),
        (out_h, out_w): (usize, usize),
    ) -> Result<Vec<f32>> {
        let sum;
        let stream = match (skip, &self.rcu1) {
            (Some(skip), Some(rcu1)) => {
                let res = rcu1.forward(skip, h, w);
                let mut s = vec![0.0f32; x.len()];
                add_into(x, &res, &mut s);
                sum = s;
                &sum[..]
            }
            (None, _) => x,
            (Some(_), None) => {
                return Err(MlError::Unsupported(
                    "a skip input for the coarsest fusion block".into(),
                ));
            }
        };
        let fused = self.rcu2.forward(stream, h, w);
        let up = bilinear(&fused, h, w, FEATURES, out_h, out_w);
        let mut out = vec![0.0f32; out_h * out_w * FEATURES];
        gemm_par(
            &up,
            out_h * out_w,
            &self.out_conv,
            Act::None,
            None,
            &mut out,
        );
        Ok(out)
    }
}

/// What a traced run of the head keeps (maps are `[y][x][channel]`).
#[derive(Debug, Default)]
pub struct HeadTrace {
    /// `layer_1` to `layer_4` after the projections and resize layers (`dpt.py` line 134): 4x4, 2x2, 1x1 and 1/2 of the patch
    /// grid, with 48, 96, 192 and 384 channels.
    pub layers: Vec<Vec<f32>>,
    /// `path_4`, `path_3`, `path_2`, `path_1` (lines 141 to 144): 1x, 2x, 4x and 8x the patch grid, 64 channels.
    pub paths: Vec<Vec<f32>>,
    /// `output_conv1(path_1)` (line 146): 8x the patch grid, 32 channels.
    pub output_conv1: Vec<f32>,
}

/// The head's weights.
#[derive(Debug)]
pub(crate) struct Head {
    projects: [Packed; 4],
    resize0: TConv,
    resize1: TConv,
    resize3: Conv3,
    /// `layer1_rn` to `layer4_rn`: 3 x 3 convolutions without bias to 64 channels.
    layer_rn: [Conv3; 4],
    /// `refinenet1` to `refinenet4` (index 0 is `refinenet1`).
    refinenet: [Fusion; 4],
    output_conv1: Conv3,
    output_conv2_0: Conv3,
    output_conv2_2: Packed,
}

/// Row scratch of the output stage.
#[derive(Default)]
struct RowScratch {
    rows: [Vec<f32>; 3],
    cols: Vec<f32>,
    hidden: Vec<f32>,
}

impl Head {
    pub(crate) fn load(w: &mut Tensors) -> Result<Self> {
        let p = "depth_head";
        let oc = OUT_CHANNELS;
        let projects = [
            w.conv1x1(&format!("{p}.projects.0"), oc[0], EMBED_DIM)?,
            w.conv1x1(&format!("{p}.projects.1"), oc[1], EMBED_DIM)?,
            w.conv1x1(&format!("{p}.projects.2"), oc[2], EMBED_DIM)?,
            w.conv1x1(&format!("{p}.projects.3"), oc[3], EMBED_DIM)?,
        ];
        // `resize_layers` (lines 61 to 81): ConvTranspose2d 4 x 4 stride 4, ConvTranspose2d 2 x 2 stride 2, Identity,
        // Conv2d 3 x 3 stride 2 padding 1.
        let resize0 = w.tconv(&format!("{p}.resize_layers.0"), oc[0], 4)?;
        let resize1 = w.tconv(&format!("{p}.resize_layers.1"), oc[1], 2)?;
        let resize3 = w.conv3(&format!("{p}.resize_layers.3"), oc[3], oc[3], true)?;
        let layer_rn = [
            w.conv3(&format!("{p}.scratch.layer1_rn"), FEATURES, oc[0], false)?,
            w.conv3(&format!("{p}.scratch.layer2_rn"), FEATURES, oc[1], false)?,
            w.conv3(&format!("{p}.scratch.layer3_rn"), FEATURES, oc[2], false)?,
            w.conv3(&format!("{p}.scratch.layer4_rn"), FEATURES, oc[3], false)?,
        ];
        let refinenet = [
            Fusion::load(w, 1)?,
            Fusion::load(w, 2)?,
            Fusion::load(w, 3)?,
            Fusion::load(w, 4)?,
        ];
        let output_conv1 = w.conv3(
            &format!("{p}.scratch.output_conv1"),
            FEATURES / 2,
            FEATURES,
            true,
        )?;
        let output_conv2_0 = w.conv3(
            &format!("{p}.scratch.output_conv2.0"),
            HEAD_FEATURES_2,
            FEATURES / 2,
            true,
        )?;
        let output_conv2_2 =
            w.conv1x1(&format!("{p}.scratch.output_conv2.2"), 1, HEAD_FEATURES_2)?;
        Ok(Self {
            projects,
            resize0,
            resize1,
            resize3,
            layer_rn,
            refinenet,
            output_conv1,
            output_conv2_0,
            output_conv2_2,
        })
    }

    /// `DPTHead.forward` (lines 117 to 150) and the final `F.relu` of `DepthAnythingV2.forward` (line 182) for the `taps`
    /// (each `[ph * pw][384]`); the map of `ph * 14` x `pw * 14` values.
    pub(crate) fn forward(
        &self,
        taps: &[Vec<f32>; 4],
        ph: usize,
        pw: usize,
        cancel: &dyn Fn() -> bool,
        mut trace: Option<&mut HeadTrace>,
    ) -> Result<Vec<f32>> {
        let n = ph * pw;
        // Lines 127 to 132: tokens to a map, the 1 x 1 projection, the resize layer.
        let mut layers: Vec<(Vec<f32>, usize, usize)> = Vec::with_capacity(4);
        for (i, tap) in taps.iter().enumerate() {
            debug_assert_eq!(tap.len(), n * EMBED_DIM);
            let mut projected = vec![0.0f32; n * OUT_CHANNELS[i]];
            gemm_par(tap, n, &self.projects[i], Act::None, None, &mut projected);
            layers.push(match i {
                0 => (self.resize0.forward(&projected, ph, pw), ph * 4, pw * 4),
                1 => (self.resize1.forward(&projected, ph, pw), ph * 2, pw * 2),
                2 => (projected, ph, pw),
                _ => {
                    let (oh, ow) = Conv3::out_size(ph, pw, 2);
                    (
                        self.resize3
                            .forward(&projected, ph, pw, 2, false, Act::None, None),
                        oh,
                        ow,
                    )
                }
            });
        }
        if cancel() {
            return Err(MlError::Cancelled);
        }
        if let Some(t) = trace.as_deref_mut() {
            t.layers = layers.iter().map(|(d, _, _)| d.clone()).collect();
        }
        // Lines 136 to 139: the 3 x 3 convolutions to 64 channels.
        let rn: Vec<Vec<f32>> = layers
            .iter()
            .enumerate()
            .map(|(i, (d, h, w))| self.layer_rn[i].forward(d, *h, *w, 1, false, Act::None, None))
            .collect();
        let sizes: Vec<(usize, usize)> = layers.iter().map(|&(_, h, w)| (h, w)).collect();
        drop(layers);
        // Lines 141 to 144. `refinenet4` resizes to `layer_3_rn`'s size, `refinenet3` to `layer_2_rn`'s,
        // `refinenet2` to `layer_1_rn`'s and `refinenet1` (no `size`) doubles its input (`scale_factor` 2, line 138).
        let path_4 = self.refinenet[3].forward(&rn[3], None, sizes[3], sizes[2])?;
        let path_3 = self.refinenet[2].forward(&path_4, Some(&rn[2]), sizes[2], sizes[1])?;
        let path_2 = self.refinenet[1].forward(&path_3, Some(&rn[1]), sizes[1], sizes[0])?;
        let (h1, w1) = sizes[0];
        let path_1 =
            self.refinenet[0].forward(&path_2, Some(&rn[0]), sizes[0], (h1 * 2, w1 * 2))?;
        if let Some(t) = trace.as_deref_mut() {
            t.paths = vec![path_4, path_3, path_2, path_1.clone()];
        }
        if cancel() {
            return Err(MlError::Cancelled);
        }
        // Line 146 and then lines 147 to 148 with the output stage of `output_conv2`.
        let (h, w) = (h1 * 2, w1 * 2);
        let conv1 = self
            .output_conv1
            .forward(&path_1, h, w, 1, false, Act::None, None);
        drop(path_1);
        if let Some(t) = trace.as_deref_mut() {
            t.output_conv1 = conv1.clone();
        }
        Ok(self.output_stage(&conv1, h, w, ph * 14, pw * 14))
    }

    /// `F.interpolate(out, (patch_h * 14, patch_w * 14), mode="bilinear", align_corners=True)` (line 147), then
    /// `output_conv2` = 3 x 3 convolution to 32, ReLU, 1 x 1 convolution to 1, ReLU (lines 109 to 115), then the ReLU of
    /// line 182 (already non-negative, kept for fidelity). Done one output row at a time: the three interpolated rows its
    /// 3 x 3 convolution reads are made on the fly instead of keeping the 32-channel map of the picture's size.
    fn output_stage(&self, x: &[f32], h: usize, w: usize, out_h: usize, out_w: usize) -> Vec<f32> {
        let c = FEATURES / 2;
        let (ax, ay) = (Axis::align_corners(w, out_w), Axis::align_corners(h, out_h));
        let mut depth = vec![0.0f32; out_h * out_w];
        depth.par_chunks_mut(out_w).enumerate().for_each_init(
            RowScratch::default,
            |s, (y, row)| {
                for (k, buf) in s.rows.iter_mut().enumerate() {
                    let yy = (y + k) as isize - 1;
                    if yy >= 0 && yy < out_h as isize {
                        buf.resize(out_w * c, 0.0);
                        bilinear_row(x, w, c, &ax, &ay, yy as usize, buf);
                    }
                }
                let rows: [Option<&[f32]>; 3] = std::array::from_fn(|k| {
                    let yy = (y + k) as isize - 1;
                    (yy >= 0 && yy < out_h as isize).then(|| &s.rows[k][..])
                });
                s.hidden.resize(out_w * HEAD_FEATURES_2, 0.0);
                self.output_conv2_0.row(
                    rows,
                    out_w,
                    1,
                    false,
                    Act::Relu,
                    None::<Residual<'_>>,
                    &mut s.cols,
                    &mut s.hidden,
                );
                gemm(
                    &s.hidden,
                    HEAD_FEATURES_2,
                    out_w,
                    &self.output_conv2_2,
                    Act::Relu,
                    None,
                    row,
                    1,
                );
                for v in row.iter_mut() {
                    *v = if *v > 0.0 { *v } else { 0.0 };
                }
            },
        );
        depth
    }
}
