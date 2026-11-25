//! From the network's outputs to a label map: the panoptic post-processing of `EomtImageProcessor.post_process_panoptic_segmentation`
//! with its defaults (`threshold` 0.8, `mask_threshold` 0.5, `overlap_mask_area_threshold` 0.8) and `stuff_classes = 80..133`, which is
//! what the POC used (`docs/34` section 4b.1), reduced to what the label map needs.
//!
//! The reference does this, for one picture whose network input was resized to `th` x `tw` (inside the 640 x 640 pad):
//!
//! 1. the mask logits are resized bilinearly (`align_corners=False`) from 160 x 160 to 640 x 640 and cropped to `th` x `tw`
//!    (`unpad_image`; the resize to the picture's own size is the identity here, because the label map is made at the fit size);
//! 2. every query gets a class and a score: the largest of `softmax(class logits)` over the 134 outputs; a query is *kept* when its
//!    class is not "none" (133) and its score is above 0.8;
//! 3. over the kept queries, every pixel goes to the one with the largest `score * sigmoid(mask logit)` (the first on a tie);
//! 4. a kept query is a *segment* when it owns at least one pixel, its mask (`sigmoid >= 0.5`) is not empty, they share a pixel, and
//!    the area it owns is more than 0.8 of its mask's area; the segment is its pixels where `sigmoid >= 0.5`. Stuff classes that
//!    several queries claim are given one segment id by the reference; the label map holds the class only, so that merge changes no
//!    pixel here;
//! 5. a pixel that no segment holds is [`UNLABELLED`].
//!
//! Only the kept queries are resized, which gives the values of the reference's full resize. Everything is integer counts or
//! per-pixel `f32` work, so the result does not depend on the thread count.

use rayon::prelude::*;

use super::network::{CLASS_OUTPUTS, IMAGE_PX, MASK_PX, QUERIES, Raw};
use super::{CLASSES, UNLABELLED};
use crate::sam2::kernels::softmax_scaled;

/// `threshold` of the reference: a query is kept when its score is above it.
pub const SCORE_THRESHOLD: f32 = 0.8;
/// `mask_threshold`: a pixel is in a query's mask when its probability is at least this.
pub const MASK_THRESHOLD: f32 = 0.5;
/// `overlap_mask_area_threshold`: a query is a segment when its area share of its mask is above this.
pub const OVERLAP_THRESHOLD: f64 = 0.8;

/// A query the post-processing keeps.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Kept {
    /// The query's index.
    pub query: usize,
    /// The class: the output of the class head with the largest probability (never 133, "none").
    pub label: u8,
    /// The softmax probability of that class.
    pub score: f32,
}

/// The queries that pass the score threshold and are not "none", in query order (`remove_low_and_no_objects`).
pub fn kept_queries(class_logits: &[f32]) -> Vec<Kept> {
    debug_assert_eq!(class_logits.len(), QUERIES * CLASS_OUTPUTS);
    let mut kept = Vec::new();
    let mut row = [0.0f32; CLASS_OUTPUTS];
    for (query, logits) in class_logits.chunks_exact(CLASS_OUTPUTS).enumerate() {
        row.copy_from_slice(logits);
        softmax_scaled(&mut row, 1.0);
        let (mut best, mut score) = (0usize, row[0]);
        for (i, &p) in row.iter().enumerate().skip(1) {
            if p > score {
                best = i;
                score = p;
            }
        }
        if best != CLASSES && score > SCORE_THRESHOLD {
            kept.push(Kept {
                query,
                label: best as u8,
                score,
            });
        }
    }
    kept
}

/// One axis of `F.interpolate(mode="bilinear", align_corners=False)` from `MASK_PX` samples to `IMAGE_PX`: for each output index the
/// two source indices and their weights, in `f32` as PyTorch computes them (the scale 0.25 makes every number exact).
struct Axis {
    i0: Vec<usize>,
    i1: Vec<usize>,
    w0: Vec<f32>,
    w1: Vec<f32>,
}

impl Axis {
    fn new(count: usize) -> Self {
        let scale = MASK_PX as f32 / IMAGE_PX as f32;
        let (mut i0, mut i1, mut w0, mut w1) = (vec![], vec![], vec![], vec![]);
        for dst in 0..count {
            let src = (scale * (dst as f32 + 0.5) - 0.5).max(0.0);
            let a = (src as usize).min(MASK_PX - 1);
            let b = if a + 1 < MASK_PX { a + 1 } else { a };
            let l = src - a as f32;
            i0.push(a);
            i1.push(b);
            w0.push(1.0 - l);
            w1.push(l);
        }
        Self { i0, i1, w0, w1 }
    }
}

/// `1 / (1 + e^-x)` of the kernels (`libm`'s `expf`), the sigmoid of the reference up to rounding.
fn sigmoid(v: f32) -> f32 {
    crate::sam2::kernels::sigmoid(v)
}

/// The label map of a run: `th * tw` class ids (0 to 132, or [`UNLABELLED`]), rows top to bottom.
pub(crate) fn label_map(raw: &Raw, th: usize, tw: usize) -> Vec<u8> {
    let kept = kept_queries(&raw.class_logits);
    let k = kept.len();
    let mut classes = vec![UNLABELLED; th * tw];
    if k == 0 {
        return classes;
    }
    let (ax, ay) = (Axis::new(tw), Axis::new(th));
    let plane = MASK_PX * MASK_PX;
    // Per row: the winning kept query of each pixel, whether its mask holds the pixel, and the counts of the row.
    let mut winners = vec![0u8; th * tw];
    let mut inside = vec![false; th * tw];
    let mut counts = vec![0u32; th * 3 * k];
    winners
        .par_chunks_mut(tw)
        .zip(inside.par_chunks_mut(tw))
        .zip(counts.par_chunks_mut(3 * k))
        .enumerate()
        .for_each_init(
            || vec![0.0f32; k * tw],
            |tmp, (y, ((win_row, in_row), count))| {
                let (y0, y1, wy0, wy1) = (ay.i0[y], ay.i1[y], ay.w0[y], ay.w1[y]);
                // The rows of every kept query, resized along x then along y.
                for (ki, kept) in kept.iter().enumerate() {
                    let p = &raw.mask_logits[kept.query * plane..][..plane];
                    let (r0, r1) = (&p[y0 * MASK_PX..][..MASK_PX], &p[y1 * MASK_PX..][..MASK_PX]);
                    let out = &mut tmp[ki * tw..][..tw];
                    for (x, v) in out.iter_mut().enumerate() {
                        let (a, b) = (ax.i0[x], ax.i1[x]);
                        let top = ax.w0[x] * r0[a] + ax.w1[x] * r0[b];
                        let bottom = ax.w0[x] * r1[a] + ax.w1[x] * r1[b];
                        *v = wy0 * top + wy1 * bottom;
                    }
                }
                let (areas, rest) = count.split_at_mut(k);
                let (masks, finals) = rest.split_at_mut(k);
                for x in 0..tw {
                    let mut best = 0usize;
                    let mut best_value = f32::NEG_INFINITY;
                    let mut best_in = false;
                    for (ki, kept) in kept.iter().enumerate() {
                        let prob = sigmoid(tmp[ki * tw + x]);
                        let member = prob >= MASK_THRESHOLD;
                        if member {
                            masks[ki] += 1;
                        }
                        let value = kept.score * prob;
                        if value > best_value {
                            best = ki;
                            best_value = value;
                            best_in = member;
                        }
                    }
                    areas[best] += 1;
                    if best_in {
                        finals[best] += 1;
                    }
                    win_row[x] = best as u8;
                    in_row[x] = best_in;
                }
            },
        );
    // A query is a segment when it owns pixels, has a mask, they overlap, and it owns more than 0.8 of its mask.
    let mut valid = vec![false; k];
    for (ki, ok) in valid.iter_mut().enumerate() {
        let sum = |part: usize| -> u32 {
            counts
                .chunks_exact(3 * k)
                .map(|row| row[part * k + ki])
                .sum()
        };
        let (area, mask, fin) = (sum(0), sum(1), sum(2));
        if area > 0 && mask > 0 && fin > 0 {
            // `mask_k_area / original_area` of two int64 tensors is a float32 true division.
            let ratio = f64::from(area as f32 / mask as f32);
            *ok = ratio > OVERLAP_THRESHOLD;
        }
    }
    for ((c, &w), &inn) in classes.iter_mut().zip(&winners).zip(&inside) {
        let w = usize::from(w);
        if inn && valid[w] {
            *c = kept[w].label;
        }
    }
    classes
}
