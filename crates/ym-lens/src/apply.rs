//! The per-tile passes composed: `Z = L2(L1(X))`, `out = (1 - a) Z + F (flare terms)` (`docs/21` section 4).

use crate::error::LensError;
use crate::image::{ImageRef, PixelRect};
use crate::plan::LensPlan;
use crate::planes::LensPlanes;
use crate::{field, l1, l3};

/// The frame at scale `k`: the full-resolution frame reduced `k` times, rounded up.
fn frame_at(plan: &LensPlan, k: usize) -> (usize, usize) {
    let f = plan.frame();
    (f.width.div_ceil(k.max(1)), f.height.div_ceil(k.max(1)))
}

/// Context needed around an output rectangle at scale `k`, in pixels at that scale: L1's, then L2's, and L3's
/// core, whichever chain reads farthest. 0 when the field optics and the other passes do nothing.
///
/// The frame at scale `k` is taken as the full frame divided by `k`, rounded up; a caller whose frame is a
/// pixel smaller needs no more.
#[must_use]
pub fn halo_px(plan: &LensPlan, planes: &LensPlanes, k: usize) -> usize {
    if plan.is_identity() {
        return 0;
    }
    let (fw, fh) = frame_at(plan, k);
    let chain = l1::halo(plan, planes, k, fw, fh) + field::halo(plan, fw, fh);
    chain.max(l3::halo(plan, planes, k, fw, fh))
}

/// Checks that the input's frame is the plan's frame at scale `k` (to the rounding of the reduction).
fn check_scale(plan: &LensPlan, k: usize, input: &ImageRef) -> Result<(), LensError> {
    if k == 0 {
        return Err(LensError::Frame("the scale k must be at least 1".into()));
    }
    let f = plan.frame();
    let fits = |at_k: usize, full: usize| at_k * k + k > full && at_k * k < full + k;
    if !fits(input.frame_width, f.width) || !fits(input.frame_height, f.height) {
        return Err(LensError::Frame(format!(
            "the input's frame is {}x{}, the plan's {}x{} frame reduced {k} times is {}x{}",
            input.frame_width,
            input.frame_height,
            f.width,
            f.height,
            f.width.div_ceil(k),
            f.height.div_ceil(k),
        )));
    }
    Ok(())
}

/// L1 to L3 on `out` (frame coordinates at scale `k`): the planar result, `3 * out.w * out.h` samples.
///
/// `input` must cover `out` grown by [`halo_px`] and clamped to the frame. An identity plan returns the
/// input's `out` rectangle bit for bit.
///
/// Every output pixel depends only on input pixels within the halo and on absolute frame coordinates, so any
/// tiling of the frame gives the same bits as the whole frame.
///
/// # Errors
///
/// [`LensError::Frame`] for a plan, planes or input that do not belong together,
/// [`LensError::InputTooSmall`] when the input does not cover the halo.
pub fn apply(
    plan: &LensPlan,
    planes: &LensPlanes,
    k: usize,
    input: &ImageRef,
    out: PixelRect,
) -> Result<Vec<f32>, LensError> {
    apply_with(plan, planes, k, input, out, field::Options::default())
}

/// [`apply`] with the CPU reference's knobs, for tests.
pub(crate) fn apply_with(
    plan: &LensPlan,
    planes: &LensPlanes,
    k: usize,
    input: &ImageRef,
    out: PixelRect,
    opts: field::Options,
) -> Result<Vec<f32>, LensError> {
    input.validate()?;
    check_scale(plan, k, input)?;
    if plan.is_identity() {
        return field::copy_rect(input, out);
    }
    if planes.key() != plan.planes_key() {
        return Err(LensError::Frame(
            "the planes were built for another lens, camera or frame".into(),
        ));
    }
    let (fw, fh) = (input.frame_width, input.frame_height);
    if out.x1() > fw || out.y1() > fh {
        return Err(LensError::Frame(format!(
            "the output {out} does not fit the {fw}x{fh} frame"
        )));
    }
    // L1 over what L2 reads, then L2 on `out`, then L3.
    let l2_halo = field::Geometry::new(plan, fw, fh, opts).map_or(0, |g| g.halo());
    let want = out.grown(l2_halo, fw, fh);
    let stage = l1::apply(plan, planes, k, input, want)?;
    let z = field::apply(plan, &stage.image(), out, opts)?;
    l3::apply(plan, planes, k, input, z, out)
}
