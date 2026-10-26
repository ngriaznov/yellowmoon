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
    field_halo_px(plan, planes, k).max(flares_halo_px(plan, planes, k))
}

/// Context [`apply_field`] needs around an output rectangle at scale `k`: L1's reach plus L2's. 0 when the
/// reshaping and the field optics do nothing.
#[must_use]
pub fn field_halo_px(plan: &LensPlan, planes: &LensPlanes, k: usize) -> usize {
    if plan.is_identity() {
        return 0;
    }
    let (fw, fh) = frame_at(plan, k);
    l1::halo(plan, planes, k, fw, fh) + field::halo(plan, fw, fh)
}

/// Context [`apply_flares`] needs around an output rectangle at scale `k`: the glow core's reach. 0 when the
/// flares do nothing.
#[must_use]
pub fn flares_halo_px(plan: &LensPlan, planes: &LensPlanes, k: usize) -> usize {
    if plan.is_identity() {
        return 0;
    }
    let (fw, fh) = frame_at(plan, k);
    l3::halo(plan, planes, k, fw, fh)
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
    let z = field_with(plan, planes, k, input, out, opts)?;
    l3::apply(plan, planes, k, input, &z, out)
}

/// The checks every pass makes before it reads the planes: the planes belong to the plan, and `out` lies in
/// the frame.
fn check_planes(
    plan: &LensPlan,
    planes: &LensPlanes,
    input: &ImageRef,
    out: PixelRect,
) -> Result<(), LensError> {
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
    Ok(())
}

/// `Z = L2(L1(X))` on `out`: the picture after the reshaping and the field optics, the part of the lens that
/// does not depend on Flares (`LensPlan::field_key`). The planar result has `3 * out.w * out.h` samples.
///
/// `input` must cover `out` grown by [`field_halo_px`] and clamped to the frame. When neither the reshaping nor
/// the field optics act (an identity plan, or Flares alone) the result is the input's `out` rectangle bit for
/// bit.
///
/// # Errors
///
/// [`LensError::Frame`] for a plan, planes or input that do not belong together,
/// [`LensError::InputTooSmall`] when the input does not cover the halo.
pub fn apply_field(
    plan: &LensPlan,
    planes: &LensPlanes,
    k: usize,
    input: &ImageRef,
    out: PixelRect,
) -> Result<Vec<f32>, LensError> {
    input.validate()?;
    check_scale(plan, k, input)?;
    if plan.is_identity() {
        return field::copy_rect(input, out);
    }
    field_with(plan, planes, k, input, out, field::Options::default())
}

/// The samples `data` as the image over `rect` of a frame of `frame` pixels.
fn over(rect: PixelRect, frame: (usize, usize), data: &[f32]) -> ImageRef<'_> {
    ImageRef {
        width: rect.w,
        height: rect.h,
        x0: rect.x,
        y0: rect.y,
        frame_width: frame.0,
        frame_height: frame.1,
        data,
    }
}

fn field_with(
    plan: &LensPlan,
    planes: &LensPlanes,
    k: usize,
    input: &ImageRef,
    out: PixelRect,
    opts: field::Options,
) -> Result<Vec<f32>, LensError> {
    check_planes(plan, planes, input, out)?;
    let (fw, fh) = (input.frame_width, input.frame_height);
    // L1 spreads the light's colour, not the clip pattern's cast.
    let lit = crate::light::lit_input(plan, input);
    let lit_image = lit
        .as_deref()
        .map_or(*input, |data| ImageRef { data, ..*input });
    // L1 over what L2 reads, then L2 on `out`.
    let l2_halo = field::Geometry::new(plan, fw, fh, opts).map_or(0, |g| g.halo());
    let want = out.grown(l2_halo, fw, fh);
    let stage = l1::apply(plan, planes, k, &lit_image, want)?;
    let image = stage.image();
    // L2 spreads the picture, and besides it the light in its own colour with the headroom of the clipped small
    // lights, which reaches the pixels the lens moves it to.
    let moved = matches!(stage, l1::Stage::New { .. });
    let l2 = crate::light::l2_input(plan, planes, k, input, moved.then_some(&image));
    match &l2 {
        Some(l2) => {
            let extra = over(l2.rect, (fw, fh), &l2.extra);
            let picture = over(l2.rect, (fw, fh), &l2.own);
            field::apply(plan, &picture, Some(&extra), out, opts)
        }
        None => field::apply(plan, &image, None, out, opts),
    }
}

/// The flare terms of L3 on `out`, `F (a_g K_g * R + a_v K_v * R + sum a_i G_i[H] + ...)`: the light the
/// flares add, planar `f64`, `3 * out.w * out.h` samples, or `None` when Flares is 0. They read the picture
/// before L1 and not `Z`, so they do not change with Edges; add them to `Z` with [`add_flares`].
///
/// `input` must cover `out` grown by [`flares_halo_px`] and clamped to the frame.
///
/// # Errors
///
/// As [`apply_flares`].
pub fn flare_terms(
    plan: &LensPlan,
    planes: &LensPlanes,
    k: usize,
    input: &ImageRef,
    out: PixelRect,
) -> Result<Option<Vec<f64>>, LensError> {
    input.validate()?;
    check_scale(plan, k, input)?;
    if plan.is_identity() {
        return Ok(None);
    }
    check_planes(plan, planes, input, out)?;
    l3::terms(plan, planes, k, input, out)
}

/// [`flare_terms`] with every ghost drawn at every pixel: the exact sum the coarse grids of the ghosts that vary
/// slowly stand for. Slow (minutes for a photo with hundreds of lights); for tests and for measuring the grids'
/// error.
///
/// # Errors
///
/// As [`flare_terms`].
pub fn flare_terms_exact(
    plan: &LensPlan,
    planes: &LensPlanes,
    k: usize,
    input: &ImageRef,
    out: PixelRect,
) -> Result<Option<Vec<f64>>, LensError> {
    input.validate()?;
    check_scale(plan, k, input)?;
    if plan.is_identity() {
        return Ok(None);
    }
    check_planes(plan, planes, input, out)?;
    l3::terms_exact(plan, planes, k, input, out)
}

/// `(1 - a) Z + terms`: the picture after L3 from `z` ([`apply_field`]) and the flare terms of the same
/// rectangle ([`flare_terms`]). [`apply_flares`] is `flare_terms` and this, so a cached `terms` gives the
/// bits a fresh run gives.
///
/// # Errors
///
/// [`LensError::Frame`] when `z` and `terms` do not have the same number of samples.
pub fn add_flares(plan: &LensPlan, z: &[f32], terms: &[f64]) -> Result<Vec<f32>, LensError> {
    l3::combine(plan, z, terms)
}

/// L3 on `out`: `out = (1 - a) Z + F (flare terms)`, given `z = L2(L1(X))` over `out` ([`apply_field`]).
///
/// `input` is the picture before L1 (the glow's core reads it) and must cover `out` grown by
/// [`flares_halo_px`] and clamped to the frame. With Flares at 0 the result is `z`.
///
/// # Errors
///
/// [`LensError::Frame`] for a plan, planes, `z` or input that do not belong together,
/// [`LensError::InputTooSmall`] when the input does not cover the halo.
pub fn apply_flares(
    plan: &LensPlan,
    planes: &LensPlanes,
    k: usize,
    input: &ImageRef,
    z: &[f32],
    out: PixelRect,
) -> Result<Vec<f32>, LensError> {
    input.validate()?;
    check_scale(plan, k, input)?;
    if plan.is_identity() {
        return Ok(z.to_vec());
    }
    check_planes(plan, planes, input, out)?;
    l3::apply(plan, planes, k, input, z, out)
}
