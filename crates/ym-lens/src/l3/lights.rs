//! How much brighter than its clip level a clipped light really is (`docs/21` section 4, L3, *As built (flare
//! strength)*).
//!
//! A pixel the sensor clipped says only that the scene was at least as bright as the clip level. The flares
//! need the light the scene held, and what the photo can tell is the size of the clipped patch: a clipped patch
//! that is small is a lamp, a window, the sun or a specular glint, and those are tens to thousands of times
//! brighter than the clip level of the exposure they are photographed at, while a large clipped patch is a
//! surface (a white wall, a cloud, the sky) that is a little above it. The extra headroom therefore falls with
//! the size of the clipped patch, from [`LIGHT_BOOST_MAX_EV`] for a patch of at most [`LIGHT_BOOST_FULL_HD`]
//! half-diagonals across to nothing at [`LIGHT_BOOST_NONE_HD`].
//!
//! The map is made once per photo on the lens plane, so any tile or scale reads the same numbers.

use crate::consts::{LIGHT_BOOST_FULL_HD, LIGHT_BOOST_MAX_EV, LIGHT_BOOST_NONE_HD};
use crate::error::LensError;
use crate::image::ImageRef;
use crate::l3::restore::Restore;

/// The extra headroom, in EV, of a clipped patch whose equivalent diameter (that of the disc of the same area)
/// is `diameter_hd` half-diagonals.
///
/// [`LIGHT_BOOST_MAX_EV`] up to [`LIGHT_BOOST_FULL_HD`], 0 from [`LIGHT_BOOST_NONE_HD`], and linear in the
/// logarithm of the diameter between.
#[must_use]
pub fn boost_ev(diameter_hd: f64) -> f64 {
    if diameter_hd <= LIGHT_BOOST_FULL_HD {
        return LIGHT_BOOST_MAX_EV;
    }
    if diameter_hd >= LIGHT_BOOST_NONE_HD {
        return 0.0;
    }
    let t = libm::log(LIGHT_BOOST_NONE_HD / diameter_hd)
        / libm::log(LIGHT_BOOST_NONE_HD / LIGHT_BOOST_FULL_HD);
    LIGHT_BOOST_MAX_EV * t
}

/// The extra headroom of every pixel of the lens plane, in EV, row-major, or `None` when no pixel is clipped.
///
/// A pixel with a clip test above 0 belongs to a patch: the 8-connected pixels with a clip test above 0. Every
/// pixel of a patch gets the boost of the patch's size; a pixel whose clip test is 0 gets 0 (it is not raised
/// whatever the number says). `cancel` is polled once per row.
///
/// # Errors
///
/// [`LensError::Cancelled`].
pub(crate) fn map(
    plane: &ImageRef,
    rs: &Restore,
    half_diag: f64,
    cancel: &dyn Fn() -> bool,
) -> Result<Option<Vec<f32>>, LensError> {
    let (w, h) = (plane.width, plane.height);
    let n = w * h;
    let mut mask = vec![false; n];
    let mut any = false;
    for j in 0..h {
        if cancel() {
            return Err(LensError::Cancelled);
        }
        for i in 0..w {
            let at = j * w + i;
            let x = [
                f64::from(plane.data[at]),
                f64::from(plane.data[n + at]),
                f64::from(plane.data[2 * n + at]),
            ];
            if rs.clip_test(x) > 0.0 {
                mask[at] = true;
                any = true;
            }
        }
    }
    if !any {
        return Ok(None);
    }
    let mut out = vec![0.0f32; n];
    let mut seen = vec![false; n];
    let mut stack: Vec<usize> = Vec::new();
    let mut members: Vec<usize> = Vec::new();
    for start in 0..n {
        if start % w == 0 && cancel() {
            return Err(LensError::Cancelled);
        }
        if !mask[start] || seen[start] {
            continue;
        }
        seen[start] = true;
        stack.push(start);
        members.clear();
        while let Some(p) = stack.pop() {
            members.push(p);
            let (x, y) = (p % w, p / w);
            let (x0, x1) = (x.saturating_sub(1), (x + 1).min(w - 1));
            let (y0, y1) = (y.saturating_sub(1), (y + 1).min(h - 1));
            for yy in y0..=y1 {
                for xx in x0..=x1 {
                    let q = yy * w + xx;
                    if mask[q] && !seen[q] {
                        seen[q] = true;
                        stack.push(q);
                    }
                }
            }
        }
        // The disc of the same area: d = 2 sqrt(area / pi), the area in half-diagonals squared.
        let area = members.len() as f64 / (half_diag * half_diag);
        let diameter = 2.0 * (area / std::f64::consts::PI).sqrt();
        let ev = boost_ev(diameter) as f32;
        for p in &members {
            out[*p] = ev;
        }
    }
    Ok(Some(out))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_boost_falls_with_the_logarithm_of_the_size_and_stays_in_range() {
        assert_eq!(boost_ev(0.0), LIGHT_BOOST_MAX_EV);
        assert_eq!(boost_ev(LIGHT_BOOST_FULL_HD), LIGHT_BOOST_MAX_EV);
        assert_eq!(boost_ev(LIGHT_BOOST_NONE_HD), 0.0);
        assert_eq!(boost_ev(1.0), 0.0);
        let mid = (LIGHT_BOOST_FULL_HD * LIGHT_BOOST_NONE_HD).sqrt();
        assert!((boost_ev(mid) - 0.5 * LIGHT_BOOST_MAX_EV).abs() < 1e-9);
        let mut last = f64::INFINITY;
        for i in 0..200 {
            let d = 0.001 * 1.04f64.powi(i);
            let b = boost_ev(d);
            assert!((0.0..=LIGHT_BOOST_MAX_EV).contains(&b));
            assert!(b <= last + 1e-12, "not monotone at {d}");
            last = b;
        }
    }
}
