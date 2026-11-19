//! Lens' kernel on the CPU (`docs/33-blur.md` 3.5, BL-07b): the twin of [`crate::kernels::blur_lens`], in `f64`. It is
//! `ym_color::blur::lens`, the fast method the CPU run itself uses (the exact table up to 8 px, above it the row segments over
//! integer prefix sums), on the planar planes of the scaffold.

use ym_color::blur::{Plane4, lens};
use ym_color::mask_raster::Area;

/// The sums `N` (three planes) and `D` (one) of the weighted planes `base` (`omega X` and `omega`, four planes over `read`) by the
/// aperture of radius `r_px` (`blades`, `rotation_deg`), over `made` (inside `read`). Four planes over `made`.
#[must_use]
pub fn sums(
    base: &[f64],
    read: Area,
    made: Area,
    r_px: f64,
    blades: u32,
    rotation_deg: f64,
) -> Vec<f64> {
    let rn = read.w * read.h;
    let aos: Vec<Plane4> = (0..rn)
        .map(|i| [base[i], base[rn + i], base[2 * rn + i], base[3 * rn + i]])
        .collect();
    // The rows the made area covers: the prefix is the read area's own, a pixel's sums are its own.
    let rows = (made.y - read.y) as usize..(made.y1() - read.y) as usize;
    let all = lens::gather_rows(
        &aos,
        (read.x as usize, read.y as usize, read.w, read.h),
        (0, 0),
        (r_px, blades, rotation_deg),
        rows,
    );
    let n = made.w * made.h;
    let mut out = vec![0.0; 4 * n];
    for y in 0..made.h {
        for x in 0..made.w {
            let c = (made.x - read.x) as usize + x;
            let q = all[y * read.w + c];
            for k in 0..4 {
                out[k * n + y * made.w + x] = q[k];
            }
        }
    }
    out
}
