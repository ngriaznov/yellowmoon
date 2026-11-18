//! Motion's kernel on the CPU (`docs/33-blur.md` 3.5, BL-08): the twin of [`crate::kernels::blur_motion`], in `f64`. It is
//! `ym_color::blur::motion`, the fast method the CPU run itself uses, on the planar planes of the scaffold.

use ym_color::blur::{Plane4, motion};
use ym_color::mask_raster::Area;

/// The sums `N` (three planes) and `D` (one) of the weighted planes `base` (`omega X` and `omega`, four planes over `read`) along
/// the line of `length_px` pixels at `angle_deg`, over `made` (inside `read`). Four planes over `made`.
#[must_use]
pub fn sums(base: &[f64], read: Area, made: Area, length_px: f64, angle_deg: f64) -> Vec<f64> {
    let rn = read.w * read.h;
    let aos: Vec<Plane4> = (0..rn)
        .map(|i| [base[i], base[rn + i], base[2 * rn + i], base[3 * rn + i]])
        .collect();
    let all = motion::gather(&aos, read.w, read.h, length_px, angle_deg);
    let n = made.w * made.h;
    let mut out = vec![0.0; 4 * n];
    for y in 0..made.h {
        for x in 0..made.w {
            let r = (made.y - read.y) as usize + y;
            let c = (made.x - read.x) as usize + x;
            let q = all[r * read.w + c];
            for k in 0..4 {
                out[k * n + y * made.w + x] = q[k];
            }
        }
    }
    out
}
