//! The bright balls drawn back after the stretch (`docs/21` section 4, L1, step 5).
//!
//! Each ball is an ellipse with the radius `r` it was measured at across and `r + |s|` along its long axis,
//! where `s = (q - 1) r` along the anamorphic axis's cross direction plus `sigma(rho) e r` along the tangent
//! (the sign of the second chosen to add to the first): the same extent in every direction as the two gathers
//! give a disc. With the lens's cat's-eye the shape is the ellipse cut by a copy of itself shifted radially,
//! the two kept centred on the ball, and the brightness follows the lens's rim profile. The shape carries
//! exactly the energy the ball had, per channel: the drawn value is the energy density of the shape times the
//! profile over its area-weighted mean, times the pixel coverage of an edge one pixel wide.
//!
//! Everything is a function of the ball and the pixel's absolute position, so tiles agree.

use crate::consts::RIM_SAMPLES;

use super::Ball;

/// The largest distance between the two copies of a cat's-eye ball, in the ellipse's own semi-axes.
const MAX_CAT_OFFSET: f64 = 1.8;

/// A ball's shape in the pixels of the view being drawn.
#[derive(Clone, Debug)]
pub(super) struct Shape {
    /// Centre in pixel coordinates (pixel `i` covers `[i, i + 1)`).
    pub cx: f64,
    pub cy: f64,
    /// Semi-axes: `a` along `m`, `b` across.
    pub a: f64,
    pub b: f64,
    /// Unit vector along the long axis.
    pub m: (f64, f64),
    /// Half the cat's-eye offset between the two copies (a vector in pixels); zero without one.
    pub half_off: (f64, f64),
    /// The cat's-eye offset in the ellipse's normalised frame, in units of the semi-axes.
    pub delta: f64,
    /// Energy per pixel per channel at profile value 1 (so that the drawn values sum to the ball's energy).
    pub density: [f64; 3],
    /// Radius, in pixels, of the disc taken out of the picture.
    pub r_out: f64,
    /// The ball's own radius in pixels.
    pub r_ball: f64,
    /// Bounding radius of everything drawn, in pixels.
    pub reach: f64,
    /// The ball's plane fit: value at the centre and slopes per pixel, per channel.
    pub fit: [[f64; 3]; 3],
}

/// What one tile pass needs to shape the balls.
#[derive(Clone, Copy, Debug)]
pub(super) struct ShapeParams<'a> {
    /// The frame at this scale and its half-diagonal in pixels.
    pub fw: f64,
    pub fh: f64,
    pub hk: f64,
    /// Bokeh height over width.
    pub q: f64,
    /// Edges, 0 to 1.
    pub e: f64,
    /// True when the anamorphic axis is the buffer's x axis (so the cross direction is y).
    pub axis_x: bool,
    /// Lens field radii, swirl and cat's-eye profiles (9 entries each).
    pub radii: &'a [f64],
    pub swirl: &'a [f64],
    pub cats_eye: &'a [f64],
    /// The lens's rim profile (16 samples).
    pub rim: &'a [f64],
    /// Mean of the rim profile over a disc (area-weighted).
    pub rim_mean: f64,
}

/// Linear interpolation of a profile given at `radii`, held at the last value beyond the last radius.
pub(super) fn profile_at(radii: &[f64], values: &[f64], rho: f64) -> f64 {
    if rho <= radii[0] {
        return values[0];
    }
    for i in 1..radii.len() {
        if rho <= radii[i] {
            let t = (rho - radii[i - 1]) / (radii[i] - radii[i - 1]);
            return values[i - 1] + t * (values[i] - values[i - 1]);
        }
    }
    values[values.len() - 1]
}

/// The rim profile at normalised radius `s` (0 the centre, 1 the edge), linear between its 16 samples.
#[inline]
pub(super) fn rim_at(rim: &[f64], s: f64) -> f64 {
    let x = s.clamp(0.0, 1.0) * (RIM_SAMPLES - 1) as f64;
    let i = (x as usize).min(RIM_SAMPLES - 2);
    let t = x - i as f64;
    rim[i] + t * (rim[i + 1] - rim[i])
}

/// The area-weighted mean of the rim profile over a disc: `integral prof(s) 2 s ds`, exact for the piecewise
/// linear profile (two-point Gauss on every segment).
pub(super) fn rim_disc_mean(rim: &[f64]) -> f64 {
    let n = RIM_SAMPLES - 1;
    let g = 0.5 / 3f64.sqrt();
    let mut sum = 0.0;
    for i in 0..n {
        let (s0, s1) = (i as f64 / n as f64, (i + 1) as f64 / n as f64);
        let mid = 0.5 * (s0 + s1);
        let half = 0.5 * (s1 - s0);
        for sign in [-1.0, 1.0] {
            let s = mid + sign * 2.0 * g * half;
            let p = rim[i] + (rim[i + 1] - rim[i]) * (s - s0) / (s1 - s0);
            sum += half * 2.0 * s * p;
        }
    }
    sum
}

/// Area of the intersection of two unit discs whose centres are `delta` apart (`delta` < 2).
fn lens_area(delta: f64) -> f64 {
    let d = delta.min(2.0);
    2.0 * libm::acos(0.5 * d) - 0.5 * d * (4.0 - d * d).max(0.0).sqrt()
}

/// The mean of the rim profile over the intersection of two unit discs `delta` apart, by a fixed grid.
fn lens_mean(rim: &[f64], delta: f64) -> f64 {
    const N: usize = 96;
    let h = 0.5 * delta;
    let (mut sum, mut count) = (0.0, 0.0);
    for j in 0..N {
        let y = -1.0 + (2 * j + 1) as f64 / N as f64;
        for i in 0..N {
            let x = -1.0 + (2 * i + 1) as f64 / N as f64;
            let r1 = ((x + h).powi(2) + y * y).sqrt();
            let r2 = ((x - h).powi(2) + y * y).sqrt();
            let r = r1.max(r2);
            if r <= 1.0 {
                sum += rim_at(rim, r);
                count += 1.0;
            }
        }
    }
    if count == 0.0 { 1.0 } else { sum / count }
}

/// The shape of `ball` for this view, or `None` when the lens leaves the ball as it is.
pub(super) fn shape_of(ball: &Ball, p: &ShapeParams) -> Option<Shape> {
    let r_px = ball.radius * p.hk;
    if r_px < 0.25 {
        return None;
    }
    let cx = ball.x * p.fw;
    let cy = ball.y * p.fh;
    // Field position of the ball: radial unit vector and field radius.
    let (ux, uy) = (cx - 0.5 * p.fw, cy - 0.5 * p.fh);
    let dist = (ux * ux + uy * uy).sqrt();
    let rho = dist / p.hk;
    let (rx, ry) = if dist > 1e-9 { (ux / dist, uy / dist) } else { (1.0, 0.0) };
    let (tx, ty) = (-ry, rx);
    let sigma = profile_at(p.radii, p.swirl, rho);
    let cat = profile_at(p.radii, p.cats_eye, rho);
    // The stretch across the anamorphic axis and the swirl along the tangent.
    let (yx, yy) = if p.axis_x { (0.0, 1.0) } else { (1.0, 0.0) };
    let v = (p.q - 1.0) * r_px;
    let t = sigma * p.e * r_px;
    let (vx, vy) = (v * yx, v * yy);
    let mut sgn = 1.0;
    if vx * tx + vy * ty < 0.0 {
        sgn = -1.0;
    }
    let (sx, sy) = (vx + sgn * t * tx, vy + sgn * t * ty);
    let s_len = (sx * sx + sy * sy).sqrt();
    let off = cat * p.e * r_px;
    let stretched = s_len > 0.02 * r_px;
    let cut = off > 0.02 * r_px;
    if !stretched && !cut {
        return None;
    }
    let (a, b) = (r_px + s_len, r_px);
    let m = if stretched { (sx / s_len, sy / s_len) } else { (1.0, 0.0) };
    // The cat's-eye copies sit half the offset either side of the centre along the radial direction.
    let (half_off, delta) = if cut {
        let h = (0.5 * off * rx, 0.5 * off * ry);
        // The offset vector in the normalised frame (long axis along m, short axis across).
        let n = (-m.1, m.0);
        let full = (2.0 * h.0, 2.0 * h.1);
        let ex = (full.0 * m.0 + full.1 * m.1) / a;
        let ey = (full.0 * n.0 + full.1 * n.1) / b;
        let len = (ex * ex + ey * ey).sqrt();
        // Two copies further apart than this leave almost nothing: hold the cut at 1.8 semi-axes.
        let keep = (MAX_CAT_OFFSET / len.max(1e-12)).min(1.0);
        ((h.0 * keep, h.1 * keep), len * keep)
    } else {
        ((0.0, 0.0), 0.0)
    };
    let (area, mean) = if cut {
        (a * b * lens_area(delta), lens_mean(p.rim, delta))
    } else {
        (std::f64::consts::PI * a * b, p.rim_mean)
    };
    let scale = p.fw * p.fh / (area * mean);
    Some(Shape {
        cx,
        cy,
        a,
        b,
        m,
        half_off,
        delta,
        density: [ball.energy[0] * scale, ball.energy[1] * scale, ball.energy[2] * scale],
        r_out: ball.take_out_radius * p.hk,
        r_ball: r_px,
        reach: a + (half_off.0 * half_off.0 + half_off.1 * half_off.1).sqrt() + 2.0,
        fit: [
            fit_px(&ball.fit[0], p.hk),
            fit_px(&ball.fit[1], p.hk),
            fit_px(&ball.fit[2], p.hk),
        ],
    })
}

/// A plane fit given per half-diagonal, as value at the centre and slope per pixel.
fn fit_px(f: &[f64; 3], hk: f64) -> [f64; 3] {
    [f[0], f[1] / hk, f[2] / hk]
}

impl Shape {
    /// Coverage (0 to 1, one pixel wide edge) and normalised radius at the pixel centre `(px, py)`.
    #[inline]
    pub(super) fn coverage(&self, px: f64, py: f64) -> (f64, f64) {
        let ell = |ox: f64, oy: f64| -> (f64, f64) {
            let (dx, dy) = (px - self.cx - ox, py - self.cy - oy);
            let u = dx * self.m.0 + dy * self.m.1;
            let v = -dx * self.m.1 + dy * self.m.0;
            let g = (u / self.a) * (u / self.a) + (v / self.b) * (v / self.b);
            if g < 0.0625 {
                // Deep inside: the signed distance is at least a quarter of the smaller semi-axis.
                return (-self.b * 0.5, g.sqrt());
            }
            let gx = 2.0 * u / (self.a * self.a);
            let gy = 2.0 * v / (self.b * self.b);
            let grad = (gx * gx + gy * gy).sqrt();
            ((g - 1.0) / grad, g.sqrt())
        };
        if self.delta == 0.0 {
            let (sd, rn) = ell(0.0, 0.0);
            return ((0.5 - sd).clamp(0.0, 1.0), rn);
        }
        let (sd1, r1) = ell(-self.half_off.0, -self.half_off.1);
        let (sd2, r2) = ell(self.half_off.0, self.half_off.1);
        let sd = sd1.max(sd2);
        ((0.5 - sd).clamp(0.0, 1.0), r1.max(r2))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_disc_mean_of_a_flat_rim_is_one_and_of_a_linear_one_is_two_thirds() {
        let flat = vec![1.0; RIM_SAMPLES];
        assert!((rim_disc_mean(&flat) - 1.0).abs() < 1e-12);
        let ramp: Vec<f64> = (0..RIM_SAMPLES).map(|i| i as f64 / (RIM_SAMPLES - 1) as f64).collect();
        // integral s * 2 s ds = 2/3
        assert!((rim_disc_mean(&ramp) - 2.0 / 3.0).abs() < 1e-12);
    }

    #[test]
    fn the_lens_area_matches_the_closed_form() {
        assert!((lens_area(0.0) - std::f64::consts::PI).abs() < 1e-12);
        assert!(lens_area(2.0).abs() < 1e-12);
        // Two unit discs one radius apart overlap in 2 pi / 3 - sqrt(3) / 2 times two... = 1.2284.
        assert!((lens_area(1.0) - (2.0 * std::f64::consts::PI / 3.0 - 3f64.sqrt() / 2.0)).abs() < 1e-12);
        let flat = vec![1.0; RIM_SAMPLES];
        assert!((lens_mean(&flat, 1.0) - 1.0).abs() < 1e-12);
    }
}
