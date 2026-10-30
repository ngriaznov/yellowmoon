//! L4: the anamorphic bend, the lens's distortion and the zoom (`docs/21` section 4, L4).
//!
//! The map goes from the photo to the canvas, in coordinates centred on the canvas centre and divided by the
//! canvas's half-diagonal, with `x` along the anamorphic axis. It is the lens's radial distortion first,
//! then the attachment's bend:
//!
//! ```text
//! r' = r (1 + e (k1 r^2 + k2 r^4))          (x1, y1) = (x, y) r'/r
//! x2 = x1 (1 + G_x(x1, y1))                  y2 = y1 (1 + G_y(x1, y1))
//! G_x = a1 x^2 + a2 y^2 + a3 x^4 + a4 x^2 y^2 + a5 y^4        G_y = b1 x^2 + b2 y^2 + b3 x^4 + b4 x^2 y^2 + b5 y^4
//! ```
//!
//! with each of the ten constants the data's times `s = q - 1`. The map is odd in `x` and in `y` separately: no
//! decentre, shear or twist term, so the canvas centre stays the centre and the axes stay straight lines. The
//! canvas is the map's output times the zoom `z`: `photo_to_canvas(p) = z F(p)`, and `canvas_to_photo(c) =
//! F^-1(c / z)`, found by three Newton steps from `c / z` with the 2 x 2 Jacobian, all in `f64`. The zoom is the
//! smallest factor of at least 1 for which every point of the canvas edge, divided by `z`, maps inside the photo
//! with a margin of 1 px (so a larger `z` zooms in).
//!
//! Only `+ - * /` are used (no transcendental function), so the map has the same bits on every platform.

use crate::consts::{NEWTON_STEPS, ZOOM_EDGE_SAMPLES, ZOOM_MARGIN_PX, ZOOM_MAX};

/// `G` of one axis for the ten constants' half: `c[0] x^2 + c[1] y^2 + c[2] x^4 + c[3] x^2 y^2 + c[4] y^4`,
/// from `xx = x^2` and `yy = y^2`.
fn bend_poly(c: &[f64; 5], xx: f64, yy: f64) -> f64 {
    xx * (c[0] + c[2] * xx + c[3] * yy) + yy * (c[1] + c[4] * yy)
}

/// `(dG/dx) / x`, as `G` has only even powers: `2 (c0 + 2 c2 x^2 + c3 y^2)`, so `dG/dx = x` times it.
fn bend_poly_dx_over_x(c: &[f64; 5], xx: f64, yy: f64) -> f64 {
    2.0 * (c[0] + 2.0 * c[2] * xx + c[3] * yy)
}

/// `(dG/dy) / y`: `2 (c1 + c3 x^2 + 2 c4 y^2)`.
fn bend_poly_dy_over_y(c: &[f64; 5], xx: f64, yy: f64) -> f64 {
    2.0 * (c[1] + c[3] * xx + 2.0 * c[4] * yy)
}

/// The map `F` in normalised coordinates, without the zoom.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct BendMap {
    /// `a1 .. a5` times `s`: the terms of `x'` for `x^2, y^2, x^4, x^2 y^2, y^4`.
    pub bx: [f64; 5],
    /// `b1 .. b5` times `s`: the terms of `y'` for `x^2, y^2, x^4, x^2 y^2, y^4`.
    pub by: [f64; 5],
    /// `e k1`: the distortion's `r^2` coefficient, already scaled by Edges.
    pub d1: f64,
    /// `e k2`: the distortion's `r^4` coefficient, already scaled by Edges.
    pub d2: f64,
}

impl BendMap {
    fn is_identity(&self) -> bool {
        self.bx.iter().all(|&c| c == 0.0)
            && self.by.iter().all(|&c| c == 0.0)
            && self.d1 == 0.0
            && self.d2 == 0.0
    }

    /// `F(x, y)`.
    fn forward(&self, x: f64, y: f64) -> (f64, f64) {
        let r2 = x * x + y * y;
        let g = 1.0 + self.d1 * r2 + self.d2 * r2 * r2;
        let (x1, y1) = (x * g, y * g);
        let (xx, yy) = (x1 * x1, y1 * y1);
        let gx = bend_poly(&self.bx, xx, yy);
        let gy = bend_poly(&self.by, xx, yy);
        (x1 * (1.0 + gx), y1 * (1.0 + gy))
    }

    /// `F` and its Jacobian `[[dX/dx, dX/dy], [dY/dx, dY/dy]]`.
    fn forward_jacobian(&self, x: f64, y: f64) -> ((f64, f64), [[f64; 2]; 2]) {
        let r2 = x * x + y * y;
        let g = 1.0 + self.d1 * r2 + self.d2 * r2 * r2;
        // dg/d(r^2).
        let gp = self.d1 + 2.0 * self.d2 * r2;
        let (x1, y1) = (x * g, y * g);
        // Jacobian of (x1, y1) with respect to (x, y).
        let a = g + 2.0 * x * x * gp;
        let b = 2.0 * x * y * gp;
        let d = g + 2.0 * y * y * gp;
        // Second stage: X = x1 (1 + Gx), Y = y1 (1 + Gy), at (x1, y1).
        let (xx, yy) = (x1 * x1, y1 * y1);
        let gx = bend_poly(&self.bx, xx, yy);
        let gy = bend_poly(&self.by, xx, yy);
        let fx = x1 * (1.0 + gx);
        let fy = y1 * (1.0 + gy);
        let j00 = 1.0 + gx + xx * bend_poly_dx_over_x(&self.bx, xx, yy);
        let j01 = x1 * y1 * bend_poly_dy_over_y(&self.bx, xx, yy);
        let j10 = y1 * x1 * bend_poly_dx_over_x(&self.by, xx, yy);
        let j11 = 1.0 + gy + yy * bend_poly_dy_over_y(&self.by, xx, yy);
        let j = [
            [j00 * a + j01 * b, j00 * b + j01 * d],
            [j10 * a + j11 * b, j10 * b + j11 * d],
        ];
        ((fx, fy), j)
    }

    /// The determinant of `F`'s Jacobian at `(x, y)`.
    pub(crate) fn determinant(&self, x: f64, y: f64) -> f64 {
        let (_, j) = self.forward_jacobian(x, y);
        j[0][0] * j[1][1] - j[0][1] * j[1][0]
    }

    /// `F^-1(tx, ty)` by `steps` Newton steps from `(tx, ty)`.
    fn inverse(&self, tx: f64, ty: f64, steps: usize) -> (f64, f64) {
        let (mut x, mut y) = (tx, ty);
        for _ in 0..steps {
            let ((fx, fy), j) = self.forward_jacobian(x, y);
            let det = j[0][0] * j[1][1] - j[0][1] * j[1][0];
            if det == 0.0 || !det.is_finite() {
                break;
            }
            let (rx, ry) = (fx - tx, fy - ty);
            x -= (j[1][1] * rx - j[0][1] * ry) / det;
            y -= (-j[1][0] * rx + j[0][0] * ry) / det;
        }
        (x, y)
    }
}

/// L4 for one canvas: the map between canvas pixels and photo pixels, with the zoom.
///
/// Positions are continuous pixel positions of the canvas frame (pixel `i` covers `[i, i + 1)`), and the photo
/// has the same size as the canvas ("the same canvas frame, before straightening"). The anamorphic axis is the
/// canvas's `x`. Build it with [`LensPlan::warp`](crate::LensPlan::warp) at the full-resolution canvas and use
/// [`LensWarp::rescaled`] for reduced views, so every scale shares one zoom.
#[derive(Clone, Debug, PartialEq)]
pub struct LensWarp {
    bend: BendMap,
    zoom: f64,
    width: f64,
    height: f64,
}

impl LensWarp {
    /// Builds the warp of a canvas and fits its zoom.
    pub(crate) fn new(bend: BendMap, width: f64, height: f64) -> Self {
        let mut warp = LensWarp {
            bend,
            zoom: 1.0,
            width,
            height,
        };
        if !bend.is_identity() {
            warp.zoom = warp.fit_zoom();
        }
        warp
    }

    /// True when the map is the identity: the ten bend constants and the distortion are all 0 (and the zoom is 1).
    #[must_use]
    pub fn is_identity(&self) -> bool {
        self.bend.is_identity()
    }

    /// The zoom `z`: 1 or more.
    #[must_use]
    pub fn zoom(&self) -> f64 {
        self.zoom
    }

    /// The canvas size this warp was built for, in pixels.
    #[must_use]
    pub fn canvas_size(&self) -> (f64, f64) {
        (self.width, self.height)
    }

    /// `a1 .. a5` times `s = q - 1`: the attachment's terms of `x2 = x1 (1 + a1 x1^2 + a2 y1^2 + a3 x1^4 + a4
    /// x1^2 y1^2 + a5 y1^4)`.
    #[must_use]
    pub fn bend_x(&self) -> [f64; 5] {
        self.bend.bx
    }

    /// `b1 .. b5` times `s = q - 1`: the attachment's terms of `y2 = y1 (1 + b1 x1^2 + b2 y1^2 + b3 x1^4 + b4
    /// x1^2 y1^2 + b5 y1^4)`.
    #[must_use]
    pub fn bend_y(&self) -> [f64; 5] {
        self.bend.by
    }

    /// `e k1`: the lens distortion's `r^2` coefficient, already scaled by Edges.
    #[must_use]
    pub fn distortion_k1(&self) -> f64 {
        self.bend.d1
    }

    /// `e k2`: the lens distortion's `r^4` coefficient, already scaled by Edges.
    #[must_use]
    pub fn distortion_k2(&self) -> f64 {
        self.bend.d2
    }

    /// The same map for a canvas of another size (a reduced view), with the same zoom.
    #[must_use]
    pub fn rescaled(&self, canvas_width: f64, canvas_height: f64) -> LensWarp {
        LensWarp {
            bend: self.bend,
            zoom: self.zoom,
            width: canvas_width,
            height: canvas_height,
        }
    }

    fn half_diagonal(&self) -> f64 {
        0.5 * (self.width * self.width + self.height * self.height).sqrt()
    }

    /// Canvas pixel position to photo pixel position: `F^-1(c / z)` with three Newton steps, in `f64`.
    #[must_use]
    pub fn canvas_to_photo(&self, x: f64, y: f64) -> (f64, f64) {
        if self.is_identity() {
            return (x, y);
        }
        let (cx, cy, h) = (0.5 * self.width, 0.5 * self.height, self.half_diagonal());
        let (nx, ny) = ((x - cx) / h / self.zoom, (y - cy) / h / self.zoom);
        let (px, py) = self.bend.inverse(nx, ny, NEWTON_STEPS);
        (cx + px * h, cy + py * h)
    }

    /// Photo pixel position to canvas pixel position: the forward map `z F(p)`.
    #[must_use]
    pub fn photo_to_canvas(&self, x: f64, y: f64) -> (f64, f64) {
        if self.is_identity() {
            return (x, y);
        }
        let (cx, cy, h) = (0.5 * self.width, 0.5 * self.height, self.half_diagonal());
        let (fx, fy) = self.bend.forward((x - cx) / h, (y - cy) / h);
        (cx + fx * self.zoom * h, cy + fy * self.zoom * h)
    }

    /// Everything the map depends on, as text (exact numbers).
    #[must_use]
    pub fn key(&self) -> String {
        if self.is_identity() {
            return "warp:none".to_owned();
        }
        format!(
            "warp:{:?}:{:?}:{:?}:{:?}:z{:?}:{:?}x{:?}",
            self.bend.bx,
            self.bend.by,
            self.bend.d1,
            self.bend.d2,
            self.zoom,
            self.width,
            self.height
        )
    }

    /// True when every point of the canvas edge maps inside the photo with the margin, at zoom `z`.
    fn edge_inside(&self, z: f64) -> bool {
        let trial = LensWarp {
            zoom: z,
            ..self.clone()
        };
        let (w, h) = (self.width, self.height);
        let n = ZOOM_EDGE_SAMPLES;
        for i in 0..=n {
            let t = i as f64 / n as f64;
            for (x, y) in [(t * w, 0.0), (t * w, h), (0.0, t * h), (w, t * h)] {
                let (px, py) = trial.canvas_to_photo(x, y);
                let inside = px >= ZOOM_MARGIN_PX
                    && px <= w - ZOOM_MARGIN_PX
                    && py >= ZOOM_MARGIN_PX
                    && py <= h - ZOOM_MARGIN_PX;
                if !inside {
                    return false;
                }
            }
        }
        true
    }

    /// The smallest zoom of at least 1 for which the canvas edge maps inside the photo (bisection to 1e-13).
    fn fit_zoom(&self) -> f64 {
        if self.edge_inside(1.0) {
            return 1.0;
        }
        let mut lo: f64 = 1.0;
        let mut hi: f64 = 1.0;
        let mut step: f64 = 0.0625;
        loop {
            hi = (hi + step).min(ZOOM_MAX);
            if self.edge_inside(hi) {
                break;
            }
            lo = hi;
            if hi >= ZOOM_MAX {
                // A map that needs more than the largest zoom is held there; the data checks keep the
                // attachment's constants far from that.
                return ZOOM_MAX;
            }
            step *= 2.0;
        }
        for _ in 0..200 {
            if hi - lo <= 1e-13 {
                break;
            }
            let mid = 0.5 * (lo + hi);
            if self.edge_inside(mid) {
                hi = mid;
            } else {
                lo = mid;
            }
        }
        hi
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The two-term bend of before (`a1 = 0.06`, `b1 = -0.05`) with the lens's distortion.
    fn bend() -> BendMap {
        BendMap {
            bx: [0.06, 0.0, 0.0, 0.0, 0.0],
            by: [-0.05, 0.0, 0.0, 0.0, 0.0],
            d1: -0.02,
            d2: 0.01,
        }
    }

    /// All ten constants set, with the lens's distortion: the mustache variant of the shipped data at
    /// Anamorphic 100, a little stronger.
    fn full_bend() -> BendMap {
        BendMap {
            bx: [-0.0402, -0.0494, 0.0069, 0.0128, 0.0329],
            by: [-0.1157, -0.0196, 0.0294, 0.0297, -0.0529],
            d1: -0.02,
            d2: 0.01,
        }
    }

    #[test]
    fn jacobian_matches_finite_differences() {
        for b in [bend(), full_bend()] {
            for &(x, y) in &[(0.3, -0.2), (-0.7, 0.4), (0.05, 0.9), (0.83, -0.55)] {
                let (f, j) = b.forward_jacobian(x, y);
                assert_eq!(f, b.forward(x, y));
                let h = 1e-6;
                let (fx1, fy1) = b.forward(x + h, y);
                let (fx0, fy0) = b.forward(x - h, y);
                assert!((j[0][0] - (fx1 - fx0) / (2.0 * h)).abs() < 1e-7);
                assert!((j[1][0] - (fy1 - fy0) / (2.0 * h)).abs() < 1e-7);
                let (fx1, fy1) = b.forward(x, y + h);
                let (fx0, fy0) = b.forward(x, y - h);
                assert!((j[0][1] - (fx1 - fx0) / (2.0 * h)).abs() < 1e-7);
                assert!((j[1][1] - (fy1 - fy0) / (2.0 * h)).abs() < 1e-7);
            }
        }
    }

    /// The residual `|F(x_n) - target|` of the inverse after `steps` Newton steps, in pixels of a 6000 x 4000
    /// canvas, worst over a lattice of the canvas (`/ zoom`).
    fn residual_px(map: &BendMap, zoom: f64, steps: usize) -> f64 {
        let half_diagonal_px = 0.5 * 6000.0f64.hypot(4000.0);
        let (hx, hy) = (3000.0 / half_diagonal_px, 2000.0 / half_diagonal_px);
        let mut worst = 0.0f64;
        for j in 0..=40 {
            for i in 0..=80 {
                let tx = hx * (f64::from(i) / 40.0 - 1.0) / zoom;
                let ty = hy * (f64::from(j) / 20.0 - 1.0) / zoom;
                let (x, y) = map.inverse(tx, ty, steps);
                let (fx, fy) = map.forward(x, y);
                worst = worst.max((fx - tx).hypot(fy - ty) * half_diagonal_px);
            }
        }
        worst
    }

    #[test]
    fn three_newton_steps_are_enough_for_the_shipped_bend() {
        // The mustache constants at Anamorphic 100, with the zoom that bend needs (1.0728).
        let map = BendMap {
            bx: [-0.0402, -0.0494, 0.0069, 0.0128, 0.0329],
            by: [-0.1157, -0.0196, 0.0294, 0.0297, -0.0529],
            d1: 0.0,
            d2: 0.0,
        };
        let (two, three, four) = (
            residual_px(&map, 1.0727753943912148, 2),
            residual_px(&map, 1.0727753943912148, 3),
            residual_px(&map, 1.0727753943912148, 4),
        );
        eprintln!(
            "Newton residual, px of 6000 x 4000: 2 steps {two:e}, 3 steps {three:e}, 4 steps {four:e}"
        );
        // Two steps are already invisible, three leave rounding.
        assert!(two < 1e-3, "{two:e}");
        assert!(three < 1e-11, "{three:e}");
        assert!(four <= three * 2.0 + 1e-12, "{four:e}");
    }

    #[test]
    fn the_shipped_bend_never_folds_over_the_photo_and_a_margin() {
        // The determinant of the Jacobian stays at least 0.5 over the photo plus 25 % of its half-sides, as
        // it is and turned a quarter turn, at every position of the slider (the loader checks the same).
        let (hx, hy) = (3.0 / 13.0f64.sqrt() * 1.25, 2.0 / 13.0f64.sqrt() * 1.25);
        for s in [0.0, 0.25, 0.5, 0.75, 1.0] {
            let map = BendMap {
                bx: [-0.0402, -0.0494, 0.0069, 0.0128, 0.0329].map(|c| c * s),
                by: [-0.1157, -0.0196, 0.0294, 0.0297, -0.0529].map(|c| c * s),
                d1: 0.0,
                d2: 0.0,
            };
            let mut worst = f64::INFINITY;
            for (wx, wy) in [(hx, hy), (hy, hx)] {
                for j in 0..=40 {
                    for i in 0..=40 {
                        let (x, y) = (
                            wx * (f64::from(i) / 20.0 - 1.0),
                            wy * (f64::from(j) / 20.0 - 1.0),
                        );
                        worst = worst.min(map.determinant(x, y));
                    }
                }
            }
            assert!(worst >= 0.5, "s {s}: {worst}");
            assert!(s == 0.0 || worst < 1.0);
        }
    }

    #[test]
    fn the_map_is_odd_in_each_axis() {
        let b = full_bend();
        let (fx, fy) = b.forward(0.4, 0.3);
        assert_eq!(b.forward(-0.4, 0.3), (-fx, fy));
        assert_eq!(b.forward(0.4, -0.3), (fx, -fy));
        assert_eq!(b.forward(0.0, 0.0), (0.0, 0.0));
    }

    #[test]
    fn identity_map_is_exact() {
        let w = LensWarp::new(
            BendMap {
                bx: [0.0; 5],
                by: [-0.0, 0.0, 0.0, 0.0, 0.0],
                d1: 0.0,
                d2: 0.0,
            },
            600.0,
            400.0,
        );
        assert!(w.is_identity());
        assert_eq!(w.zoom(), 1.0);
        assert_eq!(w.canvas_to_photo(12.25, 99.5), (12.25, 99.5));
        assert_eq!(w.photo_to_canvas(12.25, 99.5), (12.25, 99.5));
    }

    #[test]
    fn zoom_is_at_least_one_and_minimal() {
        let w = LensWarp::new(bend(), 600.0, 400.0);
        assert!(w.zoom() > 1.0);
        assert!(w.edge_inside(w.zoom()));
        assert!(!w.edge_inside(w.zoom() * 0.999));
    }

    #[test]
    fn round_trip_is_tight() {
        for bend in [bend(), full_bend()] {
            round_trip_of(bend);
        }
    }

    fn round_trip_of(bend: BendMap) {
        let w = LensWarp::new(bend, 6000.0, 4000.0);
        for &(x, y) in &[
            (10.0, 10.0),
            (3000.0, 2000.0),
            (5990.0, 20.0),
            (1234.5, 3999.0),
        ] {
            let (cx, cy) = w.photo_to_canvas(x, y);
            let (px, py) = w.canvas_to_photo(cx, cy);
            assert!(
                (px - x).abs() < 1e-9 && (py - y).abs() < 1e-9,
                "{x},{y} -> {px},{py}"
            );
        }
    }
}
