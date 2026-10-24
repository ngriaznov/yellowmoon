//! L4: the anamorphic bend, the lens's distortion and the zoom (`docs/21` section 4, L4).
//!
//! The map goes from the photo to the canvas, in coordinates centred on the canvas centre and divided by the
//! canvas's half-diagonal, with `x` along the anamorphic axis. It is the lens's radial distortion first,
//! then the attachment's bend:
//!
//! ```text
//! r' = r (1 + e (k1 r^2 + k2 r^4))          (x1, y1) = (x, y) r'/r
//! x2 = x1 (1 + mu x1^2)                      y2 = y1 (1 - beta x1^2)
//! ```
//!
//! with `mu = mu0 (q - 1)`, `beta = beta0 (q - 1)`. The canvas is the map's output divided by the zoom `z`:
//! `photo_to_canvas(p) = F(p) / z`, and `canvas_to_photo(c) = F^-1(z c)`, found by three Newton steps from
//! `z c` with the 2 x 2 Jacobian, all in `f64`. The zoom is the smallest factor of at least 1 for which every
//! point of the canvas edge maps inside the photo with a margin of 1 px.
//!
//! Only `+ - * /` are used (no transcendental function), so the map has the same bits on every platform.

use crate::consts::{NEWTON_STEPS, ZOOM_EDGE_SAMPLES, ZOOM_MARGIN_PX, ZOOM_MAX};

/// The map `F` in normalised coordinates, without the zoom.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct BendMap {
    /// `mu`: the edge stretch.
    pub mu: f64,
    /// `beta`: the bowing of horizontal lines.
    pub beta: f64,
    /// `e k1`: the distortion's `r^2` coefficient, already scaled by Edges.
    pub d1: f64,
    /// `e k2`: the distortion's `r^4` coefficient, already scaled by Edges.
    pub d2: f64,
}

impl BendMap {
    fn is_identity(&self) -> bool {
        self.mu == 0.0 && self.beta == 0.0 && self.d1 == 0.0 && self.d2 == 0.0
    }

    /// `F(x, y)`.
    fn forward(&self, x: f64, y: f64) -> (f64, f64) {
        let r2 = x * x + y * y;
        let g = 1.0 + self.d1 * r2 + self.d2 * r2 * r2;
        let (x1, y1) = (x * g, y * g);
        let xx = x1 * x1;
        (x1 * (1.0 + self.mu * xx), y1 * (1.0 - self.beta * xx))
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
        // Second stage.
        let xx = x1 * x1;
        let fx = x1 * (1.0 + self.mu * xx);
        let fy = y1 * (1.0 - self.beta * xx);
        let dfx_dx1 = 1.0 + 3.0 * self.mu * xx;
        let dfy_dx1 = -2.0 * self.beta * x1 * y1;
        let dfy_dy1 = 1.0 - self.beta * xx;
        let j = [
            [dfx_dx1 * a, dfx_dx1 * b],
            [dfy_dx1 * a + dfy_dy1 * b, dfy_dx1 * b + dfy_dy1 * d],
        ];
        ((fx, fy), j)
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

    /// True when the map is the identity: the bend and the distortion are both 0 (and the zoom is 1).
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

    /// Canvas pixel position to photo pixel position: `F^-1(z c)` with three Newton steps, in `f64`.
    #[must_use]
    pub fn canvas_to_photo(&self, x: f64, y: f64) -> (f64, f64) {
        if self.is_identity() {
            return (x, y);
        }
        let (cx, cy, h) = (0.5 * self.width, 0.5 * self.height, self.half_diagonal());
        let (nx, ny) = ((x - cx) / h * self.zoom, (y - cy) / h * self.zoom);
        let (px, py) = self.bend.inverse(nx, ny, NEWTON_STEPS);
        (cx + px * h, cy + py * h)
    }

    /// Photo pixel position to canvas pixel position: the forward map `F(p) / z`.
    #[must_use]
    pub fn photo_to_canvas(&self, x: f64, y: f64) -> (f64, f64) {
        if self.is_identity() {
            return (x, y);
        }
        let (cx, cy, h) = (0.5 * self.width, 0.5 * self.height, self.half_diagonal());
        let (fx, fy) = self.bend.forward((x - cx) / h, (y - cy) / h);
        (cx + fx / self.zoom * h, cy + fy / self.zoom * h)
    }

    /// Everything the map depends on, as text (exact numbers).
    #[must_use]
    pub fn key(&self) -> String {
        if self.is_identity() {
            return "warp:none".to_owned();
        }
        format!(
            "warp:{:?}:{:?}:{:?}:{:?}:z{:?}:{:?}x{:?}",
            self.bend.mu, self.bend.beta, self.bend.d1, self.bend.d2, self.zoom, self.width, self.height
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

    fn bend() -> BendMap {
        BendMap {
            mu: 0.06,
            beta: 0.05,
            d1: -0.02,
            d2: 0.01,
        }
    }

    #[test]
    fn jacobian_matches_finite_differences() {
        let b = bend();
        for &(x, y) in &[(0.3, -0.2), (-0.7, 0.4), (0.05, 0.9)] {
            let (_, j) = b.forward_jacobian(x, y);
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

    #[test]
    fn identity_map_is_exact() {
        let w = LensWarp::new(
            BendMap {
                mu: 0.0,
                beta: 0.0,
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
        let w = LensWarp::new(bend(), 6000.0, 4000.0);
        for &(x, y) in &[(10.0, 10.0), (3000.0, 2000.0), (5990.0, 20.0), (1234.5, 3999.0)] {
            let (cx, cy) = w.photo_to_canvas(x, y);
            let (px, py) = w.canvas_to_photo(cx, cy);
            assert!((px - x).abs() < 1e-9 && (py - y).abs() < 1e-9, "{x},{y} -> {px},{py}");
        }
    }
}
