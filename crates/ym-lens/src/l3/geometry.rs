//! The frame at one scale in the lens's own coordinates (`docs/21` section 3, "Coordinates").
//!
//! Lengths are in units of the frame's half-diagonal. A pixel `i` covers `[i, i + 1)` and the optical centre is
//! the frame's centre. The **canvas** coordinates `(along, across)` put the anamorphic axis first: the buffer's
//! x axis for [`Axis::X`], its y axis for [`Axis::Y`] (the same convention as the field optics).

use crate::plan::Axis;

/// The frame at one scale.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Frame {
    /// Width in pixels.
    pub(crate) w: usize,
    /// Height in pixels.
    pub(crate) h: usize,
    /// The half-diagonal in pixels.
    pub(crate) half_diag: f64,
    /// The anamorphic axis in the buffer.
    pub(crate) axis: Axis,
    half_w: f64,
    half_h: f64,
}

/// A rectangle of pixels `[x0, x1) x [y0, y1)`, possibly empty.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Span {
    pub(crate) x0: usize,
    pub(crate) x1: usize,
    pub(crate) y0: usize,
    pub(crate) y1: usize,
}

impl Span {
    /// True when no pixel lies inside.
    pub(crate) fn is_empty(&self) -> bool {
        self.x0 >= self.x1 || self.y0 >= self.y1
    }

    /// The part both rectangles hold.
    pub(crate) fn intersect(&self, o: &Span) -> Span {
        Span {
            x0: self.x0.max(o.x0),
            x1: self.x1.min(o.x1),
            y0: self.y0.max(o.y0),
            y1: self.y1.min(o.y1),
        }
    }
}

impl Frame {
    /// The frame of `w x h` pixels with the axis `axis`.
    pub(crate) fn new(w: usize, h: usize, axis: Axis) -> Self {
        let (wf, hf) = (w as f64, h as f64);
        Frame {
            w,
            h,
            half_diag: 0.5 * (wf * wf + hf * hf).sqrt(),
            axis,
            half_w: 0.5 * wf,
            half_h: 0.5 * hf,
        }
    }

    /// The horizontal field coordinate of the continuous pixel position `x`.
    #[inline]
    pub(crate) fn u(&self, x: f64) -> f64 {
        (x - self.half_w) / self.half_diag
    }

    /// The vertical field coordinate of the continuous pixel position `y`.
    #[inline]
    pub(crate) fn v(&self, y: f64) -> f64 {
        (y - self.half_h) / self.half_diag
    }

    /// The canvas coordinates `(along, across)` of the field coordinates `(u, v)`.
    #[inline]
    pub(crate) fn canvas(&self, u: f64, v: f64) -> (f64, f64) {
        match self.axis {
            Axis::X => (u, v),
            Axis::Y => (v, u),
        }
    }

    /// The canvas coordinates of the continuous pixel position `(x, y)`.
    #[inline]
    pub(crate) fn canvas_at(&self, x: f64, y: f64) -> (f64, f64) {
        self.canvas(self.u(x), self.v(y))
    }

    /// The continuous pixel x of the field coordinate `u`.
    #[inline]
    fn x_of(&self, u: f64) -> f64 {
        u * self.half_diag + self.half_w
    }

    /// The continuous pixel y of the field coordinate `v`.
    #[inline]
    fn y_of(&self, v: f64) -> f64 {
        v * self.half_diag + self.half_h
    }

    /// The field coordinates of the two edges of pixel `p` along the axis.
    #[inline]
    pub(crate) fn along_edges(&self, p: usize) -> (f64, f64) {
        let p = p as f64;
        match self.axis {
            Axis::X => (self.u(p), self.u(p + 1.0)),
            Axis::Y => (self.v(p), self.v(p + 1.0)),
        }
    }

    /// The field coordinates of the two edges of pixel `q` across the axis.
    #[inline]
    pub(crate) fn across_edges(&self, q: usize) -> (f64, f64) {
        let q = q as f64;
        match self.axis {
            Axis::X => (self.v(q), self.v(q + 1.0)),
            Axis::Y => (self.u(q), self.u(q + 1.0)),
        }
    }

    /// The pixel `(i, j)` whose index along the axis is `p` and across it `q`.
    #[inline]
    pub(crate) fn pixel_of(&self, p: usize, q: usize) -> (usize, usize) {
        match self.axis {
            Axis::X => (p, q),
            Axis::Y => (q, p),
        }
    }

    /// The pixel columns/rows of `span` along the axis: `[p0, p1)`.
    pub(crate) fn along_range(&self, s: &Span) -> (usize, usize) {
        match self.axis {
            Axis::X => (s.x0, s.x1),
            Axis::Y => (s.y0, s.y1),
        }
    }

    /// The pixel rows/columns of `span` across the axis: `[q0, q1)`.
    pub(crate) fn across_range(&self, s: &Span) -> (usize, usize) {
        match self.axis {
            Axis::X => (s.y0, s.y1),
            Axis::Y => (s.x0, s.x1),
        }
    }

    /// The pixels whose box meets the canvas box `[a_lo, a_hi] x [c_lo, c_hi]`, clamped to the frame.
    pub(crate) fn span_of(&self, a_lo: f64, a_hi: f64, c_lo: f64, c_hi: f64) -> Span {
        let (u_lo, u_hi, v_lo, v_hi) = match self.axis {
            Axis::X => (a_lo, a_hi, c_lo, c_hi),
            Axis::Y => (c_lo, c_hi, a_lo, a_hi),
        };
        let clamp = |lo: f64, hi: f64, n: usize| -> (usize, usize) {
            let n_f = n as f64;
            let lo = libm::floor(lo).clamp(0.0, n_f);
            let hi = libm::ceil(hi).clamp(0.0, n_f);
            (lo as usize, hi as usize)
        };
        let (x0, x1) = clamp(self.x_of(u_lo), self.x_of(u_hi), self.w);
        let (y0, y1) = clamp(self.y_of(v_lo), self.y_of(v_hi), self.h);
        Span { x0, x1, y0, y1 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canvas_follows_the_axis() {
        let f = Frame::new(300, 200, Axis::X);
        let g = Frame::new(300, 200, Axis::Y);
        let (u, v) = (f.u(180.0), f.v(120.0));
        assert_eq!(f.canvas(u, v), (u, v));
        assert_eq!(g.canvas(u, v), (v, u));
        // The centre of the frame is the origin.
        assert_eq!(f.u(150.0), 0.0);
        assert_eq!(f.v(100.0), 0.0);
    }

    #[test]
    fn spans_cover_the_boxes_they_meet() {
        let f = Frame::new(100, 80, Axis::X);
        let h = f.half_diag;
        // A box from x = 10.2 to 20.7 and y = 30.2 to 30.8 meets columns 10..=20 and row 30.
        let s = f.span_of(
            (10.2 - 50.0) / h,
            (20.7 - 50.0) / h,
            (30.2 - 40.0) / h,
            (30.8 - 40.0) / h,
        );
        assert_eq!(
            s,
            Span {
                x0: 10,
                x1: 21,
                y0: 30,
                y1: 31
            }
        );
        // Clamped to the frame.
        let s = f.span_of(-10.0, 10.0, -10.0, 10.0);
        assert_eq!(
            s,
            Span {
                x0: 0,
                x1: 100,
                y0: 0,
                y1: 80
            }
        );
        // The y axis swaps the roles.
        let g = Frame::new(100, 80, Axis::Y);
        let s = g.span_of(
            (30.2 - 40.0) / h,
            (30.8 - 40.0) / h,
            (10.2 - 50.0) / h,
            (20.7 - 50.0) / h,
        );
        assert_eq!(
            s,
            Span {
                x0: 10,
                x1: 21,
                y0: 30,
                y1: 31
            }
        );
    }
}
