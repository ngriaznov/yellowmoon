//! The number types the cine rendering's core is written over: `f64` for the forward map and a forward-mode
//! dual number with three derivatives for the inverse's exact Jacobian.

use std::ops::{Add, Div, Mul, Neg, Sub};

/// What the colour terms of the model need from a number: arithmetic and the handful of functions they use.
///
/// `clamp_c` and `min_c` take plain `f64` bounds; on a dual number they return a constant outside the
/// bounds (derivative 0), which is the right derivative of the C1 junctions they appear in.
pub(super) trait Real:
    Copy
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
{
    /// A constant.
    fn c(v: f64) -> Self;
    fn exp(self) -> Self;
    fn ln(self) -> Self;
    fn sqrt(self) -> Self;
    fn sin(self) -> Self;
    fn cos(self) -> Self;
    fn clamp_c(self, lo: f64, hi: f64) -> Self;
    fn min_c(self, hi: f64) -> Self;
    /// Times a constant.
    fn sc(self, k: f64) -> Self {
        self * Self::c(k)
    }
}

impl Real for f64 {
    fn c(v: f64) -> f64 {
        v
    }
    fn exp(self) -> f64 {
        f64::exp(self)
    }
    fn ln(self) -> f64 {
        f64::ln(self)
    }
    fn sqrt(self) -> f64 {
        f64::sqrt(self)
    }
    fn sin(self) -> f64 {
        f64::sin(self)
    }
    fn cos(self) -> f64 {
        f64::cos(self)
    }
    fn clamp_c(self, lo: f64, hi: f64) -> f64 {
        self.max(lo).min(hi)
    }
    fn min_c(self, hi: f64) -> f64 {
        self.min(hi)
    }
}

/// A value and its derivatives with respect to the three inputs of the core: `(ln Ys, ca, cb)`.
#[derive(Clone, Copy)]
pub(super) struct Dual3 {
    pub(super) v: f64,
    pub(super) d: [f64; 3],
}

impl Add for Dual3 {
    type Output = Dual3;
    fn add(self, o: Dual3) -> Dual3 {
        Dual3 {
            v: self.v + o.v,
            d: [self.d[0] + o.d[0], self.d[1] + o.d[1], self.d[2] + o.d[2]],
        }
    }
}

impl Sub for Dual3 {
    type Output = Dual3;
    fn sub(self, o: Dual3) -> Dual3 {
        Dual3 {
            v: self.v - o.v,
            d: [self.d[0] - o.d[0], self.d[1] - o.d[1], self.d[2] - o.d[2]],
        }
    }
}

impl Mul for Dual3 {
    type Output = Dual3;
    fn mul(self, o: Dual3) -> Dual3 {
        Dual3 {
            v: self.v * o.v,
            d: [
                self.d[0] * o.v + self.v * o.d[0],
                self.d[1] * o.v + self.v * o.d[1],
                self.d[2] * o.v + self.v * o.d[2],
            ],
        }
    }
}

impl Div for Dual3 {
    type Output = Dual3;
    fn div(self, o: Dual3) -> Dual3 {
        let inv = 1.0 / o.v;
        let q = self.v * inv;
        Dual3 {
            v: q,
            d: [
                (self.d[0] - q * o.d[0]) * inv,
                (self.d[1] - q * o.d[1]) * inv,
                (self.d[2] - q * o.d[2]) * inv,
            ],
        }
    }
}

impl Neg for Dual3 {
    type Output = Dual3;
    fn neg(self) -> Dual3 {
        Dual3 {
            v: -self.v,
            d: [-self.d[0], -self.d[1], -self.d[2]],
        }
    }
}

impl Dual3 {
    /// The chain rule for a function with value `f` and derivative `df` at `self`.
    fn chain(self, f: f64, df: f64) -> Dual3 {
        Dual3 {
            v: f,
            d: [self.d[0] * df, self.d[1] * df, self.d[2] * df],
        }
    }

    /// The `i`-th input, with derivative 1 in its own slot.
    pub(super) fn input(v: f64, i: usize) -> Dual3 {
        let mut d = [0.0; 3];
        d[i] = 1.0;
        Dual3 { v, d }
    }
}

impl Real for Dual3 {
    fn c(v: f64) -> Dual3 {
        Dual3 { v, d: [0.0; 3] }
    }
    fn exp(self) -> Dual3 {
        let e = self.v.exp();
        self.chain(e, e)
    }
    fn ln(self) -> Dual3 {
        self.chain(self.v.ln(), 1.0 / self.v)
    }
    fn sqrt(self) -> Dual3 {
        let s = self.v.sqrt();
        self.chain(s, 0.5 / s)
    }
    fn sin(self) -> Dual3 {
        self.chain(self.v.sin(), self.v.cos())
    }
    fn cos(self) -> Dual3 {
        self.chain(self.v.cos(), -self.v.sin())
    }
    fn clamp_c(self, lo: f64, hi: f64) -> Dual3 {
        if self.v < lo {
            Dual3::c(lo)
        } else if self.v > hi {
            Dual3::c(hi)
        } else {
            self
        }
    }
    fn min_c(self, hi: f64) -> Dual3 {
        if self.v > hi { Dual3::c(hi) } else { self }
    }
}
