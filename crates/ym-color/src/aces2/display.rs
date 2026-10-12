// SPDX-License-Identifier: Apache-2.0
// Copyright Contributors to the ACES Project (the reference this file ports) and Yellowmoon contributors.

//! Display encoding: the inverse EOTFs used by the four Yellowmoon outputs.
//!
//! Port of `moncurve_fwd`, `moncurve_inv`, `bt1886_fwd`, `bt1886_inv` and `eotf_inv` in
//! `Lib.Academy.DisplayEncoding.ctl`, restricted to the two transfer functions the outputs use.

/// Transfer function of a display output.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DisplayTransfer {
    /// sRGB piecewise function (IEC 61966-2-1:1999), also used by Display P3.
    Srgb,
    /// BT.1886 with gamma 2.4, white 1.0 and black 0.0 (a pure power law of 2.4).
    Bt1886,
}

const SRGB_GAMMA: f64 = 2.4;
const SRGB_OFFSET: f64 = 0.055;

/// `moncurve_inv`: display linear to sRGB code value.
fn moncurve_inv(y: f64, gamma: f64, offs: f64) -> f64 {
    let yb = (offs * gamma / ((gamma - 1.0) * (1.0 + offs))).powf(gamma);
    let rs = ((gamma - 1.0) / offs).powf(gamma - 1.0) * ((1.0 + offs) / gamma).powf(gamma);
    if y >= yb {
        (1.0 + offs) * y.powf(1.0 / gamma) - offs
    } else {
        y * rs
    }
}

/// `moncurve_fwd`: sRGB code value to display linear.
fn moncurve_fwd(x: f64, gamma: f64, offs: f64) -> f64 {
    let fs = ((gamma - 1.0) / offs) * (offs * gamma / ((gamma - 1.0) * (1.0 + offs))).powf(gamma);
    let xb = offs / (gamma - 1.0);
    if x >= xb {
        ((x + offs) / (1.0 + offs)).powf(gamma)
    } else {
        x * fs
    }
}

impl DisplayTransfer {
    /// Inverse EOTF: display linear light (1.0 = peak white) to a code value. Negative input is
    /// clamped to 0 first, as `eotf_inv` does.
    pub fn encode(self, linear: f64) -> f64 {
        let l = linear.max(0.0);
        match self {
            DisplayTransfer::Srgb => moncurve_inv(l, SRGB_GAMMA, SRGB_OFFSET),
            // bt1886_inv with Lw = 1 and Lb = 0: a = 1, b = 0.
            DisplayTransfer::Bt1886 => l.powf(1.0 / 2.4),
        }
    }

    /// EOTF: code value to display linear light.
    pub fn decode(self, encoded: f64) -> f64 {
        match self {
            DisplayTransfer::Srgb => moncurve_fwd(encoded, SRGB_GAMMA, SRGB_OFFSET),
            DisplayTransfer::Bt1886 => encoded.max(0.0).powf(2.4),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn srgb_known_values() {
        let t = DisplayTransfer::Srgb;
        assert_eq!(t.encode(0.0), 0.0);
        assert!((t.encode(1.0) - 1.0).abs() < 1e-15);
        // 18% grey card and the linear segment slope.
        assert!((t.encode(0.18) - 0.461_356_129_5).abs() < 1e-9);
        // The continuous moncurve slope is 12.9232..., not the 12.92 of the sRGB text.
        assert!((t.encode(0.001) - 0.012_923_210_18).abs() < 1e-11);
        // Continuity at the breakpoint.
        let yb = 0.003_039_934_639_778_432;
        assert!((t.encode(yb - 1e-12) - t.encode(yb + 1e-12)).abs() < 1e-9);
        assert_eq!(t.encode(-0.5), 0.0);
    }

    #[test]
    fn bt1886_known_values() {
        let t = DisplayTransfer::Bt1886;
        assert!((t.encode(0.18) - 0.18_f64.powf(1.0 / 2.4)).abs() < 1e-15);
        assert!((t.encode(1.0) - 1.0).abs() < 1e-15);
        assert_eq!(t.encode(-1.0), 0.0);
    }

    #[test]
    fn encode_decode_round_trip() {
        for t in [DisplayTransfer::Srgb, DisplayTransfer::Bt1886] {
            for i in 0..=200 {
                let l = (i as f64 / 200.0).powi(3);
                let back = t.decode(t.encode(l));
                assert!((back - l).abs() < 1e-12, "{t:?} {l} -> {back}");
            }
        }
    }
}
