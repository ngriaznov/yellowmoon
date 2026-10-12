// SPDX-License-Identifier: Apache-2.0
// Copyright Contributors to the ACES Project (the reference this file ports) and Yellowmoon contributors.

//! ACES 2 tone scale (Daniele-style Michaelis-Menten curve with a flare term).
//!
//! Port of `Lib.Academy.Tonescale.ctl`: `init_TSParams` and `tonescale_fwd`. The inverse tone scale
//! belongs to the inverse output transform, which Yellowmoon does not need.

/// Pre-computed tone scale constants for one peak luminance.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ToneScaleParams {
    /// Peak luminance in nits.
    pub n: f64,
    /// Normalised white in nits (what 1.0 means).
    pub n_r: f64,
    /// Surround / contrast exponent.
    pub g: f64,
    /// Shadow toe, flare and glare compensation.
    pub t_1: f64,
    /// Mid grey output (normalised) the curve is anchored to.
    pub c_t: f64,
    /// Curve scale `s_2`.
    pub s_2: f64,
    /// Curve normalisation `u_2`.
    pub u_2: f64,
    /// Curve output scale `m_2`.
    pub m_2: f64,
    /// Largest AP1 channel value the forward transform accepts (8 times "r_hit").
    pub forward_limit: f64,
    /// Largest normalised output the inverse accepts.
    pub inverse_limit: f64,
    /// `log10(peak / 100)`, drives the peak-luminance dependent constants.
    pub log_peak: f64,
}

impl ToneScaleParams {
    /// `init_TSParams(peakLuminance)`.
    pub fn new(peak_luminance: f64) -> Self {
        // Preset constants that set the desired behaviour for the curve.
        let n = peak_luminance;

        let n_r = 100.0; // normalised white in nits (what 1.0 should be)
        let g = 1.15; // surround / contrast
        let c = 0.18; // anchor for 18% grey
        let c_d = 10.013; // output luminance of 18% grey (in nits)
        let w_g = 0.14; // change in grey between different peak luminance
        let t_1 = 0.04; // shadow toe or flare/glare compensation
        let r_hit_min = 128.0; // scene-referred value "hitting the roof" for SDR
        let r_hit_max = 896.0; // scene-referred value "hitting the roof" at 10000 nits

        // Calculate output constants.
        let r_hit =
            r_hit_min + (r_hit_max - r_hit_min) * ((n / n_r).ln() / (10000.0_f64 / 100.0).ln());
        let m_0 = n / n_r;
        let m_1 = 0.5 * (m_0 + (m_0 * (m_0 + 4.0 * t_1)).sqrt());
        let u = ((r_hit / m_1) / ((r_hit / m_1) + 1.0)).powf(g);
        let m = m_1 / u;
        let w_i = (n / 100.0).ln() / 2.0_f64.ln();
        let c_t = c_d / n_r * (1.0 + w_i * w_g);
        let g_ip = 0.5 * (c_t + (c_t * (c_t + 4.0 * t_1)).sqrt());
        let g_ipp2 = -(m_1 * (g_ip / m).powf(1.0 / g)) / ((g_ip / m).powf(1.0 / g) - 1.0);
        let w_2 = c / g_ipp2;
        let s_2 = w_2 * m_1;
        let u_2 = ((r_hit / m_1) / ((r_hit / m_1) + w_2)).powf(g);
        let m_2 = m_1 / u_2;

        ToneScaleParams {
            n,
            n_r,
            g,
            t_1,
            c_t,
            s_2,
            u_2,
            m_2,
            forward_limit: 8.0 * r_hit,
            inverse_limit: n / (u_2 * n_r),
            log_peak: (n / n_r).log10(),
        }
    }

    /// `tonescale_fwd`: scene-referred linear `x` (1.0 = 100 nit reference white) to luminance in nits.
    pub fn forward(&self, x: f64) -> f64 {
        // forward MM tone scale
        let f = self.m_2 * (x.max(0.0) / (x + self.s_2)).powf(self.g);
        // forward flare
        let h = (f * f / (f + self.t_1)).max(0.0);
        h * self.n_r
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grey_anchor_and_monotonic() {
        let ts = ToneScaleParams::new(100.0);
        // 18% grey lands near c_d = 10.013 nits (the flare term shifts it by a few 1e-3).
        let y = ts.forward(0.18);
        assert!((y - 10.0).abs() < 0.05, "grey maps to {y} nits");
        let mut prev = -1.0;
        for i in 0..400 {
            let x = 1e-6 * 1.06_f64.powi(i);
            let y = ts.forward(x);
            assert!(y >= prev, "not monotonic at {x}");
            prev = y;
        }
        // The curve saturates at m_2^2 / (m_2 + t_1) * 100 nits, about 1% over the peak; the
        // transform clamps to the peak after the gamut mapping.
        assert!(prev < 101.0 && prev > 100.0, "saturation level {prev}");
    }

    #[test]
    fn constants_for_100_nits() {
        let ts = ToneScaleParams::new(100.0);
        assert_eq!(ts.forward_limit, 1024.0);
        assert!((ts.log_peak).abs() < 1e-15);
        assert!((ts.c_t - 0.10013).abs() < 1e-12);
        assert_eq!(ts.forward(0.0), 0.0);
        assert_eq!(ts.forward(-1.0), 0.0);
    }
}
