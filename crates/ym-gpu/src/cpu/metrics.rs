//! Image-quality metrics for the CPU reference tests and the demosaic comparison: PSNR, CIE Lab
//! and CIEDE2000 (`docs/13-testing.md`).
//!
//! The highlight gate (`docs/05` R5) measures chroma in CIE Lab, which needs a camera-to-XYZ
//! matrix. The reference chain has no camera profile at this stage, so the tests use the stand-in
//! [`STAND_IN_CAMERA_TO_XYZ`]: white-balanced camera RGB is read as linear Rec. 709 / sRGB primaries
//! with a D65 white (neutral stays neutral, saturated colours stay saturated). It is only a way
//! to turn camera RGB into something with a perceptual metric; it is not a calibration.

use super::RgbImage;

/// Linear Rec. 709 (sRGB) primaries to XYZ, D65 white. Stand-in for a camera matrix.
pub const STAND_IN_CAMERA_TO_XYZ: [[f64; 3]; 3] = [
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041],
];

/// D65 reference white (Y = 1) in XYZ.
pub const D65_WHITE: [f64; 3] = [0.95047, 1.0, 1.08883];

/// Peak signal-to-noise ratio in dB between two images over all three channels.
///
/// Both images are clamped to `[0, peak]` first; `border` pixels are excluded on every side (the
/// usual demosaic protocol, since every method treats the border differently).
pub fn psnr(a: &RgbImage, b: &RgbImage, border: usize, peak: f64) -> f64 {
    assert_eq!(
        (a.width, a.height),
        (b.width, b.height),
        "psnr: dimension mismatch"
    );
    let (w, h) = (a.width, a.height);
    let mut sum = 0.0f64;
    let mut n = 0u64;
    for c in 0..3 {
        let (pa, pb) = (a.plane(c), b.plane(c));
        for y in border..h.saturating_sub(border) {
            for x in border..w.saturating_sub(border) {
                let i = y * w + x;
                let d = f64::from(pa[i]).clamp(0.0, peak) - f64::from(pb[i]).clamp(0.0, peak);
                sum += d * d;
                n += 1;
            }
        }
    }
    let mse = sum / n.max(1) as f64;
    if mse <= 0.0 {
        f64::INFINITY
    } else {
        10.0 * (peak * peak / mse).log10()
    }
}

/// CIE Lab (D65) of a white-balanced camera RGB triplet through [`STAND_IN_CAMERA_TO_XYZ`].
pub fn lab_from_camera_rgb(rgb: [f64; 3]) -> [f64; 3] {
    let m = &STAND_IN_CAMERA_TO_XYZ;
    let xyz: [f64; 3] =
        std::array::from_fn(|i| m[i][0] * rgb[0] + m[i][1] * rgb[1] + m[i][2] * rgb[2]);
    lab_from_xyz(xyz)
}

/// CIE Lab (D65) from XYZ.
pub fn lab_from_xyz(xyz: [f64; 3]) -> [f64; 3] {
    const EPS: f64 = 216.0 / 24389.0;
    const KAPPA: f64 = 24389.0 / 27.0;
    let f = |t: f64| {
        if t > EPS {
            t.cbrt()
        } else {
            (KAPPA * t + 16.0) / 116.0
        }
    };
    let (fx, fy, fz) = (
        f(xyz[0] / D65_WHITE[0]),
        f(xyz[1] / D65_WHITE[1]),
        f(xyz[2] / D65_WHITE[2]),
    );
    [116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)]
}

/// CIEDE2000 colour difference between two Lab colours (Sharma, Wu and Dalal 2005, kL = kC = kH = 1).
pub fn delta_e_2000(lab1: [f64; 3], lab2: [f64; 3]) -> f64 {
    let rad = |d: f64| d.to_radians();
    let [l1, a1, b1] = lab1;
    let [l2, a2, b2] = lab2;
    let c1 = a1.hypot(b1);
    let c2 = a2.hypot(b2);
    let cbar = 0.5 * (c1 + c2);
    let cbar7 = cbar.powi(7);
    let g = 0.5 * (1.0 - (cbar7 / (cbar7 + 25f64.powi(7))).sqrt());
    let a1p = (1.0 + g) * a1;
    let a2p = (1.0 + g) * a2;
    let c1p = a1p.hypot(b1);
    let c2p = a2p.hypot(b2);
    let hue = |b: f64, a: f64| {
        if a == 0.0 && b == 0.0 {
            0.0
        } else {
            let h = b.atan2(a).to_degrees();
            if h < 0.0 { h + 360.0 } else { h }
        }
    };
    let h1p = hue(b1, a1p);
    let h2p = hue(b2, a2p);
    let dlp = l2 - l1;
    let dcp = c2p - c1p;
    let dhp_deg = if c1p * c2p == 0.0 {
        0.0
    } else {
        let d = h2p - h1p;
        if d > 180.0 {
            d - 360.0
        } else if d < -180.0 {
            d + 360.0
        } else {
            d
        }
    };
    let dhp = 2.0 * (c1p * c2p).sqrt() * rad(0.5 * dhp_deg).sin();
    let lbarp = 0.5 * (l1 + l2);
    let cbarp = 0.5 * (c1p + c2p);
    let hbarp = if c1p * c2p == 0.0 {
        h1p + h2p
    } else if (h1p - h2p).abs() <= 180.0 {
        0.5 * (h1p + h2p)
    } else if h1p + h2p < 360.0 {
        0.5 * (h1p + h2p + 360.0)
    } else {
        0.5 * (h1p + h2p - 360.0)
    };
    let t = 1.0 - 0.17 * rad(hbarp - 30.0).cos()
        + 0.24 * rad(2.0 * hbarp).cos()
        + 0.32 * rad(3.0 * hbarp + 6.0).cos()
        - 0.20 * rad(4.0 * hbarp - 63.0).cos();
    let dtheta = 30.0 * (-(((hbarp - 275.0) / 25.0).powi(2))).exp();
    let cbarp7 = cbarp.powi(7);
    let rc = 2.0 * (cbarp7 / (cbarp7 + 25f64.powi(7))).sqrt();
    let sl = 1.0 + 0.015 * (lbarp - 50.0).powi(2) / (20.0 + (lbarp - 50.0).powi(2)).sqrt();
    let sc = 1.0 + 0.045 * cbarp;
    let sh = 1.0 + 0.015 * cbarp * t;
    let rt = -rad(2.0 * dtheta).sin() * rc;
    ((dlp / sl).powi(2) + (dcp / sc).powi(2) + (dhp / sh).powi(2) + rt * (dcp / sc) * (dhp / sh))
        .sqrt()
}

/// CIEDE2000 between two camera RGB triplets with the lightness forced equal, so only the chroma
/// and hue differ. Both colours are scaled to the same mean luminance first. This is the measure
/// of the R5 gate: "chroma of the reconstructed region within dE00 5 of the surround".
pub fn chroma_delta_e00(rgb1: [f64; 3], rgb2: [f64; 3]) -> f64 {
    let mut l1 = lab_from_camera_rgb(rgb1);
    let mut l2 = lab_from_camera_rgb(rgb2);
    let l = 0.5 * (l1[0] + l2[0]);
    l1[0] = l;
    l2[0] = l;
    delta_e_2000(l1, l2)
}

/// Lab hue angle in degrees (0..360) of a camera RGB triplet.
pub fn hue_degrees(rgb: [f64; 3]) -> f64 {
    let lab = lab_from_camera_rgb(rgb);
    let h = lab[2].atan2(lab[1]).to_degrees();
    if h < 0.0 { h + 360.0 } else { h }
}

/// Lab chroma `sqrt(a^2 + b^2)` of a camera RGB triplet.
pub fn chroma(rgb: [f64; 3]) -> f64 {
    let lab = lab_from_camera_rgb(rgb);
    lab[1].hypot(lab[2])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn delta_e_2000_matches_color_science_reference_pairs() {
        // Pairs and expected values computed with colour-science 0.4.7 `colour.delta_E(.., method='CIE 2000')`.
        let cases: [([f64; 3], [f64; 3], f64); 6] = [
            ([50.0, 2.6772, -79.7751], [50.0, 0.0, -82.7485], 2.0424596),
            ([50.0, 3.1571, -77.2803], [50.0, 0.0, -82.7485], 2.8615147),
            ([50.0, 2.5, 0.0], [73.0, 25.0, -18.0], 27.1492),
            ([50.0, 2.49, -0.001], [50.0, -2.49, 0.0009], 7.1792),
            (
                [60.2574, -34.0099, 36.2677],
                [60.4626, -34.1751, 39.4387],
                1.2644,
            ),
            (
                [22.7233, 20.0904, -46.694],
                [23.0331, 14.973, -42.5619],
                2.0373,
            ),
        ];
        for (a, b, want) in cases {
            let got = delta_e_2000(a, b);
            assert!(
                (got - want).abs() < 2e-3,
                "{a:?} vs {b:?}: got {got}, want {want}"
            );
        }
    }

    #[test]
    fn neutral_has_zero_chroma_and_white_is_l100() {
        let lab = lab_from_camera_rgb([1.0, 1.0, 1.0]);
        assert!((lab[0] - 100.0).abs() < 0.01 && lab[1].abs() < 0.01 && lab[2].abs() < 0.01);
        assert!(chroma([0.3, 0.3, 0.3]) < 0.01);
    }

    #[test]
    fn psnr_of_known_error() {
        let a = RgbImage::new(10, 10);
        let mut b = RgbImage::new(10, 10);
        b.data.fill(0.1);
        assert!((psnr(&a, &b, 0, 1.0) - 20.0).abs() < 1e-6);
        assert!(psnr(&a, &a, 0, 1.0).is_infinite());
    }
}
