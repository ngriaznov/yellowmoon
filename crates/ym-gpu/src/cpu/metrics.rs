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

// ---------------------------------------------------------------------------------------------
// Lateral chromatic aberration left in an image (RA-03b)
// ---------------------------------------------------------------------------------------------

/// Settings of [`lateral_ca_residual`]. The defaults suit a 1:1 render of a photograph or of a
/// synthetic scene with edges at all orientations.
#[derive(Clone, Debug, PartialEq)]
pub struct CaResidualOptions {
    /// Side of a square patch in pixels.
    pub patch_px: usize,
    /// Distance between the origins of neighbouring patches in pixels (0 counts as 1).
    pub stride_px: usize,
    /// Patches whose centre is closer to the optical centre than this fraction of the half
    /// diagonal are left out: there is nothing to measure where the CA is zero.
    pub min_radius: f64,
    /// A patch counts only when the RMS of the gradient of green along the radial direction is at
    /// least this much (per pixel, in the units of the image): flat patches fix no shift.
    pub min_gradient: f64,
    /// A patch counts only when the normalised red (or blue) and green planes correlate at least
    /// this well at the best shift: where the colours differ in structure, not just in position,
    /// the shift is not defined.
    pub min_correlation: f64,
    /// Upper edges of the radial bands, as fractions of the half diagonal, increasing; the first
    /// band starts at `min_radius`.
    pub band_edges: Vec<f64>,
}

impl Default for CaResidualOptions {
    fn default() -> Self {
        CaResidualOptions {
            patch_px: 48,
            stride_px: 32,
            min_radius: 0.2,
            min_gradient: 0.01,
            min_correlation: 0.9,
            band_edges: vec![0.4, 0.6, 0.8, 1.0],
        }
    }
}

/// The residual misregistration in one radial band of [`CaResidual`].
#[derive(Clone, Debug, PartialEq)]
pub struct CaBand {
    /// Inner edge of the band, as a fraction of the half diagonal.
    pub radius_from: f64,
    /// Outer edge of the band.
    pub radius_to: f64,
    /// Number of patches that counted in this band.
    pub patches: usize,
    /// Root mean square over the patches of the red shift against green, in pixels.
    pub red_rms_px: f64,
    /// Mean over the patches of the signed red shift (positive: outward), in pixels.
    pub red_mean_px: f64,
    /// Root mean square of the blue shift against green, in pixels.
    pub blue_rms_px: f64,
    /// Mean of the signed blue shift (positive: outward), in pixels.
    pub blue_mean_px: f64,
}

/// The shift found in one patch (see [`CaResidual::details`]).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CaPatch {
    /// Column of the patch's first pixel.
    pub x: usize,
    /// Row of the patch's first pixel.
    pub y: usize,
    /// Distance of the patch's centre from the optical centre, as a fraction of the half diagonal.
    pub radius: f64,
    /// Red against green along the radius, in pixels (positive: outward).
    pub red_px: f64,
    /// Blue against green along the radius, in pixels (positive: outward).
    pub blue_px: f64,
}

/// The lateral CA left in an image: the displacement of red and of blue against green along the
/// radius from the optical centre, per radial band (see [`lateral_ca_residual`]).
#[derive(Clone, Debug, PartialEq)]
pub struct CaResidual {
    /// One entry per band of [`CaResidualOptions::band_edges`].
    pub bands: Vec<CaBand>,
    /// Every patch that counted, in scan order, so that two renderings of the same image can be
    /// compared patch by patch.
    pub details: Vec<CaPatch>,
    /// Patches that counted, over all bands.
    pub patches: usize,
    /// Patches that were looked at but did not count (too flat, or the planes do not correlate).
    pub skipped: usize,
    /// Root mean square of the red shift over all counted patches, in pixels.
    pub red_rms_px: f64,
    /// Root mean square of the blue shift over all counted patches, in pixels.
    pub blue_rms_px: f64,
}

/// Catmull-Rom bicubic sample of `plane` (`w x h`) at the continuous position `(x, y)` (pixel `i`
/// covers `[i, i + 1)`, its centre is `i + 0.5`). Taps beyond the edge replicate the edge pixel.
pub fn sample_catmull_rom(plane: &[f32], w: usize, h: usize, x: f64, y: f64) -> f64 {
    let (fx, fy) = (x - 0.5, y - 0.5);
    let (ix, iy) = (fx.floor(), fy.floor());
    let (tx, ty) = (fx - ix, fy - iy);
    let weights = |t: f64| {
        let t2 = t * t;
        let t3 = t2 * t;
        [
            0.5 * (-t3 + 2.0 * t2 - t),
            0.5 * (3.0 * t3 - 5.0 * t2 + 2.0),
            0.5 * (-3.0 * t3 + 4.0 * t2 + t),
            0.5 * (t3 - t2),
        ]
    };
    let (wx, wy) = (weights(tx), weights(ty));
    let mut acc = 0.0;
    for (j, wyj) in wy.iter().enumerate() {
        let yy = (iy as isize + j as isize - 1).clamp(0, h as isize - 1) as usize;
        let mut row = 0.0;
        for (i, wxi) in wx.iter().enumerate() {
            let xx = (ix as isize + i as isize - 1).clamp(0, w as isize - 1) as usize;
            row += wxi * f64::from(plane[yy * w + xx]);
        }
        acc += wyj * row;
    }
    acc
}

/// The shift `s` (pixels, positive outward) of the patch of `x_plane` against `g_plane` along the
/// unit vector `u`, and the correlation of the two at that shift: the Gauss-Newton fit of
/// `X(p + s u) = G(p)` over the patch pixels, with both planes normalised to zero mean and unit
/// variance inside the patch (a colour difference between red and green is mostly a gain and an
/// offset, not a different structure), the red (or blue) plane sampled bicubically. `None` when
/// the patch has no gradient along `u` or no variance.
#[allow(clippy::too_many_arguments)]
fn patch_shift(
    x_plane: &[f32],
    g_plane: &[f32],
    w: usize,
    h: usize,
    origin: (usize, usize),
    side: usize,
    u: (f64, f64),
    min_gradient: f64,
) -> Option<(f64, f64)> {
    let n = side * side;
    let mut pos = Vec::with_capacity(n);
    let mut gv = Vec::with_capacity(n);
    for y in origin.1..origin.1 + side {
        for x in origin.0..origin.0 + side {
            pos.push((x as f64 + 0.5, y as f64 + 0.5));
            gv.push(f64::from(g_plane[y * w + x]));
        }
    }
    // The gradient of green along u, to decide whether the patch fixes a shift at all.
    let grad_g: f64 = pos
        .iter()
        .map(|&(px, py)| {
            let a = sample_catmull_rom(g_plane, w, h, px + 0.5 * u.0, py + 0.5 * u.1);
            let b = sample_catmull_rom(g_plane, w, h, px - 0.5 * u.0, py - 0.5 * u.1);
            (a - b) * (a - b)
        })
        .sum::<f64>();
    if (grad_g / n as f64).sqrt() < min_gradient {
        return None;
    }
    let normalise = |v: &mut [f64]| -> Option<f64> {
        let mean = v.iter().sum::<f64>() / n as f64;
        let var = v.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / n as f64;
        if var <= 1e-24 {
            return None;
        }
        let sd = var.sqrt();
        for x in v.iter_mut() {
            *x = (*x - mean) / sd;
        }
        Some(sd)
    };
    normalise(&mut gv)?;
    let mut s = 0.0f64;
    let mut correlation = 0.0;
    for _ in 0..8 {
        let mut xv: Vec<f64> = pos
            .iter()
            .map(|&(px, py)| sample_catmull_rom(x_plane, w, h, px + s * u.0, py + s * u.1))
            .collect();
        let mut gx: Vec<f64> = pos
            .iter()
            .map(|&(px, py)| {
                let a =
                    sample_catmull_rom(x_plane, w, h, px + (s + 0.5) * u.0, py + (s + 0.5) * u.1);
                let b =
                    sample_catmull_rom(x_plane, w, h, px + (s - 0.5) * u.0, py + (s - 0.5) * u.1);
                a - b
            })
            .collect();
        let sd = normalise(&mut xv)?;
        for g in &mut gx {
            *g /= sd;
        }
        correlation = xv.iter().zip(&gv).map(|(a, b)| a * b).sum::<f64>() / n as f64;
        let num: f64 = xv
            .iter()
            .zip(&gv)
            .zip(&gx)
            .map(|((x, g), d)| (g - x) * d)
            .sum();
        let den: f64 = gx.iter().map(|d| d * d).sum();
        if den <= 1e-12 {
            return None;
        }
        let step = num / den;
        s = (s + step).clamp(-4.0, 4.0);
        if step.abs() < 1e-5 {
            break;
        }
    }
    Some((s, correlation))
}

/// The lateral chromatic aberration left in `img`: how far red and blue are displaced against
/// green along the radius from the optical centre `centre_px` (in the pixels of `img`), in
/// patches of [`CaResidualOptions::patch_px`] on a grid, grouped in radial bands of
/// `half_diagonal_px`.
///
/// In every patch whose green has radial gradient the displacement `s` solves
/// `X(p + s u) = G(p)` in the least-squares sense (`u` is the unit vector away from the centre, `X`
/// red or blue; see the private fit for the details), so a positive value means the colour plane is
/// displaced outward. After a perfect correction the numbers are zero up to the interpolation and
/// the demosaic's own error; a lens with CA that is not corrected shows its full displacement.
///
/// This is a measurement tool for tests and the local evaluation of RA-03b, not part of the render.
pub fn lateral_ca_residual(
    img: &RgbImage,
    centre_px: (f64, f64),
    half_diagonal_px: f64,
    options: &CaResidualOptions,
) -> CaResidual {
    let (w, h) = (img.width, img.height);
    let side = options.patch_px;
    let stride = options.stride_px.max(1);
    let margin = 4usize;
    let mut edges = vec![options.min_radius];
    edges.extend(options.band_edges.iter().copied());
    let nb = edges.len() - 1;
    // Per band: count, sum and sum of squares of red and blue.
    let mut acc = vec![[0.0f64; 5]; nb];
    let mut counts = vec![0usize; nb];
    let (mut total, mut skipped) = (0usize, 0usize);
    let (mut sum_r2, mut sum_b2) = (0.0f64, 0.0f64);
    let mut details = Vec::new();
    let mut y0 = margin;
    while y0 + side + margin <= h {
        let mut x0 = margin;
        while x0 + side + margin <= w {
            let (cx, cy) = (x0 as f64 + 0.5 * side as f64, y0 as f64 + 0.5 * side as f64);
            let (dx, dy) = (cx - centre_px.0, cy - centre_px.1);
            let r = dx.hypot(dy);
            let frac = r / half_diagonal_px;
            // A patch centred on the optical centre has no radial direction to measure along.
            let band = (r > 0.0)
                .then(|| (0..nb).find(|&b| frac >= edges[b] && frac < edges[b + 1]))
                .flatten();
            if let Some(b) = band {
                let u = (dx / r, dy / r);
                let red = patch_shift(
                    img.plane(0),
                    img.plane(1),
                    w,
                    h,
                    (x0, y0),
                    side,
                    u,
                    options.min_gradient,
                );
                let blue = patch_shift(
                    img.plane(2),
                    img.plane(1),
                    w,
                    h,
                    (x0, y0),
                    side,
                    u,
                    options.min_gradient,
                );
                match (red, blue) {
                    (Some((sr, cr)), Some((sb, cb)))
                        if cr >= options.min_correlation && cb >= options.min_correlation =>
                    {
                        let a = &mut acc[b];
                        a[0] += sr;
                        a[1] += sr * sr;
                        a[2] += sb;
                        a[3] += sb * sb;
                        counts[b] += 1;
                        total += 1;
                        sum_r2 += sr * sr;
                        sum_b2 += sb * sb;
                        details.push(CaPatch {
                            x: x0,
                            y: y0,
                            radius: frac,
                            red_px: sr,
                            blue_px: sb,
                        });
                    }
                    _ => skipped += 1,
                }
            }
            x0 += stride;
        }
        y0 += stride;
    }
    let bands = (0..nb)
        .map(|b| {
            let n = counts[b].max(1) as f64;
            CaBand {
                radius_from: edges[b],
                radius_to: edges[b + 1],
                patches: counts[b],
                red_rms_px: (acc[b][1] / n).sqrt(),
                red_mean_px: acc[b][0] / n,
                blue_rms_px: (acc[b][3] / n).sqrt(),
                blue_mean_px: acc[b][2] / n,
            }
        })
        .collect();
    let n = total.max(1) as f64;
    CaResidual {
        bands,
        details,
        patches: total,
        skipped,
        red_rms_px: (sum_r2 / n).sqrt(),
        blue_rms_px: (sum_b2 / n).sqrt(),
    }
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

    /// A smooth test pattern with edges at many orientations, analytic so that a plane can be
    /// sampled at any shifted position.
    fn pattern(x: f64, y: f64) -> f64 {
        let g = (0.071 * x + 0.043 * y).sin()
            + (0.031 * x - 0.089 * y + 1.0).sin()
            + (0.113 * x + 0.017 * y + 2.0).sin()
            + 0.7 * (0.052 * (x + y) + 0.5).sin();
        0.5 + 0.35 * (g * 1.3).tanh()
    }

    #[test]
    fn the_ca_residual_measures_a_radial_displacement() {
        // Green is the pattern; red is displaced outward by 0.6 r / hd px and blue inward by
        // 0.4 r / hd px (the pattern sampled at p - s u): the estimator must find them back.
        let (w, h) = (360usize, 240usize);
        let (cx, cy) = (w as f64 / 2.0, h as f64 / 2.0);
        let hd = cx.hypot(cy);
        let mut img = RgbImage::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let (px, py) = (x as f64 + 0.5, y as f64 + 0.5);
                let (dx, dy) = (px - cx, py - cy);
                let r = dx.hypot(dy);
                let u = if r > 0.0 {
                    (dx / r, dy / r)
                } else {
                    (0.0, 0.0)
                };
                let shift = |amount: f64| amount * r / hd;
                let sr = shift(0.6);
                let sb = shift(-0.4);
                img.plane_mut(0)[y * w + x] = pattern(px - sr * u.0, py - sr * u.1) as f32;
                img.plane_mut(1)[y * w + x] = pattern(px, py) as f32;
                img.plane_mut(2)[y * w + x] = pattern(px - sb * u.0, py - sb * u.1) as f32;
            }
        }
        let opts = CaResidualOptions {
            patch_px: 32,
            stride_px: 16,
            min_radius: 0.3,
            band_edges: vec![0.6, 0.8, 1.0],
            ..CaResidualOptions::default()
        };
        let res = lateral_ca_residual(&img, (cx, cy), hd, &opts);
        assert!(res.patches > 20, "{} patches", res.patches);
        for band in &res.bands {
            if band.patches == 0 {
                continue;
            }
            let mid = 0.5 * (band.radius_from + band.radius_to);
            // The shift varies across a band; compare with the value at the band's middle.
            assert!(
                (band.red_mean_px - 0.6 * mid).abs() < 0.12,
                "red {band:?} want {}",
                0.6 * mid
            );
            assert!(
                (band.blue_mean_px + 0.4 * mid).abs() < 0.12,
                "blue {band:?} want {}",
                -0.4 * mid
            );
        }
        // Nothing displaced: nothing found.
        for c in [0, 2] {
            let g = img.plane(1).to_vec();
            img.plane_mut(c).copy_from_slice(&g);
        }
        let zero = lateral_ca_residual(&img, (cx, cy), hd, &opts);
        assert!(zero.patches > 20);
        assert!(
            zero.red_rms_px < 0.01 && zero.blue_rms_px < 0.01,
            "{zero:?}"
        );
    }

    #[test]
    fn a_zero_stride_terminates_and_a_patch_at_the_centre_is_skipped() {
        let (w, h) = (96usize, 96usize);
        let mut img = RgbImage::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let v = pattern(x as f64 + 0.5, y as f64 + 0.5) as f32;
                for c in 0..3 {
                    img.plane_mut(c)[y * w + x] = v;
                }
            }
        }
        // Stride 0 would never advance; it counts as 1. The patch whose centre is the optical
        // centre (r = 0) has no radial direction and does not count, with `min_radius` 0.
        let opts = CaResidualOptions {
            patch_px: 32,
            stride_px: 0,
            min_radius: 0.0,
            min_gradient: 0.0,
            ..CaResidualOptions::default()
        };
        let res = lateral_ca_residual(&img, (48.0, 48.0), 68.0, &opts);
        assert!(res.patches > 100, "{} patches", res.patches);
        assert!(res.details.iter().all(|p| p.radius > 0.0));
        assert!(res.red_rms_px < 0.01 && res.blue_rms_px < 0.01, "{res:?}");
    }

    #[test]
    fn the_ca_residual_skips_flat_patches() {
        let mut img = RgbImage::new(128, 128);
        img.data.fill(0.4);
        let res = lateral_ca_residual(&img, (64.0, 64.0), 90.0, &CaResidualOptions::default());
        assert_eq!(res.patches, 0);
        assert_eq!(res.red_rms_px, 0.0);
    }
}
