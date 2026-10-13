//! Evaluation helpers for the R8 A/B of `docs/algorithms/03-diffraction-deconvolution.md` section
//! 6.3: edge profile, MTF50, rise distance, overshoot and flat-patch noise, measured on a luma
//! plane. Used by `examples/ra07_ab.rs` (synthetic scenes) and by the local real-file comparison
//! in `ym-core` (`pipeline::raw` tests, ignored); nothing in the render path calls them.
//!
//! The edge measurement is the slanted-edge idea reduced to what an A/B needs: every pixel is
//! projected on the edge normal, the edge spread function (ESF) is the mean luma in bins of 1/4
//! pixel, the line spread function its difference, and the MTF the magnitude of its windowed DFT.
//! The levels on both sides are the means of the ESF 12 to 24 pixels from the edge, so that the
//! normalised profile runs from 0 to 1.

use std::f64::consts::PI;

/// Half-width of the edge spread function in pixels.
pub const ESF_HALF_PX: f64 = 24.0;
/// Width of an ESF bin in pixels.
pub const ESF_STEP_PX: f64 = 0.25;
/// Distance from the edge, in pixels, beyond which the ESF gives the two levels.
const PLATEAU_PX: f64 = 12.0;

/// Where an edge is: the unit normal (pointing from the dark side to the bright side) and a point
/// on it, in pixel centres `(x + 0.5, y + 0.5)`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EdgeGeometry {
    /// Unit normal `(nx, ny)`.
    pub normal: (f64, f64),
    /// A point of the edge `(cx, cy)`.
    pub centre: (f64, f64),
}

/// What the edge measurement gives.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct EdgeStats {
    /// Frequency at which the MTF falls to 0.5, in cycles per pixel (`NaN` if it never does).
    pub mtf50: f64,
    /// 10 to 90 % rise distance in pixels (`NaN` if the profile never crosses).
    pub rise: f64,
    /// Overshoot in percent of the step (the normalised profile above 1 within 12 px of the edge).
    pub overshoot_pct: f64,
    /// Undershoot in percent of the step (the normalised profile below 0 within 12 px of the edge).
    pub undershoot_pct: f64,
    /// The normalised edge spread function from -24 to +24 px along the normal, at 1/4 px.
    pub esf: Vec<f64>,
}

/// The dominant edge of a window from its gradients: the normal is the dominant gradient direction
/// (structure tensor, sign chosen so that it points to the brighter side) and the point is the
/// centroid of the gradient energy. `None` for a flat window.
#[must_use]
pub fn edge_geometry(luma: &[f64], w: usize, h: usize) -> Option<EdgeGeometry> {
    if w < 3 || h < 3 {
        return None;
    }
    let (mut jxx, mut jyy, mut jxy) = (0.0f64, 0.0f64, 0.0f64);
    let (mut sx, mut sy, mut sw) = (0.0f64, 0.0f64, 0.0f64);
    let (mut gx_sum, mut gy_sum) = (0.0f64, 0.0f64);
    for y in 1..h - 1 {
        for x in 1..w - 1 {
            let gx = 0.5 * (luma[y * w + x + 1] - luma[y * w + x - 1]);
            let gy = 0.5 * (luma[(y + 1) * w + x] - luma[(y - 1) * w + x]);
            let e = gx * gx + gy * gy;
            jxx += gx * gx;
            jyy += gy * gy;
            jxy += gx * gy;
            sx += e * (x as f64 + 0.5);
            sy += e * (y as f64 + 0.5);
            sw += e;
            gx_sum += gx;
            gy_sum += gy;
        }
    }
    if sw <= 0.0 {
        return None;
    }
    let angle = 0.5 * (2.0 * jxy).atan2(jxx - jyy);
    let mut normal = (angle.cos(), angle.sin());
    // Point towards increasing luma.
    if normal.0 * gx_sum + normal.1 * gy_sum < 0.0 {
        normal = (-normal.0, -normal.1);
    }
    Some(EdgeGeometry {
        normal,
        centre: (sx / sw, sy / sw),
    })
}

/// Measures the edge of `geometry` in a `w x h` luma plane, ignoring pixels closer than
/// `margin_px` to the border (use it where the plane's own border is a mirror of the image).
#[must_use]
pub fn edge_stats(
    luma: &[f64],
    w: usize,
    h: usize,
    geometry: &EdgeGeometry,
    margin_px: usize,
) -> EdgeStats {
    let (n, c) = (geometry.normal, geometry.centre);
    let bins = (2.0 * ESF_HALF_PX / ESF_STEP_PX) as usize;
    let mut sum = vec![0.0f64; bins];
    let mut cnt = vec![0.0f64; bins];
    for y in margin_px..h.saturating_sub(margin_px) {
        for x in margin_px..w.saturating_sub(margin_px) {
            let s = n.0 * (x as f64 + 0.5 - c.0) + n.1 * (y as f64 + 0.5 - c.1);
            if s.abs() >= ESF_HALF_PX {
                continue;
            }
            let b = ((s + ESF_HALF_PX) / ESF_STEP_PX) as usize;
            sum[b] += luma[y * w + x];
            cnt[b] += 1.0;
        }
    }
    let mut esf: Vec<f64> = sum
        .iter()
        .zip(&cnt)
        .map(|(s, c)| if *c > 0.0 { s / c } else { f64::NAN })
        .collect();
    if esf.iter().all(|v| v.is_nan()) {
        return EdgeStats {
            mtf50: f64::NAN,
            rise: f64::NAN,
            overshoot_pct: f64::NAN,
            undershoot_pct: f64::NAN,
            esf,
        };
    }
    for i in 0..bins {
        if esf[i].is_nan() {
            let a = (0..i)
                .rev()
                .find(|&j| !esf[j].is_nan())
                .map_or(f64::NAN, |j| esf[j]);
            let b = (i + 1..bins)
                .find(|&j| !esf[j].is_nan())
                .map_or(a, |j| esf[j]);
            esf[i] = if a.is_nan() { b } else { 0.5 * (a + b) };
        }
    }
    let at = |s: f64| ((s + ESF_HALF_PX) / ESF_STEP_PX) as usize;
    let mean = |a: usize, b: usize| esf[a..b].iter().sum::<f64>() / (b - a) as f64;
    let lo = mean(0, at(-PLATEAU_PX));
    let hi = mean(at(PLATEAU_PX), bins);
    let norm: Vec<f64> = esf.iter().map(|v| (v - lo) / (hi - lo)).collect();
    let cross = |level: f64| -> f64 {
        for i in at(-PLATEAU_PX)..at(PLATEAU_PX) {
            if norm[i] < level && norm[i + 1] >= level {
                return i as f64 + (level - norm[i]) / (norm[i + 1] - norm[i]);
            }
        }
        f64::NAN
    };
    let rise = (cross(0.9) - cross(0.1)) * ESF_STEP_PX;
    let (mut over, mut under) = (0.0f64, 0.0f64);
    for v in &norm[at(-PLATEAU_PX)..at(PLATEAU_PX)] {
        over = over.max(v - 1.0);
        under = under.max(-v);
    }
    EdgeStats {
        mtf50: mtf50(&norm),
        rise,
        overshoot_pct: 100.0 * over,
        undershoot_pct: 100.0 * under,
        esf: norm,
    }
}

/// MTF50 in cycles per pixel from a normalised ESF at [`ESF_STEP_PX`]: the line spread function is
/// the difference, windowed with a Hann window, zero padded to 2048 points and transformed.
fn mtf50(esf: &[f64]) -> f64 {
    let lsf: Vec<f64> = (0..esf.len() - 1).map(|i| esf[i + 1] - esf[i]).collect();
    let m = lsf.len();
    let win: Vec<f64> = (0..m)
        .map(|i| 0.5 - 0.5 * (2.0 * PI * (i as f64 + 0.5) / m as f64).cos())
        .collect();
    let nfft = 2048usize;
    let per_px = 1.0 / ESF_STEP_PX; // samples per pixel
    let dft = |k: usize| -> f64 {
        let (mut re, mut im) = (0.0, 0.0);
        for (i, (&l, &wv)) in lsf.iter().zip(&win).enumerate() {
            let a = -2.0 * PI * (k * i) as f64 / nfft as f64;
            re += l * wv * a.cos();
            im += l * wv * a.sin();
        }
        re.hypot(im)
    };
    let dc = dft(0);
    if dc.is_nan() || dc == 0.0 {
        return f64::NAN;
    }
    let mut prev = 1.0;
    for k in 1..nfft / 2 {
        let v = dft(k) / dc;
        if v <= 0.5 {
            let f_prev = (k - 1) as f64 / nfft as f64 * per_px;
            let f_now = k as f64 / nfft as f64 * per_px;
            return f_prev + (prev - 0.5) / (prev - v) * (f_now - f_prev);
        }
        prev = v;
    }
    f64::NAN
}

/// Standard deviation of a luma plane over the windows `(x, y, w, h)`, pooled (each window's own
/// mean removed).
#[must_use]
pub fn patch_sd(luma: &[f64], w: usize, windows: &[(usize, usize, usize, usize)]) -> f64 {
    let mut dev = Vec::new();
    for &(x0, y0, ww, wh) in windows {
        let mut v = Vec::with_capacity(ww * wh);
        for y in y0..y0 + wh {
            for x in x0..x0 + ww {
                v.push(luma[y * w + x]);
            }
        }
        let m = v.iter().sum::<f64>() / v.len() as f64;
        dev.extend(v.iter().map(|x| x - m));
    }
    (dev.iter().map(|d| d * d).sum::<f64>() / dev.len() as f64).sqrt()
}

/// Like [`patch_sd`] for one window `(x, y, w, h)`, with the best-fit plane `a + b x + c y`
/// removed first: real flat areas (a sky, a wall) have a slow gradient that is not noise.
#[must_use]
pub fn patch_sd_detrended(
    luma: &[f64],
    stride: usize,
    window: (usize, usize, usize, usize),
) -> f64 {
    let (x0, y0, ww, wh) = window;
    let n = (ww * wh) as f64;
    // Normal equations of the plane fit on centred coordinates (the cross terms vanish).
    let (cx, cy) = ((ww as f64 - 1.0) / 2.0, (wh as f64 - 1.0) / 2.0);
    let (mut sz, mut sxz, mut syz, mut sxx, mut syy) = (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for y in 0..wh {
        for x in 0..ww {
            let z = luma[(y0 + y) * stride + x0 + x];
            let (dx, dy) = (x as f64 - cx, y as f64 - cy);
            sz += z;
            sxz += dx * z;
            syz += dy * z;
            sxx += dx * dx;
            syy += dy * dy;
        }
    }
    let (a, b, c) = (sz / n, sxz / sxx, syz / syy);
    let mut ss = 0.0f64;
    for y in 0..wh {
        for x in 0..ww {
            let z = luma[(y0 + y) * stride + x0 + x];
            let r = z - (a + b * (x as f64 - cx) + c * (y as f64 - cy));
            ss += r * r;
        }
    }
    (ss / n).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A slanted edge blurred by a known Gaussian: the measures must match the analytic values.
    fn gaussian_edge(size: usize, sigma: f64, angle_deg: f64) -> (Vec<f64>, EdgeGeometry) {
        let theta = angle_deg.to_radians();
        let n = (theta.cos(), theta.sin());
        let c = (size as f64 / 2.0, size as f64 / 2.0);
        let mut v = vec![0.0f64; size * size];
        for y in 0..size {
            for x in 0..size {
                let d = n.0 * (x as f64 + 0.5 - c.0) + n.1 * (y as f64 + 0.5 - c.1);
                // The ESF of a Gaussian blur of a unit step is its integrated Gaussian (erf).
                v[y * size + x] = 0.1 + 0.4 * 0.5 * (1.0 + erf(d / (sigma * 2f64.sqrt())));
            }
        }
        (
            v,
            EdgeGeometry {
                normal: n,
                centre: c,
            },
        )
    }

    /// Abramowitz and Stegun 7.1.26, error under 1.5e-7.
    fn erf(x: f64) -> f64 {
        let t = 1.0 / (1.0 + 0.327_591_1 * x.abs());
        let y = 1.0
            - (((((1.061_405_429 * t - 1.453_152_027) * t) + 1.421_413_741) * t - 0.284_496_736)
                * t
                + 0.254_829_592)
                * t
                * (-x * x).exp();
        if x >= 0.0 { y } else { -y }
    }

    #[test]
    fn a_gaussian_edge_gives_the_analytic_rise_and_mtf() {
        for sigma in [1.0f64, 1.5, 2.5] {
            let (luma, g) = gaussian_edge(160, sigma, 5.0);
            let s = edge_stats(&luma, 160, 160, &g, 32);
            // 10-90 % rise of a Gaussian-blurred step: 2 * 1.2816 sigma.
            assert!(
                (s.rise - 2.563 * sigma).abs() < 0.12,
                "sigma {sigma}: rise {}",
                s.rise
            );
            // MTF of a Gaussian: exp(-2 pi^2 sigma^2 f^2) = 0.5 at f = sqrt(ln 2 / 2) / (pi sigma).
            let want = (std::f64::consts::LN_2 / 2.0).sqrt() / (PI * sigma);
            assert!(
                (s.mtf50 - want).abs() / want < 0.03,
                "sigma {sigma}: mtf50 {} vs {want}",
                s.mtf50
            );
            assert!(s.overshoot_pct < 0.5 && s.undershoot_pct < 0.5);
        }
    }

    #[test]
    fn overshoot_and_undershoot_are_read_from_the_profile() {
        // A step with a ringing profile: 8 % over and 5 % under.
        let size = 128;
        let n = (1.0, 0.0);
        let c = (64.0, 64.0);
        let mut luma = vec![0.0f64; size * size];
        for y in 0..size {
            for x in 0..size {
                let d = x as f64 + 0.5 - c.0;
                let step = if d > 0.0 { 1.0 } else { 0.0 };
                let ring = if (0.0..2.0).contains(&d) {
                    0.08
                } else if (-2.0..0.0).contains(&d) {
                    -0.05
                } else {
                    0.0
                };
                luma[y * size + x] = 0.2 + 0.6 * (step + ring);
            }
        }
        let g = EdgeGeometry {
            normal: n,
            centre: c,
        };
        let s = edge_stats(&luma, size, size, &g, 0);
        assert!((s.overshoot_pct - 8.0).abs() < 0.5, "{}", s.overshoot_pct);
        assert!((s.undershoot_pct - 5.0).abs() < 0.5, "{}", s.undershoot_pct);
    }

    #[test]
    fn the_geometry_of_a_tilted_edge_is_found() {
        let (luma, g) = gaussian_edge(96, 1.2, 12.0);
        let found = edge_geometry(&luma, 96, 96).expect("an edge");
        assert!(
            (found.normal.0 - g.normal.0).abs() < 0.01
                && (found.normal.1 - g.normal.1).abs() < 0.01
        );
        // The point lies on the edge: its distance from the true edge is under a pixel.
        let d =
            g.normal.0 * (found.centre.0 - g.centre.0) + g.normal.1 * (found.centre.1 - g.centre.1);
        assert!(d.abs() < 0.3, "{d}");
        // A flat window has none.
        assert!(edge_geometry(&vec![0.3; 64 * 64], 64, 64).is_none());
        // A bright-to-dark edge gets its normal flipped to point at the bright side.
        let flipped: Vec<f64> = luma.iter().map(|v| 0.6 - v).collect();
        let f = edge_geometry(&flipped, 96, 96).expect("an edge");
        assert!(f.normal.0 * g.normal.0 + f.normal.1 * g.normal.1 < -0.99);
    }

    #[test]
    fn a_detrended_sd_ignores_a_slow_gradient() {
        let w = 64;
        let mut v = vec![0.0f64; w * w];
        for y in 0..w {
            for x in 0..w {
                // A gradient of 0.3 across the window plus a checker of +-0.01.
                v[y * w + x] = 0.2
                    + 0.3 * (x + y) as f64 / (2.0 * w as f64)
                    + if (x + y) % 2 == 0 { 0.01 } else { -0.01 };
            }
        }
        let plain = patch_sd(&v, w, &[(8, 8, 48, 48)]);
        let flat = patch_sd_detrended(&v, w, (8, 8, 48, 48));
        assert!(plain > 0.03, "{plain}");
        assert!((flat - 0.01).abs() < 5e-4, "{flat}");
    }

    #[test]
    fn patch_sd_pools_windows_without_their_means() {
        let w = 20;
        let mut v = vec![0.0f64; w * 10];
        // Window one around 1.0 with deviation +-0.1, window two around 5.0 with +-0.1.
        for y in 0..10 {
            for x in 0..10 {
                v[y * w + x] = 1.0 + if (x + y) % 2 == 0 { 0.1 } else { -0.1 };
                v[y * w + 10 + x] = 5.0 + if (x + y) % 2 == 0 { 0.1 } else { -0.1 };
            }
        }
        let sd = patch_sd(&v, w, &[(0, 0, 10, 10), (10, 0, 10, 10)]);
        assert!((sd - 0.1).abs() < 1e-12, "{sd}");
    }
}
