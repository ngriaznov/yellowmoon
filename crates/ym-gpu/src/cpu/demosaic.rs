//! R6 classical demosaic, the preview and fallback path (`docs/05-raw-processing.md` section R6).
//!
//! Three methods, selectable with [`Method`]:
//!
//! * [`Method::Bilinear`]: plain bilinear interpolation. Cheap, soft, zippers on edges; it exists
//!   for the GPU spike (M0-07) and as the PSNR floor.
//! * [`Method::Menon`]: Menon, Andriani and Calvagno (2007), "Demosaicing with directional
//!   filtering and a posteriori decision" (DDFAPD), including the refining step. Ported from the
//!   `colour-demosaicing` Python package (`demosaicing_CFA_Bayer_Menon2007`), Copyright 2015
//!   Colour Developers, **BSD-3-Clause** (notice below). The port keeps the package's filters, its (one-sided) direction window,
//!   its mirror boundary handling and its order of operations, so the output matches the Python
//!   reference to f32 rounding (`tests/cpu_demosaic.rs` checks this against fixtures produced by
//!   the package).
//! * [`Method::Lmmse`]: a directional LMMSE colour-difference interpolation in the spirit of
//!   Zhang and Wu (2005), "Color demosaicking via directional linear minimum mean square-error
//!   estimation". Getreuer's IPOL reference code (BSD-2) could not be fetched here (ipol.im is
//!   unreachable), so this is our own implementation from the paper's idea: horizontal and
//!   vertical colour-difference estimates are Wiener-smoothed with local statistics and fused with
//!   weights inversely proportional to their estimated error variances. It is not a port of the
//!   IPOL code and is not expected to be bit-comparable with it.
//!
//! # Licence notice for the Menon port
//!
//! The port follows `colour-demosaicing` 0.2.7 (<https://github.com/colour-science/colour-demosaicing>),
//! distributed under the BSD-3-Clause licence, whose text is reproduced here as that licence requires
//! (and must also be listed in the repository's `NOTICE`):
//!
//! > Copyright 2015 Colour Developers
//! >
//! > Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
//! >
//! > 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
//! >
//! > 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
//! >
//! > 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
//! >
//! > THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS “AS IS” AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE
//!
//! All methods work for the four Bayer phases, mirror at the borders (parity-preserving), do not
//! clip, and are tile-consistent with the halos in [`Method::halo`]. The output keeps the input's
//! [`Placement`](super::Placement).

use rayon::prelude::*;

use super::filters::{self, BAND_ROWS};
use super::{CfaColor, Mosaic, RgbImage, mirror};

/// A classical demosaic method.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Method {
    /// Bilinear interpolation.
    Bilinear,
    /// Menon 2007 DDFAPD with the refining step.
    Menon,
    /// Directional LMMSE colour-difference interpolation (after Zhang and Wu 2005).
    Lmmse,
}

/// The default classical method: Menon, because the pipeline demosaics linear light, where Menon is
/// scale-invariant and LMMSE loses PSNR in dark images (see `docs/05-raw-processing.md` section R6).
pub const DEFAULT_METHOD: Method = Method::Menon;

impl Method {
    /// All methods.
    pub const ALL: [Method; 3] = [Method::Bilinear, Method::Menon, Method::Lmmse];

    /// Receptive-field radius in pixels: output pixel `p` depends on mosaic pixels within this
    /// Chebyshev distance of `p`. Verified by the perturbation test in `tests/cpu_demosaic.rs`.
    pub const fn halo(self) -> usize {
        match self {
            Method::Bilinear => BILINEAR_HALO,
            Method::Menon => MENON_HALO,
            Method::Lmmse => LMMSE_HALO,
        }
    }

    /// Short lower-case name.
    pub const fn name(self) -> &'static str {
        match self {
            Method::Bilinear => "bilinear",
            Method::Menon => "menon",
            Method::Lmmse => "lmmse",
        }
    }
}

/// Receptive-field radius of [`bilinear`].
pub const BILINEAR_HALO: usize = 1;
/// Receptive-field radius of [`menon`] (with the refining step).
pub const MENON_HALO: usize = 8;
/// Receptive-field radius of [`lmmse`].
pub const LMMSE_HALO: usize = 8;

/// Demosaics `m` with `method`, in parallel bands (`halo` rows of context per band).
pub fn demosaic(m: &Mosaic, method: Method) -> RgbImage {
    if m.width < 2 || m.height < 2 {
        return bilinear(m);
    }
    filters::run_mosaic_bands(m, method.halo(), BAND_ROWS, |band| {
        demosaic_whole(band, method)
    })
}

/// Demosaics a buffer as one piece (no banding). The reference the banded runner must match.
pub fn demosaic_whole(m: &Mosaic, method: Method) -> RgbImage {
    match method {
        Method::Bilinear => bilinear(m),
        Method::Menon => menon(m, true),
        Method::Lmmse => lmmse(m),
    }
}

#[inline(always)]
fn col_map(m: &Mosaic) -> Vec<u8> {
    let (w, h) = (m.width, m.height);
    let mut col = vec![0u8; w * h];
    for y in 0..h {
        for x in 0..w {
            col[y * w + x] = m.cfa.color_at(x, y).plane() as u8;
        }
    }
    col
}

const R: u8 = CfaColor::Red as u8;
const G: u8 = CfaColor::Green as u8;
const B: u8 = CfaColor::Blue as u8;

/// Bilinear demosaic.
pub fn bilinear(m: &Mosaic) -> RgbImage {
    let (w, h) = (m.width, m.height);
    let mut out = RgbImage::new(w, h);
    out.place = m.place;
    if w == 0 || h == 0 {
        return out;
    }
    let col = col_map(m);
    let n = w * h;
    let d = &m.data;
    let at = |x: isize, y: isize| d[mirror(y, h) * w + mirror(x, w)];
    let (r_plane, rest) = out.data.split_at_mut(n);
    let (g_plane, b_plane) = rest.split_at_mut(n);
    r_plane
        .par_chunks_mut(w)
        .zip(g_plane.par_chunks_mut(w))
        .zip(b_plane.par_chunks_mut(w))
        .enumerate()
        .for_each(|(y, ((rr, gg), bb))| {
            let yi = y as isize;
            for x in 0..w {
                let xi = x as isize;
                let v = d[y * w + x];
                let cross =
                    || 0.25 * (at(xi - 1, yi) + at(xi + 1, yi) + at(xi, yi - 1) + at(xi, yi + 1));
                let diag = || {
                    0.25 * (at(xi - 1, yi - 1)
                        + at(xi + 1, yi - 1)
                        + at(xi - 1, yi + 1)
                        + at(xi + 1, yi + 1))
                };
                let horiz = || 0.5 * (at(xi - 1, yi) + at(xi + 1, yi));
                let vert = || 0.5 * (at(xi, yi - 1) + at(xi, yi + 1));
                // The colour found left/right of a green photosite is the one that owns its row.
                let row_colour = |xx: isize| col[y * w + mirror(xx, w)];
                match col[y * w + x] {
                    R => {
                        rr[x] = v;
                        gg[x] = cross();
                        bb[x] = diag();
                    }
                    B => {
                        bb[x] = v;
                        gg[x] = cross();
                        rr[x] = diag();
                    }
                    _ => {
                        gg[x] = v;
                        if row_colour(xi + 1) == R {
                            rr[x] = horiz();
                            bb[x] = vert();
                        } else {
                            bb[x] = horiz();
                            rr[x] = vert();
                        }
                    }
                }
            }
        });
    out
}

// ---------------------------------------------------------------------------------------------
// Menon 2007 (DDFAPD)
// ---------------------------------------------------------------------------------------------

/// Menon 2007 DDFAPD demosaic (port of `colour_demosaicing.demosaicing_CFA_Bayer_Menon2007`).
/// `refine` selects the refining step (the package default and ours).
pub fn menon(m: &Mosaic, refine: bool) -> RgbImage {
    let (w, h) = (m.width, m.height);
    let mut out = RgbImage::new(w, h);
    out.place = m.place;
    if w < 3 || h < 3 {
        return bilinear(m);
    }
    let n = w * h;
    let cfa = &m.data;
    let col = col_map(m);
    let wi = w as isize;
    let hi = h as isize;
    let idx = |x: isize, y: isize| mirror(y, h) * w + mirror(x, w);

    // Directional green estimates and colour differences at red / blue photosites.
    let mut g_h = vec![0.0f32; n];
    let mut g_v = vec![0.0f32; n];
    let mut c_h = vec![0.0f32; n];
    let mut c_v = vec![0.0f32; n];
    for y in 0..hi {
        for x in 0..wi {
            let i = (y as usize) * w + x as usize;
            if col[i] == G {
                g_h[i] = cfa[i];
                g_v[i] = cfa[i];
            } else {
                // h_0 + h_1 = [-1/4, 1/2, 1/2, 1/2, -1/4]
                let gh = 0.5 * (cfa[idx(x - 1, y)] + cfa[idx(x + 1, y)]) + 0.5 * cfa[i]
                    - 0.25 * (cfa[idx(x - 2, y)] + cfa[idx(x + 2, y)]);
                let gv = 0.5 * (cfa[idx(x, y - 1)] + cfa[idx(x, y + 1)]) + 0.5 * cfa[i]
                    - 0.25 * (cfa[idx(x, y - 2)] + cfa[idx(x, y + 2)]);
                g_h[i] = gh;
                g_v[i] = gv;
                c_h[i] = cfa[i] - gh;
                c_v[i] = cfa[i] - gv;
            }
        }
    }

    // Directional gradients of the colour differences, two samples apart (reflect at the border).
    let mut d_h = vec![0.0f32; n];
    let mut d_v = vec![0.0f32; n];
    for y in 0..hi {
        for x in 0..wi {
            let i = (y as usize) * w + x as usize;
            d_h[i] = (c_h[i] - c_h[idx(x + 2, y)]).abs();
            d_v[i] = (c_v[i] - c_v[idx(x, y + 2)]).abs();
        }
    }
    drop(c_h);
    drop(c_v);

    // Direction decision: weighted sums of the gradients over the package's 5x5 kernel (zero padded).
    let get = |a: &[f32], x: isize, y: isize| -> f32 {
        if x < 0 || y < 0 || x >= wi || y >= hi {
            0.0
        } else {
            a[(y as usize) * w + x as usize]
        }
    };
    let mut horizontal = vec![false; n];
    let mut g = vec![0.0f32; n];
    for y in 0..hi {
        for x in 0..wi {
            let i = (y as usize) * w + x as usize;
            let dh = get(&d_h, x, y - 2)
                + get(&d_h, x, y + 2)
                + 3.0 * get(&d_h, x, y)
                + get(&d_h, x - 1, y - 1)
                + get(&d_h, x - 1, y + 1)
                + get(&d_h, x - 2, y - 2)
                + get(&d_h, x - 2, y + 2)
                + 3.0 * get(&d_h, x - 2, y);
            let dv = get(&d_v, x + 2, y)
                + get(&d_v, x + 2, y - 2)
                + get(&d_v, x + 1, y - 1)
                + 3.0 * get(&d_v, x, y)
                + 3.0 * get(&d_v, x, y - 2)
                + get(&d_v, x - 1, y - 1)
                + get(&d_v, x - 2, y)
                + get(&d_v, x - 2, y - 2);
            let hz = dv >= dh;
            horizontal[i] = hz;
            g[i] = if col[i] == G {
                cfa[i]
            } else if hz {
                g_h[i]
            } else {
                g_v[i]
            };
        }
    }
    drop((d_h, d_v, g_h, g_v));

    // Red / blue planes with only the measured samples.
    let mut r: Vec<f32> = (0..n)
        .map(|i| if col[i] == R { cfa[i] } else { 0.0 })
        .collect();
    let mut b: Vec<f32> = (0..n)
        .map(|i| if col[i] == B { cfa[i] } else { 0.0 })
        .collect();
    let row_has = |y: usize, c: u8| col[y * w] == c || col[y * w + 1] == c;

    // Red and blue at green photosites (colour-difference interpolation along the row / column).
    for y in 0..h {
        let red_row = row_has(y, R);
        let yi = y as isize;
        for x in 0..w {
            let i = y * w + x;
            if col[i] != G {
                continue;
            }
            let xi = x as isize;
            let (l, rt, u, d) = (
                idx(xi - 1, yi),
                idx(xi + 1, yi),
                idx(xi, yi - 1),
                idx(xi, yi + 1),
            );
            let hz = |p: &[f32]| 0.5 * (p[l] + p[rt]);
            let vt = |p: &[f32]| 0.5 * (p[u] + p[d]);
            if red_row {
                r[i] = g[i] + hz(&r) - hz(&g);
                b[i] = g[i] + vt(&b) - vt(&g);
            } else {
                r[i] = g[i] + vt(&r) - vt(&g);
                b[i] = g[i] + hz(&b) - hz(&g);
            }
        }
    }
    // Red at blue and blue at red photosites, along the decided direction.
    for y in 0..h {
        let yi = y as isize;
        for x in 0..w {
            let i = y * w + x;
            let xi = x as isize;
            let (l, rt, u, d) = (
                idx(xi - 1, yi),
                idx(xi + 1, yi),
                idx(xi, yi - 1),
                idx(xi, yi + 1),
            );
            if col[i] == B {
                r[i] = if horizontal[i] {
                    b[i] + 0.5 * (r[l] + r[rt]) - 0.5 * (b[l] + b[rt])
                } else {
                    b[i] + 0.5 * (r[u] + r[d]) - 0.5 * (b[u] + b[d])
                };
            } else if col[i] == R {
                b[i] = if horizontal[i] {
                    r[i] + 0.5 * (b[l] + b[rt]) - 0.5 * (r[l] + r[rt])
                } else {
                    r[i] + 0.5 * (b[u] + b[d]) - 0.5 * (r[u] + r[d])
                };
            }
        }
    }

    if refine {
        menon_refine(w, h, &col, &horizontal, &mut r, &mut g, &mut b);
    }

    let (rp, rest) = out.data.split_at_mut(n);
    let (gp, bp) = rest.split_at_mut(n);
    rp.copy_from_slice(&r);
    gp.copy_from_slice(&g);
    bp.copy_from_slice(&b);
    out
}

/// The refining step of Menon 2007 (`refining_step_Menon2007`).
fn menon_refine(
    w: usize,
    h: usize,
    col: &[u8],
    horizontal: &[bool],
    r: &mut [f32],
    g: &mut [f32],
    b: &mut [f32],
) {
    let n = w * h;
    let idx = |x: isize, y: isize| mirror(y, h) * w + mirror(x, w);
    let third = 1.0f32 / 3.0;
    let mean3_h = |p: &[f32], x: isize, y: isize| {
        third * (p[idx(x - 1, y)] + p[idx(x, y)] + p[idx(x + 1, y)])
    };
    let mean3_v = |p: &[f32], x: isize, y: isize| {
        third * (p[idx(x, y - 1)] + p[idx(x, y)] + p[idx(x, y + 1)])
    };
    let row_has = |y: usize, c: u8| col[y * w] == c || col[y * w + 1] == c;
    let col_has = |x: usize, c: u8| col[x] == c || col[w + x] == c;

    // Update of the green component at red / blue photosites.
    let rg: Vec<f32> = (0..n).map(|i| r[i] - g[i]).collect();
    let bg: Vec<f32> = (0..n).map(|i| b[i] - g[i]).collect();
    let mut g_new = g.to_vec();
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            let (xi, yi) = (x as isize, y as isize);
            match col[i] {
                R => {
                    let m = if horizontal[i] {
                        mean3_h(&rg, xi, yi)
                    } else {
                        mean3_v(&rg, xi, yi)
                    };
                    g_new[i] = r[i] - m;
                }
                B => {
                    let m = if horizontal[i] {
                        mean3_h(&bg, xi, yi)
                    } else {
                        mean3_v(&bg, xi, yi)
                    };
                    g_new[i] = b[i] - m;
                }
                _ => {}
            }
        }
    }
    g.copy_from_slice(&g_new);

    // Update of red and blue at green photosites, using the new green.
    let rg: Vec<f32> = (0..n).map(|i| r[i] - g[i]).collect();
    let bg: Vec<f32> = (0..n).map(|i| b[i] - g[i]).collect();
    for y in 0..h {
        let (blue_row, red_row) = (row_has(y, B), row_has(y, R));
        for x in 0..w {
            let i = y * w + x;
            if col[i] != G {
                continue;
            }
            let (xi, yi) = (x as isize, y as isize);
            let (blue_col, red_col) = (col_has(x, B), col_has(x, R));
            let hz = |p: &[f32]| 0.5 * (p[idx(xi - 1, yi)] + p[idx(xi + 1, yi)]);
            let vt = |p: &[f32]| 0.5 * (p[idx(xi, yi - 1)] + p[idx(xi, yi + 1)]);
            if blue_row {
                r[i] = g[i] + vt(&rg);
            }
            if blue_col {
                r[i] = g[i] + hz(&rg);
            }
            if red_row {
                b[i] = g[i] + vt(&bg);
            }
            if red_col {
                b[i] = g[i] + hz(&bg);
            }
        }
    }

    // Update of red at blue photosites and blue at red photosites.
    let rb: Vec<f32> = (0..n).map(|i| r[i] - b[i]).collect();
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            let (xi, yi) = (x as isize, y as isize);
            match col[i] {
                B => {
                    let m = if horizontal[i] {
                        mean3_h(&rb, xi, yi)
                    } else {
                        mean3_v(&rb, xi, yi)
                    };
                    r[i] = b[i] + m;
                }
                R => {
                    let m = if horizontal[i] {
                        mean3_h(&rb, xi, yi)
                    } else {
                        mean3_v(&rb, xi, yi)
                    };
                    b[i] = r[i] - m;
                }
                _ => {}
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Directional LMMSE
// ---------------------------------------------------------------------------------------------

/// Half-width of the Gaussian window of the LMMSE statistics (9 taps).
const LMMSE_WIN: usize = 4;
/// Standard deviation of that window in samples.
const LMMSE_SIGMA: f32 = 2.0;
/// Regularisation added to the signal and noise variances (picked by PSNR on Kodak and McMaster: 1e-8 gave 39.00 / 35.02 dB, 1e-5 gives 39.16 / 35.09, 1e-4 gives 38.96 / 35.02).
const LMMSE_EPS: f32 = 1e-5;

/// Per-row estimate of the colour difference `G - X` along the rows of `data` (`w * h`, pattern
/// `cfa`), Wiener-smoothed with local statistics. Returns the estimate and its error variance.
fn lmmse_rows(data: &[f32], w: usize, h: usize, col: &[u8]) -> (Vec<f32>, Vec<f32>) {
    let n = w * h;
    let kern = filters::gaussian_kernel(LMMSE_SIGMA, LMMSE_WIN);
    let mut est = vec![0.0f32; n];
    let mut var = vec![0.0f32; n];
    let pad = LMMSE_WIN + 1;
    est.par_chunks_mut(w)
        .zip(var.par_chunks_mut(w))
        .enumerate()
        .for_each(|(y, (erow, vrow))| {
            let row = &data[y * w..(y + 1) * w];
            let crow = &col[y * w..(y + 1) * w];
            let at = |x: isize| row[mirror(x, w)];
            // Colour difference G - X at every photosite, from the [-1 2 2 2 -1]/4 filter.
            let mut sig = vec![0.0f32; w];
            for x in 0..w {
                let xi = x as isize;
                let f = 0.5 * (at(xi - 1) + at(xi + 1)) + 0.5 * row[x]
                    - 0.25 * (at(xi - 2) + at(xi + 2));
                sig[x] = if crow[x] == G { row[x] - f } else { f - row[x] };
            }
            // Mirror-padded copy so the 9-tap windows below need no index mapping.
            let sp: Vec<f32> = (0..w + 2 * pad)
                .map(|i| sig[mirror(i as isize - pad as isize, w)])
                .collect();
            // Second difference energy: interpolation errors alternate in sign between neighbours.
            let hp2: Vec<f32> = (0..w + 2 * pad)
                .map(|i| {
                    if i == 0 || i + 1 >= sp.len() {
                        0.0
                    } else {
                        let hp = sp[i] - 0.5 * (sp[i - 1] + sp[i + 1]);
                        hp * hp
                    }
                })
                .collect();
            for x in 0..w {
                // Window of 2 * LMMSE_WIN + 1 samples centred on x (padded index x + pad).
                let win_s = &sp[x + pad - LMMSE_WIN..=x + pad + LMMSE_WIN];
                let win_h = &hp2[x + pad - LMMSE_WIN..=x + pad + LMMSE_WIN];
                let mut mean = 0.0f32;
                for (&kw, &sv) in kern.iter().zip(win_s) {
                    mean += kw * sv;
                }
                let mut total = 0.0f32;
                let mut noise = 0.0f32;
                for ((&kw, &sv), &hv) in kern.iter().zip(win_s).zip(win_h) {
                    let d = sv - mean;
                    total += kw * d * d;
                    noise += kw * hv;
                }
                noise *= 2.0 / 3.0;
                let signal = (total - noise).max(0.0);
                let kappa = signal / (signal + noise + LMMSE_EPS);
                erow[x] = mean + kappa * (sig[x] - mean);
                vrow[x] = kappa * noise + LMMSE_EPS;
            }
        });
    (est, var)
}

fn transpose(src: &[f32], w: usize, h: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; w * h];
    out.par_chunks_mut(h).enumerate().for_each(|(x, orow)| {
        for (y, o) in orow.iter_mut().enumerate() {
            *o = src[y * w + x];
        }
    });
    out
}

/// Directional LMMSE demosaic (after Zhang and Wu 2005; see the module documentation).
pub fn lmmse(m: &Mosaic) -> RgbImage {
    let (w, h) = (m.width, m.height);
    let mut out = RgbImage::new(w, h);
    out.place = m.place;
    if w < 5 || h < 5 {
        return bilinear(m);
    }
    let n = w * h;
    let cfa = &m.data;
    let col = col_map(m);

    // Horizontal pass on the image, vertical pass on its transpose.
    let (est_h, var_h) = lmmse_rows(cfa, w, h, &col);
    let cfa_t = transpose(cfa, w, h);
    let col_t = {
        let colf: Vec<f32> = col.iter().map(|&c| f32::from(c)).collect();
        transpose(&colf, w, h)
            .iter()
            .map(|&c| c as u8)
            .collect::<Vec<u8>>()
    };
    let (est_vt, var_vt) = lmmse_rows(&cfa_t, h, w, &col_t);
    let est_v = transpose(&est_vt, h, w);
    let var_v = transpose(&var_vt, h, w);

    // Fuse the two directions with inverse-variance weights and form the green plane.
    let mut g = vec![0.0f32; n];
    for i in 0..n {
        g[i] = if col[i] == G {
            cfa[i]
        } else {
            let (ph, pv) = (var_h[i], var_v[i]);
            let diff = (pv * est_h[i] + ph * est_v[i]) / (ph + pv);
            cfa[i] + diff
        };
    }
    drop((est_h, est_v, var_h, var_v, cfa_t, col_t));

    // Red and blue from interpolated colour differences R - G and B - G.
    let idx = |x: isize, y: isize| mirror(y, h) * w + mirror(x, w);
    let mut r = vec![0.0f32; n];
    let mut b = vec![0.0f32; n];
    let row_has = |y: usize, c: u8| col[y * w] == c || col[y * w + 1] == c;
    // Known samples first.
    for i in 0..n {
        match col[i] {
            R => r[i] = cfa[i],
            B => b[i] = cfa[i],
            _ => {}
        }
    }
    // Differences at known sites.
    let dr: Vec<f32> = (0..n)
        .map(|i| if col[i] == R { cfa[i] - g[i] } else { 0.0 })
        .collect();
    let db: Vec<f32> = (0..n)
        .map(|i| if col[i] == B { cfa[i] - g[i] } else { 0.0 })
        .collect();
    // First the green photosites (neighbours on the row / column are known samples).
    for y in 0..h {
        let red_row = row_has(y, R);
        for x in 0..w {
            let i = y * w + x;
            if col[i] != G {
                continue;
            }
            let (xi, yi) = (x as isize, y as isize);
            let hz = |p: &[f32]| 0.5 * (p[idx(xi - 1, yi)] + p[idx(xi + 1, yi)]);
            let vt = |p: &[f32]| 0.5 * (p[idx(xi, yi - 1)] + p[idx(xi, yi + 1)]);
            if red_row {
                r[i] = g[i] + hz(&dr);
                b[i] = g[i] + vt(&db);
            } else {
                r[i] = g[i] + vt(&dr);
                b[i] = g[i] + hz(&db);
            }
        }
    }
    // Then red at blue and blue at red photosites: average the colour difference along the
    // diagonal with the smaller variation of green plus colour difference.
    let dr_full: Vec<f32> = (0..n).map(|i| r[i] - g[i]).collect();
    let db_full: Vec<f32> = (0..n).map(|i| b[i] - g[i]).collect();
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            if col[i] == G {
                continue;
            }
            let (xi, yi) = (x as isize, y as isize);
            let (nw, ne, sw, se) = (
                idx(xi - 1, yi - 1),
                idx(xi + 1, yi - 1),
                idx(xi - 1, yi + 1),
                idx(xi + 1, yi + 1),
            );
            let (plane, own) = if col[i] == B {
                (&dr_full, &mut r[i])
            } else {
                (&db_full, &mut b[i])
            };
            let g1 = (g[nw] - g[se]).abs() + (plane[nw] - plane[se]).abs();
            let g2 = (g[ne] - g[sw]).abs() + (plane[ne] - plane[sw]).abs();
            let w1 = 1.0 / (1e-4 + g1);
            let w2 = 1.0 / (1e-4 + g2);
            let d1 = 0.5 * (plane[nw] + plane[se]);
            let d2 = 0.5 * (plane[ne] + plane[sw]);
            *own = g[i] + (w1 * d1 + w2 * d2) / (w1 + w2);
        }
    }

    let (rp, rest) = out.data.split_at_mut(n);
    let (gp, bp) = rest.split_at_mut(n);
    rp.copy_from_slice(&r);
    gp.copy_from_slice(&g);
    bp.copy_from_slice(&b);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu::Cfa;
    use crate::cpu::synth;

    fn smooth_ramp(w: usize, h: usize) -> RgbImage {
        let mut img = RgbImage::new(w, h);
        for y in 0..h {
            for x in 0..w {
                img.set_pixel(
                    x,
                    y,
                    [
                        0.2 + 0.5 * x as f32 / w as f32,
                        0.3 + 0.4 * y as f32 / h as f32,
                        0.6 - 0.3 * (x + y) as f32 / (w + h) as f32,
                    ],
                );
            }
        }
        img
    }

    #[test]
    fn constant_image_is_reproduced_by_every_method_and_phase() {
        for cfa in Cfa::ALL {
            let mut img = RgbImage::new(24, 20);
            for (c, v) in [0.3f32, 0.5, 0.7].iter().enumerate() {
                img.plane_mut(c).fill(*v);
            }
            let m = synth::mosaic_from_rgb(&img, cfa);
            for method in Method::ALL {
                let out = demosaic_whole(&m, method);
                assert!(out.max_abs_diff(&img) < 1e-5, "{method:?} {}", cfa.name());
            }
        }
    }

    #[test]
    fn smooth_ramps_are_nearly_exact_in_the_interior() {
        for cfa in Cfa::ALL {
            let img = smooth_ramp(40, 32);
            let m = synth::mosaic_from_rgb(&img, cfa);
            for method in Method::ALL {
                let out = demosaic_whole(&m, method);
                let mut worst = 0.0f32;
                for y in 6..26 {
                    for x in 6..34 {
                        for c in 0..3 {
                            worst = worst.max((out.pixel(x, y)[c] - img.pixel(x, y)[c]).abs());
                        }
                    }
                }
                assert!(worst < 2e-3, "{method:?} {}: {worst}", cfa.name());
            }
        }
    }

    #[test]
    fn measured_samples_are_kept() {
        let img = synth::natural(48, 36, 4);
        for cfa in Cfa::ALL {
            let m = synth::mosaic_from_rgb(&img, cfa);
            for method in [Method::Bilinear, Method::Lmmse] {
                let out = demosaic_whole(&m, method);
                for y in 0..36 {
                    for x in 0..48 {
                        let c = cfa.color_at(x, y).plane();
                        assert_eq!(out.pixel(x, y)[c], m.get(x, y), "{method:?} at ({x},{y})");
                    }
                }
            }
        }
    }

    #[test]
    fn banded_equals_whole() {
        let img = synth::natural(70, 300, 8);
        let m = synth::mosaic_from_rgb(&img, Cfa::GBRG);
        for method in Method::ALL {
            let whole = demosaic_whole(&m, method);
            let banded = demosaic(&m, method);
            assert!(
                whole.max_abs_diff(&banded) < 1e-5,
                "{method:?}: {}",
                whole.max_abs_diff(&banded)
            );
        }
    }
}
