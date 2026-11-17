//! What the two dev tools of S-17 share: reading a JPEG into the pipeline's picture, and the colour-gate teeth mask of S-16
//! (`docs/27` 5.6 at ab4c63cd) as the baseline.
#![allow(dead_code)]

use std::path::Path;

use ym_retouch::mediapipe::AWG4_TO_REC709;
use ym_retouch::{LinearAwg4Image, Mask, feather, mul_masks, smoothstep};

pub fn decode(path: &Path) -> (usize, usize, Vec<u8>) {
    let mut d = zune_jpeg::JpegDecoder::new(std::io::Cursor::new(
        std::fs::read(path).expect("read the photo"),
    ));
    let rgb = d.decode().expect("decode the photo");
    let info = d.info().expect("jpeg header");
    (usize::from(info.width), usize::from(info.height), rgb)
}

fn invert(m: [[f32; 3]; 3]) -> [[f32; 3]; 3] {
    let a = m.map(|r| r.map(f64::from));
    let det = a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
        - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
        + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);
    let c = |r: usize, c: usize| {
        let (r1, r2, c1, c2) = ((r + 1) % 3, (r + 2) % 3, (c + 1) % 3, (c + 2) % 3);
        a[r1][c1] * a[r2][c2] - a[r1][c2] * a[r2][c1]
    };
    let mut out = [[0.0f32; 3]; 3];
    for (r, row) in out.iter_mut().enumerate() {
        for (k, v) in row.iter_mut().enumerate() {
            *v = (c(k, r) / det) as f32;
        }
    }
    out
}

pub fn to_linear_awg4(w: usize, h: usize, rgb: &[u8]) -> LinearAwg4Image {
    let inv = invert(AWG4_TO_REC709);
    let decode = |v: u8| {
        let e = f64::from(v) / 255.0;
        (if e <= 0.04045 {
            e / 12.92
        } else {
            ((e + 0.055) / 1.055).powf(2.4)
        }) as f32
    };
    let data = rgb
        .chunks_exact(3)
        .map(|p| {
            let l = [decode(p[0]), decode(p[1]), decode(p[2])];
            [0, 1, 2].map(|r| inv[r][0] * l[0] + inv[r][1] * l[1] + inv[r][2] * l[2])
        })
        .collect();
    LinearAwg4Image::new(w as u32, h as u32, data).expect("image")
}

/// The teeth mask of S-16 (`docs/27` 5.6 at ab4c63cd), kept here as the baseline of the evaluation: three colour gates inside
/// the mouth's interior. `closed` adds the final feather of 5.6; without it the mask is what the pipeline now feathers itself.
pub fn old_teeth_mask(mouth: &Mask, image: &LinearAwg4Image, extent_px: f64, closed: bool) -> Mask {
    let gate = |pixel: [f32; 3], y_mouth: f64| -> u8 {
        let (r, g, b) = (
            f64::from(pixel[0]).max(0.0),
            f64::from(pixel[1]).max(0.0),
            f64::from(pixel[2]).max(0.0),
        );
        let y = (r + g + b) / 3.0;
        if y_mouth.is_nan() || y_mouth <= 0.0 || !y.is_finite() {
            return 0;
        }
        let max = r.max(g).max(b);
        let sat = if max > 0.0 {
            1.0 - r.min(g).min(b) / max
        } else {
            0.0
        };
        let bright = smoothstep(0.6, 1.0, y / y_mouth);
        let pale = 1.0 - smoothstep(0.55, 0.85, sat);
        let notred = 1.0 - smoothstep(0.12, 0.26, (r - g) / y.max(1e-4));
        (255.0 * bright * pale * notred).round().clamp(0.0, 255.0) as u8
    };
    let mouth = feather(mouth, -0.004 * extent_px).expect("feather");
    let w = image.width_px as usize;
    let mut ys: Vec<f64> = mouth
        .data
        .iter()
        .enumerate()
        .filter(|(_, v)| **v != 0)
        .map(|(i, _)| {
            let p = image.data[i];
            (f64::from(p[0]).max(0.0) + f64::from(p[1]).max(0.0) + f64::from(p[2]).max(0.0)) / 3.0
        })
        .collect();
    ys.sort_unstable_by(f64::total_cmp);
    let y_mouth = if ys.is_empty() {
        0.0
    } else {
        ys[((0.8 * ys.len() as f64).ceil() as usize).clamp(1, ys.len()) - 1]
    };
    let mut g = Mask::zeros(image.width_px, image.height_px).expect("mask");
    for (i, out) in g.data.iter_mut().enumerate() {
        if mouth.data[i] != 0 {
            *out = gate(image.data[i], y_mouth);
        }
    }
    let _ = w;
    let teeth = mul_masks(&mouth, &g).expect("mul");
    if closed {
        feather(&teeth, 0.002 * extent_px).expect("feather")
    } else {
        teeth
    }
}

/// Box-averaged shrink so that the long edge is at most `max` pixels.
pub fn shrink(w: usize, h: usize, rgb: &[u8], max: usize) -> (usize, usize, Vec<u8>) {
    let long = w.max(h);
    if long <= max {
        return (w, h, rgb.to_vec());
    }
    let f = max as f64 / long as f64;
    let (nw, nh) = (
        ((w as f64 * f).round() as usize).max(1),
        ((h as f64 * f).round() as usize).max(1),
    );
    let mut out = vec![0u8; nw * nh * 3];
    for y in 0..nh {
        let (y0, y1) = (
            y * h / nh,
            ((y + 1) * h).div_ceil(nh).min(h).max(y * h / nh + 1),
        );
        for x in 0..nw {
            let (x0, x1) = (
                x * w / nw,
                ((x + 1) * w).div_ceil(nw).min(w).max(x * w / nw + 1),
            );
            for c in 0..3 {
                let mut sum = 0u32;
                for yy in y0..y1 {
                    for xx in x0..x1 {
                        sum += u32::from(rgb[(yy * w + xx) * 3 + c]);
                    }
                }
                out[(y * nw + x) * 3 + c] = (sum / ((y1 - y0) * (x1 - x0)) as u32) as u8;
            }
        }
    }
    (nw, nh, out)
}
