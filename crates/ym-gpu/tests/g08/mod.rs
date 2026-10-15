//! Helpers shared by the G-08 test files (`gpu_resize.rs`, `gpu_tail.rs`): the CPU reference of an
//! export (sampler, colour chain, Lanczos-3 resize, output sharpening, C13) written from the same
//! pieces the CPU export uses, scenes and grids, and error statistics that treat a NaN as an
//! infinite error.
//!
//! A directory module (`tests/g08/mod.rs`), so cargo does not build it as a test binary of its own;
//! the files include it with `#[path = "g08/mod.rs"] mod g08;`.

#![allow(dead_code)]

use std::io::Write as _;
use std::sync::Arc;

use ym_color::display::{ColorChain, Frame, Layout};
use ym_color::encode::{self, OutputSpace};
use ym_gpu::cpu::output_sharpen::{OutputSharpen, scale_for_edges};
use ym_gpu::cpu::sampler::{SamplerGrid, sample_to_camera};
use ym_gpu::cpu::{RgbImage, synth};
use ym_gpu::kernels::resize::Taps;

/// A line for the log and, on GitHub Actions, an annotation on the run's summary page: the numbers a
/// Metal runner measured (the first Apple run supplies the hardware columns).
pub fn ci_note(what: &str, text: &str) {
    let mut err = std::io::stderr();
    let _ = writeln!(err, "[G-08] {text}");
    if std::env::var_os("GITHUB_ACTIONS").is_some() {
        let _ = writeln!(err, "::notice title=G-08 {what}::{text}");
    }
}

/// A picture to export: the camera image, the geometry that maps the picture's pixels to it, and
/// the picture's size.
pub struct Scene {
    pub camera: Arc<RgbImage>,
    pub grid: SamplerGrid,
    pub w: usize,
    pub h: usize,
}

fn swaps(exif: u16, turns: u8) -> bool {
    ((5..=8).contains(&exif)) != (turns % 2 == 1)
}

/// A natural scene in camera RGB (`synth::natural`, scene-linear values around 0.2 with a few
/// highlights).
pub fn camera(w: usize, h: usize, seed: u64) -> Arc<RgbImage> {
    let mut img = synth::natural(w, h, seed);
    for v in &mut img.data {
        *v *= 0.8;
    }
    Arc::new(img)
}

/// One buffer pixel per picture pixel, EXIF orientation `exif` and `turns` quarter turns.
pub fn exact_scene(bw: usize, bh: usize, exif: u16, turns: u8, seed: u64) -> Scene {
    let (w, h) = if swaps(exif, turns) {
        (bh, bw)
    } else {
        (bw, bh)
    };
    Scene {
        camera: camera(bw, bh, seed),
        grid: SamplerGrid::new(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            (w as f64, h as f64),
            (bw as f64, bh as f64),
            exif,
            turns,
            true,
        ),
        w,
        h,
    }
}

/// A picture straightened by `angle_deg` (so its corners are outside the photo: the letterbox) and
/// scaled by `scale`, `w x h` output pixels from a `bw x bh` buffer.
pub fn rotated_scene(
    bw: usize,
    bh: usize,
    (w, h): (usize, usize),
    angle_deg: f64,
    scale: f64,
    seed: u64,
) -> Scene {
    let (s, c) = (-angle_deg).to_radians().sin_cos();
    let (cx, cy) = (bw as f64 * 0.5, bh as f64 * 0.5);
    let t = (
        (bw as f64 - w as f64 * scale) * 0.5,
        (bh as f64 - h as f64 * scale) * 0.5,
    );
    let to_canvas = [
        [c * scale, -s * scale, c * (t.0 - cx) - s * (t.1 - cy) + cx],
        [s * scale, c * scale, s * (t.0 - cx) + c * (t.1 - cy) + cy],
    ];
    Scene {
        camera: camera(bw, bh, seed),
        grid: SamplerGrid::new(
            to_canvas,
            (bw as f64, bh as f64),
            (bw as f64, bh as f64),
            1,
            0,
            SamplerGrid::is_unrotated_angle(angle_deg),
        ),
        w,
        h,
    }
}

/// Whether every pixel of the `w x h` tile at `off` is outside the photo (such a tile reads nothing
/// of the camera image and an export does not add it).
pub fn tile_is_outside(sc: &Scene, off: (usize, usize), w: usize, h: usize) -> bool {
    let s = sample_to_camera(&sc.camera, (0, 0), &sc.grid, off, w, h);
    s.outside.is_some_and(|o| o.iter().all(|&v| v != 0))
}

/// The rendered picture the CPU export makes: sampler, colour chain C1 to C12, zero outside the
/// photo. Planar, `3 * w * h` samples.
pub fn cpu_rendered(sc: &Scene, chain: &ColorChain) -> Vec<f32> {
    let (w, h) = (sc.w, sc.h);
    let sampled = sample_to_camera(&sc.camera, (0, 0), &sc.grid, (0, 0), w, h);
    let frame = Frame::planar(w, h, &sampled.img.data).expect("frame");
    let mut out = vec![0.0f32; 3 * w * h];
    chain
        .render_display_linear_f32(&frame, Layout::Planar, &mut out, 1)
        .expect("render");
    if let Some(outside) = &sampled.outside {
        let n = w * h;
        for (i, &o) in outside.iter().enumerate() {
            if o != 0 {
                for c in 0..3 {
                    out[c * n + i] = 0.0;
                }
            }
        }
    }
    out
}

/// `ym_export::resize::resize_lanczos3` on a planar image: the horizontal pass into an `f32`
/// intermediate image, then the vertical pass, every sum in `f64`.
pub fn cpu_resize(src: &[f32], w: usize, h: usize, out_w: usize, out_h: usize) -> Vec<f32> {
    let xs = Taps::new(w, out_w);
    let ys = Taps::new(h, out_h);
    let mut mid = vec![0.0f32; 3 * out_w * h];
    for c in 0..3 {
        for y in 0..h {
            for x in 0..out_w {
                let (first, weights) = xs.row(x);
                let mut acc = 0.0f64;
                for (k, &wt) in weights.iter().enumerate() {
                    acc += wt * f64::from(src[c * w * h + y * w + first + k]);
                }
                mid[c * out_w * h + y * out_w + x] = acc as f32;
            }
        }
    }
    let mut out = vec![0.0f32; 3 * out_w * out_h];
    for c in 0..3 {
        for y in 0..out_h {
            let (first, weights) = ys.row(y);
            for x in 0..out_w {
                let mut acc = 0.0f64;
                for (k, &wt) in weights.iter().enumerate() {
                    acc += wt * f64::from(mid[c * out_w * h + (first + k) * out_w + x]);
                }
                out[c * out_w * out_h + y * out_w + x] = acc as f32;
            }
        }
    }
    out
}

/// Planar to interleaved.
pub fn interleave(planar: &[f32], w: usize, h: usize) -> Vec<f32> {
    let n = w * h;
    let mut out = vec![0.0f32; 3 * n];
    for i in 0..n {
        for c in 0..3 {
            out[3 * i + c] = planar[c * n + i];
        }
    }
    out
}

/// What the CPU export writes, as codes: the rendered picture, resized to `out` (when it differs),
/// sharpened with `sharpen_pct` (when above 0; the stage at the scale of the written long edge over
/// the rendered one) and encoded with C13 (8 bit dithered, or 16 bit). Interleaved RGB rows.
pub struct CpuCodes {
    pub eight: Option<Vec<u8>>,
    pub sixteen: Option<Vec<u16>>,
    pub out: (usize, usize),
}

pub fn cpu_codes(
    sc: &Scene,
    chain: &ColorChain,
    out: (usize, usize),
    sharpen_pct: f64,
    space: OutputSpace,
    bits16: bool,
) -> CpuCodes {
    let rendered = cpu_rendered(sc, chain);
    let planar = if out == (sc.w, sc.h) {
        rendered
    } else {
        cpu_resize(&rendered, sc.w, sc.h, out.0, out.1)
    };
    let mut rgb = interleave(&planar, out.0, out.1);
    if sharpen_pct > 0.0 {
        let scale = scale_for_edges(out.0.max(out.1), sc.w.max(sc.h));
        OutputSharpen::screen()
            .with_amount_pct(sharpen_pct as f32)
            .apply_interleaved_in_place(out.0, out.1, &mut rgb, scale)
            .expect("sharpen");
    }
    let mut eight = None;
    let mut sixteen = None;
    if bits16 {
        let mut v = Vec::with_capacity(rgb.len());
        for p in rgb.chunks_exact(3) {
            let lin = [f64::from(p[0]), f64::from(p[1]), f64::from(p[2])];
            v.extend_from_slice(&encode::encode_quantized(lin, space, 16));
        }
        sixteen = Some(v);
    } else {
        let mut v = Vec::with_capacity(rgb.len());
        for (i, p) in rgb.chunks_exact(3).enumerate() {
            let lin = [f64::from(p[0]), f64::from(p[1]), f64::from(p[2])];
            v.extend_from_slice(&encode::encode_8bit_dithered(
                lin,
                space,
                i % out.0,
                i / out.0,
            ));
        }
        eight = Some(v);
    }
    CpuCodes { eight, sixteen, out }
}

/// How two code arrays differ.
#[derive(Clone, Copy, Debug)]
pub struct CodeDiff {
    pub n: usize,
    pub exact: usize,
    pub max: u32,
}

impl CodeDiff {
    pub fn exact_share(&self) -> f64 {
        self.exact as f64 / self.n.max(1) as f64
    }
}

pub fn diff_codes(a: &[u32], b: &[u32]) -> CodeDiff {
    assert_eq!(a.len(), b.len(), "lengths differ");
    let mut d = CodeDiff {
        n: a.len(),
        exact: 0,
        max: 0,
    };
    for (&x, &y) in a.iter().zip(b) {
        let e = x.abs_diff(y);
        if e == 0 {
            d.exact += 1;
        }
        d.max = d.max.max(e);
    }
    d
}
