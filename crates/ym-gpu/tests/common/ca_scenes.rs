//! The CA scenes of `ca_green_residual` as a library of functions (`docs/29-raw-constants.md` RC-05): the
//! analytic grey scene, the lens tags, the mosaic made consistent with them, the five configurations
//! (three CA strengths without distortion, two with 1.5 % barrel distortion) and the variants that run on
//! them. `ca_green_residual` prints the tables from them and `fit_demosaic` takes the residual and the
//! PSNR of the CA path under a given `Tuning`. Ours: nothing here reads a dataset.

#![allow(dead_code)]

use std::time::Instant;

use rayon::prelude::*;
use ym_gpu::cpu::demosaic::adaptive_ratio::{self, Params, Tuning};
use ym_gpu::cpu::demosaic::{self, CaCurves, DemosaicInputs, Method};
use ym_gpu::cpu::lens::{LensGeometry, LensSwitches, LensTags};
use ym_gpu::cpu::metrics::{CaResidualOptions, lateral_ca_residual, psnr};
use ym_gpu::cpu::synth::Rng;
use ym_gpu::cpu::tiling::Rect;
use ym_gpu::cpu::{Cfa, Mosaic, NoiseDn, Placement, RgbImage, resample};
use ym_noise::model::NoiseModel;

/// An analytic grey scene: soft-edged level sets of two random sums of plane waves.
pub struct Scene {
    waves: Vec<[f64; 4]>,
    split: usize,
}

impl Scene {
    pub fn new(seed: u64) -> Scene {
        let mut rng = Rng::new(seed);
        let mut waves = Vec::new();
        // Coarse field (periods 18 to 200 px), then a finer one (periods 9 to 40 px).
        for (count, lo, hi) in [(8usize, 0.03f64, 0.35f64), (8, 0.15, 0.7)] {
            for _ in 0..count {
                let k = lo * (hi / lo).powf(f64::from(rng.range(0.0, 1.0)));
                let a = f64::from(rng.range(0.0, std::f32::consts::TAU));
                let phase = f64::from(rng.range(0.0, std::f32::consts::TAU));
                let amp = f64::from(rng.range(0.5, 1.0));
                waves.push([k * a.cos(), k * a.sin(), phase, amp]);
            }
        }
        Scene { waves, split: 8 }
    }

    /// The scene value at the continuous position `(x, y)` (pixel `i` covers `[i, i + 1)`).
    pub fn eval(&self, x: f64, y: f64) -> f64 {
        let level = |waves: &[[f64; 4]]| {
            let (mut g, mut gx, mut gy) = (0.0, 0.0, 0.0);
            for w in waves {
                let arg = w[0] * x + w[1] * y + w[2];
                g += w[3] * arg.cos();
                gx -= w[3] * w[0] * arg.sin();
                gy -= w[3] * w[1] * arg.sin();
            }
            // Signed distance to the level set, to first order, in pixels: a uniform edge width.
            let d = g / (gx * gx + gy * gy + 0.02f64 * 0.02).sqrt();
            1.0 / (1.0 + (-d / 0.55).exp())
        };
        let (coarse, fine) = self.waves.split_at(self.split);
        let v = 0.5 + 0.30 * (level(coarse) - 0.5) * 2.0 + 0.16 * (level(fine) - 0.5) * 2.0;
        v.clamp(0.04, 0.96)
    }
}

/// The lens tags for a corner shift of about `strength * 1.2` px for red and `strength * 1.0` px for
/// blue on a frame of half diagonal `hd`: 8 knots per colour, K - 1 growing with the square of the
/// radius; optionally a barrel correction of 1.5 % at the corner.
pub fn tags_for(strength: f64, hd: f64, distortion: bool) -> LensTags {
    let n = 8usize;
    let rho = |j: usize| (j as f64 + 0.5) / (n as f64 - 1.0);
    let knot = |amount_px: f64, j: usize| -> i16 {
        let k_minus_1 = amount_px / hd * rho(j) * rho(j);
        (k_minus_1 * 2f64.powi(21)).round() as i16
    };
    let mut ca: Vec<i16> = (0..n).map(|j| knot(1.2 * strength, j)).collect();
    ca.extend((0..n).map(|j| knot(-strength, j)));
    let dist = if distortion {
        (0..n)
            .map(|j| (0.015 * rho(j) * rho(j) * 2f64.powi(14)).round() as i16)
            .collect()
    } else {
        vec![]
    };
    LensTags {
        vignetting: vec![],
        ca,
        distortion: dist,
    }
}

/// The source position `q` of colour `c` is sampled from the scene at the output position `p` with
/// `source_position(c, p) = q`: found by iterating `p <- p + (q - source_position(c, p))`.
pub fn mosaic_of_scene(scene: &Scene, geo: &LensGeometry, cfa: Cfa, noise_sigma: f32) -> Mosaic {
    let (w, h) = (geo.frame_width(), geo.frame_height());
    let mut m = Mosaic::new(w, h, cfa);
    m.place = Placement::whole(w, h);
    m.data.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        for (x, v) in row.iter_mut().enumerate() {
            let c = cfa.color_at(x, y).plane();
            let q = (x as f64 + 0.5, y as f64 + 0.5);
            let mut p = q;
            for _ in 0..8 {
                let s = geo.source_position(c, p.0, p.1);
                p = (p.0 + (q.0 - s.0), p.1 + (q.1 - s.1));
            }
            *v = scene.eval(p.0, p.1) as f32;
        }
    });
    if noise_sigma > 0.0 {
        let mut rng = Rng::new(99);
        for v in &mut m.data {
            *v += noise_sigma * rng.gaussian();
        }
    }
    m
}

/// One variant: the image after R9, the demosaic's CPU seconds.
pub struct Run {
    /// The image after R9.
    pub out: RgbImage,
    /// CPU seconds of the demosaic.
    pub seconds: f64,
}

/// Demosaics `m` with `method`, then runs R9 with `geo_r9`. The adaptive ratio runs under `tuning`
/// (what `demosaic::demosaic_with` does at `Tuning::DEFAULT`, which is what `ca_green_residual` passes).
pub fn run_variant(
    m: &Mosaic,
    method: Method,
    ca: Option<&CaCurves>,
    noise: NoiseDn,
    geo_r9: &LensGeometry,
    tuning: &Tuning,
) -> Run {
    let inputs = DemosaicInputs {
        gains: [1.0; 3],
        noise,
        ca,
    };
    let t = Instant::now();
    let img = if method == Method::AdaptiveRatio {
        let params = Params {
            tuning: *tuning,
            ..Params::default()
        };
        adaptive_ratio::demosaic_balanced(m, &inputs, &params)
            .unwrap_or_else(|_| demosaic::demosaic_with(m, Method::Menon, &inputs))
    } else {
        demosaic::demosaic_with(m, method, &inputs)
    };
    let seconds = t.elapsed().as_secs_f64();
    let (cw, ch) = geo_r9.output_size();
    let out = resample::resample_lens(&img, geo_r9, Rect::whole(cw, ch)).expect("R9");
    Run { out, seconds }
}

/// The noise model of the CA scenes: Gaussian of `noise_sigma` in the signal-independent term, or the
/// nominal low-noise model of spec 01 section 6.3 when it is 0.
pub fn ca_noise_model(noise_sigma: f32) -> NoiseModel {
    if noise_sigma > 0.0 {
        NoiseModel {
            a: 0.0,
            b: f64::from(noise_sigma) * f64::from(noise_sigma),
            delta: [0.0; 4],
            sigma_row: 0.0,
            q: 0.0,
        }
    } else {
        NoiseModel {
            a: 2e-5,
            b: 1e-8,
            delta: [0.0; 4],
            sigma_row: 0.0,
            q: 1.0 / 65535.0,
        }
    }
}

/// The five configurations of `ca_green_residual`: `(strength, barrel distortion)`.
pub const CA_CONFIGS: [(f64, bool); 5] = [
    (0.5, false),
    (1.0, false),
    (2.0, false),
    (1.0, true),
    (2.0, true),
];

/// One configuration: the lens, the mosaic made consistent with its tags and the ideal output.
pub struct CaScene {
    /// Corner shift of red against blue, in units of the table (0.5, 1, 2).
    pub strength: f64,
    /// Whether the lens also has 1.5 % barrel distortion.
    pub distortion: bool,
    /// Frame width and height.
    pub frame: (usize, usize),
    /// Half the frame diagonal, pixels.
    pub half_diagonal: f64,
    /// R9 geometry with the CA correction.
    pub geo: LensGeometry,
    /// R9 geometry without it (the scale reference).
    pub geo_no_ca: LensGeometry,
    /// The CA curves of the lens.
    pub curves: CaCurves,
    /// Optical centre in output pixels.
    pub centre: (f64, f64),
    /// `|R - B|` at the corner, pixels.
    pub corner_px: f64,
    /// The mosaic.
    pub mosaic: Mosaic,
    /// The ideal output (grey).
    pub truth: RgbImage,
}

/// Builds the scene of one configuration (`strength`, `distortion`) at frame size `frame`.
pub fn ca_scene(
    scene: &Scene,
    frame: (usize, usize),
    strength: f64,
    distortion: bool,
    noise_sigma: f32,
) -> CaScene {
    let (fw, fh) = frame;
    let hd = 0.5 * (fw as f64).hypot(fh as f64);
    let tags = tags_for(strength, hd, distortion);
    let sw_ca = LensSwitches::default();
    let sw_no_ca = LensSwitches {
        lens_corrections: true,
        chromatic_aberration: false,
    };
    let geo_src = LensGeometry::new(fw, fh, &tags, sw_ca).expect("lens");
    let geo = geo_src.clone().with_auto_crop();
    let geo_no_ca = LensGeometry::new(fw, fh, &tags, sw_no_ca)
        .expect("lens")
        .with_crop(geo.crop());
    let curves = geo.ca_curves().expect("curves").expect("the lens has CA");
    let crop = geo.crop();
    let centre = (
        fw as f64 / 2.0 - crop.x as f64,
        fh as f64 / 2.0 - crop.y as f64,
    );
    let corner_px = curves.misregistration_px(hd);
    let mosaic = mosaic_of_scene(scene, &geo_src, Cfa::RGGB, noise_sigma);
    // Ideal output: the scene at the output pixel centres of the cropped output.
    let (cw, ch) = geo.output_size();
    let mut truth = RgbImage::new(cw, ch);
    for y in 0..ch {
        for x in 0..cw {
            let v = scene.eval((crop.x + x) as f64 + 0.5, (crop.y + y) as f64 + 0.5) as f32;
            for c in 0..3 {
                truth.plane_mut(c)[y * cw + x] = v;
            }
        }
    }
    CaScene {
        strength,
        distortion,
        frame,
        half_diagonal: hd,
        geo,
        geo_no_ca,
        curves,
        centre,
        corner_px,
        mosaic,
        truth,
    }
}

/// The five scenes of `ca_green_residual` for `seed` at `frame` size.
pub fn ca_scene_set(frame: (usize, usize), noise_sigma: f32, seed: u64) -> Vec<CaScene> {
    let scene = Scene::new(seed);
    CA_CONFIGS
        .iter()
        .map(|&(s, d)| ca_scene(&scene, frame, s, d, noise_sigma))
        .collect()
}

/// The measure of the CA path on one scene: the residual of red and blue against green after R9
/// (RMS over the patches of `lateral_ca_residual`, pixels) and the PSNR of all three planes.
#[derive(Clone, Copy, Debug)]
pub struct CaPathResult {
    /// Residual of red, pixels.
    pub red_rms_px: f64,
    /// Residual of blue, pixels.
    pub blue_rms_px: f64,
    /// PSNR of the three planes against the ideal scene, 16 px border off, dB.
    pub psnr_rgb_db: f64,
    /// The image after R9.
    pub out_seconds: f64,
}

/// Runs the adaptive ratio with the CA path under `tuning` on `s` and measures it.
pub fn ca_path_result(s: &CaScene, noise: NoiseDn, tuning: &Tuning) -> (CaPathResult, RgbImage) {
    let opts = CaResidualOptions {
        patch_px: 48,
        stride_px: 48,
        ..CaResidualOptions::default()
    };
    let run = run_variant(
        &s.mosaic,
        Method::AdaptiveRatio,
        Some(&s.curves),
        noise,
        &s.geo,
        tuning,
    );
    let r = lateral_ca_residual(&run.out, s.centre, s.half_diagonal, &opts);
    let res = CaPathResult {
        red_rms_px: r.red_rms_px,
        blue_rms_px: r.blue_rms_px,
        psnr_rgb_db: psnr(&run.out, &s.truth, 16, 1.0),
        out_seconds: run.seconds,
    };
    (res, run.out)
}

/// The colour error at the edges of the scene: mean `|R - G|` and mean `|B - G|` over the pixels
/// (16 px border off) where the ideal green's gradient exceeds 0.03 per pixel.
pub fn edge_chroma(out: &RgbImage, truth_g: &[f32]) -> (f64, f64) {
    let (w, h) = (out.width, out.height);
    let (mut sr, mut sb, mut n) = (0.0f64, 0.0f64, 0usize);
    for y in 16..h - 16 {
        for x in 16..w - 16 {
            let gx = f64::from(truth_g[y * w + x + 1] - truth_g[y * w + x - 1]) * 0.5;
            let gy = f64::from(truth_g[(y + 1) * w + x] - truth_g[(y - 1) * w + x]) * 0.5;
            if gx.hypot(gy) < 0.03 {
                continue;
            }
            let i = y * w + x;
            let g = f64::from(out.plane(1)[i]);
            sr += (f64::from(out.plane(0)[i]) - g).abs();
            sb += (f64::from(out.plane(2)[i]) - g).abs();
            n += 1;
        }
    }
    (sr / n.max(1) as f64, sb / n.max(1) as f64)
}

/// `img` with all three planes equal to plane `c` of `src` (for the PSNR of one plane).
pub fn grey_of(src: &RgbImage, c: usize) -> RgbImage {
    let mut g = RgbImage::new(src.width, src.height);
    for k in 0..3 {
        g.plane_mut(k).copy_from_slice(src.plane(c));
    }
    g
}
