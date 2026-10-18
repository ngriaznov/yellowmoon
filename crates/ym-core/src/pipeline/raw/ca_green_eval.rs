//! The CA-aware green refinement of the adaptive ratio demosaic through the whole render
//! (`docs/algorithms/01-adaptive-ratio-demosaic.md` section 3.6, `INTEGRATION.md` section 6).
//!
//! * `ca_tables_in_the_file_drive_the_ca_path_of_the_full_chain`: a synthetic ARW with CA tables
//!   (made by `ym_raw::testgen`) goes through `RawPlan` and `FullChain`. Since the real-file
//!   evaluation of R9 (`docs/05` R9, "Real files") the plan takes the CA out of R9 and the chain
//!   corrects it on the mosaic (R5b, `lens::correct_mosaic_ca`) before the demosaic, for every
//!   method, exactly when the file has a non-neutral 0x7035 table *and*
//!   `decode.chromatic_aberration` is on (checked against R1, R4, R5b, a whole-frame
//!   `demosaic_whole_with` and R9 done by hand). The demosaic then gets no CA curves: the CA-aware
//!   path of the adaptive ratio demosaic (section 3.6) is not reached from the chain any more, and
//!   stays covered by its own tests (`ym-gpu/tests/cpu_adaptive_ratio_ca.rs`).
//! * `the_ca_path_leaves_no_more_ca_after_r9_on_a_consistent_synthetic_file`: the mosaic of that
//!   file is made *consistent with its tables* (every site holds the scene at the position R9
//!   will move it from), so the correct output has no colour fringes at all. The displacement of
//!   red and blue against green left after R9 (`metrics::lateral_ca_residual`) must be no larger
//!   with the CA path than without it.
//! * `ca_green_real_files_residual` (ignored, local files only): the same measurement on the
//!   photographs of `testdata/real/files` that carry a 0x7035 table, patch by patch, with the CA
//!   path and without it, and with R9 *without* the CA correction as the scale of what there was
//!   to correct. Results go to `testdata/ca_green_residual/real/results.json` (git-ignored):
//!
//! ```text
//! cargo test -p ym-core --lib ca_green_real_files -- --ignored --nocapture
//! ```

use std::path::{Path, PathBuf};

use serde_json::{Value, json};
use ym_gpu::cpu::demosaic::{self, CaCurves, DemosaicInputs};
use ym_gpu::cpu::metrics::{
    CaResidual, CaResidualOptions, lateral_ca_residual, sample_catmull_rom,
};
use ym_gpu::cpu::{gains, resample, synth};
use ym_raw::testgen::{ArwBuilder, LensTables};

use super::*;
use crate::auto::{AutoEstimates, AutoInputs};
use crate::camera::PhotoCamera;
use crate::session::SessionEnv;

/// Black and white level of the synthetic files (14 bits).
const BLACK: u16 = 512;
const WHITE: u16 = 16000;

/// The stored lens tables of a synthetic file: no vignetting, a CA table of 8 knots per colour
/// whose `K - 1` grows with the square of the radius to `corner_px` of shift for red and
/// `-0.85 * corner_px` for blue at the corner of a frame of half diagonal `hd`, and optionally a
/// barrel distortion of 1.5 % at the corner. Returns the stored tables and the same knots as
/// `LensTags` (without the leading counts).
fn ca_tables(corner_px: f64, hd: f64, distortion: bool) -> (LensTables, LensTags) {
    let n = 8usize;
    let rho = |j: usize| (j as f64 + 0.5) / (n as f64 - 1.0);
    let knot = |amount_px: f64, j: usize| -> i16 {
        (amount_px / hd * rho(j) * rho(j) * 2f64.powi(21)).round() as i16
    };
    let mut ca: Vec<i16> = (0..n).map(|j| knot(corner_px, j)).collect();
    ca.extend((0..n).map(|j| knot(-0.85 * corner_px, j)));
    let dist: Vec<i16> = if distortion {
        (0..n)
            .map(|j| (0.015 * rho(j) * rho(j) * 2f64.powi(14)).round() as i16)
            .collect()
    } else {
        vec![]
    };
    let lead = |count: usize, knots: &[i16]| -> Option<Vec<i16>> {
        (!knots.is_empty()).then(|| {
            let mut v = vec![count as i16];
            v.extend_from_slice(knots);
            v
        })
    };
    let tables = LensTables {
        vignetting: None,
        chromatic_aberration: lead(ca.len(), &ca),
        distortion: lead(dist.len(), &dist),
    };
    let tags = LensTags {
        vignetting: vec![],
        ca,
        distortion: dist,
    };
    (tables, tags)
}

/// The raw values of a grey scene (`synth::natural`, luma) as seen through the lens of `geo`: the
/// site of colour `c` at source position `q` holds the scene at the output position `p` with
/// `geo.source_position(c, p) = q`, found by iterating `p <- p + (q - source_position(c, p))`.
/// The scene is read with a Catmull-Rom interpolation at `p`.
fn consistent_raw(w: usize, h: usize, geo: &LensGeometry, seed: u64) -> Vec<u16> {
    let rgb = synth::natural(w, h, seed);
    let scene: Vec<f32> = (0..w * h)
        .map(|i| 0.3 * rgb.plane(0)[i] + 0.55 * rgb.plane(1)[i] + 0.15 * rgb.plane(2)[i])
        .collect();
    let cfa = Cfa::RGGB;
    let mut data = vec![0u16; w * h];
    for y in 0..h {
        for x in 0..w {
            let c = cfa.color_at(x, y).plane();
            let q = (x as f64 + 0.5, y as f64 + 0.5);
            let mut p = q;
            for _ in 0..8 {
                let s = geo.source_position(c, p.0, p.1);
                p = (p.0 + (q.0 - s.0), p.1 + (q.1 - s.1));
            }
            let v = sample_catmull_rom(&scene, w, h, p.0, p.1).clamp(0.0, 1.0);
            data[y * w + x] = (f64::from(BLACK) + v * f64::from(WHITE - BLACK)).round() as u16;
        }
    }
    data
}

fn decode(bytes: &[u8]) -> Arc<RawImage> {
    Arc::new(
        ym_raw::decode::decode_bytes(bytes, &ym_raw::decode::DecodeOptions::default())
            .expect("decode"),
    )
}

/// The settings of a synthetic render: the gains of a neutral scene, no R5, no denoise, no R7, no R8.
fn bare_params(chromatic_aberration: bool) -> RawParams {
    RawParams {
        g0: [1.0; 3],
        neural_demosaic: false,
        highlight_recovery: false,
        lens_corrections: true,
        chromatic_aberration,
        vignetting: true,
        noise_reduction: NoiseReduction {
            luminance: 0.0,
            color: 0.0,
            detail: 0.0,
        },
        sharpen: SharpenParams {
            amount: 0.0,
            radius_px: 0.8,
        },
        iso: 100.0,
        noise: NoiseModel {
            a: 2.0e-5,
            b: 1.0e-8,
            delta: [0.0; 4],
            sigma_row: 0.0,
            q: 1.0 / 65535.0,
        },
        diffraction: None,
    }
}

/// A full chain with the adaptive ratio demosaic (or `method`) for `raw` with `params`.
fn build(
    raw: &Arc<RawImage>,
    params: &RawParams,
    method: Method,
) -> (FrameData, RawPlan, FullChain) {
    let frame = FrameData::new(raw).expect("frame");
    let plan = RawPlan::new(raw, &frame, params).expect("plan");
    let sigma = SigmaPlane {
        width: frame.width / 2,
        height: frame.height / 2,
        data: vec![0.01; (frame.width / 2) * (frame.height / 2)],
    };
    let full = FullChain::with_method(&frame, &plan, params, &sigma, method, &CancelToken::new())
        .expect("full chain");
    (frame, plan, full)
}

/// R1, R4, R5b (when the plan's lens corrects its CA on the mosaic), the whole-frame demosaic of
/// `method` with `ca`, and R9 over the whole output: the chain of a file without vignetting, R5,
/// denoise, R7 and R8, by hand.
fn by_hand(
    frame: &FrameData,
    plan: &RawPlan,
    params: &RawParams,
    method: Method,
    ca: Option<&CaCurves>,
) -> RgbImage {
    by_hand_with(frame, plan, params, method, ca, true)
}

/// [`by_hand`] with R5b switched on or off.
fn by_hand_with(
    frame: &FrameData,
    plan: &RawPlan,
    params: &RawParams,
    method: Method,
    ca: Option<&CaCurves>,
    r5b: bool,
) -> RgbImage {
    let (mut m, _) = normalize::normalize(
        frame.data(),
        frame.width,
        frame.height,
        frame.cfa,
        &plan.levels,
        None,
    )
    .expect("normalize");
    gains::apply_gains(&mut m, params.g0);
    if r5b && let Some(curves) = plan.lens.as_ref().and_then(LensGeometry::mosaic_ca_curves) {
        ym_gpu::cpu::lens::correct_mosaic_ca(&mut m, &curves).expect("R5b");
    }
    let img = demosaic::demosaic_whole_with(
        &m,
        method,
        &DemosaicInputs {
            gains: params.g0,
            noise: NoiseDn::from_model(&params.noise),
            ca,
        },
    );
    match &plan.lens {
        Some(geo) => {
            let (ow, oh) = geo.output_size();
            resample::resample_lens(&img, geo, Rect::whole(ow, oh)).expect("R9")
        }
        None => img,
    }
}

fn bits(img: &RgbImage) -> Vec<u32> {
    img.data.iter().map(|v| v.to_bits()).collect()
}

#[test]
fn ca_tables_in_the_file_drive_the_ca_path_of_the_full_chain() {
    let (w, h) = (192usize, 128usize);
    let hd = 0.5 * (w as f64).hypot(h as f64);
    let (tables, tags) = ca_tables(2.0, hd, true);
    let geo = LensGeometry::new(w, h, &tags, LensSwitches::default()).expect("lens");
    let data = consistent_raw(w, h, &geo, 5);
    let build_file = |tables: LensTables| {
        decode(
            &ArwBuilder::new("ILCE-7RM4", w as u32, h as u32)
                .black([BLACK; 4])
                .white([WHITE; 3])
                .lens_tables(tables)
                .build(&data),
        )
    };

    // A file with CA tables: the CA path.
    let raw = build_file(tables.clone());
    let params = bare_params(true);
    let (frame, plan, full) = build(&raw, &params, Method::AdaptiveRatio);
    assert_eq!((frame.width, frame.height), (w, h));
    let curves = plan
        .lens
        .as_ref()
        .expect("R9 geometry")
        .ca_curves()
        .expect("curves")
        .expect("the file's table is not neutral");
    assert!(curves.misregistration_px(hd) > 1.0);
    let (ow, oh) = (full.out_w, full.out_h);
    let got = full
        .run_tile(&frame, Rect::whole(ow, oh))
        .expect("whole frame");
    // The plan took the CA out of R9: the chain corrects it on the mosaic (R5b) and the demosaic
    // gets no curves.
    assert!(plan.lens.as_ref().expect("R9 geometry").ca_on_mosaic());
    let want = by_hand(&frame, &plan, &params, Method::AdaptiveRatio, None);
    assert_eq!(
        bits(&got),
        bits(&want),
        "the full chain corrects the CA on the mosaic"
    );
    let no_r5b = by_hand_with(&frame, &plan, &params, Method::AdaptiveRatio, None, false);
    assert!(
        got.max_abs_diff(&no_r5b) > 1e-3,
        "without R5b the image would differ"
    );

    // Another method with the same file gets the same R5b.
    let (frame_m, plan_m, full_m) = build(&raw, &params, Method::Menon);
    let got_m = full_m
        .run_tile(&frame_m, Rect::whole(full_m.out_w, full_m.out_h))
        .expect("menon");
    assert_eq!(
        bits(&got_m),
        bits(&by_hand(&frame_m, &plan_m, &params, Method::Menon, None))
    );

    // `decode.chromatic_aberration = false`: no CA in the geometry, so the plain path.
    let off = bare_params(false);
    let (frame_o, plan_o, full_o) = build(&raw, &off, Method::AdaptiveRatio);
    assert_eq!(
        plan_o
            .lens
            .as_ref()
            .expect("distortion remains")
            .ca_curves(),
        Ok(None)
    );
    let got_o = full_o
        .run_tile(&frame_o, Rect::whole(full_o.out_w, full_o.out_h))
        .expect("off");
    assert_eq!(
        bits(&got_o),
        bits(&by_hand(
            &frame_o,
            &plan_o,
            &off,
            Method::AdaptiveRatio,
            None
        ))
    );

    // A neutral table (every knot zero): no CA either.
    let (neutral, _) = {
        let (mut t, tg) = ca_tables(2.0, hd, true);
        t.chromatic_aberration = Some({
            let mut v = vec![16i16];
            v.extend([0i16; 16]);
            v
        });
        (t, tg)
    };
    let raw_n = build_file(neutral);
    let (frame_n, plan_n, full_n) = build(&raw_n, &params, Method::AdaptiveRatio);
    assert_eq!(
        plan_n
            .lens
            .as_ref()
            .expect("distortion remains")
            .ca_curves(),
        Ok(None)
    );
    let got_n = full_n
        .run_tile(&frame_n, Rect::whole(full_n.out_w, full_n.out_h))
        .expect("neutral");
    assert_eq!(
        bits(&got_n),
        bits(&by_hand(
            &frame_n,
            &plan_n,
            &params,
            Method::AdaptiveRatio,
            None
        ))
    );

    // Tiles of the chain equal the whole frame (the curves belong to the frame).
    let mut tiled = RgbImage::new(ow, oh);
    for core in ym_gpu::cpu::tiling::tile_grid(ow, oh, 71, 53, 1) {
        let t = full.run_tile(&frame, core).expect("tile");
        tiled.paste(&t, Rect::whole(core.w, core.h), core.x, core.y);
    }
    assert!(got.max_abs_diff(&tiled) <= 1e-4);
}

/// The CA residual of the chain with the adaptive ratio demosaic over `out`: with the CA path
/// (`on`), without it (`off`) and with R9 not correcting CA at all (`reference`, plain demosaic).
struct Variants {
    on: CaResidual,
    off: CaResidual,
    reference: CaResidual,
}

fn residual_options() -> CaResidualOptions {
    CaResidualOptions {
        patch_px: 48,
        stride_px: 48,
        ..CaResidualOptions::default()
    }
}

/// The residuals of the three variants over the whole output of the file `raw`.
fn whole_frame_variants(raw: &Arc<RawImage>, params: &RawParams) -> Variants {
    let (frame, plan, full) = build(raw, params, Method::AdaptiveRatio);
    let geo = plan.lens.clone().expect("R9 geometry");
    let (ow, oh) = geo.output_size();
    let crop = geo.crop();
    let hd = 0.5 * (frame.width as f64).hypot(frame.height as f64);
    let centre = (
        frame.width as f64 / 2.0 - crop.x as f64,
        frame.height as f64 / 2.0 - crop.y as f64,
    );
    let on = full.run_tile(&frame, Rect::whole(ow, oh)).expect("CA path");
    // Without the CA path: the chain without its lens stage (so `ca = None`), then R9 by hand.
    let mut chain = full.cpu_chain().clone();
    chain.lens = None;
    let src = chain
        .run_tile(
            &frame.as_raw_frame(),
            &full.prepared,
            Rect::whole(frame.width, frame.height),
        )
        .expect("source");
    let off = resample::resample_lens(&src, &geo, Rect::whole(ow, oh)).expect("R9");
    // R9 without the CA correction: the scale of what there was to correct.
    let no_ca = LensGeometry::new(
        frame.width,
        frame.height,
        &plan.tags,
        LensSwitches {
            lens_corrections: params.lens_corrections,
            chromatic_aberration: false,
        },
    )
    .expect("lens")
    .with_crop(crop);
    let reference = resample::resample_lens(&src, &no_ca, Rect::whole(ow, oh)).expect("R9");
    let opts = residual_options();
    Variants {
        on: lateral_ca_residual(&on, centre, hd, &opts),
        off: lateral_ca_residual(&off, centre, hd, &opts),
        reference: lateral_ca_residual(&reference, centre, hd, &opts),
    }
}

#[test]
fn the_ca_path_leaves_no_more_ca_after_r9_on_a_consistent_synthetic_file() {
    // 704 x 480: hd = 426 px, corner shifts of red and blue 1.5 and 1.3 px, with barrel distortion.
    let (w, h) = (704usize, 480usize);
    let hd = 0.5 * (w as f64).hypot(h as f64);
    let (tables, tags) = ca_tables(1.5, hd, true);
    let geo = LensGeometry::new(w, h, &tags, LensSwitches::default()).expect("lens");
    let data = consistent_raw(w, h, &geo, 11);
    let bytes = ArwBuilder::new("ILCE-7RM4", w as u32, h as u32)
        .black([BLACK; 4])
        .white([WHITE; 3])
        .lens_tables(tables)
        .build(&data);
    let raw = decode(&bytes);
    let v = whole_frame_variants(&raw, &bare_params(true));
    let line = |name: &str, r: &CaResidual| {
        let bands: Vec<String> = r
            .bands
            .iter()
            .map(|b| format!("{:.3}/{:.3}", b.red_rms_px, b.blue_rms_px))
            .collect();
        println!(
            "{name:<34} {} patches, R/B rms {:.3}/{:.3} px; bands {}",
            r.patches,
            r.red_rms_px,
            r.blue_rms_px,
            bands.join("  ")
        );
    };
    line("R9 without the CA correction", &v.reference);
    line("plain adaptive ratio, R9", &v.off);
    line("CA path adaptive ratio, R9", &v.on);
    assert!(v.on.patches > 30 && v.off.patches > 30);
    // R9 with the tables does not leave the CA it started with: a demosaic that interpolates red
    // and blue as differences from green pulls their edges toward green's, so what R9 corrects
    // afterwards is more than what is left, and the residual is mostly an over-correction. The CA
    // path pulls less (docs/algorithms/01 section 11). It is no worse than the plain path.
    assert!(
        v.on.red_rms_px <= v.off.red_rms_px && v.on.blue_rms_px <= v.off.blue_rms_px,
        "CA path {:.4}/{:.4} px, plain {:.4}/{:.4} px",
        v.on.red_rms_px,
        v.on.blue_rms_px,
        v.off.red_rms_px,
        v.off.blue_rms_px
    );
}

// ---------------------------------------------------------------------------------------------
// Real files
// ---------------------------------------------------------------------------------------------

/// Side of an analysis window in output pixels.
const WINDOW_PX: usize = 256;

/// Centres of the analysis windows in output coordinates: toward the four corners and the four
/// edges at a few radii, clamped to the output.
fn window_origins(ow: usize, oh: usize, centre: (f64, f64), hd: f64) -> Vec<(usize, usize)> {
    let mut out: Vec<(usize, usize)> = Vec::new();
    let half = (WINDOW_PX / 2) as f64;
    for angle_deg in [34.0f64, 146.0, 214.0, 326.0, 0.0, 90.0, 180.0, 270.0] {
        let (s, c) = angle_deg.to_radians().sin_cos();
        for frac in [0.3, 0.5, 0.7, 0.85, 0.97] {
            let cx = (centre.0 + frac * hd * c).clamp(half + 4.0, ow as f64 - half - 4.0);
            let cy = (centre.1 + frac * hd * s).clamp(half + 4.0, oh as f64 - half - 4.0);
            // On a 4-pixel grid so that clamped windows that coincide are found out.
            let o = (
                (((cx - half) / 4.0).round() as usize) * 4,
                (((cy - half) / 4.0).round() as usize) * 4,
            );
            if !out.contains(&o) {
                out.push(o);
            }
        }
    }
    out
}

fn paired_stats(
    on: &[ym_gpu::cpu::metrics::CaPatch],
    off: &[ym_gpu::cpu::metrics::CaPatch],
) -> Value {
    // Patches of the same place (the grid of every window is the same for both variants).
    let (mut n, mut d_red, mut d_blue, mut better_red, mut better_blue) = (0usize, 0.0, 0.0, 0, 0);
    for a in on {
        if let Some(b) = off.iter().find(|b| (b.x, b.y) == (a.x, a.y)) {
            n += 1;
            d_red += a.red_px.abs() - b.red_px.abs();
            d_blue += a.blue_px.abs() - b.blue_px.abs();
            better_red += usize::from(a.red_px.abs() < b.red_px.abs());
            better_blue += usize::from(a.blue_px.abs() < b.blue_px.abs());
        }
    }
    let nn = n.max(1) as f64;
    json!({
        "paired_patches": n,
        "mean_abs_shift_change_red_px": d_red / nn,
        "mean_abs_shift_change_blue_px": d_blue / nn,
        "share_red_closer_to_zero": better_red as f64 / nn,
        "share_blue_closer_to_zero": better_blue as f64 / nn,
    })
}

/// Sum of the squared central differences of the green plane over the interior of `img`, and the
/// number of samples: the gradient energy (a plain sharpness measure; compare two renderings of
/// the same window).
fn gradient_energy(img: &RgbImage) -> (f64, usize) {
    let (w, h) = (img.width, img.height);
    let g = img.plane(1);
    let mut e = 0.0f64;
    for y in 1..h - 1 {
        for x in 1..w - 1 {
            let dx = f64::from(g[y * w + x + 1] - g[y * w + x - 1]) * 0.5;
            let dy = f64::from(g[(y + 1) * w + x] - g[(y - 1) * w + x]) * 0.5;
            e += dx * dx + dy * dy;
        }
    }
    (e, (w - 2) * (h - 2))
}

fn to_srgb(v: f64) -> u8 {
    let v = v.clamp(0.0, 1.0);
    let e = if v <= 0.003_130_8 {
        12.92 * v
    } else {
        1.055 * v.powf(1.0 / 2.4) - 0.055
    };
    (e * 255.0 + 0.5) as u8
}

/// The same `crop` of every image side by side, 4x nearest, one exposure (the brightest 0.5 % of
/// the first image is white).
fn write_strip(path: &Path, images: &[&RgbImage], crop: Rect, zoom: usize) {
    let n = images[0].plane_len();
    let mut luma: Vec<f64> = (0..n)
        .map(|i| {
            0.3 * f64::from(images[0].data[i])
                + 0.55 * f64::from(images[0].data[n + i])
                + 0.15 * f64::from(images[0].data[2 * n + i])
        })
        .collect();
    luma.sort_by(f64::total_cmp);
    let k = 0.9 / luma[(luma.len() as f64 * 0.995) as usize].max(1e-6);
    let (ow, oh) = ((crop.w * zoom + 4) * images.len() - 4, crop.h * zoom);
    let mut buf = vec![255u8; ow * oh * 3];
    for (m, img) in images.iter().enumerate() {
        for y in 0..oh {
            for x in 0..crop.w * zoom {
                let p = img.pixel(crop.x + x / zoom, crop.y + y / zoom);
                let o = (y * ow + m * (crop.w * zoom + 4) + x) * 3;
                for c in 0..3 {
                    buf[o + c] = to_srgb(f64::from(p[c]) * k);
                }
            }
        }
    }
    let file = std::fs::File::create(path).expect("png file");
    let mut enc = png::Encoder::new(std::io::BufWriter::new(file), ow as u32, oh as u32);
    enc.set_color(png::ColorType::Rgb);
    enc.set_depth(png::BitDepth::Eight);
    enc.write_header()
        .expect("png header")
        .write_image_data(&buf)
        .expect("png data");
}

/// The crop of one window kept for a picture: the patch where the tables change the red and blue
/// shifts most.
struct Strip {
    /// Sum of the changes of the red and blue shifts of that patch, in pixels.
    difference: f64,
    /// The crop inside the window.
    crop: Rect,
    /// The window without the CA correction, with the tables, with the tables and the CA path.
    images: [RgbImage; 3],
    /// Origin of the window in the output.
    window: (usize, usize),
}

#[test]
#[ignore = "local evaluation on testdata/real/files, which is never committed: cargo test -p ym-core --lib ca_green_real_files -- --ignored --nocapture"]
fn ca_green_real_files_residual() {
    let repo: PathBuf = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let files_dir = repo.join("testdata/real/files");
    let Ok(rd) = std::fs::read_dir(&files_dir) else {
        eprintln!("no {}: nothing to evaluate", files_dir.display());
        return;
    };
    let mut files: Vec<PathBuf> = rd
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|e| e.eq_ignore_ascii_case("arw")))
        .collect();
    files.sort();
    let out_dir = repo.join("testdata/ca_green_residual/real");
    std::fs::create_dir_all(&out_dir).expect("output directory");
    let env = SessionEnv::load(&repo.join("data")).expect("the data folder");
    let cancel = CancelToken::new();
    let opts = residual_options();
    let mut records = Vec::new();
    println!(
        "| file | lens | CA at corner px | patches | R9 without CA (R/B rms px) | R9 flipped sign (R/B) | plain (R/B) | CA path (R/B) | paired |"
    );
    println!("|---|---|---|---|---|---|---|---|---|");
    for path in files {
        let name = path
            .file_stem()
            .unwrap_or_default()
            .to_string_lossy()
            .into_owned();
        let raw = match ym_raw::decode::decode(&path) {
            Ok(r) => Arc::new(r),
            Err(e) => {
                eprintln!("{name}: cannot decode ({e}); skipped");
                records.push(json!({"file": name, "error": e.to_string()}));
                continue;
            }
        };
        if raw.is_reduced() {
            eprintln!("{name}: camera-processed file; skipped");
            continue;
        }
        let camera = PhotoCamera::new(&env.profiles, &raw.meta, None);
        let profile = env.profiles.resolve(&camera.matched).clone();
        let est = AutoEstimates::compute(
            &raw,
            &AutoInputs {
                profile: &profile,
                illuminant_model: None,
                priors: &env.priors,
                autoset: None,
            },
        )
        .expect("estimates");
        let g0 = est.initial_gains().expect("gains");
        let params = RawParams {
            g0: [g0[0] as f32, g0[1] as f32, g0[2] as f32],
            neural_demosaic: false,
            highlight_recovery: true,
            lens_corrections: true,
            chromatic_aberration: true,
            vignetting: true,
            noise_reduction: NoiseReduction {
                luminance: est.noise_reduction.luminance as f32,
                color: est.noise_reduction.color as f32,
                detail: est.noise_reduction.detail as f32,
            },
            sharpen: SharpenParams {
                amount: est.sharpen.amount as f32,
                radius_px: est.sharpen.radius_px as f32,
            },
            iso: raw.meta.iso.map_or(100.0, |i| i as f32),
            noise: est.noise.model,
            diffraction: None,
        };
        let frame = FrameData::new(&raw).expect("frame");
        let plan = RawPlan::new(&raw, &frame, &params).expect("plan");
        let lens = raw.meta.lens_model.clone().unwrap_or_default();
        let tag_summary = format!(
            "ca {} knots, dist {} knots",
            plan.tags.ca.len(),
            plan.tags.distortion.len()
        );
        let Some(geo) = plan.lens.clone() else {
            println!("| {name} | {lens} | none | | | | | | no lens table ({tag_summary}) |");
            records.push(json!({"file": name, "lens": lens, "ca": null, "tags": tag_summary}));
            continue;
        };
        let curves = match geo.ca_curves() {
            Ok(Some(c)) => c,
            other => {
                println!(
                    "| {name} | {lens} | none | | | | | | no usable CA table ({tag_summary}; {other:?}) |"
                );
                records.push(json!({"file": name, "lens": lens, "ca": null, "tags": tag_summary}));
                continue;
            }
        };
        let (fw, fh) = (frame.width, frame.height);
        let hd = 0.5 * (fw as f64).hypot(fh as f64);
        let corner_px = curves.misregistration_px(hd);
        let predicted: Vec<String> = [0.25, 0.5, 0.75, 1.0]
            .iter()
            .map(|rho| {
                format!(
                    "{rho}: {:+.2}/{:+.2}",
                    curves.red.shift_at(rho * hd),
                    curves.blue.shift_at(rho * hd)
                )
            })
            .collect();
        eprintln!(
            "{name}: the tables move red/blue (px, at source radius) {}",
            predicted.join("  ")
        );
        let (ow, oh) = geo.output_size();
        let crop = geo.crop();
        let centre = (
            fw as f64 / 2.0 - crop.x as f64,
            fh as f64 / 2.0 - crop.y as f64,
        );
        let sigma = sigma_plane(&frame, &plan, &params, &cancel).expect("sigma");
        let full = FullChain::with_method(
            &frame,
            &plan,
            &params,
            &sigma,
            Method::AdaptiveRatio,
            &cancel,
        )
        .expect("chain");
        let mut plain_chain = full.cpu_chain().clone();
        plain_chain.lens = None;
        let no_ca = LensGeometry::new(
            fw,
            fh,
            &plan.tags,
            LensSwitches {
                lens_corrections: true,
                chromatic_aberration: false,
            },
        )
        .expect("lens")
        .with_crop(crop);

        // The same tables with the sign of every CA knot flipped: if R9 had the direction of the
        // tag wrong, this would leave less than not correcting.
        let flipped = LensTags {
            ca: plan.tags.ca.iter().map(|c| -c).collect(),
            ..plan.tags.clone()
        };
        let inverted = LensGeometry::new(fw, fh, &flipped, LensSwitches::default())
            .expect("lens")
            .with_crop(crop);

        let origins = window_origins(ow, oh, centre, hd);
        let (mut on_all, mut off_all, mut ref_all, mut inv_all) =
            (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        // Per band accumulators over the windows: (patches, sum r2, sum b2) per variant.
        let nb = opts.band_edges.len();
        let mut acc = vec![[[0.0f64; 3]; 4]; nb];
        let (mut sharp_on, mut sharp_off) = (0.0f64, 0.0f64);
        let mut best_strip: Option<Strip> = None;
        for (ox, oy) in origins {
            let out = Rect::new(ox, oy, WINDOW_PX, WINDOW_PX);
            let win_centre = (centre.0 - ox as f64, centre.1 - oy as f64);
            let on = full.run_tile(&frame, out).expect("window with the CA path");
            let src_rect = geo.source_rect(out);
            let src = plain_chain
                .run_tile(&frame.as_raw_frame(), &full.prepared, src_rect)
                .expect("source window");
            let off = resample::resample_lens(&src, &geo, out).expect("R9");
            let reference = resample::resample_lens(&src, &no_ca, out).expect("R9");
            let flipped_img = resample::resample_lens(&src, &inverted, out).expect("R9");
            let mut results = [
                lateral_ca_residual(&on, win_centre, hd, &opts),
                lateral_ca_residual(&off, win_centre, hd, &opts),
                lateral_ca_residual(&reference, win_centre, hd, &opts),
                lateral_ca_residual(&flipped_img, win_centre, hd, &opts),
            ];
            for (k, r) in results.iter().enumerate() {
                for (b, band) in r.bands.iter().enumerate() {
                    let n = band.patches as f64;
                    acc[b][k][0] += n;
                    acc[b][k][1] += n * band.red_rms_px * band.red_rms_px;
                    acc[b][k][2] += n * band.blue_rms_px * band.blue_rms_px;
                }
            }
            // Patch origins in output coordinates, so that windows do not collide.
            for r in &mut results {
                for p in &mut r.details {
                    p.x += ox;
                    p.y += oy;
                }
            }
            // The patch where the tables change the red-green shift most: its crop is saved.
            for p in &results[1].details {
                if let Some(q) = results[2].details.iter().find(|q| (q.x, q.y) == (p.x, p.y)) {
                    let d = (p.red_px - q.red_px).abs() + (p.blue_px - q.blue_px).abs();
                    if best_strip.as_ref().is_none_or(|b| d > b.difference) {
                        let cx = (p.x + 24).saturating_sub(48).min(WINDOW_PX - 96);
                        let cy = (p.y + 24).saturating_sub(48).min(WINDOW_PX - 96);
                        best_strip = Some(Strip {
                            difference: d,
                            crop: Rect::new(cx, cy, 96, 96),
                            images: [reference.clone(), off.clone(), on.clone()],
                            window: (ox, oy),
                        });
                    }
                }
            }
            let (ge_on, _) = gradient_energy(&on);
            let (ge_off, _) = gradient_energy(&off);
            sharp_on += ge_on;
            sharp_off += ge_off;
            let [a, b, c, d] = results;
            on_all.extend(a.details);
            off_all.extend(b.details);
            ref_all.extend(c.details);
            inv_all.extend(d.details);
        }
        if let Some(strip) = &best_strip {
            let path = out_dir.join(format!("{name}_no_ca__tables__ca_path.png"));
            let [a, b, c] = &strip.images;
            write_strip(&path, &[a, b, c], strip.crop, 4);
            eprintln!(
                "{name}: strip (R9 without CA, R9 with the tables, with the tables and the CA path) of the window at {:?} in {}",
                strip.window,
                path.display()
            );
        }
        let rms = |p: &[ym_gpu::cpu::metrics::CaPatch]| -> (f64, f64) {
            let n = p.len().max(1) as f64;
            (
                (p.iter().map(|q| q.red_px * q.red_px).sum::<f64>() / n).sqrt(),
                (p.iter().map(|q| q.blue_px * q.blue_px).sum::<f64>() / n).sqrt(),
            )
        };
        let (on_rms, off_rms, ref_rms, inv_rms) =
            (rms(&on_all), rms(&off_all), rms(&ref_all), rms(&inv_all));
        let paired = paired_stats(&on_all, &off_all);
        println!(
            "| {name} | {lens} | {corner_px:.2} | {} / {} / {} | {:.3} / {:.3} | {:.3} / {:.3} | {:.3} / {:.3} | {:.3} / {:.3} | red closer {:.0} %, blue closer {:.0} % of {} |",
            ref_all.len(),
            off_all.len(),
            on_all.len(),
            ref_rms.0,
            ref_rms.1,
            inv_rms.0,
            inv_rms.1,
            off_rms.0,
            off_rms.1,
            on_rms.0,
            on_rms.1,
            100.0 * paired["share_red_closer_to_zero"].as_f64().unwrap_or(0.0),
            100.0 * paired["share_blue_closer_to_zero"].as_f64().unwrap_or(0.0),
            paired["paired_patches"],
        );
        println!(
            "|   | green gradient energy, CA path over plain | {:.4} | | | | | | |",
            sharp_on / sharp_off.max(1e-30)
        );
        let bands_json: Vec<Value> = (0..nb)
            .map(|b| {
                let f = |k: usize| {
                    let n = acc[b][k][0].max(1.0);
                    json!({"patches": acc[b][k][0], "red_rms_px": (acc[b][k][1] / n).sqrt(), "blue_rms_px": (acc[b][k][2] / n).sqrt()})
                };
                json!({
                    "from": if b == 0 { opts.min_radius } else { opts.band_edges[b - 1] },
                    "to": opts.band_edges[b],
                    "ca_path": f(0), "plain": f(1), "r9_without_ca": f(2), "r9_flipped_sign": f(3),
                })
            })
            .collect();
        records.push(json!({
            "file": name, "lens": lens, "model": raw.meta.model, "size": [fw, fh],
            "tags": tag_summary, "ca_corner_px": corner_px,
            "patches": {"ca_path": on_all.len(), "plain": off_all.len(), "r9_without_ca": ref_all.len()},
            "rms_px": {
                "ca_path": {"red": on_rms.0, "blue": on_rms.1},
                "plain": {"red": off_rms.0, "blue": off_rms.1},
                "r9_without_ca": {"red": ref_rms.0, "blue": ref_rms.1},
                "r9_flipped_sign": {"red": inv_rms.0, "blue": inv_rms.1},
            },
            "paired_ca_path_vs_plain": paired,
            "green_gradient_energy_ca_path_over_plain": sharp_on / sharp_off.max(1e-30),
            "bands": bands_json,
        }));
    }
    std::fs::write(
        out_dir.join("results.json"),
        serde_json::to_string_pretty(&records).expect("json"),
    )
    .expect("write results");
}
