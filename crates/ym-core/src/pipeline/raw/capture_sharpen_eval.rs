//! Local A/B of R8 Richardson-Lucy and diffraction deconvolution on the real files of
//! `testdata/real/files` (`docs/algorithms/03-diffraction-deconvolution.md` section 6.3).
//!
//! The raw files are never committed, so this test is ignored and skips what it cannot find:
//!
//! ```text
//! cargo test -p ym-core --lib capture_sharpen_real_files -- --ignored --nocapture
//! ```
//!
//! For every decodable file it runs R1 to R7 (lens resample off, so that R8 is the last stage and
//! the variants differ only by it) and then R8 four ways: off, Richardson-Lucy with the Auto values
//! the photo would get, diffraction deconvolution (the f-number from the file, the pixel pitch from
//! the registry or the crop factor), and diffraction with the noise-gate variant of spec section 7.1
//! item 1. On two windows picked from the raw data it measures the 10 to 90 % rise distance, MTF50,
//! overshoot and undershoot of the strongest clean edge and the luma noise of the flattest area.
//! Results go to `testdata/capture_sharpen_ab/real/` (`results.json` and PNG strips, git-ignored).

use std::path::{Path, PathBuf};

use serde_json::{Value, json};
use ym_gpu::cpu::sharpen::diffraction::eval::{EdgeGeometry, edge_stats, patch_sd_detrended};
use ym_gpu::cpu::tiling::Halo;

use super::*;
use crate::auto::{AutoEstimates, AutoInputs};
use crate::camera::PhotoCamera;
use crate::session::SessionEnv;

/// Side of an analysis window in pixels (the edge spread function reaches 24 px on each side).
const WINDOW_PX: usize = 96;
/// Step of the window scan in pixels.
const STRIDE_PX: usize = 32;
/// Context a variant that is applied outside the chain needs on each side (the stage's halo is at
/// most 25).
const CONTEXT_PX: usize = 32;
/// The region of the frame searched for windows: the middle of the frame, which the vignetting and
/// the corners' lens blur affect least.
const REGION_PX: (usize, usize) = (2560, 1664);

/// One scanned window of the R7 luma.
struct Window {
    x: usize,
    y: usize,
    /// Mean gradient energy of the window.
    energy: f64,
    mean: f64,
    max: f64,
    /// The edge in the window, when it has one that is clean enough: the line, the step, the
    /// elongation of the line and its sharpness (the peak gradient over the step, 1 / px).
    edge: Option<(EdgeGeometry, f64, f64, f64)>,
}

/// Box-smooths a plane over 3 x 3 (borders repeated).
fn smooth3(v: &[f64], w: usize, h: usize) -> Vec<f64> {
    let mut out = vec![0.0f64; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut s = 0.0;
            for dy in -1isize..=1 {
                for dx in -1isize..=1 {
                    let yy = (y as isize + dy).clamp(0, h as isize - 1) as usize;
                    let xx = (x as isize + dx).clamp(0, w as isize - 1) as usize;
                    s += v[yy * w + xx];
                }
            }
            out[y * w + x] = s / 9.0;
        }
    }
    out
}

/// Scans the luma of the region in windows. The edge of a window is the line of its strongest
/// gradients: principal axes of the positions of the top 6 % gradient pixels (elongated, thin), the
/// gradients across it, and flat plateaus with a step of at least 0.1 between them.
fn scan(luma: &[f64], w: usize, h: usize) -> Vec<Window> {
    let ys = smooth3(luma, w, h);
    let mut e = vec![0.0f64; w * h];
    let mut gxs = vec![0.0f64; w * h];
    let mut gys = vec![0.0f64; w * h];
    for y in 1..h - 1 {
        for x in 1..w - 1 {
            let gx = 0.5 * (ys[y * w + x + 1] - ys[y * w + x - 1]);
            let gy = 0.5 * (ys[(y + 1) * w + x] - ys[(y - 1) * w + x]);
            gxs[y * w + x] = gx;
            gys[y * w + x] = gy;
            e[y * w + x] = gx * gx + gy * gy;
        }
    }
    let mut out = Vec::new();
    let mut wy = 1;
    while wy + WINDOW_PX < h {
        let mut wx = 1;
        while wx + WINDOW_PX < w {
            let idx = |x: usize, y: usize| (wy + y) * w + wx + x;
            let mut ev: Vec<f64> = Vec::with_capacity(WINDOW_PX * WINDOW_PX);
            let (mut energy, mut mean, mut max) = (0.0f64, 0.0f64, f64::MIN);
            for y in 0..WINDOW_PX {
                for x in 0..WINDOW_PX {
                    ev.push(e[idx(x, y)]);
                    energy += e[idx(x, y)];
                    mean += luma[idx(x, y)];
                    max = max.max(luma[idx(x, y)]);
                }
            }
            let n = (WINDOW_PX * WINDOW_PX) as f64;
            let (energy, mean) = (energy / n, mean / n);
            ev.sort_by(|a, b| b.total_cmp(a));
            let thr = ev[ev.len() * 6 / 100];
            // Principal axes of the top gradient pixels.
            let (mut sw, mut sx, mut sy) = (0.0f64, 0.0f64, 0.0f64);
            for y in 0..WINDOW_PX {
                for x in 0..WINDOW_PX {
                    let ee = e[idx(x, y)];
                    if ee >= thr && ee > 0.0 {
                        sw += ee;
                        sx += ee * x as f64;
                        sy += ee * y as f64;
                    }
                }
            }
            let mut edge = None;
            if sw > 0.0 {
                let (cx, cy) = (sx / sw, sy / sw);
                let (mut sxx, mut syy, mut sxy) = (0.0f64, 0.0f64, 0.0f64);
                for y in 0..WINDOW_PX {
                    for x in 0..WINDOW_PX {
                        let ee = e[idx(x, y)];
                        if ee >= thr && ee > 0.0 {
                            let (dx, dy) = (x as f64 - cx, y as f64 - cy);
                            sxx += ee * dx * dx;
                            syy += ee * dy * dy;
                            sxy += ee * dx * dy;
                        }
                    }
                }
                let (sxx, syy, sxy) = (sxx / sw, syy / sw, sxy / sw);
                let tr = sxx + syy;
                let det = sxx * syy - sxy * sxy;
                let disc = (tr * tr / 4.0 - det).max(0.0).sqrt();
                let (l_major, l_minor) = (tr / 2.0 + disc, (tr / 2.0 - disc).max(1e-9));
                // The minor axis is the normal of the line.
                let ang = 0.5 * (2.0 * sxy).atan2(sxx - syy); // major axis angle
                let mut normal = (-ang.sin(), ang.cos());
                let elong = (l_major / l_minor).sqrt();
                // Gradients must run across the line, and the plateaus on both sides be flat.
                let (mut agree, mut wsum) = (0.0f64, 0.0f64);
                for y in 0..WINDOW_PX {
                    for x in 0..WINDOW_PX {
                        let ee = e[idx(x, y)];
                        if ee >= thr && ee > 0.0 {
                            let g = (gxs[idx(x, y)], gys[idx(x, y)]);
                            let gm = ee.sqrt();
                            agree += ee * (g.0 * normal.0 + g.1 * normal.1).abs() / gm;
                            wsum += ee;
                        }
                    }
                }
                let agree = agree / wsum;
                let (mut lo_s, mut lo_ss, mut lo_n) = (0.0f64, 0.0f64, 0.0f64);
                let (mut hi_s, mut hi_ss, mut hi_n) = (0.0f64, 0.0f64, 0.0f64);
                for y in 0..WINDOW_PX {
                    for x in 0..WINDOW_PX {
                        let s = normal.0 * (x as f64 - cx) + normal.1 * (y as f64 - cy);
                        let v = luma[idx(x, y)];
                        if (-34.0..-14.0).contains(&s) {
                            lo_s += v;
                            lo_ss += v * v;
                            lo_n += 1.0;
                        } else if (14.0..34.0).contains(&s) {
                            hi_s += v;
                            hi_ss += v * v;
                            hi_n += 1.0;
                        }
                    }
                }
                if lo_n > 200.0 && hi_n > 200.0 {
                    let (mut lo, mut hi) = (lo_s / lo_n, hi_s / hi_n);
                    let mut sd_lo = (lo_ss / lo_n - lo * lo).max(0.0).sqrt();
                    let mut sd_hi = (hi_ss / hi_n - hi * hi).max(0.0).sqrt();
                    if hi < lo {
                        normal = (-normal.0, -normal.1);
                        std::mem::swap(&mut lo, &mut hi);
                        std::mem::swap(&mut sd_lo, &mut sd_hi);
                    }
                    let step = hi - lo;
                    if step >= 0.1
                        && sd_lo <= 0.2 * step
                        && sd_hi <= 0.2 * step
                        && max <= 0.95
                        && elong >= 2.0
                        && l_minor.sqrt() <= 4.0
                        && agree >= 0.75
                    {
                        edge = Some((
                            EdgeGeometry {
                                normal,
                                centre: (cx + 0.5, cy + 0.5),
                            },
                            step,
                            elong,
                            ev[0].sqrt() / step,
                        ));
                    }
                }
            }
            out.push(Window {
                x: wx,
                y: wy,
                energy,
                mean,
                max,
                edge,
            });
            wx += STRIDE_PX;
        }
        wy += STRIDE_PX;
    }
    out
}

/// The sharpest clean edge (the largest peak gradient over its step: only an edge that is already
/// sharp shows what a PSF-scale correction does), and up to `n_flat` flat windows, flattest first,
/// with some signal, none overlapping another or a window that holds an edge candidate.
fn pick(w: &[Window], n_flat: usize) -> (Option<&Window>, Vec<&Window>) {
    let edge = w.iter().filter(|x| x.edge.is_some()).max_by(|a, b| {
        let sharp = |x: &Window| x.edge.map_or(0.0, |(_, _, _, s)| s);
        sharp(a).total_cmp(&sharp(b))
    });
    let near =
        |a: &Window, b: &Window| a.x.abs_diff(b.x) < WINDOW_PX && a.y.abs_diff(b.y) < WINDOW_PX;
    let mut candidates: Vec<&Window> = w
        .iter()
        .filter(|x| (0.06..=0.5).contains(&x.mean) && x.max <= 0.9)
        .filter(|x| !w.iter().any(|e| e.edge.is_some() && near(x, e)))
        .collect();
    candidates.sort_by(|a, b| a.energy.total_cmp(&b.energy));
    let mut flats: Vec<&Window> = Vec::new();
    for c in candidates {
        if flats.len() == n_flat {
            break;
        }
        if !flats.iter().any(|f| near(c, f)) {
            flats.push(c);
        }
    }
    (edge, flats)
}

fn luma_of(img: &RgbImage) -> Vec<f64> {
    let n = img.plane_len();
    let w = ym_gpu::cpu::sharpen::diffraction::SPEC_LUMA;
    (0..n)
        .map(|i| {
            f64::from(w[0]) * f64::from(img.data[i])
                + f64::from(w[1]) * f64::from(img.data[n + i])
                + f64::from(w[2]) * f64::from(img.data[2 * n + i])
        })
        .collect()
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

/// A strip of the same crop of every variant side by side, one exposure for all, 4x nearest.
fn write_strip(path: &Path, variants: &[&RgbImage], crop: Rect, zoom: usize) {
    let lum = luma_of(variants[0]);
    let mut v: Vec<f64> = lum.clone();
    v.sort_by(f64::total_cmp);
    let k = 0.9 / v[(v.len() as f64 * 0.995) as usize].max(1e-6);
    let (ow, oh) = ((crop.w * zoom + 4) * variants.len() - 4, crop.h * zoom);
    let mut buf = vec![255u8; ow * oh * 3];
    for (n, img) in variants.iter().enumerate() {
        for y in 0..oh {
            for x in 0..crop.w * zoom {
                let p = img.pixel(crop.x + x / zoom, crop.y + y / zoom);
                let o = (y * ow + n * (crop.w * zoom + 4) + x) * 3;
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

fn stats_json(s: &ym_gpu::cpu::sharpen::diffraction::eval::EdgeStats, sd_ratio: f64) -> Value {
    json!({
        "mtf50_cpp": s.mtf50, "rise_10_90_px": s.rise,
        "overshoot_pct": s.overshoot_pct, "undershoot_pct": s.undershoot_pct,
        "flat_noise_ratio": sd_ratio, "esf": s.esf,
    })
}

#[test]
#[ignore = "local evaluation on testdata/real/files, which is never committed: cargo test -p ym-core --lib capture_sharpen_real_files -- --ignored --nocapture"]
fn capture_sharpen_real_files_ab() {
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
    let out_dir = repo.join("testdata/capture_sharpen_ab/real");
    std::fs::create_dir_all(&out_dir).expect("output directory");
    // Crops of an earlier run (a file that has no clean edge now) must not stay behind.
    for entry in std::fs::read_dir(&out_dir)
        .expect("read the output directory")
        .flatten()
    {
        if entry.path().extension().is_some_and(|e| e == "png") {
            std::fs::remove_file(entry.path()).expect("remove an old crop");
        }
    }
    let env = SessionEnv::load(&repo.join("data")).expect("the data folder");
    let cancel = CancelToken::new();
    let mut records = Vec::new();
    println!(
        "| file | f | ISO | pitch um (source) | R8 | edge | method | MTF50 | rise px | over % | under % | flat noise x |\n|---|---|---|---|---|---|---|---|---|---|---|---|"
    );
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
                embedded_jpeg: None,
            },
        )
        .expect("estimates");
        let g0 = est.initial_gains().expect("gains");
        let sharpen_rl = SharpenParams {
            amount: est.sharpen.amount as f32,
            radius_px: est.sharpen.radius_px as f32,
        };
        let diffraction = diffraction_params(
            &raw.meta,
            camera.matched.sensor.as_ref(),
            raw.data.width,
            raw.data.height,
            false,
        );
        let pitch = pixel_pitch(
            &raw.meta,
            camera.matched.sensor.as_ref(),
            raw.data.width,
            raw.data.height,
            false,
        );
        let fallback_pitch = pixel_pitch(&raw.meta, None, raw.data.width, raw.data.height, false);
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
                amount: 0.0,
                ..sharpen_rl
            },
            iso: raw.meta.iso.map_or(100.0, |i| i as f32),
            noise: est.noise.model,
            diffraction: None,
        };
        let frame = FrameData::new(&raw).expect("frame");
        let plan = RawPlan::new(&raw, &frame, &params).expect("plan");
        let sigma = sigma_plane(&frame, &plan, &params, &cancel).expect("sigma");
        let full = FullChain::new(&frame, &plan, &params, &sigma, &cancel).expect("chain");
        // R8 is the last stage: the variants differ only by it.
        let base = {
            let mut c = full.cpu_chain().clone();
            c.lens = None;
            c
        };
        let (fw, fh) = (frame.width, frame.height);
        let frame_rect = Rect::whole(fw, fh);

        // The region the windows are searched in, from the R7 output (R8 off).
        let (rw, rh) = (REGION_PX.0.min(fw & !1), REGION_PX.1.min(fh & !1));
        let region = Rect::new(((fw - rw) / 2) & !1, ((fh - rh) / 2) & !1, rw, rh);
        let r7_region = {
            let mut c = base.clone();
            c.sharpen = None;
            c.run_tile(&frame.as_raw_frame(), &full.prepared, region)
                .expect("region")
        };
        let windows = scan(&luma_of(&r7_region), rw, rh);
        let (edge_w, flat_ws) = pick(&windows, 6);
        let edge_candidates = windows.iter().filter(|x| x.edge.is_some()).count();
        let f_number = raw.meta.f_number;
        let iso = params.iso;
        let pitch_text = pitch.map_or("none".to_owned(), |(p, _, s)| {
            format!(
                "{p:.2} ({})",
                if s == PitchSource::Registry {
                    "registry"
                } else {
                    "crop factor"
                }
            )
        });
        let mut record = json!({
            "file": name, "model": raw.meta.model, "f_number": f_number, "iso": iso,
            "focal_length_mm": raw.meta.focal_length_mm, "focal_length_35mm": raw.meta.focal_length_35mm,
            "size": [fw, fh],
            "pitch_um": pitch.map(|p| p.0), "pitch_source": pitch.map(|p| format!("{:?}", p.2)),
            "pitch_crop_factor_um": fallback_pitch.map(|p| p.0),
            "rl_auto": {"amount": sharpen_rl.amount, "radius_px": sharpen_rl.radius_px},
            "diffraction_active": diffraction.is_some(),
        });
        let Some(d) = diffraction else {
            println!(
                "| {name} | {f_number:?} | {iso} | {pitch_text} | RL (diffraction inactive) | | | | | | | |"
            );
            records.push(record);
            continue;
        };
        let noise_dn = NoiseDn::from_model(&params.noise);
        let dplan = DiffractionPlan::new(&d, params.g0, &noise_dn, SPEC_LUMA).expect("active plan");

        eprintln!(
            "plan {name}: sigma0 {:.1} DN, sigma1 {:.3} DN, R {}, halo {}, noise g {:.3} r {:.3}",
            dplan.sigma0_dn(),
            dplan.sigma1_dn(),
            dplan.radius(),
            dplan.halo_px(),
            noise_dn.g,
            noise_dn.r
        );
        // The four variants of a window, as tiles of the same rectangle.
        let variants = |core: Rect| -> [RgbImage; 4] {
            let run = |sharpen: Option<CaptureSharpen>| {
                let mut c = base.clone();
                c.sharpen = sharpen;
                c.run_tile(&frame.as_raw_frame(), &full.prepared, core)
                    .expect("tile")
            };
            let none = run(None);
            let rl = run(Some(CaptureSharpen::RichardsonLucy(sharpen_rl)));
            let dd = run(Some(CaptureSharpen::Diffraction(d)));
            // The gate variant: the R7 tile with context, then the stage, then the core.
            let ext = core.grow(Halo::uniform(CONTEXT_PX), frame_rect);
            let r7 = {
                let mut c = base.clone();
                c.sharpen = None;
                c.run_tile(&frame.as_raw_frame(), &full.prepared, ext)
                    .expect("context tile")
            };
            let gated = dplan.clone().with_noise_gate(true).apply_whole(&r7);
            let gate = gated.extract(core.relative_to(&ext));
            [none, rl, dd, gate]
        };
        let labels = ["none", "rl_auto", "diffraction", "diffraction_gate"];
        let mut edge_json = Value::Null;
        let to_rect = |w: &Window| Rect::new(region.x + w.x, region.y + w.y, WINDOW_PX, WINDOW_PX);
        if let Some(e) = edge_w {
            let core = to_rect(e);
            let (geo, step, elong, sharpness) = e.edge.expect("an edge window");
            let v = variants(core);
            let lumas: Vec<Vec<f64>> = v.iter().map(luma_of).collect();
            let mut per = serde_json::Map::new();
            for (k, l) in labels.iter().zip(&lumas) {
                let s = edge_stats(l, WINDOW_PX, WINDOW_PX, &geo, 0);
                println!(
                    "| {name} | {f_number:?} | {iso} | {pitch_text} | diffraction | edge | {k} | {:.3} | {:.2} | {:.1} | {:.1} | |",
                    s.mtf50, s.rise, s.overshoot_pct, s.undershoot_pct
                );
                per.insert((*k).to_owned(), stats_json(&s, f64::NAN));
            }
            edge_json = json!({
                "rect": [core.x, core.y, core.w, core.h],
                "normal": [geo.normal.0, geo.normal.1], "centre": [geo.centre.0, geo.centre.1],
                "step": step, "elongation": elong, "sharpness_per_px": sharpness,
                "candidates": edge_candidates, "variants": per,
            });
            let crop = Rect::new(
                (geo.centre.0 as usize)
                    .saturating_sub(24)
                    .min(WINDOW_PX - 48),
                (geo.centre.1 as usize)
                    .saturating_sub(24)
                    .min(WINDOW_PX - 48),
                48,
                48,
            );
            let refs: Vec<&RgbImage> = v.iter().collect();
            write_strip(&out_dir.join(format!("{name}_edge.png")), &refs, crop, 4);
        } else {
            println!(
                "| {name} | {f_number:?} | {iso} | {pitch_text} | diffraction | edge | no clean edge in the region | | | | | |"
            );
        }
        let mut flats_json = Vec::new();
        let (mut sum_d, mut sum_g, mut max_d, mut max_g) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
        for (n, fl) in flat_ws.iter().enumerate() {
            let core = to_rect(fl);
            let v = variants(core);
            let lumas: Vec<Vec<f64>> = v.iter().map(luma_of).collect();
            let win = (16usize, 16usize, 64usize, 64usize);
            let sd: Vec<f64> = lumas
                .iter()
                .map(|l| patch_sd_detrended(l, WINDOW_PX, win))
                .collect();
            let ratios: Vec<f64> = sd.iter().map(|x| x / sd[0]).collect();
            sum_d += ratios[2];
            sum_g += ratios[3];
            max_d = max_d.max(ratios[2]);
            max_g = max_g.max(ratios[3]);
            flats_json.push(json!({
                "rect": [core.x, core.y, core.w, core.h], "mean_luma": fl.mean,
                "luma_sd_r7": sd[0],
                "ratio_to_r7": {"rl_auto": ratios[1], "diffraction": ratios[2], "diffraction_gate": ratios[3]},
            }));
            if n == 0 {
                let refs: Vec<&RgbImage> = v.iter().collect();
                write_strip(
                    &out_dir.join(format!("{name}_flat.png")),
                    &refs,
                    Rect::new(16, 16, 64, 64),
                    3,
                );
            }
        }
        if !flat_ws.is_empty() {
            let k = flat_ws.len() as f64;
            println!(
                "| {name} | {f_number:?} | {iso} | {pitch_text} | diffraction | flat x{} | diffraction | | | | | mean {:.3}, max {:.3} (gate: mean {:.3}, max {:.3}) |",
                flat_ws.len(),
                sum_d / k,
                max_d,
                sum_g / k,
                max_g
            );
        }
        let flat_json = json!(flats_json);
        record["edge"] = edge_json;
        record["flat"] = flat_json;
        record["variants"] = json!(labels);
        records.push(record);
    }
    let doc = json!({
        "description": "Capture sharpening A/B on the real files of testdata/real/files: R1-R7 (lens resample off), then R8 off / Richardson-Lucy Auto / diffraction / diffraction with the noise-gate variant",
        "records": records,
    });
    std::fs::write(
        out_dir.join("results.json"),
        serde_json::to_string_pretty(&doc).expect("json"),
    )
    .expect("results.json");
    eprintln!("wrote {}", out_dir.display());
}
