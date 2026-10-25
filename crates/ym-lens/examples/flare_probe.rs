//! Draws the flare terms of L3 on the lens plane of a real photo and prints how bright and how coloured they
//! are: a developer tool for tuning the ghosts, glow and streaks (`docs/21` section 4, L3), the quick loop
//! next to the full renders of `ym-core`'s `lens_sheet`.
//!
//! ```text
//! cargo run -p ym-lens --example flare_probe -- [options] <plane dir>...
//!
//!   --lenses DIR      a lens folder (default <repo>/data/lenses)
//!   --kind K          the lens (default default)
//!   --flares N        Flares (default 60)
//!   --anamorphic N    Anamorphic (default 0)
//!   --region X0,Y0,X1,Y1  also print the mean plane and the mean terms (plane units) over this rectangle
//!   --out PREFIX      write PREFIX-<photo>-<kind>.ppm (the plane with the flares added, tone curve
//!                     `v (1 + v / 9) / (1 + v)` on the white-balanced plane, G clip at 1) and
//!                     PREFIX-<photo>-<kind>-terms.ppm (the terms alone, 8 times brighter)
//! ```
//!
//! `<plane dir>` is a folder with `lens_plane.f32` and `lens_plane.json`, as `lens_sheet --dump-planes` writes
//! them. One line is printed per photo: the light the terms add per channel in plane units (the chromaticity
//! is the sum over the frame divided by its mean, 1 1 1 is neutral), the largest term in clip levels (the
//! channel's value over its clip level `g0`) and where it lies.

use std::path::{Path, PathBuf};

use ym_lens::{
    Axis, CameraResponse, FrameInfo, ImageRef, LensAmounts, LensKind, LensLibrary, LensPlan,
    LensPlanes, PixelRect, flare_terms,
};

fn srgb(v: f64) -> u8 {
    let d = v.clamp(0.0, 1.0);
    let e = if d <= 0.003_130_8 {
        12.92 * d
    } else {
        1.055 * d.powf(1.0 / 2.4) - 0.055
    };
    (e * 255.0 + 0.5) as u8
}

fn tone(v: f64) -> f64 {
    let v = v.max(0.0);
    v * (1.0 + v / 9.0) / (1.0 + v)
}

fn main() {
    let mut lens_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../data/lenses");
    let mut kind = LensKind::Default;
    let (mut flares, mut anamorphic) = (60.0f64, 0.0f64);
    let mut out: Option<String> = None;
    let mut region: Option<[usize; 4]> = None;
    let mut dirs = Vec::new();
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--lenses" => lens_dir = PathBuf::from(args.next().expect("--lenses DIR")),
            "--kind" => {
                let v = args.next().expect("--kind K");
                kind = LensKind::ALL
                    .into_iter()
                    .find(|k| k.to_string() == v)
                    .expect("a lens name");
            }
            "--flares" => flares = args.next().expect("--flares N").parse().expect("number"),
            "--anamorphic" => {
                anamorphic = args
                    .next()
                    .expect("--anamorphic N")
                    .parse()
                    .expect("number");
            }
            "--out" => out = Some(args.next().expect("--out PREFIX")),
            "--region" => {
                let v: Vec<usize> = args
                    .next()
                    .expect("--region X0,Y0,X1,Y1")
                    .split(',')
                    .map(|n| n.parse().expect("number"))
                    .collect();
                region = Some([v[0], v[1], v[2], v[3]]);
            }
            dir => dirs.push(PathBuf::from(dir)),
        }
    }
    let lib = LensLibrary::from_dir(&lens_dir).expect("the lens folder");
    let curves = ym_color::spectral::SpectralCurves::from_path(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../data/cameras/sony/Sony_ILCE-7RM4_380_780_5.json"
    ))
    .expect("curves");
    let camera = CameraResponse::from_curves(&curves).expect("camera");

    for dir in &dirs {
        let json: serde_json::Value = serde_json::from_slice(
            &std::fs::read(dir.join("lens_plane.json")).expect("lens_plane.json"),
        )
        .expect("json");
        let n = |k: &str| json[k].as_u64().expect(k) as usize;
        let (w, h, k) = (n("width"), n("height"), n("k_L"));
        let bytes = std::fs::read(dir.join("lens_plane.f32")).expect("lens_plane.f32");
        let data: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        let g0: Vec<f32> = json["g0"]
            .as_array()
            .expect("g0")
            .iter()
            .map(|v| v.as_f64().expect("g0") as f32)
            .collect();
        let name = dir
            .file_name()
            .map_or_else(String::new, |s| s.to_string_lossy().into_owned());
        let axis = if json["axis"].as_str() == Some("Y") {
            Axis::Y
        } else {
            Axis::X
        };
        let frame = FrameInfo {
            width: n("frame_width"),
            height: n("frame_height"),
            axis,
            f_number: json["f_number"].as_f64(),
            clip: [g0[0], g0[1], g0[2]],
        };
        let input = ImageRef {
            width: w,
            height: h,
            x0: 0,
            y0: 0,
            frame_width: w,
            frame_height: h,
            data: &data,
        };
        let amounts = LensAmounts {
            kind,
            flares,
            edges: 0.0,
            anamorphic,
        };
        let plan = LensPlan::new(&lib, &amounts, frame, &camera).expect("plan");
        let planes = LensPlanes::build(
            &LensPlan::new(
                &lib,
                &LensAmounts {
                    anamorphic: 0.0,
                    ..amounts
                },
                frame,
                &camera,
            )
            .expect("plan"),
            &input,
            None,
            &|| false,
        )
        .expect("planes");
        let t0 = std::time::Instant::now();
        let terms = flare_terms(&plan, &planes, k, &input, PixelRect::new(0, 0, w, h))
            .expect("terms")
            .unwrap_or_else(|| vec![0.0; 3 * w * h]);
        let secs = t0.elapsed().as_secs_f64();
        let plane = w * h;
        let mut sum = [0.0f64; 3];
        let mut peak = (0.0f64, 0usize);
        for i in 0..plane {
            for c in 0..3 {
                sum[c] += terms[c * plane + i];
                let level = terms[c * plane + i] / f64::from(g0[c]);
                if level > peak.0 {
                    peak = (level, i);
                }
            }
        }
        let mean = (sum[0] + sum[1] + sum[2]) / 3.0;
        println!(
            "{name}\t{kind}\tF{flares} A{anamorphic}\tadded {:.4e}\tchroma {:.2} {:.2} {:.2}\tpeak {:.3e} clip levels at ({}, {})\t{secs:.1}s",
            sum[0] + sum[1] + sum[2],
            sum[0] / mean,
            sum[1] / mean,
            sum[2] / mean,
            peak.0,
            peak.1 % w,
            peak.1 / w
        );
        if let Some([x0, y0, x1, y1]) = region {
            let mut p = [0.0f64; 3];
            let mut t = [0.0f64; 3];
            let count = ((x1 - x0) * (y1 - y0)) as f64;
            for y in y0..y1 {
                for x in x0..x1 {
                    for c in 0..3 {
                        p[c] += f64::from(data[c * plane + y * w + x]) / count;
                        t[c] += terms[c * plane + y * w + x] / count;
                    }
                }
            }
            println!(
                "  region plane {:.4} {:.4} {:.4}  terms {:.4} {:.4} {:.4}",
                p[0], p[1], p[2], t[0], t[1], t[2]
            );
        }
        if let Some(prefix) = &out {
            let stem = format!("{prefix}-{name}-{kind}");
            if let Some(parent) = Path::new(&stem).parent() {
                std::fs::create_dir_all(parent).expect("dir");
            }
            let mut scene = format!("P6\n{w} {h}\n255\n").into_bytes();
            let mut only = format!("P6\n{w} {h}\n255\n").into_bytes();
            for i in 0..plane {
                for c in 0..3 {
                    let v = f64::from(data[c * plane + i]) + terms[c * plane + i];
                    scene.push(srgb(tone(v)));
                }
                for c in 0..3 {
                    only.push(srgb(tone(8.0 * terms[c * plane + i])));
                }
            }
            std::fs::write(format!("{stem}.ppm"), scene).expect("ppm");
            std::fs::write(format!("{stem}-terms.ppm"), only).expect("ppm");
        }
    }
}
