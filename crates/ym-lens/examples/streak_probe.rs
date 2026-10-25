//! Draws the streaks of L3 alone on the lens plane of a real photo and prints where they are and how bright:
//! a developer tool for the anamorphic streaks (`docs/21` section 4, L3, *Streaks*), the quick loop next to the
//! full renders of `ym-core`'s `lens_sheet`.
//!
//! ```text
//! cargo run -p ym-lens --example streak_probe -- [options] <plane dir>...
//!
//!   --lenses DIR      a lens folder (default <repo>/data/lenses)
//!   --kind K          the lens (default default)
//!   --flares N        Flares (default 60)
//!   --anamorphic N    Anamorphic (default 100)
//!   --out PREFIX      write PREFIX-<photo>-<kind>.ppm (the plane with the flares added, tone curve
//!                     `v (1 + v / 9) / (1 + v)` on the white-balanced plane, G clip at 1) and
//!                     PREFIX-<photo>-<kind>-streaks.ppm (the streaks alone, 16 times brighter)
//! ```
//!
//! `<plane dir>` is a folder with `lens_plane.f32` and `lens_plane.json`, as `lens_sheet --dump-planes` writes
//! them. The streaks are the flare terms at the Anamorphic amount minus those at 0 (with the attachment's own
//! ghosts there as well: point `--lenses` at a copy of the folder whose `anamorphic.json` has `ghosts: []` and
//! `ghost_stretch: 1` to see the streaks alone). One block is printed per band of rows the streaks cross: the
//! brightest channel's value over its clip level (`peak`), where it is, how many rows add 2 codes or more on a
//! plain display of the plane (the plane's value as the background, +1.11 EV, the sRGB curve), how far to either
//! side of the peak the streak still adds 2 codes, and, at 8, 40, 120 and 300 px from the peak, the value and the
//! colour of the added light (blue over red and green over red in plane units, 1 for a neutral).

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

/// The plain display of a plane value in clip levels: +1.11 EV of exposure and a shoulder.
fn tone2(v: f64) -> f64 {
    let d = 2.16 * v.max(0.0);
    d / (1.0 + 0.5 * d)
}

fn main() {
    let mut lens_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../data/lenses");
    let mut kind = LensKind::Default;
    let (mut flares, mut anamorphic) = (60.0f64, 100.0f64);
    let mut out: Option<String> = None;
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
        let amounts = |a: f64| LensAmounts {
            kind,
            flares,
            edges: 0.0,
            anamorphic: a,
        };
        let plan_with = LensPlan::new(&lib, &amounts(anamorphic), frame, &camera).expect("plan");
        let plan_without = LensPlan::new(&lib, &amounts(0.0), frame, &camera).expect("plan");
        let planes = LensPlanes::build(&plan_without, &input, None, &|| false).expect("planes");
        let run = |plan: &LensPlan| -> Vec<f64> {
            flare_terms(plan, &planes, k, &input, PixelRect::new(0, 0, w, h))
                .expect("terms")
                .unwrap_or_else(|| vec![0.0; 3 * w * h])
        };
        let t0 = std::time::Instant::now();
        let with = run(&plan_with);
        let without = run(&plan_without);
        let secs = t0.elapsed().as_secs_f64();
        let plane = w * h;
        let streak: Vec<f64> = with.iter().zip(&without).map(|(a, b)| a - b).collect();
        // The brightest channel over its clip level, per pixel, and the change in code values it makes on a
        // plain display of the plane (the plane's own value as the background, +1.11 EV of exposure, the
        // sRGB curve): the proxy of "adds more than 2 codes".
        let level: Vec<f64> = (0..plane)
            .map(|i| {
                (0..3)
                    .map(|c| streak[c * plane + i] / f64::from(g0[c]))
                    .fold(0.0f64, f64::max)
            })
            .collect();
        let code = |v: f64| -> f64 { f64::from(srgb(tone2(v))) };
        let codes: Vec<f64> = (0..plane)
            .map(|i| {
                (0..3)
                    .map(|c| {
                        let bg = f64::from(data[c * plane + i]) / f64::from(g0[c]);
                        code(bg + streak[c * plane + i] / f64::from(g0[c])) - code(bg)
                    })
                    .fold(0.0f64, f64::max)
            })
            .collect();
        println!("== {name}\t{kind}\tF{flares} A{anamorphic}\t{w}x{h}\t{secs:.1}s");
        let (across, along) = if axis == Axis::X { (h, w) } else { (w, h) };
        let at = |v: &[f64], q: usize, p: usize| -> f64 {
            if axis == Axis::X {
                v[q * w + p]
            } else {
                v[p * w + q]
            }
        };
        // Bands of rows (across the axis) where some pixel is over 0.001 of the clip level.
        let rows: Vec<Vec<f64>> = (0..across)
            .map(|q| (0..along).map(|p| at(&level, q, p)).collect())
            .collect();
        let row_codes: Vec<Vec<f64>> = (0..across)
            .map(|q| (0..along).map(|p| at(&codes, q, p)).collect())
            .collect();
        let mut q = 0;
        while q < across {
            let top = rows[q].iter().cloned().fold(0.0f64, f64::max);
            if top < 0.001 {
                q += 1;
                continue;
            }
            let start = q;
            let mut peak = (0.0f64, q, 0usize);
            while q < across {
                let t = rows[q].iter().cloned().fold(0.0f64, f64::max);
                if t < 0.001 {
                    break;
                }
                if t > peak.0 {
                    let p = rows[q].iter().position(|v| *v == t).unwrap_or(0);
                    peak = (t, q, p);
                }
                q += 1;
            }
            let end = q - 1;
            let pq = peak.1;
            let r = &rows[pq];
            let rc = &row_codes[pq];
            // How far from the peak, to each side, the streak still adds 2 codes: the farthest pixel over 2 codes
            // that is reached through gaps of at most 60 px (the light itself and what it clips are skipped).
            let side = |dir: isize| -> usize {
                let (mut d, mut last, mut gap) = (0usize, 0usize, 0usize);
                loop {
                    d += 1;
                    let p = peak.2 as isize + dir * d as isize;
                    if p < 0 || p >= along as isize {
                        break last;
                    }
                    if rc[p as usize] >= 2.0 {
                        last = d;
                        gap = 0;
                    } else if d > 16 {
                        gap += 1;
                        if gap > 60 {
                            break last;
                        }
                    }
                }
            };
            let thick = (start..=end)
                .filter(|q| row_codes[*q].iter().cloned().fold(0.0f64, f64::max) >= 2.0)
                .count();
            let plane_at = |p: usize, c: usize| -> f64 {
                let i = if axis == Axis::X {
                    pq * w + p
                } else {
                    p * w + pq
                };
                streak[c * plane + i]
            };
            let probe = |d: usize| -> String {
                let p = if peak.2 >= d { peak.2 - d } else { peak.2 + d };
                let p = p.min(along - 1);
                let (rr, gg, bb) = (plane_at(p, 0), plane_at(p, 1), plane_at(p, 2));
                format!(
                    "{d}px: {:.4} b/r {:.2} g/r {:.2}",
                    r[p],
                    bb / rr.max(1e-12),
                    gg / rr.max(1e-12)
                )
            };
            println!(
                "rows {start}..{end} ({} rows over 2 codes) peak {:.4} at (along {}, across {pq}) visible: {} px toward 0, {} px toward the end (frame along {along} px)\n    {}  {}  {}  {}",
                thick,
                peak.0,
                peak.2,
                side(-1),
                side(1),
                probe(8),
                probe(40),
                probe(120),
                probe(300),
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
                    let v = f64::from(data[c * plane + i]) + with[c * plane + i];
                    scene.push(srgb(tone(v)));
                }
                for c in 0..3 {
                    only.push(srgb(tone(16.0 * streak[c * plane + i])));
                }
            }
            std::fs::write(format!("{stem}.ppm"), scene).expect("ppm");
            std::fs::write(format!("{stem}-streaks.ppm"), only).expect("ppm");
        }
    }
}
