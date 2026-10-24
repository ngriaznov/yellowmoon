//! Measures how much light each element of L3 adds, on a real lens plane or a synthetic frame with lights
//! (`docs/21` section 4, L3, *As built (flare strength)*): a developer tool for the flare strengths.
//!
//! ```text
//! cargo run -p ym-lens --example flare_measure -- [options] <target>...
//!
//!   --lenses DIR      a lens folder (default <repo>/data/lenses); the element-only copies the review made
//!                     (every fraction but one set to 0) give each element on its own
//!   --kinds a,b,c     lenses to measure (default all seven)
//!   --flares N        Flares (default 100)
//!   --anamorphic N    Anamorphic (default 0)
//!   --scale N         the plane's scale k (default: the plane's own, 4 for a synthetic frame)
//!
//!   <target>          `synthetic`, or a folder with `lens_plane.f32` and `lens_plane.json` (the files
//!                     `ym-core`'s `lens_sheet --dump-planes` writes)
//! ```
//!
//! For every lens one line of tab-separated numbers: the lights found (clipped components of at most 4000
//! pixels), the light added in clip levels times pixels (mean over the channels, the plane normalised so that
//! every channel clips at 1), the peak of the added light and the peak at least 10 px from a clipped pixel
//! ("off"), the share of the frame where the added light is at least 5 % of the picture there (plus 1 % of
//! the clip level), and the added light along the axis and on rings around the largest light.
//!
//! The terms are linear in Flares, so Flares 30 or 60 is the Flares 100 line times 0.3 or 0.6.

use std::path::{Path, PathBuf};

use ym_lens::{
    Axis, CameraResponse, FrameInfo, ImageRef, LensAmounts, LensKind, LensLibrary, LensPlan,
    LensPlanes, PixelRect, flare_terms,
};

struct Target {
    name: String,
    /// The plane's size, which is also the frame's size at its scale.
    width: usize,
    height: usize,
    /// The full-resolution frame.
    frame_width: usize,
    frame_height: usize,
    k: usize,
    g0: [f32; 3],
    f_number: Option<f64>,
    data: Vec<f32>,
}

fn synthetic() -> Target {
    let (w, h) = (1500usize, 1000usize);
    let mut data = vec![0.04f32; 3 * w * h];
    let mut put = |cx: f64, cy: f64, r: f64| {
        for y in 0..h {
            for x in 0..w {
                let (dx, dy) = (x as f64 + 0.5 - cx, y as f64 + 0.5 - cy);
                if dx * dx + dy * dy <= r * r {
                    for c in 0..3 {
                        data[c * w * h + y * w + x] = 1.0;
                    }
                }
            }
        }
    };
    // Three point lights of 3, 5 and 9 px and two discs of 12 and 30 px radius, off the centre.
    put(300.0, 250.0, 1.6);
    put(900.0, 200.0, 2.8);
    put(1250.0, 330.0, 4.8);
    put(380.0, 720.0, 30.0);
    put(1100.0, 760.0, 12.0);
    Target {
        name: "synthetic".into(),
        width: w,
        height: h,
        frame_width: 4 * w,
        frame_height: 4 * h,
        k: 4,
        g0: [1.0; 3],
        f_number: Some(8.0),
        data,
    }
}

fn load(dir: &Path) -> Target {
    let json: serde_json::Value = serde_json::from_slice(
        &std::fs::read(dir.join("lens_plane.json")).expect("lens_plane.json"),
    )
    .expect("json");
    let n = |k: &str| json[k].as_u64().expect(k) as usize;
    let (w, h) = (n("width"), n("height"));
    let bytes = std::fs::read(dir.join("lens_plane.f32")).expect("lens_plane.f32");
    let data: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    assert_eq!(data.len(), 3 * w * h);
    let g0: Vec<f32> = json["g0"]
        .as_array()
        .expect("g0")
        .iter()
        .map(|v| v.as_f64().expect("g0") as f32)
        .collect();
    Target {
        name: dir
            .file_name()
            .map_or_else(String::new, |s| s.to_string_lossy().into_owned()),
        width: w,
        height: h,
        frame_width: n("frame_width"),
        frame_height: n("frame_height"),
        k: n("k_L"),
        g0: [g0[0], g0[1], g0[2]],
        f_number: json["f_number"].as_f64(),
        data,
    }
}

/// The clipped components (8-connected, pixels at or above 0.9 of the clip in every channel) of at most
/// 400 pixels: `(centroid x, centroid y, pixels)`, and the mask of all clipped pixels.
fn lights(t: &Target) -> (Vec<(f64, f64, usize)>, Vec<bool>) {
    let (w, h) = (t.width, t.height);
    let plane = w * h;
    let mut mask = vec![false; plane];
    for (i, m) in mask.iter_mut().enumerate() {
        *m = (0..3).all(|c| t.data[c * plane + i] >= 0.9 * t.g0[c]);
    }
    let mut seen = vec![false; plane];
    let mut out = Vec::new();
    for start in 0..plane {
        if !mask[start] || seen[start] {
            continue;
        }
        let mut stack = vec![start];
        seen[start] = true;
        let (mut sx, mut sy, mut n) = (0.0, 0.0, 0usize);
        while let Some(p) = stack.pop() {
            let (x, y) = (p % w, p / w);
            sx += x as f64 + 0.5;
            sy += y as f64 + 0.5;
            n += 1;
            for dy in -1i64..=1 {
                for dx in -1i64..=1 {
                    let (nx, ny) = (x as i64 + dx, y as i64 + dy);
                    if nx < 0 || ny < 0 || nx >= w as i64 || ny >= h as i64 {
                        continue;
                    }
                    let q = ny as usize * w + nx as usize;
                    if mask[q] && !seen[q] {
                        seen[q] = true;
                        stack.push(q);
                    }
                }
            }
        }
        if n <= 4000 {
            out.push((sx / n as f64, sy / n as f64, n));
        }
    }
    (out, mask)
}

fn main() {
    let mut lens_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../data/lenses");
    let mut kinds: Vec<LensKind> = LensKind::ALL.to_vec();
    let (mut flares, mut anamorphic) = (100.0f64, 0.0f64);
    let mut scale: Option<usize> = None;
    let mut targets = Vec::new();
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--lenses" => lens_dir = PathBuf::from(args.next().expect("--lenses DIR")),
            "--kinds" => {
                let v = args.next().expect("--kinds a,b");
                kinds = LensKind::ALL
                    .into_iter()
                    .filter(|k| v.split(',').any(|n| n == k.to_string()))
                    .collect();
            }
            "--flares" => flares = args.next().expect("--flares N").parse().expect("number"),
            "--anamorphic" => {
                anamorphic = args
                    .next()
                    .expect("--anamorphic N")
                    .parse()
                    .expect("number");
            }
            "--scale" => scale = Some(args.next().expect("--scale N").parse().expect("number")),
            "synthetic" => targets.push(synthetic()),
            dir => targets.push(load(Path::new(dir))),
        }
    }
    let lib = LensLibrary::from_dir(&lens_dir).expect("the lens folder");
    let curves = ym_color::spectral::SpectralCurves::from_path(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../data/cameras/sony/Sony_ILCE-7RM4_380_780_5.json"
    ))
    .expect("curves");
    let camera = CameraResponse::from_curves(&curves).expect("camera");

    for t in &targets {
        let (w, h) = (t.width, t.height);
        let plane = w * h;
        let k = scale.unwrap_or(t.k);
        let (lights, mask) = lights(t);
        // The normalised picture: every channel's clip at 1.
        let norm = |c: usize, i: usize| f64::from(t.data[c * plane + i]) / f64::from(t.g0[c]);
        let z_mean: Vec<f64> = (0..plane)
            .map(|i| (norm(0, i) + norm(1, i) + norm(2, i)) / 3.0)
            .collect();
        // Distance to the nearest clipped pixel is at most 10 px: a dilation by 10.
        let mut near = vec![false; plane];
        {
            let mut rows = vec![false; plane];
            for y in 0..h {
                for x in 0..w {
                    if mask[y * w + x] {
                        for xx in x.saturating_sub(10)..(x + 11).min(w) {
                            rows[y * w + xx] = true;
                        }
                    }
                }
            }
            for y in 0..h {
                for x in 0..w {
                    if rows[y * w + x] {
                        for yy in y.saturating_sub(10)..(y + 11).min(h) {
                            near[yy * w + x] = true;
                        }
                    }
                }
            }
        }
        let big = lights.iter().max_by_key(|l| l.2).copied();
        let frame = FrameInfo {
            width: t.frame_width,
            height: t.frame_height,
            axis: Axis::X,
            f_number: t.f_number,
            clip: t.g0,
        };
        let input = ImageRef {
            width: w,
            height: h,
            x0: 0,
            y0: 0,
            frame_width: w,
            frame_height: h,
            data: &t.data,
        };
        let whole = PixelRect::new(0, 0, w, h);
        for kind in &kinds {
            let amounts = LensAmounts {
                kind: *kind,
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
            let terms = flare_terms(&plan, &planes, k, &input, whole).expect("terms");
            let secs = t0.elapsed().as_secs_f64();
            let Some(terms) = terms else {
                println!("{}\t{kind}\tflares 0", t.name);
                continue;
            };
            let tm: Vec<f64> = (0..plane)
                .map(|i| {
                    (0..3)
                        .map(|c| terms[c * plane + i] / f64::from(t.g0[c]))
                        .sum::<f64>()
                        / 3.0
                })
                .collect();
            let total: f64 = tm.iter().sum();
            let peak = tm.iter().copied().fold(0.0, f64::max);
            let peak_off = tm
                .iter()
                .zip(&near)
                .filter(|(_, n)| !**n)
                .map(|(v, _)| *v)
                .fold(0.0, f64::max);
            let visible = tm
                .iter()
                .zip(&z_mean)
                .filter(|(v, z)| **v >= 0.05 * (**z + 0.01))
                .count();
            let z_total: f64 = z_mean.iter().sum();
            // Profiles around the largest light: along the axis (rows within 1 px, the larger side) and on
            // rings.
            let mut axis_profile = String::new();
            let mut ring_profile = String::new();
            if let Some((cx, cy, _)) = big {
                let at = |x: f64, y: f64| -> f64 {
                    let (xi, yi) = (x.round() as i64, y.round() as i64);
                    if xi < 0 || yi < 0 || xi >= w as i64 || yi >= h as i64 {
                        return 0.0;
                    }
                    tm[yi as usize * w + xi as usize]
                };
                for d in [8.0, 20.0, 50.0, 100.0, 200.0, 400.0] {
                    let m = (-1..=1)
                        .map(|dy| {
                            at(cx + d, cy + f64::from(dy)).max(at(cx - d, cy + f64::from(dy)))
                        })
                        .fold(0.0, f64::max);
                    axis_profile.push_str(&format!("{m:.2e} "));
                }
                for r in [6.0, 12.0, 24.0, 48.0, 96.0, 192.0] {
                    let n = 64;
                    let s: f64 = (0..n)
                        .map(|i| {
                            let a = std::f64::consts::TAU * f64::from(i) / f64::from(n);
                            at(cx + r * a.cos(), cy + r * a.sin())
                        })
                        .sum::<f64>()
                        / f64::from(n);
                    ring_profile.push_str(&format!("{s:.2e} "));
                }
            }
            println!(
                "{}\t{kind}\tlights {}\tadded {:.4e}\tpicture {:.4e}\tpeak {:.3e}\toff {:.3e}\tvisible {:.3}%\taxis(8,20,50,100,200,400) {}\tring(6,12,24,48,96,192) {}\t{secs:.1}s",
                t.name,
                lights.len(),
                total,
                z_total,
                peak,
                peak_off,
                100.0 * visible as f64 / plane as f64,
                axis_profile.trim_end(),
                ring_profile.trim_end(),
            );
        }
    }
}
