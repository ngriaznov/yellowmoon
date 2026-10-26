//! Measures the light the field stages (L1, L2) add around one light of a real photo's lens plane: a developer
//! tool for the colour of the light the lens spreads (`docs/21` section 4, *As built (the light's colour)*).
//!
//! ```text
//! cargo run -p ym-lens --example light_probe -- [options] <plane dir>
//!
//!   --lenses DIR      a lens folder (default <repo>/data/lenses)
//!   --kind K          the lens (default vintage)
//!   --edges N         Edges (default 100)
//!   --anamorphic N    Anamorphic (default 0)
//!   --star X,Y        the light's pixel on the lens plane (default 169,650)
//!   --neutral R       replace the plane within R px of the star by its mean over the channels (a neutral
//!                     light of the same brightness): the colour the lens's own spread gives a white light
//!   --patch X,Y,R     print the picture and the stage's output with and without the light rule over the
//!                     square of half-side R at (X, Y), channel by channel (the rule is switched off through
//!                     `YM_NO_LIGHT`, which the library reads)
//! ```
//!
//! `<plane dir>` is a folder with `lens_plane.f32` and `lens_plane.json`, as `lens_sheet --dump-planes`
//! writes them. Prints, per annulus of the star (0 to 3, 3 to 6, 6 to 30, 6 to 60 px), the light the stages
//! add (output minus input, plane units, positive part) as channel over mean (1 1 1 is neutral), and the
//! light the input holds there.

use std::path::PathBuf;

use ym_lens::{
    Axis, CameraResponse, FrameInfo, ImageRef, LensAmounts, LensKind, LensLibrary, LensPlan,
    LensPlanes, PixelRect, apply_field,
};

fn main() {
    let mut lens_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../data/lenses");
    let mut kind = LensKind::Vintage;
    let (mut edges, mut anamorphic) = (100.0f64, 0.0f64);
    let mut star = (169usize, 650usize);
    let mut neutral: Option<f64> = None;
    let mut window: Option<(usize, usize, usize)> = None;
    let mut dir = None;
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
            "--edges" => edges = args.next().expect("--edges N").parse().expect("number"),
            "--anamorphic" => {
                anamorphic = args.next().expect("--anamorphic N").parse().expect("number");
            }
            "--star" => {
                let v: Vec<usize> = args
                    .next()
                    .expect("--star X,Y")
                    .split(',')
                    .map(|n| n.parse().expect("number"))
                    .collect();
                star = (v[0], v[1]);
            }
            "--patch" => {
                let v: Vec<usize> = args
                    .next()
                    .expect("--patch X,Y,R")
                    .split(',')
                    .map(|n| n.parse().expect("number"))
                    .collect();
                window = Some((v[0], v[1], v[2]));
            }
            "--neutral" => neutral = Some(args.next().expect("--neutral R").parse().expect("number")),
            d => dir = Some(PathBuf::from(d)),
        }
    }
    let dir = dir.expect("a plane dir");
    let lib = LensLibrary::from_dir(&lens_dir).expect("the lens folder");
    let curves = ym_color::spectral::SpectralCurves::from_path(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../data/cameras/sony/Sony_ILCE-7RM4_380_780_5.json"
    ))
    .expect("curves");
    let camera = CameraResponse::from_curves(&curves).expect("camera");
    let json: serde_json::Value = serde_json::from_slice(
        &std::fs::read(dir.join("lens_plane.json")).expect("lens_plane.json"),
    )
    .expect("json");
    let n = |k: &str| json[k].as_u64().expect(k) as usize;
    let (w, h, k) = (n("width"), n("height"), n("k_L"));
    let bytes = std::fs::read(dir.join("lens_plane.f32")).expect("lens_plane.f32");
    let mut data: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    let g0: Vec<f32> = json["g0"]
        .as_array()
        .expect("g0")
        .iter()
        .map(|v| v.as_f64().expect("g0") as f32)
        .collect();
    let frame = FrameInfo {
        width: n("frame_width"),
        height: n("frame_height"),
        axis: if json["axis"].as_str() == Some("Y") {
            Axis::Y
        } else {
            Axis::X
        },
        f_number: json["f_number"].as_f64(),
        clip: [g0[0], g0[1], g0[2]],
    };
    let plane = w * h;
    if let Some(r) = neutral {
        for y in 0..h {
            for x in 0..w {
                let (dx, dy) = (x as f64 - star.0 as f64, y as f64 - star.1 as f64);
                if dx.hypot(dy) <= r {
                    let i = y * w + x;
                    let m = (data[i] + data[plane + i] + data[2 * plane + i]) / 3.0;
                    data[i] = m;
                    data[plane + i] = m;
                    data[2 * plane + i] = m;
                }
            }
        }
    }
    let amounts = LensAmounts {
        kind,
        flares: 0.0,
        edges,
        anamorphic,
    };
    let plan = LensPlan::new(&lib, &amounts, frame, &camera).expect("plan");
    let input = ImageRef {
        width: w,
        height: h,
        x0: 0,
        y0: 0,
        frame_width: w,
        frame_height: h,
        data: &data,
    };
    let planes = LensPlanes::build(&plan, &input, None, &|| false).expect("planes");
    let map = planes.light_map_data();
    let patch = map.ev.iter().filter(|v| **v > 0.0).count();
    println!(
        "cap scale {:.3}; light map: {patch} plane pixels with headroom, EV at the star {:.2}",
        planes.light_boost_scale(),
        map.ev.get(star.1 * w + star.0).copied().unwrap_or(0.0)
    );
    let out = apply_field(&plan, &planes, k, &input, PixelRect::new(0, 0, w, h)).expect("field");
    if let Some((px, py, r)) = window {
        // SAFETY: single-threaded here; the library reads the variable on every call.
        unsafe { std::env::set_var("YM_NO_LIGHT", "1") };
        let plain = apply_field(&plan, &planes, k, &input, PixelRect::new(0, 0, w, h)).expect("field");
        unsafe { std::env::remove_var("YM_NO_LIGHT") };
        for (c, name) in ["R", "G", "B"].iter().enumerate() {
            println!("channel {name}: picture | without the rule | with the rule (x {}..{}, y {}..{})", px - r, px + r, py - r, py + r);
            for y in py - r..=py + r {
                let row = |d: &[f32]| {
                    (px - r..=px + r)
                        .map(|x| format!("{:6.3}", d[c * plane + y * w + x]))
                        .collect::<Vec<_>>()
                        .join(" ")
                };
                println!("{y:4} {} | {} | {}", row(&data), row(&plain), row(&out));
            }
        }
    }
    println!(
        "{kind} Edges {edges} Anamorphic {anamorphic}, star at {star:?}{}",
        neutral.map_or(String::new(), |r| format!(" made neutral within {r} px"))
    );
    for (lo, hi) in [(0.0, 3.0), (3.0, 6.0), (6.0, 30.0), (6.0, 60.0), (0.0, 60.0)] {
        let (mut add, mut inp) = ([0.0f64; 3], [0.0f64; 3]);
        for y in 0..h {
            for x in 0..w {
                let r = (x as f64 - star.0 as f64).hypot(y as f64 - star.1 as f64);
                if r < lo || r >= hi {
                    continue;
                }
                let i = y * w + x;
                for c in 0..3 {
                    let (a, b) = (f64::from(out[c * plane + i]), f64::from(data[c * plane + i]));
                    add[c] += (a - b).max(0.0);
                    inp[c] += b;
                }
            }
        }
        let rel = |v: [f64; 3]| {
            let m = (v[0] + v[1] + v[2]) / 3.0;
            [v[0] / m, v[1] / m, v[2] / m]
        };
        let (a, b) = (rel(add), rel(inp));
        println!(
            "  {lo:>3.0}..{hi:<3.0} px: added {:.2} {:.2} {:.2} (mean {:.3}); input {:.2} {:.2} {:.2} (mean {:.3})",
            a[0],
            a[1],
            a[2],
            (add[0] + add[1] + add[2]) / 3.0,
            b[0],
            b[1],
            b[2],
            (inp[0] + inp[1] + inp[2]) / 3.0
        );
    }
}
