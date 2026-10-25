//! Scratch probe of agent G1 (light colour): applies the lens to a dumped lens plane and prints, around one
//! pixel, the picture before the lens, after L1/L2 (Z), the flare terms and the finished picture, in camera RGB
//! with their chromaticity (1 1 1 is neutral). Not committed.
//!
//! ```text
//! g1_probe <plane dir> <kind> <flares> <edges> <anamorphic> <cx> <cy> <radius>
//! ```

use ym_lens::{
    Axis, CameraResponse, FrameInfo, ImageRef, LensAmounts, LensKind, LensLibrary, LensPlan,
    LensPlanes, PixelRect, apply, apply_field, flare_terms,
};

fn smooth(t: f64) -> f64 {
    let t = t.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Experiment: raises the saturated channels of a pixel toward the opposed estimate of the others.
fn lift(x: [f64; 3], g0: [f64; 3], from: f64, to: f64) -> [f64; 3] {
    let n = [x[0] / g0[0], x[1] / g0[1], x[2] / g0[2]];
    let s = n.map(|v| smooth((v - from) / (to - from)));
    let mut y = x;
    for c in 0..3 {
        if s[c] <= 0.0 {
            continue;
        }
        let (mut wsum, mut acc) = (0.0, 0.0);
        for o in 0..3 {
            if o != c {
                let w = 1.0 - s[o];
                wsum += w;
                acc += w * x[o].max(0.0).cbrt();
            }
        }
        if wsum <= 0.0 {
            continue;
        }
        let est = (acc / wsum).powi(3);
        let weight = s[c] * (wsum / 2.0);
        y[c] = x[c] + weight * (est - x[c]).max(0.0);
    }
    y
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let dir = std::path::PathBuf::from(&a[1]);
    let kind = LensKind::ALL
        .into_iter()
        .find(|k| k.to_string() == a[2])
        .expect("a lens name");
    let num = |i: usize| a[i].parse::<f64>().expect("number");
    let (flares, edges, anam) = (num(3), num(4), num(5));
    let (cx, cy, rad) = (num(6) as usize, num(7) as usize, num(8) as usize);
    let lens_dir = std::env::var("YELLOWMOON_DATA").map_or_else(
        |_| std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../data"),
        std::path::PathBuf::from,
    );
    let lib = LensLibrary::from_dir(&lens_dir.join("lenses")).expect("the lens folder");
    let curves = ym_color::spectral::SpectralCurves::from_path(
        lens_dir.join("cameras/sony/Sony_ILCE-7RM4_380_780_5.json"),
    )
    .expect("curves");
    let camera = CameraResponse::from_curves(&curves).expect("camera");
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
    let raw = data.clone();
    let mut data = data;
    if std::env::var("LIFT").is_ok() {
        let g = [f64::from(g0[0]), f64::from(g0[1]), f64::from(g0[2])];
        for i in 0..w * h {
            let x = [0, 1, 2].map(|c| f64::from(data[c * w * h + i]));
            let y = lift(x, g, 0.90, 1.0);
            for c in 0..3 {
                data[c * w * h + i] = y[c] as f32;
            }
        }
    }
    let input = ImageRef {
        width: w,
        height: h,
        x0: 0,
        y0: 0,
        frame_width: w,
        frame_height: h,
        data: &data,
    };
    let _ = &raw;
    let amounts = LensAmounts {
        kind,
        flares,
        edges,
        anamorphic: anam,
    };
    let plan = LensPlan::new(&lib, &amounts, frame, &camera).expect("plan");
    let planes = LensPlanes::build(&plan, &input, None, &|| false).expect("planes");
    let whole = PixelRect::new(0, 0, w, h);
    let z = apply_field(&plan, &planes, k, &input, whole).expect("z");
    let terms = flare_terms(&plan, &planes, k, &input, whole).expect("terms");
    let out = apply(&plan, &planes, k, &input, whole).expect("apply");
    let plane = w * h;
    let show = |name: &str, f: &dyn Fn(usize, usize) -> [f64; 3]| {
        let (mut s, mut peak) = ([0.0f64; 3], [0.0f64; 3]);
        for y in cy.saturating_sub(rad)..(cy + rad + 1).min(h) {
            for x in cx.saturating_sub(rad)..(cx + rad + 1).min(w) {
                let v = f(x, y);
                for c in 0..3 {
                    s[c] += v[c];
                    peak[c] = peak[c].max(v[c]);
                }
            }
        }
        let m = (s[0] + s[1] + s[2]) / 3.0;
        println!(
            "{name:<8} sum {:.3} {:.3} {:.3}  chroma {:.3} {:.3} {:.3}  peak {:.3} {:.3} {:.3}",
            s[0],
            s[1],
            s[2],
            s[0] / m,
            s[1] / m,
            s[2] / m,
            peak[0],
            peak[1],
            peak[2]
        );
    };
    let px = |buf: &[f32], x: usize, y: usize| {
        [0, 1, 2].map(|c| f64::from(buf[c * plane + y * w + x]))
    };
    show("before", &|x, y| px(&data, x, y));
    show("Z", &|x, y| px(&z, x, y));
    if let Some(t) = &terms {
        show("terms", &|x, y| [0, 1, 2].map(|c| t[c * plane + y * w + x]));
    }
    show("after", &|x, y| px(&out, x, y));
    // The light the flares add over the dark pixels of the plane (mean of the channels under DARK clip levels).
    if let (Ok(thr), Some(t)) = (std::env::var("DARK"), &terms) {
        let thr: f64 = thr.parse().expect("DARK");
        let mut per: [Vec<f64>; 3] = Default::default();
        for i in 0..plane {
            let m = (0..3)
                .map(|c| f64::from(data[c * plane + i]) / f64::from(g0[c]))
                .sum::<f64>()
                / 3.0;
            if m < thr {
                for c in 0..3 {
                    per[c].push(t[c * plane + i]);
                }
            }
        }
        let med = |v: &mut Vec<f64>| {
            v.sort_by(f64::total_cmp);
            v.get(v.len() / 2).copied().unwrap_or(0.0)
        };
        let m = [med(&mut per[0]), med(&mut per[1]), med(&mut per[2])];
        let mean = (m[0] + m[1] + m[2]) / 3.0;
        println!(
            "dark ({} px under {thr}) median terms {:.5} {:.5} {:.5}  chroma {:.3} {:.3} {:.3}",
            per[0].len(),
            m[0],
            m[1],
            m[2],
            m[0] / mean,
            m[1] / mean,
            m[2] / mean
        );
        // Over the whole frame: sum chroma.
        let mut sum = [0.0f64; 3];
        for i in 0..plane {
            for c in 0..3 {
                sum[c] += t[c * plane + i];
            }
        }
        let mean = (sum[0] + sum[1] + sum[2]) / 3.0;
        println!(
            "frame sum of terms {:.3} {:.3} {:.3}  chroma {:.3} {:.3} {:.3}",
            sum[0], sum[1], sum[2], sum[0] / mean, sum[1] / mean, sum[2] / mean
        );
    }
    // Peak pixel of the finished picture.
    let (mut best, mut at) = (0.0f64, (cx, cy));
    for y in cy.saturating_sub(rad)..(cy + rad + 1).min(h) {
        for x in cx.saturating_sub(rad)..(cx + rad + 1).min(w) {
            let v = px(&out, x, y);
            let l = v[0] + v[1] + v[2];
            if l > best {
                best = l;
                at = (x, y);
            }
        }
    }
    let v = px(&out, at.0, at.1);
    let m = (v[0] + v[1] + v[2]) / 3.0;
    println!(
        "brightest after at {at:?}: {:.3} {:.3} {:.3} chroma {:.3} {:.3} {:.3}",
        v[0],
        v[1],
        v[2],
        v[0] / m,
        v[1] / m,
        v[2] / m
    );
}
