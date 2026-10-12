//! R3 vignetting and R9 distortion / CA: the image checks of `docs/05` section Verifying lens
//! corrections (straight lines become straight, corners become flat), the auto crop, tiling,
//! agreement with the independent Python model in `tools/lens/lens_model.py`, and agreement with
//! what `darktable-cli` 4.6.1 measurably does (`tests/cpu_data/darktable_oracle.json`, recorded by
//! `tools/lens/record_oracle.py`).

use ym_gpu::cpu::lens::{LensGeometry, LensSwitches, LensTags};
use ym_gpu::cpu::resample::resample_lens;
use ym_gpu::cpu::spline::MonotoneCubic;
use ym_gpu::cpu::synth::lines_value;
use ym_gpu::cpu::tiling::{Rect, tile_grid};
use ym_gpu::cpu::vignette::{VignetteProfile, apply_vignette, knot_attenuation};
use ym_gpu::cpu::{Cfa, Mosaic, Placement, RgbImage};

const W: usize = 600;
const H: usize = 400;
const SPACING: f64 = 50.0;
const OFFSET: f64 = 25.0;
const LINE_SIGMA: f64 = 1.3;

fn barrel() -> LensTags {
    // Source radius up to 10% larger than the output radius at the corner (quadratic profile).
    let d: Vec<i16> = (0..9)
        .map(|i| (1640.0 * (i as f64 / 8.0).powi(2)) as i16)
        .collect();
    // Red sampled up to 0.76% further out, blue the same inward (K - 1 = 16000 / 2^21 at the
    // corner): about 2.7 px of lateral CA there.
    let red: Vec<i16> = (0..9)
        .map(|i| (16000.0 * (i as f64 / 8.0).powi(2)) as i16)
        .collect();
    let blue: Vec<i16> = red.iter().map(|v| -v).collect();
    LensTags {
        distortion: d,
        ca: [red, blue].concat(),
        vignetting: vec![],
    }
}

/// The "sensor" image of the straight grid for channel `c`: every source pixel `q` shows the grid
/// at the output position `p` with `source_position(c, p) = q`. Found by bisection on the output
/// radius along the ray from the centre (the maps are radial and monotone).
fn distorted_lines(geo: &LensGeometry, channels_differ: bool, vertical: bool) -> RgbImage {
    let place = Placement::whole(W, H);
    let (cx, cy) = place.centre();
    let hd = place.half_diagonal();
    let mut img = RgbImage::new(W, H);
    for c in 0..3 {
        if !channels_differ && c != 1 {
            continue; // filled from the green plane below
        }
        let cc = if channels_differ { c } else { 1 };
        let plane = img.plane_mut(c);
        for y in 0..H {
            for x in 0..W {
                let (qx, qy) = (x as f64 + 0.5, y as f64 + 0.5);
                let (dx, dy) = (qx - cx, qy - cy);
                let rq = dx.hypot(dy);
                let value = if rq < 1e-9 {
                    lines_value(qx, qy, vertical, SPACING, OFFSET, LINE_SIGMA)
                } else {
                    let (ux, uy) = (dx / rq, dy / rq);
                    let (mut lo, mut hi) = (0.0, rq * 1.2 + 5.0);
                    for _ in 0..40 {
                        let mid = 0.5 * (lo + hi);
                        let (sx, sy) = geo.source_position(cc, cx + ux * mid, cy + uy * mid);
                        if (sx - cx).hypot(sy - cy) < rq {
                            lo = mid;
                        } else {
                            hi = mid;
                        }
                    }
                    let r = 0.5 * (lo + hi);
                    lines_value(
                        cx + ux * r,
                        cy + uy * r,
                        vertical,
                        SPACING,
                        OFFSET,
                        LINE_SIGMA,
                    )
                };
                plane[y * W + x] = value as f32;
            }
        }
    }
    let _ = hd;
    if !channels_differ {
        let green = img.plane(1).to_vec();
        img.plane_mut(0).copy_from_slice(&green);
        img.plane_mut(2).copy_from_slice(&green);
    }
    img
}

/// Sub-pixel centroid track of a dark line: starting at line index `x0` on the middle row of
/// `plane` (`pw x ph`), follow the darkness centroid row by row with a +-8 px window. Returns
/// `(row, centroid x)` pairs for rows in `rows`.
fn trace_vertical(
    plane: &[f32],
    pw: usize,
    ph: usize,
    x_guess: f64,
    rows: std::ops::Range<usize>,
) -> Vec<(f64, f64)> {
    let centroid = |row: usize, around: f64| -> f64 {
        let lo = ((around - 8.0).floor().max(0.0)) as usize;
        let hi = ((around + 8.0).ceil() as usize).min(pw - 1);
        let (mut sw, mut sx) = (0.0f64, 0.0f64);
        for i in lo..=hi {
            let wgt = (1.0 - f64::from(plane[row * pw + i])).max(0.0);
            sw += wgt;
            sx += wgt * (i as f64 + 0.5);
        }
        sx / sw
    };
    let mid = ph / 2;
    let start = centroid(mid, x_guess);
    let mut out = Vec::new();
    let mut cur = start;
    for r in (rows.start..=mid).rev() {
        cur = centroid(r, cur);
        out.push((r as f64 + 0.5, cur));
    }
    cur = start;
    for r in mid + 1..rows.end {
        cur = centroid(r, cur);
        out.push((r as f64 + 0.5, cur));
    }
    out
}

/// RMS distance (px) of a traced line from its best straight-line fit `x = a y + b`.
fn line_residual(track: &[(f64, f64)]) -> f64 {
    let n = track.len() as f64;
    let (sy, sx) = track
        .iter()
        .fold((0.0, 0.0), |(a, b), &(y, x)| (a + y, b + x));
    let (my, mx) = (sy / n, sx / n);
    let (mut syy, mut sxy) = (0.0, 0.0);
    for &(y, x) in track {
        syy += (y - my) * (y - my);
        sxy += (y - my) * (x - mx);
    }
    let a = sxy / syy;
    let b = mx - a * my;
    (track
        .iter()
        .map(|&(y, x)| (x - (a * y + b)).powi(2))
        .sum::<f64>()
        / n)
        .sqrt()
}

/// Worst residual over all vertical grid lines inside the image, in the plane `c` of `img`.
fn worst_vertical_residual(img: &RgbImage, c: usize, crop_x: usize) -> f64 {
    let plane = img.plane(c);
    let (pw, ph) = (img.width, img.height);
    let margin = 30.0;
    let mut worst = 0.0f64;
    let mut k = 0;
    loop {
        let x_scene = OFFSET + SPACING * k as f64 - crop_x as f64;
        k += 1;
        if x_scene > pw as f64 + 400.0 {
            break;
        }
        if x_scene < margin || x_scene > pw as f64 - margin {
            continue;
        }
        let track = trace_vertical(plane, pw, ph, x_scene, 6..ph - 6);
        worst = worst.max(line_residual(&track));
    }
    worst
}

#[test]
fn straight_lines_are_straight_after_distortion_correction() {
    let tags = LensTags {
        distortion: barrel().distortion,
        ..Default::default()
    };
    let geo = LensGeometry::new(W, H, &tags, LensSwitches::default()).unwrap();
    let sensor = distorted_lines(&geo, false, true);
    let before = worst_vertical_residual(&sensor, 1, 0);

    let geo_crop = geo.clone().with_auto_crop();
    let corrected = resample_lens(
        &sensor,
        &geo_crop,
        Rect::whole(geo_crop.output_size().0, geo_crop.output_size().1),
    )
    .unwrap();
    let after = worst_vertical_residual(&corrected, 1, geo_crop.crop().x);
    println!(
        "line-fit residual: {before:.3} px before, {after:.4} px after; crop {:?}",
        geo_crop.crop()
    );
    assert!(
        before > 1.5,
        "the synthetic distortion is too weak to test ({before} px)"
    );
    assert!(after < 0.1, "lines still bent after correction: {after} px");
    assert!(after < before / 20.0);
}

#[test]
fn horizontal_lines_are_straight_too() {
    let tags = LensTags {
        distortion: barrel().distortion,
        ..Default::default()
    };
    let geo = LensGeometry::new(W, H, &tags, LensSwitches::default())
        .unwrap()
        .with_auto_crop();
    let sensor = distorted_lines(&geo, false, false);
    let (cw, ch) = geo.output_size();
    let corrected = resample_lens(&sensor, &geo, Rect::whole(cw, ch)).unwrap();
    // Transpose so that horizontal lines become vertical ones and reuse the tracker.
    let transpose = |img: &RgbImage| {
        let mut t = RgbImage::new(img.height, img.width);
        for c in 0..3 {
            for y in 0..img.height {
                for x in 0..img.width {
                    t.plane_mut(c)[x * img.height + y] = img.plane(c)[y * img.width + x];
                }
            }
        }
        t
    };
    let before = worst_vertical_residual(&transpose(&sensor), 1, 0);
    let after = worst_vertical_residual(&transpose(&corrected), 1, geo.crop().y);
    println!("horizontal lines: {before:.3} px before, {after:.4} px after");
    assert!(
        before > 1.5 && after < 0.1 && after < before / 10.0,
        "{before} -> {after}"
    );
}

#[test]
fn lateral_ca_is_removed() {
    let geo = LensGeometry::new(W, H, &barrel(), LensSwitches::default()).unwrap();
    let sensor = distorted_lines(&geo, true, true);
    let geo_crop = geo.clone().with_auto_crop();
    let (cw, ch) = geo_crop.output_size();
    let corrected = resample_lens(&sensor, &geo_crop, Rect::whole(cw, ch)).unwrap();

    // Distance between the red / blue and green centroids of the outermost vertical line, on the
    // middle row, before (sensor) and after (corrected).
    let offset = |img: &RgbImage, crop_x: usize, c: usize| -> f64 {
        let x_scene = OFFSET + SPACING * 10.0 - crop_x as f64; // line k = 10, near the right edge
        let row = img.height / 4; // away from the optical centre
        let t = |plane: &[f32]| {
            trace_vertical(plane, img.width, img.height, x_scene, row..row + 2)[0].1
        };
        t(img.plane(c)) - t(img.plane(1))
    };
    let (red_before, blue_before) = (offset(&sensor, 0, 0), offset(&sensor, 0, 2));
    let (red_after, blue_after) = (
        offset(&corrected, geo_crop.crop().x, 0),
        offset(&corrected, geo_crop.crop().x, 2),
    );
    println!(
        "CA offset of line 10: red {red_before:.3} -> {red_after:.3} px, blue {blue_before:.3} -> {blue_after:.3} px"
    );
    assert!(
        red_before.abs() > 0.4 && blue_before.abs() > 0.4,
        "synthetic CA too weak: {red_before}, {blue_before}"
    );
    assert!(
        red_after.abs() < 0.08 && blue_after.abs() < 0.08,
        "CA left: red {red_after}, blue {blue_after}"
    );
}

#[test]
fn auto_crop_output_samples_only_valid_source_positions() {
    let tags = LensTags {
        distortion: barrel().distortion,
        ca: barrel().ca,
        ..Default::default()
    };
    let geo = LensGeometry::new(W, H, &tags, LensSwitches::default())
        .unwrap()
        .with_auto_crop();
    let (cw, ch) = geo.output_size();
    assert!(cw < W && ch < H, "barrel correction must crop");
    assert!((cw as f64 / ch as f64 - W as f64 / H as f64).abs() < 0.01);
    for y in 0..ch {
        for x in 0..cw {
            for c in 0..3 {
                let (sx, sy) = geo.source_position_cropped(c, x, y);
                assert!(
                    sx >= 0.5 && sx <= W as f64 - 0.5 && sy >= 0.5 && sy <= H as f64 - 0.5,
                    "({x},{y}) c{c} -> ({sx},{sy})"
                );
            }
        }
    }
}

#[test]
fn tiled_resample_equals_whole() {
    let geo = LensGeometry::new(W, H, &barrel(), LensSwitches::default())
        .unwrap()
        .with_auto_crop();
    let sensor = distorted_lines(&geo, true, true);
    let (cw, ch) = geo.output_size();
    let whole = resample_lens(&sensor, &geo, Rect::whole(cw, ch)).unwrap();
    let mut tiled = RgbImage::new(cw, ch);
    for core in tile_grid(cw, ch, 150, 120, 1) {
        let need = geo.source_rect(core);
        let tile_src = sensor.extract(need);
        let out = resample_lens(&tile_src, &geo, core).unwrap();
        tiled.paste(&out, Rect::whole(core.w, core.h), core.x, core.y);
    }
    assert!(
        whole.max_abs_diff(&tiled) <= 1e-6,
        "tiled R9 differs from whole by {}",
        whole.max_abs_diff(&tiled)
    );
}

#[test]
fn vignetting_correction_flattens_corners() {
    // Strong, Sony-like profile: the lens leaves 2^-0.5 or so of the light at the corner.
    let knots: [i16; 12] = [
        0, 150, 500, 1000, 1700, 2600, 3700, 5000, 6500, 8000, 9500, 11000,
    ];
    let profile = VignetteProfile::from_tag(&knots).unwrap();
    let mut m = Mosaic::new(W, H, Cfa::RGGB);
    for y in 0..H {
        for x in 0..W {
            m.data[y * W + x] =
                (0.6 * profile.attenuation_at(VignetteProfile::radius(&m.place, x, y))) as f32;
        }
    }
    let ratio = |m: &Mosaic| {
        let centre = (m.get(W / 2, H / 2)
            + m.get(W / 2 + 1, H / 2)
            + m.get(W / 2, H / 2 + 1)
            + m.get(W / 2 + 1, H / 2 + 1))
            / 4.0;
        let corner = (m.get(0, 0) + m.get(1, 0) + m.get(0, 1) + m.get(1, 1)) / 4.0;
        corner / centre
    };
    let before = ratio(&m);
    apply_vignette(&mut m, &profile);
    let after = ratio(&m);
    println!("corner / centre: {before:.4} -> {after:.4}");
    assert!(before < 0.7);
    assert!(
        (after - 1.0).abs() < 0.005,
        "corner/centre {after} after correction"
    );
}

// ---------------------------------------------------------------------------------------------
// Agreement with the independent Python model (tools/lens/lens_model.py).
// ---------------------------------------------------------------------------------------------

fn vectors() -> serde_json::Value {
    let path =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/cpu_data/lens_vectors.json");
    serde_json::from_str(&std::fs::read_to_string(path).expect("lens_vectors.json")).expect("json")
}

fn i16s(v: &serde_json::Value) -> Vec<i16> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_i64().unwrap() as i16)
        .collect()
}

#[test]
fn matches_the_python_model() {
    let fx = vectors();
    for case in fx["cases"].as_array().unwrap() {
        let tags = LensTags {
            vignetting: i16s(&case["vignetting"]),
            distortion: i16s(&case["distortion"]),
            ca: i16s(&case["ca"]),
        };
        let (w, h) = (
            case["width"].as_u64().unwrap() as usize,
            case["height"].as_u64().unwrap() as usize,
        );
        // Vignetting: relative illumination G(r) and the multiplier 1 / G(r).
        let prof = VignetteProfile::from_tag(&tags.vignetting).unwrap();
        for pair in case["attenuation"].as_array().unwrap() {
            let (r, want) = (pair[0].as_f64().unwrap(), pair[1].as_f64().unwrap());
            assert!(
                (prof.attenuation_at(r) - want).abs() < 1e-12,
                "G({r}): {} vs {want}",
                prof.attenuation_at(r)
            );
        }
        for pair in case["gain"].as_array().unwrap() {
            let (r, want) = (pair[0].as_f64().unwrap(), pair[1].as_f64().unwrap());
            assert!(
                (prof.gain_at(r) - want).abs() < 1e-12,
                "1/G({r}): {} vs {want}",
                prof.gain_at(r)
            );
        }
        for (v, want) in tags
            .vignetting
            .iter()
            .zip(case["knot_attenuation"].as_array().unwrap())
        {
            assert!((knot_attenuation(*v) - want.as_f64().unwrap()).abs() < 1e-12);
        }
        // Source positions for each channel.
        let geo = LensGeometry::new(w, h, &tags, LensSwitches::default()).unwrap();
        for item in case["positions"].as_array().unwrap() {
            let (px, py) = (item["x"].as_f64().unwrap(), item["y"].as_f64().unwrap());
            for (c, key) in ["red", "green", "blue"].iter().enumerate() {
                let want = (
                    item[*key][0].as_f64().unwrap(),
                    item[*key][1].as_f64().unwrap(),
                );
                let got = geo.source_position(c, px, py);
                assert!(
                    (got.0 - want.0).abs() < 1e-9 && (got.1 - want.1).abs() < 1e-9,
                    "{key} at ({px},{py}): {got:?} vs {want:?}"
                );
            }
        }
    }
    // The monotone cubic (kept in `spline`, no longer used by the lens stages) against SciPy on a
    // non-uniform case.
    let s = &fx["pchip"];
    let xs: Vec<f64> = s["xs"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();
    let ys: Vec<f64> = s["ys"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();
    let spline = MonotoneCubic::new(&xs, &ys).unwrap();
    for pair in s["samples"].as_array().unwrap() {
        let (x, want) = (pair[0].as_f64().unwrap(), pair[1].as_f64().unwrap());
        assert!((spline.eval(x) - want).abs() < 1e-12, "pchip({x})");
    }
}

// ---------------------------------------------------------------------------------------------
// Agreement with darktable-cli 4.6.1 (recorded measurements, tools/lens/record_oracle.py).
//
// darktable counts pixel *indices* from the frame centre (W/2, H/2). Our functions depend on the
// offset from the centre only, so feeding them darktable's index coordinates reproduces its numbers
// exactly; the half-pixel difference to pixel-centre coordinates is a shift of the whole frame.
// ---------------------------------------------------------------------------------------------

fn oracle() -> serde_json::Value {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/cpu_data/darktable_oracle.json");
    serde_json::from_str(&std::fs::read_to_string(path).expect("darktable_oracle.json"))
        .expect("json")
}

fn f64s(v: &serde_json::Value) -> Vec<f64> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap())
        .collect()
}

/// darktable's auto-scale for a table of `cor` values at the Sony knots: the largest value over
/// 200 radii from the short half-side to the corner (see `lens_model.darktable_zoom`).
fn darktable_zoom(w: usize, h: usize, cor: &[Vec<f64>]) -> f64 {
    let (cx, cy) = (w as f64 / 2.0, h as f64 / 2.0);
    let srr = cx.min(cy) / cx.hypot(cy);
    let mut zoom = 0.0f64;
    for table in cor {
        let knots = ym_gpu::cpu::spline::LinearKnots::sony(table).unwrap();
        for i in 0..200 {
            zoom = zoom.max(knots.eval(srr + (1.0 - srr) * i as f64 / 199.0));
        }
    }
    zoom
}

#[test]
fn vignetting_matches_darktable() {
    let fx = oracle();
    let v = &fx["vignetting"];
    let tags: Vec<i16> = f64s(&v["tags"]).iter().map(|&x| x as i16).collect();
    let prof = VignetteProfile::from_tag(&tags).unwrap();
    let mut worst = 0.0f64;
    let samples = v["samples"].as_array().unwrap();
    assert!(samples.len() > 60);
    for pair in samples {
        let (r, measured) = (pair[0].as_f64().unwrap(), pair[1].as_f64().unwrap());
        worst = worst.max((prof.gain_at(r) / measured - 1.0).abs());
    }
    println!("vignetting gain vs darktable: worst relative deviation {worst:.2e}");
    // docs/13 allows 0.5%; the measurement agrees to the 16-bit float noise of the render.
    assert!(worst < 2e-4, "{worst}");
    // The recorded flat field is corrected *up* in the corner: positive knots mean a darker corner
    // (G < 1), and darktable divides by G.
    assert!(samples.last().unwrap()[1].as_f64().unwrap() > 1.15);
}

#[test]
fn distortion_matches_darktable() {
    let fx = oracle();
    let (w, h) = (
        fx["frame"][0].as_u64().unwrap() as usize,
        fx["frame"][1].as_u64().unwrap() as usize,
    );
    let d = &fx["distortion"];
    let tags = LensTags {
        distortion: f64s(&d["tags"]).iter().map(|&x| x as i16).collect(),
        ..Default::default()
    };
    let geo = LensGeometry::new(w, h, &tags, LensSwitches::default()).unwrap();
    // darktable's zoom, from our own table, must be the one it applied.
    let table: Vec<f64> = tags
        .distortion
        .iter()
        .map(|&k| ym_gpu::cpu::lens::distortion_factor(k))
        .collect();
    let zoom = darktable_zoom(w, h, &[table]);
    assert!(
        (zoom - d["zoom"].as_f64().unwrap()).abs() < 1e-5,
        "zoom {zoom}"
    );
    let (cx, cy) = (w as f64 / 2.0, h as f64 / 2.0);
    let mut worst = 0.0f64;
    let samples = d["samples"].as_array().unwrap();
    assert!(samples.len() > 60);
    for s in samples {
        let (y, x_out, x_src) = (
            s[0].as_f64().unwrap(),
            s[1].as_f64().unwrap(),
            s[2].as_f64().unwrap(),
        );
        // The unzoomed position of the corrected pixel, then our map back to the source.
        let (px, py) = (cx + (x_out - cx) / zoom, cy + (y - cy) / zoom);
        let (sx, _) = geo.source_position(1, px, py);
        worst = worst.max((sx - x_src).abs());
    }
    println!("distortion vs darktable: worst line-position deviation {worst:.4} px");
    assert!(worst < 0.03, "{worst} px (docs/13: 0.2 px)");
}

#[test]
fn lateral_ca_matches_darktable() {
    let fx = oracle();
    let (w, h) = (
        fx["frame"][0].as_u64().unwrap() as usize,
        fx["frame"][1].as_u64().unwrap() as usize,
    );
    let c = &fx["ca"];
    let ca: Vec<i16> = f64s(&c["tags"]).iter().map(|&x| x as i16).collect();
    let tags = LensTags {
        ca: ca.clone(),
        ..Default::default()
    };
    let geo = LensGeometry::new(w, h, &tags, LensSwitches::default()).unwrap();
    // darktable zooms by the largest of the three channels' factors (green's is 1).
    let n = ca.len() / 2;
    let red: Vec<f64> = ca[..n]
        .iter()
        .map(|&k| ym_gpu::cpu::lens::ca_factor(k))
        .collect();
    let blue: Vec<f64> = ca[n..]
        .iter()
        .map(|&k| ym_gpu::cpu::lens::ca_factor(k))
        .collect();
    let zoom = darktable_zoom(w, h, &[red, vec![1.0; n], blue]);
    assert!(
        (zoom - c["zoom"].as_f64().unwrap()).abs() < 1e-5,
        "zoom {zoom}"
    );
    let (cx, cy) = (w as f64 / 2.0, h as f64 / 2.0);
    let mut worst = [0.0f64; 3];
    let samples = c["samples"].as_array().unwrap();
    assert!(samples.len() > 60);
    for s in samples {
        let y = s[0].as_f64().unwrap();
        let x_src = s[1].as_f64().unwrap();
        for ch in 0..3 {
            let x_out = s[2 + ch].as_f64().unwrap();
            let (px, py) = (cx + (x_out - cx) / zoom, cy + (y - cy) / zoom);
            let (sx, _) = geo.source_position(ch, px, py);
            worst[ch] = worst[ch].max((sx - x_src).abs());
        }
    }
    println!("CA vs darktable: worst deviation (red, green, blue) {worst:.4?} px");
    // The recorded CA is 8x the size of a strong real one; the lines are located to 0.01 px.
    assert!(
        worst.iter().all(|&e| e < 0.03),
        "{worst:?} px (docs/13: 0.2 px)"
    );
}
