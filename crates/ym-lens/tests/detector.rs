//! The bokeh detector on synthetic scenes with known blur (`docs/21` section 14).

mod common;
mod scenes;

use common::{plan, whole};
use scenes::{Noise, Scene, night_scene, sharp_scene, subject_scene};
use ym_lens::{Axis, BokehPlanes, LensKind, LensPlanes};

fn planes_of(scene: &Scene) -> (ym_lens::LensPlan, LensPlanes) {
    let (w, h) = (scene.img.w, scene.img.h);
    let p = plan(LensKind::Default, 0.0, 0.0, 100.0, w, h, Axis::X);
    assert_eq!(p.lens_plane_scale(), 1);
    let planes = LensPlanes::build(&p, &whole(w, h, &scene.img.data), scene.focus, &|| false).expect("planes");
    (p, planes)
}

#[test]
fn explore() {
    let t = std::time::Instant::now();
    let scene = subject_scene(1000, 667, [5.0, 9.0, 14.0], Noise::LOW, 1);
    eprintln!("scene {:?}", t.elapsed());
    let t = std::time::Instant::now();
    let (_p, planes) = planes_of(&scene);
    eprintln!("planes {:?}", t.elapsed());
    let bp: &BokehPlanes = planes.bokeh().unwrap();
    eprintln!("report {:?} edges {} frac {} active {} bmax {}", planes.report(), bp.edge_count(), bp.blurred_fraction(), bp.active(), bp.max_blur());
    let (w, h) = bp.size();
    let hd = 0.5 * ((w * w + h * h) as f64).sqrt();
    let b = bp.blur();
    let c = bp.confidence();
    for band in 0..3 {
        let (mut sum, mut cnt) = (0.0f64, 0usize);
        let mut vals = Vec::new();
        for y in 0..h {
            for x in 0..w {
                let i = y * w + x;
                if scene.trusted[i] && scene.truth_r[i] > 0.0 && (x * 3 / w) == band && c[i] > 0.5 && b[i] > 0.0 {
                    let est = f64::from(b[i]) * hd;
                    vals.push(est / f64::from(scene.truth_r[i]));
                    sum += est;
                    cnt += 1;
                }
            }
        }
        vals.sort_by(|a, b| a.total_cmp(b));
        let total = (0..h).flat_map(|y| (0..w).map(move |x| (x, y))).filter(|&(x, y)| scene.trusted[y * w + x] && (x * 3 / w) == band).count();
        if !vals.is_empty() {
            eprintln!("band {band}: confident {cnt}/{total}, ratio p10 {:.2} p50 {:.2} p90 {:.2}", vals[vals.len() / 10], vals[vals.len() / 2], vals[vals.len() * 9 / 10]);
        } else {
            eprintln!("band {band}: no confident pixel of {total}");
        }
    }
    let mut worst = 0.0f32;
    for i in 0..w * h {
        if scene.subject[i] || scene.strand[i] {
            worst = worst.max(b[i]);
        }
    }
    {
        let (mut ns, mut nviol, mut nst, mut nstv) = (0usize, 0usize, 0usize, 0usize);
        let mut bins = [0usize; 6];
        for i in 0..w * h {
            if scene.subject[i] {
                ns += 1;
                if f64::from(b[i]) * hd > 2.4 {
                    nviol += 1;
                    let v = f64::from(b[i]) * hd;
                    bins[((v / 5.0) as usize).min(5)] += 1;
                }
            }
            if scene.strand[i] {
                nst += 1;
                if f64::from(b[i]) * hd > 2.4 {
                    nstv += 1;
                }
            }
        }
        eprintln!("subject pixels {ns}, violating {nviol} (b px in 5-wide bins {bins:?}); strand pixels {nst}, violating {nstv}");
    }
    eprintln!("sharp worst b*hd = {:.2} (b_min px = {:.2})", f64::from(worst) * hd, 0.004 * hd);
    let _ = (night_scene, sharp_scene);
    if let Ok(dir) = std::env::var("YM_L1_DUMP") {
        let f32s = |v: &[f32]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
        let px: Vec<f32> = b.iter().map(|v| (f64::from(*v) * hd) as f32).collect();
        std::fs::write(format!("{dir}/lum.f32"), f32s(&scene.img.luminance())).unwrap();
        std::fs::write(format!("{dir}/truth.f32"), f32s(&scene.truth_r)).unwrap();
        std::fs::write(format!("{dir}/b.f32"), f32s(&px)).unwrap();
        std::fs::write(format!("{dir}/conf.f32"), f32s(c)).unwrap();
        std::fs::write(format!("{dir}/size.txt"), format!("{w} {h}")).unwrap();
    }
    // Sample-level statistics.
    let (sw, sh, shd, nv, samples) = ym_lens::debug_samples(&whole(scene.img.w, scene.img.h, &scene.img.data));
    eprintln!("work grid {sw}x{sh} hd {shd:.1} noise_v {nv:.4} samples {}", samples.len());
    for band in 0..3 {
      for pair in 0..5u8 {
        let mut ratios = Vec::new();
        let mut wsum = 0.0;
        for &(idx, wt, ln_r, pr) in &samples {
            if pr != pair { continue; }
            let (x, y) = (idx as usize % sw, idx as usize / sw);
            let i = y * sw + x;
            if scene.trusted[i] && scene.truth_r[i] > 0.0 && (x * 3 / sw) == band {
                ratios.push((f64::from(ln_r).exp() / f64::from(scene.truth_r[i]), wt));
                wsum += f64::from(wt);
            }
        }
        ratios.sort_by(|a, b| a.0.total_cmp(&b.0));
        let q = |p: f64| ratios[((ratios.len() - 1) as f64 * p) as usize].0;
        if !ratios.is_empty() {
            eprintln!("band {band} pair {pair}: {} samples (weight {:.0}), ratio p10 {:.2} p25 {:.2} p50 {:.2} p75 {:.2} p90 {:.2}", ratios.len(), wsum, q(0.1), q(0.25), q(0.5), q(0.75), q(0.9));
        }
      }
    }
}

#[test]
fn explore_step() {
    use scenes::{Rgb, disc_blur};
    let (w, h) = (400usize, 120usize);
    for r in [3.0, 5.0, 9.0, 14.0, 20.0] {
        let mut step = vec![0.1f32; w * h];
        for y in 0..h {
            for x in 200..w {
                step[y * w + x] = 0.8;
            }
        }
        let blurred = disc_blur(&step, w, h, r);
        let mut img = Rgb::new(w, h);
        for c in 0..3 {
            img.plane_mut(c).copy_from_slice(&blurred);
        }
        let (sw, _sh, _hd, _nv, samples) = ym_lens::debug_samples(&whole(w, h, &img.data));
        let mut v: Vec<(f64, f32, usize)> = samples
            .iter()
            .filter(|(idx, ..)| {
                let (x, y) = (*idx as usize % sw, *idx as usize / sw);
                y > 30 && y < 90 && (x as i64 - 200).abs() < 3
            })
            .map(|(idx, wt, ln, _)| (f64::from(*ln).exp() / r, *wt, *idx as usize % sw))
            .collect();
        v.sort_by(|a, b| a.0.total_cmp(&b.0));
        let wsum: f64 = v.iter().map(|s| f64::from(s.1)).sum();
        let mean: f64 = v.iter().map(|s| s.0.ln() * f64::from(s.1)).sum::<f64>() / wsum;
        eprintln!("r {r}: {} samples, weight {wsum:.1}, weighted geo-mean ratio {:.3}, min {:.2} max {:.2}", v.len(), mean.exp(), v.first().map_or(0.0, |s| s.0), v.last().map_or(0.0, |s| s.0));
    }
}

#[test]
fn explore_night() {
    let radii = [6.0, 8.0, 10.0, 12.0, 14.0, 7.0, 9.0, 11.0, 13.0, 15.0, 5.5, 16.0];
    for (name, noise) in [("none", Noise::NONE), ("low", Noise::LOW), ("mid", Noise::MID), ("high", Noise::HIGH)] {
        let scene = night_scene(1000, 667, &radii, noise, 7);
        let t = std::time::Instant::now();
        let (_p, planes) = planes_of(&scene);
        let bp = planes.bokeh().unwrap();
        let hd = 601.0;
        eprintln!("night {name}: {:?}, balls {} (of {}), report {:?}, frac {:.3} edges {}", t.elapsed(), bp.balls().len(), scene.discs.len(), planes.report(), bp.blurred_fraction(), bp.edge_count());
        let (w, h) = bp.size();
        for &(cx, cy, r) in &scene.discs {
            let found = bp.balls().iter().find(|b| (b.x * w as f64 - cx).hypot(b.y * h as f64 - cy) < 0.6 * r);
            match found {
                Some(b) => {
                    let est = b.radius * hd;
                    let true_energy = std::f64::consts::PI * r * r;
                    let _ = true_energy;
                    eprintln!("  disc r {r:5.1} at ({cx:.0},{cy:.0}): ball r {est:.2} ({:+.1}%), centre err {:.2} px, energy {:.3?} supported {}", (est / r - 1.0) * 100.0, (b.x * w as f64 - cx).hypot(b.y * h as f64 - cy), b.energy.map(|e| e * (w * h) as f64), b.supported);
                }
                None => eprintln!("  disc r {r:5.1} at ({cx:.0},{cy:.0}): NOT FOUND"),
            }
        }
        let false_balls = bp.balls().iter().filter(|b| !scene.discs.iter().any(|&(cx, cy, r)| (b.x * w as f64 - cx).hypot(b.y * h as f64 - cy) < 0.6 * r)).count();
        eprintln!("  false balls {false_balls}");
    }
}

#[test]
fn explore_sharp_and_flat() {
    for (name, scene) in [
        ("sharp none", sharp_scene(1000, 667, Noise::NONE, 3)),
        ("sharp mid", sharp_scene(1000, 667, Noise::MID, 3)),
        ("sharp high", sharp_scene(1000, 667, Noise::HIGH, 3)),
        ("flat low", scenes::flat_scene(1000, 667, Noise::LOW, 3)),
    ] {
        let (_p, planes) = planes_of(&scene);
        let bp = planes.bokeh().unwrap();
        let hd = 601.0;
        if name == "sharp mid" {
            if let Ok(dir) = std::env::var("YM_L1_DUMP") {
                let f32s = |v: &[f32]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
                let px: Vec<f32> = bp.blur().iter().map(|v| (f64::from(*v) * hd) as f32).collect();
                std::fs::write(format!("{dir}/lum.f32"), f32s(&scene.img.luminance())).unwrap();
                std::fs::write(format!("{dir}/truth.f32"), f32s(&scene.truth_r)).unwrap();
                std::fs::write(format!("{dir}/b.f32"), f32s(&px)).unwrap();
                std::fs::write(format!("{dir}/conf.f32"), f32s(bp.confidence())).unwrap();
                std::fs::write(format!("{dir}/size.txt"), "1000 667").unwrap();
            }
        }
        let nz = bp.blur().iter().filter(|b| f64::from(**b) * hd > 2.4).count();
        let mx = bp.blur().iter().fold(0.0f32, |m, v| m.max(*v));
        eprintln!("{name}: report {:?} frac {:.4} edges {} pixels above b_min {} ({:.3}%), max b px {:.1}, balls {}", planes.report(), bp.blurred_fraction(), bp.edge_count(), nz, 100.0 * nz as f64 / bp.blur().len() as f64, f64::from(mx) * hd, bp.balls().len());
    }
}

#[test]
fn explore_conf() {
    let hd = 601.0;
    let pct = |v: &mut Vec<f32>, p: f64| -> f32 { v.sort_by(|a, b| a.total_cmp(b)); if v.is_empty() { 0.0 } else { v[((v.len() - 1) as f64 * p) as usize] } };
    let sharp = sharp_scene(1000, 667, Noise::MID, 3);
    let (_p, planes) = planes_of(&sharp);
    let bp = planes.bokeh().unwrap();
    let mut cs: Vec<f32> = bp.blur().iter().zip(bp.confidence()).filter(|(b, _)| f64::from(**b) * hd > 2.4).map(|(_, c)| *c).collect();
    eprintln!("sharp mid false-positive confidences: n {} p10 {:.2} p50 {:.2} p90 {:.2} p99 {:.2}", cs.len(), pct(&mut cs, 0.1), pct(&mut cs, 0.5), pct(&mut cs, 0.9), pct(&mut cs, 0.99));
    let scene = subject_scene(1000, 667, [5.0, 9.0, 14.0], Noise::MID, 2);
    let (_p, planes) = planes_of(&scene);
    let bp = planes.bokeh().unwrap();
    let mut cs: Vec<f32> = (0..bp.blur().len()).filter(|&i| scene.trusted[i] && scene.truth_r[i] > 0.0 && f64::from(bp.blur()[i]) * hd > 2.4).map(|i| bp.confidence()[i]).collect();
    eprintln!("subject-scene background confidences: n {} p10 {:.2} p25 {:.2} p50 {:.2} p90 {:.2}", cs.len(), pct(&mut cs, 0.1), pct(&mut cs, 0.25), pct(&mut cs, 0.5), pct(&mut cs, 0.9));
    for thr in [0.3f32, 0.4, 0.5, 0.6, 0.7, 0.8] {
        let (mut n, mut ok) = (0usize, 0usize);
        let mut errs = Vec::new();
        for i in 0..bp.blur().len() {
            if scene.trusted[i] && scene.truth_r[i] > 0.0 && bp.confidence()[i] >= thr && bp.blur()[i] > 0.0 {
                n += 1;
                let ratio = f64::from(bp.blur()[i]) * hd / f64::from(scene.truth_r[i]);
                if (ratio - 1.0).abs() <= 0.25 { ok += 1; }
                errs.push((ratio - 1.0).abs());
            }
        }
        errs.sort_by(|a, b| a.total_cmp(b));
        eprintln!("conf >= {thr}: {n} pixels, within 25%: {:.1}%, median |err| {:.1}%", 100.0 * ok as f64 / n as f64, 100.0 * errs[errs.len() / 2]);
    }
}
