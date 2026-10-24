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
    eprintln!("sharp worst b*hd = {:.2} (b_min px = {:.2})", f64::from(worst) * hd, 0.004 * hd);
    let _ = (night_scene, sharp_scene);
}
