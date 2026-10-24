//! L1, the bokeh reshaping, on synthetic scenes (`docs/21` sections 4 and 14).

mod common;
mod scenes;

use common::{apply_tile, plan, planes_for, whole};
use scenes::{Noise, Scene, night_scene, subject_scene};
use ym_lens::{Axis, ImageRef, LensKind, LensPlan, LensPlanes, PixelRect, apply, halo_px, reshape, reshape_halo};

fn planes_of(p: &LensPlan, scene: &Scene) -> LensPlanes {
    LensPlanes::build(p, &whole(scene.img.w, scene.img.h, &scene.img.data), scene.focus, &|| false).expect("planes")
}

/// L1 alone over `rect` of a full-frame image at scale `k`, given exactly the context it needs.
fn reshape_tile(p: &LensPlan, planes: &LensPlanes, k: usize, data: &[f32], w: usize, h: usize, rect: PixelRect) -> Vec<f32> {
    let halo = reshape_halo(p, planes, k);
    let need = rect.grown(halo, w, h);
    let buf = common::crop(data, w, h, need);
    let input = ImageRef { width: need.w, height: need.h, x0: need.x, y0: need.y, frame_width: w, frame_height: h, data: &buf };
    reshape(p, planes, k, &input, rect).expect("reshape")
}

#[test]
fn explore_subject() {
    let scene = subject_scene(1000, 667, [5.0, 9.0, 14.0], Noise::LOW, 1);
    let (w, h) = (scene.img.w, scene.img.h);
    let p = plan(LensKind::Default, 0.0, 0.0, 100.0, w, h, Axis::X);
    let planes = planes_of(&p, &scene);
    let t = std::time::Instant::now();
    let out = reshape_tile(&p, &planes, 1, &scene.img.data, w, h, PixelRect::new(0, 0, w, h));
    eprintln!("reshape {:?}, halo {}", t.elapsed(), reshape_halo(&p, &planes, 1));
    let n = w * h;
    let mut max_subj = 0.0f32;
    let mut max_strand = 0.0f32;
    let (mut changed_bg, mut bg) = (0usize, 0usize);
    for i in 0..n {
        for c in 0..3 {
            let d = (out[c * n + i] - scene.img.data[c * n + i]).abs();
            if scene.subject[i] { max_subj = max_subj.max(d); }
            if scene.strand[i] { max_strand = max_strand.max(d); }
        }
        if !scene.subject[i] && !scene.strand[i] {
            bg += 1;
            if (0..3).any(|c| (out[c * n + i] - scene.img.data[c * n + i]).abs() > 1e-6) { changed_bg += 1; }
        }
    }
    eprintln!("max change on subject {max_subj:e}, on strands {max_strand:e}; background pixels changed {changed_bg}/{bg}");
    if let Ok(dir) = std::env::var("YM_L1_DUMP") {
        let f32s = |v: &[f32]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
        std::fs::write(format!("{dir}/in.f32"), f32s(&scene.img.data)).unwrap();
        std::fs::write(format!("{dir}/out.f32"), f32s(&out)).unwrap();
        std::fs::write(format!("{dir}/size.txt"), format!("{w} {h}")).unwrap();
    }
}

#[test]
fn explore_discs() {
    let radii = [6.0, 8.0, 10.0, 12.0, 14.0, 7.0, 9.0, 11.0, 13.0, 15.0, 5.5, 16.0];
    let scene = night_scene(1000, 667, &radii, Noise::LOW, 7);
    let (w, h) = (scene.img.w, scene.img.h);
    for (axis, anam) in [(Axis::X, 100.0), (Axis::X, 50.0), (Axis::Y, 100.0)] {
        let p = plan(LensKind::Default, 0.0, 0.0, anam, w, h, axis);
        let planes = planes_of(&p, &scene);
        let t = std::time::Instant::now();
        let out = reshape_tile(&p, &planes, 1, &scene.img.data, w, h, PixelRect::new(0, 0, w, h));
        eprintln!("axis {axis:?} anamorphic {anam}: halo {} apply {:?} q {}", halo_px(&p, &planes, 1), t.elapsed(), p.q());
        let n = w * h;
        for &(cx, cy, r) in &scene.discs {
            let (x0, x1) = ((cx - 3.5 * r).max(0.0) as usize, ((cx + 3.5 * r) as usize).min(w));
            let (y0, y1) = ((cy - 3.5 * r).max(0.0) as usize, ((cy + 3.5 * r) as usize).min(h));
            let mut d_energy = [0.0f64; 3];
            let mut in_energy = [0.0f64; 3];
            for c in 0..3 {
                for y in y0..y1 {
                    for x in x0..x1 {
                        let i = c * n + y * w + x;
                        d_energy[c] += f64::from(out[i]) - f64::from(scene.img.data[i]);
                        let g = 0.006;
                        in_energy[c] += f64::from(scene.img.data[i]) - g;
                    }
                }
            }
            // Extent of the output along the vertical and horizontal lines through the centre at 35% of its peak.
            let lum = |x: usize, y: usize| -> f64 { f64::from(out[n + y * w + x]) };
            let (ix, iy) = (cx as usize, cy as usize);
            let peak = lum(ix, iy);
            let thr = 0.35 * peak;
            let extent = |dx: i64, dy: i64| -> f64 {
                let mut len = 0;
                let (mut x, mut y) = (ix as i64, iy as i64);
                while x >= 0 && y >= 0 && (x as usize) < w && (y as usize) < h && lum(x as usize, y as usize) > thr {
                    len += 1;
                    x += dx;
                    y += dy;
                }
                len as f64
            };
            let width = extent(1, 0) + extent(-1, 0) - 1.0;
            let height = extent(0, 1) + extent(0, -1) - 1.0;
            eprintln!(
                "  r {r:5.1}: width {width:5.1} height {height:5.1} ratio {:.3} (flat disc {:.1}); energy change {:+.2}% {:+.2}% {:+.2}%",
                if axis == Axis::X { height / width } else { width / height },
                2.0 * r,
                100.0 * d_energy[0] / in_energy[0],
                100.0 * d_energy[1] / in_energy[1],
                100.0 * d_energy[2] / in_energy[2],
            );
        }
    }
    let _ = (planes_for, apply, apply_tile, halo_px);
}

#[test]
fn explore_column() {
    let scene = night_scene(1000, 667, &[8.0, 12.0, 9.0, 6.0, 10.0, 14.0, 11.0, 7.0, 13.0, 15.0, 5.5, 16.0], Noise::NONE, 7);
    let (w, h) = (scene.img.w, scene.img.h);
    let p = plan(LensKind::Default, 0.0, 0.0, 100.0, w, h, Axis::X);
    let planes = planes_of(&p, &scene);
    let out = apply_tile(&p, &planes, 1, &scene.img.data, w, h, PixelRect::new(0, 0, w, h));
    let (cx, cy, r) = scene.discs[0];
    let n = w * h;
    let bp = planes.bokeh().unwrap();
    eprintln!("ball list {:?}", bp.balls().iter().map(|b| (b.x * w as f64, b.y * h as f64, b.radius * 601.0)).collect::<Vec<_>>());
    eprintln!("disc ({cx},{cy},{r}) in/out along the column:");
    for dy in -30..=30i64 {
        let (x, y) = (cx as usize, (cy as i64 + dy) as usize);
        eprintln!("  dy {dy:3}: in {:.3} out {:.3} b {:.4}", scene.img.data[n + y * w + x], out[n + y * w + x], bp.blur()[y * w + x] * 601.0);
    }
}

/// The scene reduced `k` times by box means (the frame at scale `k`).
fn reduce(data: &[f32], w: usize, h: usize, k: usize) -> (Vec<f32>, usize, usize) {
    let (rw, rh) = (w.div_ceil(k), h.div_ceil(k));
    let mut out = vec![0.0f32; 3 * rw * rh];
    for c in 0..3 {
        for y in 0..rh {
            for x in 0..rw {
                let (mut acc, mut cnt) = (0.0f64, 0.0f64);
                for dy in 0..k {
                    for dx in 0..k {
                        let (sx, sy) = (x * k + dx, y * k + dy);
                        if sx < w && sy < h {
                            acc += f64::from(data[c * w * h + sy * w + sx]);
                            cnt += 1.0;
                        }
                    }
                }
                out[c * rw * rh + y * rw + x] = (acc / cnt) as f32;
            }
        }
    }
    (out, rw, rh)
}

#[test]
fn explore_tiles() {
    let scene = subject_scene(1000, 667, [5.0, 9.0, 14.0], Noise::LOW, 1);
    let (w, h) = (scene.img.w, scene.img.h);
    for (kind, anam, edges) in [(LensKind::Default, 100.0, 0.0), (LensKind::Petzval, 60.0, 100.0)] {
        let p = plan(kind, 0.0, edges, anam, w, h, Axis::X);
        let planes = planes_of(&p, &scene);
        for k in [1usize, 4] {
            let (data, fw, fh) = if k == 1 { (scene.img.data.clone(), w, h) } else { reduce(&scene.img.data, w, h, k) };
            let t = std::time::Instant::now();
            let whole_out = reshape_tile(&p, &planes, k, &data, fw, fh, PixelRect::new(0, 0, fw, fh));
            let t_whole = t.elapsed();
            for (nx, ny) in [(2usize, 2usize), (3, 3)] {
                let mut assembled = vec![0.0f32; 3 * fw * fh];
                for rect in common::tiles(fw, fh, nx, ny) {
                    let tile = reshape_tile(&p, &planes, k, &data, fw, fh, rect);
                    for c in 0..3 {
                        for y in 0..rect.h {
                            let dst = c * fw * fh + (rect.y + y) * fw + rect.x;
                            assembled[dst..dst + rect.w].copy_from_slice(&tile[c * rect.w * rect.h + y * rect.w..][..rect.w]);
                        }
                    }
                }
                let diff = assembled.iter().zip(&whole_out).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
                let changed = whole_out.iter().zip(&data).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
                eprintln!("{kind:?} k {k} tiles {nx}x{ny}: {diff} samples differ (whole frame took {t_whole:?}, {changed} samples changed by L1, halo {})", reshape_halo(&p, &planes, k));
            }
        }
    }
}

#[test]
fn explore_swirl() {
    let radii = [10.0; 10];
    let scene = night_scene(1000, 667, &radii, Noise::LOW, 11);
    let (w, h) = (scene.img.w, scene.img.h);
    let n = w * h;
    for (kind, edges) in [(LensKind::Petzval, 100.0), (LensKind::Vintage, 100.0)] {
        let p = plan(kind, 0.0, edges, 0.0, w, h, Axis::X);
        let planes = planes_of(&p, &scene);
        let out = reshape_tile(&p, &planes, 1, &scene.img.data, w, h, PixelRect::new(0, 0, w, h));
        eprintln!("{kind:?} edges {edges}: balls {}, halo {}", planes.bokeh().unwrap().balls().len(), reshape_halo(&p, &planes, 1));
        for &(cx, cy, r) in &scene.discs {
            let (x0, x1) = ((cx - 3.0 * r).max(0.0) as usize, ((cx + 3.0 * r) as usize).min(w));
            let (y0, y1) = ((cy - 3.0 * r).max(0.0) as usize, ((cy + 3.0 * r) as usize).min(h));
            let (mut m, mut mx, mut my, mut sxx, mut sxy, mut syy) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
            for y in y0..y1 {
                for x in x0..x1 {
                    let v = (f64::from(out[n + y * w + x]) - 0.006).max(0.0);
                    m += v;
                    mx += v * (x as f64 + 0.5);
                    my += v * (y as f64 + 0.5);
                }
            }
            let (mx, my) = (mx / m, my / m);
            for y in y0..y1 {
                for x in x0..x1 {
                    let v = (f64::from(out[n + y * w + x]) - 0.006).max(0.0);
                    let (dx, dy) = (x as f64 + 0.5 - mx, y as f64 + 0.5 - my);
                    sxx += v * dx * dx;
                    sxy += v * dx * dy;
                    syy += v * dy * dy;
                }
            }
            let (sxx, sxy, syy) = (sxx / m, sxy / m, syy / m);
            let tr = sxx + syy;
            let det = sxx * syy - sxy * sxy;
            let d = (tr * tr / 4.0 - det).max(0.0).sqrt();
            let (l1, l2) = (tr / 2.0 + d, tr / 2.0 - d);
            let angle = 0.5 * (2.0 * sxy).atan2(sxx - syy).to_degrees();
            let (ux, uy) = (cx - 500.0, cy - 333.5);
            let tangent = (ux.atan2(-uy)).to_degrees(); // direction of (-uy, ux): angle of the tangent vector
            let tang = (-(uy)).atan2(-ux * 0.0 + ux).to_degrees();
            let _ = tangent;
            eprintln!("  ball ({cx:.0},{cy:.0}) rho {:.2}: elongation {:.2}, major axis {:.0} deg, tangent {:.0} deg", (ux * ux + uy * uy).sqrt() / 601.0, (l1 / l2).sqrt(), angle, tang);
        }
    }
}
