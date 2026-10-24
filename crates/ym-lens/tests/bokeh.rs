//! L1, the bokeh reshaping, on synthetic scenes (`docs/21` sections 4 and 14).

mod common;
mod scenes;

use common::{apply_tile, plan, planes_for, whole};
use scenes::{Noise, Scene, night_scene, subject_scene};
use ym_lens::{Axis, LensKind, LensPlan, LensPlanes, PixelRect, apply, halo_px};

fn planes_of(p: &LensPlan, scene: &Scene) -> LensPlanes {
    LensPlanes::build(p, &whole(scene.img.w, scene.img.h, &scene.img.data), scene.focus, &|| false).expect("planes")
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
        let out = apply_tile(&p, &planes, 1, &scene.img.data, w, h, PixelRect::new(0, 0, w, h));
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
    let _ = (planes_for, apply, subject_scene);
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
