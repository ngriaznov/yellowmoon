//! Times L1 (`docs/21` section 9): the planes' build on a 1500 px lens plane and the per-tile pass over a 24 MP
//! frame (6000 x 4000) about half of which is out of focus, at Anamorphic 100.
//!
//! ```text
//! cargo run --release -p ym-lens --example bench_l1
//! ```
//!
//! CPU times (user plus system, all threads) are read from `/proc/self/stat` where it exists, because wall
//! times on a shared machine say little.

#[path = "../tests/scenes/mod.rs"]
mod scenes;

use std::time::Instant;

use scenes::{Noise, subject_scene};
use ym_lens::test_support::write_synthetic_lenses;
use ym_lens::{
    Axis, CameraResponse, FrameInfo, ImageRef, LensAmounts, LensKind, LensLibrary, LensPlan,
    LensPlanes, PixelRect, apply, halo_px, reshape, reshape_halo,
};

/// CPU seconds of this process (user plus system), when the platform tells.
fn cpu_seconds() -> Option<f64> {
    let stat = std::fs::read_to_string("/proc/self/stat").ok()?;
    let rest = stat.rsplit_once(')')?.1;
    let fields: Vec<&str> = rest.split_whitespace().collect();
    // After the command: state is field 3, utime 14 and stime 15 (1-based), so indexes 11 and 12 here.
    let ticks: f64 = fields.get(11)?.parse::<f64>().ok()? + fields.get(12)?.parse::<f64>().ok()?;
    Some(ticks / 100.0)
}

fn timed<T>(f: impl FnOnce() -> T) -> (T, f64, Option<f64>) {
    let (c0, t0) = (cpu_seconds(), Instant::now());
    let out = f();
    let wall = t0.elapsed().as_secs_f64();
    (out, wall, c0.zip(cpu_seconds()).map(|(a, b)| b - a))
}

fn main() {
    let dir = std::env::temp_dir().join("ym-lens-bench-l1");
    std::fs::create_dir_all(&dir).expect("temp dir");
    write_synthetic_lenses(&dir).expect("lenses");
    let lib = LensLibrary::from_dir(&dir).expect("library");
    let curves = ym_color::spectral::SpectralCurves::from_path(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../data/cameras/sony/Sony_ILCE-7RM4_380_780_5.json"
    ))
    .expect("curves");
    let camera = CameraResponse::from_curves(&curves).expect("camera");

    // The lens plane of a 6000 x 4000 frame is 1500 x 1000: a subject in the middle (sharp), blurred bands.
    let (pw, ph) = (1500usize, 1000usize);
    let (scene, wall, _) = timed(|| subject_scene(pw, ph, [8.0, 14.0, 22.0], Noise::LOW, 1));
    println!("scene {pw}x{ph} drawn in {wall:.1} s");
    let (fw, fh) = (6000usize, 4000usize);
    let amounts = LensAmounts {
        kind: LensKind::Default,
        flares: 0.0,
        edges: 0.0,
        anamorphic: 100.0,
    };
    let frame = FrameInfo {
        width: fw,
        height: fh,
        axis: Axis::X,
        f_number: None,
        clip: [1.0; 3],
    };
    let plan = LensPlan::new(&lib, &amounts, frame, &camera).expect("plan");
    assert_eq!(plan.lens_plane_scale(), 4);
    let plane = ImageRef {
        width: pw,
        height: ph,
        x0: 0,
        y0: 0,
        frame_width: pw,
        frame_height: ph,
        data: &scene.img.data,
    };
    let (planes, wall, cpu) =
        timed(|| LensPlanes::build(&plan, &plane, scene.focus, &|| false).expect("planes"));
    let bp = planes.bokeh().expect("blur map");
    let blurred = bp.blur().iter().filter(|b| **b > 0.0).count() as f64 / bp.blur().len() as f64;
    println!(
        "planes on a {pw}x{ph} lens plane: {:.0} ms wall, {} CPU; {:.0} % of the frame has a blur, {} balls, report {:?}",
        wall * 1e3,
        cpu.map_or("n/a".into(), |c| format!("{:.0} ms", c * 1e3)),
        100.0 * blurred,
        bp.balls().len(),
        planes.report()
    );

    // The full-resolution frame: the lens plane scaled up 4x (bilinear) with a little grain.
    let (full, wall, _) = timed(|| {
        let mut out = vec![0.0f32; 3 * fw * fh];
        for c in 0..3 {
            for y in 0..fh {
                let fy = ((y as f64 + 0.5) / 4.0 - 0.5).clamp(0.0, (ph - 1) as f64);
                let (y0, ty) = (fy.floor() as usize, fy.fract() as f32);
                let y1 = (y0 + 1).min(ph - 1);
                for x in 0..fw {
                    let fx = ((x as f64 + 0.5) / 4.0 - 0.5).clamp(0.0, (pw - 1) as f64);
                    let (x0, tx) = (fx.floor() as usize, fx.fract() as f32);
                    let x1 = (x0 + 1).min(pw - 1);
                    let p = &scene.img.data[c * pw * ph..(c + 1) * pw * ph];
                    let top = p[y0 * pw + x0] * (1.0 - tx) + p[y0 * pw + x1] * tx;
                    let bot = p[y1 * pw + x0] * (1.0 - tx) + p[y1 * pw + x1] * tx;
                    out[c * fw * fh + y * fw + x] = top * (1.0 - ty) + bot * ty;
                }
            }
        }
        out
    });
    println!("full-resolution frame made in {wall:.1} s");

    let tile = 1024usize;
    let mut tiles = Vec::new();
    for y in (0..fh).step_by(tile) {
        for x in (0..fw).step_by(tile) {
            tiles.push(PixelRect::new(x, y, tile.min(fw - x), tile.min(fh - y)));
        }
    }
    let halo_l1 = reshape_halo(&plan, &planes, 1);
    let halo_all = halo_px(&plan, &planes, 1);
    println!(
        "{} tiles of {tile} px, halo {halo_l1} px (L1) and {halo_all} px (whole lens)",
        tiles.len()
    );
    let crop = |rect: PixelRect| -> (Vec<f32>, PixelRect) {
        let need = rect.grown(halo_all, fw, fh);
        let mut buf = Vec::with_capacity(3 * need.w * need.h);
        for c in 0..3 {
            for y in need.y..need.y1() {
                let row = c * fw * fh + y * fw + need.x;
                buf.extend_from_slice(&full[row..row + need.w]);
            }
        }
        (buf, need)
    };
    let megapixels = (fw * fh) as f64 / 1e6;
    for (name, l1_only) in [("L1 alone", true), ("whole lens (L1 + L2)", false)] {
        let (_, wall, cpu) = timed(|| {
            for &rect in &tiles {
                let (buf, need) = crop(rect);
                let input = ImageRef {
                    width: need.w,
                    height: need.h,
                    x0: need.x,
                    y0: need.y,
                    frame_width: fw,
                    frame_height: fh,
                    data: &buf,
                };
                if l1_only {
                    reshape(&plan, &planes, 1, &input, rect).expect("reshape");
                } else {
                    apply(&plan, &planes, 1, &input, rect).expect("apply");
                }
            }
        });
        println!(
            "{name} over {megapixels:.0} MP: {:.1} s wall ({:.0} ms/MP), {}",
            wall,
            wall * 1e3 / megapixels,
            cpu.map_or("CPU n/a".into(), |c| format!(
                "{:.1} s CPU ({:.0} ms/MP CPU)",
                c,
                c * 1e3 / megapixels
            ))
        );
    }
}
