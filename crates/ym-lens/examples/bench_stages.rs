//! Times the stages of the lens on the shipped lens data: the CPU reference's cost per stage on a fit-view group
//! (a 1500 x 1000 image of a 6000 x 4000 frame, `k` = 4), so the stage caching and the kernels can be judged
//! against it.
//!
//! ```text
//! cargo run -p ym-lens --example bench_stages
//! ```
//!
//! CPU times (user plus system, all threads) are read from `/proc/self/stat` where it exists, because wall
//! times on a shared machine say little.

#[path = "../tests/scenes/mod.rs"]
mod scenes;

use std::time::Instant;

use scenes::{Noise, subject_scene};
use ym_lens::{
    Axis, CameraResponse, FrameInfo, ImageRef, LensAmounts, LensKind, LensLibrary, LensPlan,
    LensPlanes, PixelRect, apply, apply_field, apply_flares, reshape,
};

/// CPU seconds of this process (user plus system), when the platform tells.
fn cpu_seconds() -> Option<f64> {
    let stat = std::fs::read_to_string("/proc/self/stat").ok()?;
    let rest = stat.rsplit_once(')')?.1;
    let fields: Vec<&str> = rest.split_whitespace().collect();
    let ticks: f64 = fields.get(11)?.parse::<f64>().ok()? + fields.get(12)?.parse::<f64>().ok()?;
    Some(ticks / 100.0)
}

fn timed<T>(what: &str, f: impl FnOnce() -> T) -> T {
    let (c0, t0) = (cpu_seconds(), Instant::now());
    let out = f();
    let wall = t0.elapsed().as_secs_f64();
    let cpu = c0.zip(cpu_seconds()).map(|(a, b)| b - a);
    println!(
        "{what:<52} {:>8.0} ms wall {:>8}",
        wall * 1e3,
        cpu.map_or("n/a".into(), |c| format!("{:.0} ms CPU", c * 1e3))
    );
    out
}

fn main() {
    let lib =
        LensLibrary::load(&std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data"))
            .expect("the shipped lens data");
    let curves = ym_color::spectral::SpectralCurves::from_path(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../data/cameras/sony/Sony_ILCE-7RM4_380_780_5.json"
    ))
    .expect("curves");
    let camera = CameraResponse::from_curves(&curves).expect("camera");
    let (gw, gh) = (1500usize, 1000usize);
    let (fw, fh) = (6000usize, 4000usize);
    let mut scene = subject_scene(gw, gh, [8.0, 14.0, 22.0], Noise::LOW, 1);
    // Below the flares' knee (half the clip level) everywhere but in a few clipped lights, as a photo has.
    for v in &mut scene.img.data {
        *v = (*v * 0.4).min(0.45);
    }
    // A few clipped lights, as a photo has.
    for (cx, cy, r) in [(300usize, 250usize, 3usize), (1100, 300, 5), (800, 800, 2)] {
        for c in 0..3 {
            for y in cy - r..=cy + r {
                for x in cx - r..=cx + r {
                    scene.img.plane_mut(c)[y * gw + x] = 1.0;
                }
            }
        }
    }
    let input = ImageRef {
        width: gw,
        height: gh,
        x0: 0,
        y0: 0,
        frame_width: gw,
        frame_height: gh,
        data: &scene.img.data,
    };
    let frame = FrameInfo {
        width: fw,
        height: fh,
        axis: Axis::X,
        f_number: Some(8.0),
        clip: [1.0; 3],
    };
    let whole = PixelRect::new(0, 0, gw, gh);
    let amounts = |kind, flares, edges, anamorphic| LensAmounts {
        kind,
        flares,
        edges,
        anamorphic,
    };
    let plan = |a: LensAmounts| LensPlan::new(&lib, &a, frame, &camera).expect("plan");
    for kind in [LensKind::Vintage, LensKind::Default] {
        println!("--- {kind}");
        let full = plan(amounts(kind, 60.0, 50.0, 50.0));
        let planes = timed("planes (blur map, balls, tail plane, sources)", || {
            LensPlanes::build(&full, &ImageRef { ..input }, scene.focus, &|| false).expect("planes")
        });
        println!("reshape report: {:?}", planes.report());
        let k = 4;
        let l1 = timed("L1 alone, Vintage 60/50/50", || {
            reshape(&full, &planes, k, &input, whole).expect("reshape")
        });
        drop(l1);
        let z = timed("Z = L2(L1(X)), 60/50/50", || {
            apply_field(&full, &planes, k, &input, whole).expect("field")
        });
        timed("L3 on Z, 60/50/50", || {
            apply_flares(&full, &planes, k, &input, &z, whole).expect("flares")
        });
        timed("the whole lens, 60/50/50", || {
            apply(&full, &planes, k, &input, whole).expect("apply")
        });
        let edges_only = plan(amounts(kind, 0.0, 50.0, 0.0));
        let p = LensPlanes::build(&edges_only, &input, scene.focus, &|| false).expect("planes");
        timed("Z, Edges 50 alone (L2, and L1 on a swirl)", || {
            apply_field(&edges_only, &p, k, &input, whole).expect("field")
        });
        let anam_only = plan(amounts(kind, 0.0, 0.0, 50.0));
        let p = LensPlanes::build(&anam_only, &input, scene.focus, &|| false).expect("planes");
        timed("Z, Anamorphic 50 alone (L1 and L2)", || {
            apply_field(&anam_only, &p, k, &input, whole).expect("field")
        });
        let flares_only = plan(amounts(kind, 60.0, 0.0, 0.0));
        let p = LensPlanes::build(&flares_only, &input, scene.focus, &|| false).expect("planes");
        timed("whole lens, Flares 60 alone (L3)", || {
            apply(&flares_only, &p, k, &input, whole).expect("apply")
        });
    }
}
