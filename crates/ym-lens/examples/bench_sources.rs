//! Times L3 on frames with many bright lights, and checks the light budget on a uniform frame: the CPU
//! reference's cost against the number of sources (`cargo run -p ym-lens --example bench_sources`).
//!
//! The scene is a flat grey below the flares' knee with `n` clipped squares of 3 to 7 px at seeded places, on
//! a 1500 x 1000 group of a 6000 x 4000 frame (`k` = 4), the fit view of a 24 MP photo.

#[path = "../tests/scenes/mod.rs"]
mod scenes;

use std::time::Instant;

use scenes::Rng;
use ym_lens::{
    Axis, CameraResponse, FrameInfo, ImageRef, LensAmounts, LensKind, LensLibrary, LensPlan,
    LensPlanes, PixelRect, apply, flare_terms,
};

fn main() {
    let lib = LensLibrary::load(&std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data"))
        .expect("the shipped lens data");
    let curves = ym_color::spectral::SpectralCurves::from_path(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../data/cameras/sony/Sony_ILCE-7RM4_380_780_5.json"
    ))
    .expect("curves");
    let camera = CameraResponse::from_curves(&curves).expect("camera");
    let (gw, gh) = (1500usize, 1000usize);
    let (fw, fh) = (6000usize, 4000usize);
    let frame = FrameInfo {
        width: fw,
        height: fh,
        axis: Axis::X,
        f_number: Some(8.0),
        clip: [1.0; 3],
    };
    let whole = PixelRect::new(0, 0, gw, gh);
    let counts: Vec<usize> = std::env::var("YM_BENCH_SOURCES")
        .ok()
        .map(|s| s.split(',').filter_map(|v| v.parse().ok()).collect())
        .unwrap_or_else(|| vec![0, 8, 32, 128, 256]);
    for kind in [LensKind::Default, LensKind::Vintage] {
        for &n in &counts {
            let mut data = vec![0.25f32; 3 * gw * gh];
            let mut rng = Rng::new(7);
            for _ in 0..n {
                let r = 1 + (rng.unit() * 3.0) as usize;
                let cx = 20 + (rng.unit() * (gw - 40) as f64) as usize;
                let cy = 20 + (rng.unit() * (gh - 40) as f64) as usize;
                for c in 0..3 {
                    for y in cy - r..=cy + r {
                        for x in cx - r..=cx + r {
                            data[c * gw * gh + y * gw + x] = 1.0;
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
                data: &data,
            };
            let amounts = LensAmounts {
                kind,
                flares: 60.0,
                edges: 40.0,
                anamorphic: 50.0,
            };
            let plan = LensPlan::new(&lib, &amounts, frame, &camera).expect("plan");
            let planes = LensPlanes::build(&plan, &input, None, &|| false).expect("planes");
            let t = Instant::now();
            let terms = flare_terms(&plan, &planes, 4, &input, whole).expect("terms");
            let ms = t.elapsed().as_secs_f64() * 1e3;
            let sum: f64 = terms.as_ref().map_or(0.0, |t| t.iter().sum());
            println!(
                "{kind:<8} {n:>4} clipped lights: L3 terms {ms:>9.0} ms, light added {sum:.1}"
            );
        }
        // The light budget on a uniform grey frame (no lights): the mean of the output against the input's.
        let grey = vec![0.2f32; 3 * gw * gh];
        let input = ImageRef {
            width: gw,
            height: gh,
            x0: 0,
            y0: 0,
            frame_width: gw,
            frame_height: gh,
            data: &grey,
        };
        for (flares, edges, anamorphic) in [
            (60.0, 0.0, 0.0),
            (0.0, 40.0, 0.0),
            (60.0, 40.0, 0.0),
            (60.0, 40.0, 50.0),
        ] {
            let amounts = LensAmounts {
                kind,
                flares,
                edges,
                anamorphic,
            };
            let plan = LensPlan::new(&lib, &amounts, frame, &camera).expect("plan");
            let planes = LensPlanes::build(&plan, &input, None, &|| false).expect("planes");
            let out = apply(&plan, &planes, 4, &input, whole).expect("apply");
            let mean = |v: &[f32], c: usize| -> f64 {
                v[c * gw * gh..(c + 1) * gw * gh]
                    .iter()
                    .map(|x| f64::from(*x))
                    .sum::<f64>()
                    / (gw * gh) as f64
            };
            println!(
                "{kind:<8} uniform grey 0.2, Flares {flares} Edges {edges} Anamorphic {anamorphic}: mean out {:.5} {:.5} {:.5}",
                mean(&out, 0),
                mean(&out, 1),
                mean(&out, 2)
            );
        }
    }
}
