//! Times the CPU reference of the field optics (L2) at Edges 100 on a synthetic lens folder.
//!
//! `cargo run --release -p ym-lens --features test-support --example bench_field -- [width height [lens [anamorphic]]]`

use std::time::Instant;

use ym_lens::test_support::write_synthetic_lenses;
use ym_lens::{
    Axis, CameraResponse, FrameInfo, ImageRef, LensAmounts, LensKind, LensLibrary, LensPlan,
    LensPlanes, PixelRect, apply, halo_px,
};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let width: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(6000);
    let height: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(4000);
    let kind: LensKind = args
        .get(3)
        .and_then(|s| s.parse().ok())
        .unwrap_or(LensKind::Default);
    let anamorphic: f64 = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(0.0);

    let dir = tempfile::tempdir().expect("temp dir");
    write_synthetic_lenses(dir.path()).expect("write lenses");
    let lib = LensLibrary::from_dir(dir.path()).expect("load lenses");

    let amounts = LensAmounts {
        kind,
        flares: 0.0,
        edges: 100.0,
        anamorphic,
    };
    let frame = FrameInfo {
        width,
        height,
        axis: Axis::X,
        f_number: None,
        clip: [1.0; 3],
    };
    let t = Instant::now();
    let plan = LensPlan::new(&lib, &amounts, frame, &CameraResponse::neutral()).expect("plan");
    println!("plan: {:.1} ms", t.elapsed().as_secs_f64() * 1e3);
    let planes = {
        let k = plan.lens_plane_scale();
        let (w, h) = (width.div_ceil(k), height.div_ceil(k));
        let data = vec![0.5f32; 3 * w * h];
        let img = ImageRef {
            width: w,
            height: h,
            x0: 0,
            y0: 0,
            frame_width: w,
            frame_height: h,
            data: &data,
        };
        LensPlanes::build(&plan, &img, None, &|| false).expect("planes")
    };

    // A deterministic noisy image with some structure.
    let mut data = vec![0.0f32; 3 * width * height];
    let mut state = 0x9E37_79B9_7F4A_7C15u64;
    for (i, v) in data.iter_mut().enumerate() {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let noise = (state >> 40) as f32 / (1u64 << 24) as f32;
        *v = 0.3 + 0.4 * noise + 0.2 * (((i % width) / 16 + (i / width) / 16) % 2) as f32;
    }
    let img = ImageRef {
        width,
        height,
        x0: 0,
        y0: 0,
        frame_width: width,
        frame_height: height,
        data: &data,
    };
    let out = PixelRect::new(0, 0, width, height);
    println!("halo: {} px", halo_px(&plan, &planes, 1));
    let t = Instant::now();
    let result = apply(&plan, &planes, 1, &img, out).expect("apply");
    let secs = t.elapsed().as_secs_f64();
    let mp = (width * height) as f64 / 1e6;
    println!(
        "{kind} {width}x{height} edges 100 anamorphic {anamorphic}: {:.2} s, {:.0} ms/MP ({} samples, first {})",
        secs,
        secs * 1e3 / mp,
        result.len(),
        result[0]
    );
}
