//! Times the CPU reference of the highlight optics (L3) on a synthetic lens folder and a frame with lights.
//!
//! `cargo run --release -p ym-lens --features test-support --example bench_flare -- [width height [lights [anamorphic]]]`
//!
//! The lens is the synthetic Classic with glow and veil out to a whole half-diagonal and ghosts that are not cut
//! by their clipping circles, so every light draws all its items. Flares are at 100.

use std::fs;
use std::time::Instant;

use serde_json::{Value, json};
use ym_lens::test_support::{coating, write_sums, write_synthetic_lenses};
use ym_lens::{
    Axis, CameraResponse, FrameInfo, ImageRef, LensAmounts, LensKind, LensLibrary, LensPlan,
    LensPlanes, PixelRect, apply, halo_px,
};

fn ghost(m: f64, n: f64, s: f64, fraction: f64, nm: f64) -> Value {
    json!({ "m": m, "n": n, "s": s, "clip_radius": 2.0, "fraction": fraction, "soft": 0.1,
            "coating": coating(nm) })
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let width: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(6000);
    let height: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(4000);
    let lights: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(50);
    let anamorphic: f64 = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(100.0);

    let dir = tempfile::tempdir().expect("temp dir");
    write_synthetic_lenses(dir.path()).expect("write lenses");
    let path = dir.path().join("classic/lens.json");
    let mut lens: Value = serde_json::from_slice(&fs::read(&path).expect("read")).expect("json");
    lens["flare"]["ghosts"] = json!([
        ghost(-0.5, 0.3, 0.05, 0.010, 450.0),
        ghost(0.7, -0.2, 0.07, 0.008, 600.0),
        ghost(1.4, 0.0, 0.04, 0.006, 520.0),
        ghost(-1.1, 0.1, 0.03, 0.004, 560.0),
    ]);
    fs::write(&path, serde_json::to_string_pretty(&lens).expect("json")).expect("write");
    let path = dir.path().join("anamorphic.json");
    let mut att: Value = serde_json::from_slice(&fs::read(&path).expect("read")).expect("json");
    att["ghosts"] = json!([
        ghost(-0.4, 0.2, 0.06, 0.006, 470.0),
        ghost(0.9, 0.1, 0.05, 0.004, 560.0)
    ]);
    fs::write(&path, serde_json::to_string_pretty(&att).expect("json")).expect("write");
    write_sums(dir.path()).expect("sums");
    let lib = LensLibrary::from_dir(dir.path()).expect("load lenses");

    let amounts = |flares: f64, anamorphic: f64| LensAmounts {
        kind: LensKind::Classic,
        flares,
        edges: 0.0,
        anamorphic,
    };
    let frame = FrameInfo {
        width,
        height,
        axis: Axis::X,
        f_number: Some(8.0),
        clip: [1.0; 3],
    };
    let camera = CameraResponse::neutral();
    // The planes do not depend on the amounts (their key leaves them out). They are built here with Anamorphic
    // 0, which leaves out L1's blur map, so that the time is L3's alone.
    let plan_planes = LensPlan::new(&lib, &amounts(100.0, 0.0), frame, &camera).expect("plan");
    let plan = LensPlan::new(&lib, &amounts(100.0, anamorphic), frame, &camera).expect("plan");
    let plan_off = LensPlan::new(&lib, &amounts(0.0, anamorphic), frame, &camera).expect("plan");

    // A deterministic frame: dim noise and `lights` clipped squares of 4 to 40 px.
    let mut data = vec![0.0f32; 3 * width * height];
    let mut state = 0x9E37_79B9_7F4A_7C15u64;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 11) as f64 / (1u64 << 53) as f64
    };
    for v in data.iter_mut() {
        *v = (0.05 + 0.25 * next()) as f32;
    }
    for i in 0..lights {
        let size = 4 + (next() * 36.0) as usize + (i % 3) * 2;
        let x0 = (next() * (width - size) as f64) as usize;
        let y0 = (next() * (height - size) as f64) as usize;
        let rgb = [
            1.0,
            (0.8 + 0.2 * next()) as f32,
            (0.5 + 0.5 * next()) as f32,
        ];
        for (c, v) in rgb.iter().enumerate() {
            for y in y0..y0 + size {
                for x in x0..x0 + size {
                    data[c * width * height + y * width + x] = if i % 2 == 0 { 1.0 } else { *v };
                }
            }
        }
    }

    // The lens plane: the frame box-reduced `k` times.
    let k = plan_planes.lens_plane_scale();
    let (pw, ph) = (width.div_ceil(k), height.div_ceil(k));
    let mut plane = vec![0.0f32; 3 * pw * ph];
    for c in 0..3 {
        for y in 0..ph {
            for x in 0..pw {
                let mut s = 0.0f32;
                let mut n = 0.0f32;
                for dy in 0..k {
                    for dx in 0..k {
                        let (sx, sy) = (x * k + dx, y * k + dy);
                        if sx < width && sy < height {
                            s += data[c * width * height + sy * width + sx];
                            n += 1.0;
                        }
                    }
                }
                plane[c * pw * ph + y * pw + x] = s / n;
            }
        }
    }
    let img = ImageRef {
        width: pw,
        height: ph,
        x0: 0,
        y0: 0,
        frame_width: pw,
        frame_height: ph,
        data: &plane,
    };
    let t = Instant::now();
    let planes = LensPlanes::build(&plan_planes, &img, None, &|| false).expect("planes");
    println!(
        "planes ({pw} x {ph} lens plane, k = {k}): {:.0} ms, {} KiB",
        t.elapsed().as_secs_f64() * 1e3,
        planes.bytes() / 1024
    );

    let img = ImageRef {
        width,
        height,
        x0: 0,
        y0: 0,
        frame_width: width,
        frame_height: height,
        data: &data,
    };
    let mp = (width * height) as f64 / 1e6;
    let all = PixelRect::new(0, 0, width, height);
    println!("halo: {} px", halo_px(&plan, &planes, 1));
    // The same frame without the flares (L2 alone) and with them: the difference is L3.
    let t = Instant::now();
    let base = apply(&plan_off, &planes, 1, &img, all).expect("apply");
    let base_ms = t.elapsed().as_secs_f64() * 1e3;
    let t = Instant::now();
    let out = apply(&plan, &planes, 1, &img, all).expect("apply");
    let ms = t.elapsed().as_secs_f64() * 1e3;
    println!(
        "{width} x {height}, {lights} lights, Anamorphic {anamorphic}: flares off {base_ms:.0} ms, on {ms:.0} ms, \
         L3 {:.0} ms = {:.0} ms/MP",
        ms - base_ms,
        (ms - base_ms) / mp
    );
    // The same in 1024 x 1024 tiles, each given its input with the halo.
    let halo = halo_px(&plan, &planes, 1);
    let mut tiles_ms = |p: &LensPlan| {
        let t = Instant::now();
        for ty in (0..height).step_by(1024) {
            for tx in (0..width).step_by(1024) {
                let tile = PixelRect::new(tx, ty, 1024.min(width - tx), 1024.min(height - ty));
                let need = tile.grown(halo, width, height);
                let mut buf = Vec::with_capacity(3 * need.w * need.h);
                for c in 0..3 {
                    for y in need.y..need.y1() {
                        let row = c * width * height + y * width + need.x;
                        buf.extend_from_slice(&data[row..row + need.w]);
                    }
                }
                let input = ImageRef {
                    width: need.w,
                    height: need.h,
                    x0: need.x,
                    y0: need.y,
                    frame_width: width,
                    frame_height: height,
                    data: &buf,
                };
                let _ = apply(p, &planes, 1, &input, tile).expect("tile");
            }
        }
        t.elapsed().as_secs_f64() * 1e3
    };
    let (off_t, on_t) = (tiles_ms(&plan_off), tiles_ms(&plan));
    println!(
        "1024 px tiles: flares off {off_t:.0} ms, on {on_t:.0} ms, L3 {:.0} ms = {:.0} ms/MP",
        on_t - off_t,
        (on_t - off_t) / mp
    );
    let _ = base;
    let sum: f64 = out.iter().map(|v| f64::from(*v)).sum();
    println!("checksum {sum:.3}");
}
