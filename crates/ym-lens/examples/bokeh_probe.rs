//! Runs the bokeh detector and L1 on a dumped lens plane (a developer tool for local review; never shipped).
//!
//! ```text
//! cargo run -p ym-lens --example bokeh_probe -- [--out DIR] [--anamorphic 100] [--no-l1] [--map DIR] [--crop NAME]... <dump-dir>...
//! ```
//!
//! A dump folder holds `lens_plane.f32` (planar R, G, B, little-endian f32, camera RGB balanced with g0) and
//! `lens_plane.json` (width, height, `k_L`, frame size, g0, f-number, axis), and optionally `bokeh.json` with the
//! camera's `focus_fraction`, as written by `crates/ym-core/examples/lens_sheet.rs --dump-planes`. For every folder
//! the tool writes into `<out>/<name>/`: `stats.json` (the detector's counters and timings), `balls.json`,
//! `b.f32` and `conf.f32` (the blur map and its confidence), and `a100.f32` (L1 over the whole plane at
//! Anamorphic `--anamorphic`, planar, same size as the plane) unless `--no-l1`.
//!
//! With `--map DIR` the blur map of the learned model is used in place of the classical detector's: `DIR/<name>/`
//! holds `model_b.f32` and `model_p.f32` (the cells' radius in half-diagonals and probability of being out of focus,
//! as `crates/ym-core/examples/bokeh_gate.rs --write-maps` writes them), and the planes are built with
//! [`LensPlanes::build_with_map`] and the lens's own rule.
//!
//! With `--crop NAME` the folder also holds `NAME_l1in.f32` and `.json` (a full-resolution window of the raw group
//! grown by L1's halo, written by `crates/ym-core/examples/bokeh_sheet.rs --dump-planes --crop`): L1 runs on it at
//! scale 1, and `NAME_l1base.f32` and `NAME_l1out.f32` (the window before and after, planar) are written.

use std::path::{Path, PathBuf};
use std::time::Instant;

use ym_lens::{
    Axis, BlurMapInput, CameraResponse, FrameInfo, ImageRef, LensAmounts, LensKind, LensLibrary,
    LensPlan, LensPlanes, PixelRect, reshape, reshape_halo,
};

/// The size of a plane after the network's box reduction to a long edge of 1024 pixels (`ym_ml::bokeh_net`).
fn reduced_size(w: usize, h: usize) -> (usize, usize) {
    let long = w.max(h);
    if long <= 1024 {
        return (w, h);
    }
    let s = 1024.0 / long as f64;
    (
        ((w as f64 * s).round() as usize).max(1),
        ((h as f64 * s).round() as usize).max(1),
    )
}

type Fallible<T> = Result<T, Box<dyn std::error::Error>>;

fn read_f32(path: &Path) -> Fallible<Vec<f32>> {
    let bytes = std::fs::read(path)?;
    Ok(bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

fn write_f32(path: &Path, data: &[f32]) -> Fallible<()> {
    let mut bytes = Vec::with_capacity(data.len() * 4);
    for v in data {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    std::fs::write(path, bytes)?;
    Ok(())
}

fn main() -> Fallible<()> {
    let mut out = PathBuf::from("bokeh-probe");
    let mut anamorphic = 100.0;
    let mut run_l1 = true;
    let mut crops: Vec<String> = Vec::new();
    let mut map_dir: Option<PathBuf> = None;
    let mut dirs: Vec<PathBuf> = Vec::new();
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--out" => out = PathBuf::from(it.next().ok_or("--out needs a folder")?),
            "--anamorphic" => {
                anamorphic = it.next().ok_or("--anamorphic needs a value")?.parse()?
            }
            "--no-l1" => run_l1 = false,
            "--map" => map_dir = Some(PathBuf::from(it.next().ok_or("--map needs a folder")?)),
            "--crop" => crops.push(it.next().ok_or("--crop needs a name")?),
            other => dirs.push(PathBuf::from(other)),
        }
    }
    let data = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data");
    let lib = LensLibrary::load(&data)?;
    let camera = CameraResponse::neutral();
    for dir in dirs {
        let name = dir
            .file_name()
            .map_or_else(|| "photo".into(), |n| n.to_string_lossy().into_owned());
        let meta: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(dir.join("lens_plane.json"))?)?;
        let num = |k: &str| meta[k].as_u64().ok_or(format!("{k} missing"));
        let (w, h) = (num("width")? as usize, num("height")? as usize);
        let (fw, fh) = (num("frame_width")? as usize, num("frame_height")? as usize);
        let k = num("k_L")? as usize;
        let axis = if meta["axis"].as_str() == Some("Y") {
            Axis::Y
        } else {
            Axis::X
        };
        let g0: Vec<f64> = meta["g0"]
            .as_array()
            .ok_or("g0 missing")?
            .iter()
            .filter_map(serde_json::Value::as_f64)
            .collect();
        let focus = std::fs::read_to_string(dir.join("bokeh.json"))
            .ok()
            .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok())
            .and_then(|v| {
                let f = v["focus_fraction"].as_array()?;
                Some((f.first()?.as_f64()?, f.get(1)?.as_f64()?))
            });
        let plane = read_f32(&dir.join("lens_plane.f32"))?;
        let image = ImageRef {
            width: w,
            height: h,
            x0: 0,
            y0: 0,
            frame_width: w,
            frame_height: h,
            data: &plane,
        };
        let amounts = LensAmounts {
            kind: LensKind::Default,
            flares: 0.0,
            edges: 0.0,
            anamorphic,
        };
        let plan = LensPlan::new(
            &lib,
            &amounts,
            FrameInfo {
                width: fw,
                height: fh,
                axis,
                f_number: meta["f_number"].as_f64(),
                clip: [g0[0] as f32, g0[1] as f32, g0[2] as f32],
            },
            &camera,
        )?;
        assert_eq!(
            plan.lens_plane_scale(),
            k,
            "the plan's k_L differs from the dump's"
        );
        let t = Instant::now();
        let map = match &map_dir {
            Some(m) => {
                let b = read_f32(&m.join(&name).join("model_b.f32"))?;
                let p = read_f32(&m.join(&name).join("model_p.f32"))?;
                let (pw, ph) = reduced_size(w, h);
                let (cw, ch) = (pw.div_ceil(2), ph.div_ceil(2));
                assert_eq!(b.len(), cw * ch, "the map's cells do not fit the plane");
                Some(BlurMapInput {
                    width: cw,
                    height: ch,
                    plane_width: pw,
                    plane_height: ph,
                    b,
                    p,
                })
            }
            None => None,
        };
        let planes = LensPlanes::build_with_map(&plan, &image, focus, map.as_ref(), &|| false)?;
        let planes_ms = t.elapsed().as_secs_f64() * 1e3;
        let bp = planes.bokeh().ok_or("no blur map")?;
        let o = out.join(&name);
        std::fs::create_dir_all(&o)?;
        let (bw, bh) = bp.size();
        write_f32(&o.join("b.f32"), bp.blur())?;
        write_f32(&o.join("conf.f32"), bp.confidence())?;
        let balls: Vec<_> = bp
            .balls()
            .iter()
            .map(|b| {
                serde_json::json!({
                    "x": b.x, "y": b.y, "radius": b.radius, "peak": b.peak,
                    "supported": b.supported, "energy": b.energy,
                    "take_out_radius": b.take_out_radius, "softness": b.softness,
                })
            })
            .collect();
        std::fs::write(o.join("balls.json"), serde_json::to_string(&balls)?)?;
        let mut l1_ms = 0.0;
        if run_l1 {
            let t = Instant::now();
            let reshaped = reshape(&plan, &planes, k, &image, PixelRect::new(0, 0, w, h))?;
            l1_ms = t.elapsed().as_secs_f64() * 1e3;
            write_f32(&o.join("a100.f32"), &reshaped)?;
        }
        for c in &crops {
            let cj: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(
                dir.join(format!("{c}_l1in.json")),
            )?)?;
            let n = |k: &str| {
                cj[k]
                    .as_u64()
                    .ok_or(format!("{k} missing"))
                    .map(|v| v as usize)
            };
            let (x0, y0, cw, ch) = (n("x0")?, n("y0")?, n("width")?, n("height")?);
            let want = &cj["want"];
            let wn = |k: &str| want[k].as_u64().map(|v| v as usize).unwrap_or(0);
            let (fw, fh) = (n("frame_width")?, n("frame_height")?);
            // The window is shrunk where the dump's margin is smaller than L1's halo now (away from the frame's edge).
            let halo = reshape_halo(&plan, &planes, 1);
            let lo = |start: usize, input_start: usize| {
                if input_start == 0 {
                    start
                } else {
                    start.max(input_start + halo)
                }
            };
            let hi = |end: usize, input_end: usize, frame: usize| {
                if input_end == frame {
                    end
                } else {
                    end.min(input_end - halo)
                }
            };
            let (wx0, wy0) = (lo(wn("x"), x0), lo(wn("y"), y0));
            let (wx1, wy1) = (
                hi(wn("x") + wn("w"), x0 + cw, fw),
                hi(wn("y") + wn("h"), y0 + ch, fh),
            );
            let rect = PixelRect::new(wx0, wy0, wx1 - wx0, wy1 - wy0);
            let data = read_f32(&dir.join(format!("{c}_l1in.f32")))?;
            let input = ImageRef {
                width: cw,
                height: ch,
                x0,
                y0,
                frame_width: fw,
                frame_height: fh,
                data: &data,
            };
            let t = Instant::now();
            let out_px = reshape(&plan, &planes, 1, &input, rect)?;
            println!("  crop {c}: L1 {:.0} ms", t.elapsed().as_secs_f64() * 1e3);
            let mut base = Vec::with_capacity(3 * rect.w * rect.h);
            for ch_i in 0..3 {
                for y in rect.y..rect.y + rect.h {
                    let row = ch_i * cw * ch + (y - y0) * cw + (rect.x - x0);
                    base.extend_from_slice(&data[row..row + rect.w]);
                }
            }
            write_f32(&o.join(format!("{c}_l1base.f32")), &base)?;
            write_f32(&o.join(format!("{c}_l1out.f32")), &out_px)?;
            std::fs::write(
                o.join(format!("{c}_l1.json")),
                serde_json::to_string(&serde_json::json!({"w": rect.w, "h": rect.h}))?,
            )?;
        }
        let stats = serde_json::json!({
            "size": [bw, bh], "plane": [w, h], "reduce": bp.reduce(), "axis": format!("{axis:?}"),
            "active": bp.active(), "blurred_fraction": bp.blurred_fraction(),
            "edges": bp.edge_count(), "balls": bp.balls().len(), "max_blur": bp.max_blur(),
            "nonzero_fraction": bp.blur().iter().filter(|v| **v > 0.0).count() as f64 / (bw * bh) as f64,
            "report": format!("{:?}", planes.report()),
            "planes_ms": planes_ms, "l1_ms": l1_ms,
        });
        std::fs::write(o.join("stats.json"), serde_json::to_string_pretty(&stats)?)?;
        println!(
            "{name}: active {} blurred {:.1} % edges {} balls {} planes {planes_ms:.0} ms l1 {l1_ms:.0} ms",
            bp.active(),
            100.0 * bp.blurred_fraction(),
            bp.edge_count(),
            bp.balls().len()
        );
    }
    Ok(())
}
