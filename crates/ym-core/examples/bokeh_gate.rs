//! Runs the bokeh map network on dumped lens planes and says what the lens's planes make of its map at several
//! lower-bound factors `k` (`b_low = b * exp(-k * beta)`, `beta = -0.25 ln c`), next to the classical detector. A
//! developer tool for local review (never shipped).
//!
//! ```text
//! cargo run -p ym-core --example bokeh_gate -- --data DIR [--ks 0,0.8,1.61,2.5] [--ts 1,1.5,2] [--write-maps OUT] <dump-dir>...
//! ```
//!
//! A dump folder holds `lens_plane.f32` and `lens_plane.json` as `bokeh_probe` reads them. `DIR` is a data folder
//! with `models/bokeh/` and `lenses/`. One JSON line per photo goes to stdout: the network's timings, the share of
//! cells with `b_low >= 2 b_min`, and for the classical detector and each `k` whether the planes are active, their
//! edge count, blurred fraction, balls and the share of the map above zero. With `--write-maps` the `k = 1.61`
//! planes' map and the network's raw `b` and `c` go to `OUT/<name>/` as little-endian f32 (`model_b.f32`,
//! `model_c.f32`, `blur_b.f32`, `conf.f32`, `balls.json`).

use std::path::{Path, PathBuf};
use std::time::Instant;

use serde_json::json;
use ym_lens::{
    Axis, BlurMapInput, CameraResponse, FrameInfo, ImageRef, LensAmounts, LensKind, LensLibrary,
    LensPlan, LensPlanes,
};
use ym_ml::bokeh_net::{BokehMap, BokehNet, PlanarRgbRef};

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

fn input(map: &BokehMap, k: f64, t: f64) -> BlurMapInput {
    let b_low = map
        .b
        .iter()
        .zip(&map.c)
        .map(|(&b, &c)| {
            let beta = -0.25 * libm::log(f64::from(c).max(1e-6));
            (f64::from(b) * libm::exp(-k * beta) * 2.0 / t) as f32
        })
        .collect();
    BlurMapInput {
        width: map.width,
        height: map.height,
        plane_width: map.plane_width,
        plane_height: map.plane_height,
        b: map.b.clone(),
        b_low,
    }
}

fn stats(planes: &LensPlanes, ms: f64) -> serde_json::Value {
    let mut v = json!({"ms": ms, "reason": planes.report().reason.map(|r| format!("{r:?}"))});
    if let Some(b) = planes.bokeh() {
        let nonzero = b.blur().iter().filter(|x| **x > 0.0).count();
        v["active"] = json!(b.active());
        v["edges"] = json!(b.edge_count());
        v["blurred_fraction"] = json!(b.blurred_fraction());
        v["balls"] = json!(b.balls().len());
        v["nonzero_fraction"] = json!(nonzero as f64 / b.blur().len().max(1) as f64);
    }
    v
}

fn main() -> Fallible<()> {
    let mut data = PathBuf::new();
    let mut ks = vec![0.0, 0.8, 1.2, 1.61, 2.0, 2.5];
    let mut ts = vec![2.0];
    let mut write_maps: Option<PathBuf> = None;
    let mut dirs: Vec<PathBuf> = Vec::new();
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--data" => data = PathBuf::from(it.next().ok_or("--data needs a folder")?),
            "--ks" => {
                ks = it
                    .next()
                    .ok_or("--ks needs a list")?
                    .split(',')
                    .map(str::parse)
                    .collect::<Result<_, _>>()?;
            }
            "--ts" => {
                ts = it
                    .next()
                    .ok_or("--ts needs a list")?
                    .split(',')
                    .map(str::parse)
                    .collect::<Result<_, _>>()?;
            }
            "--write-maps" => {
                write_maps = Some(PathBuf::from(
                    it.next().ok_or("--write-maps needs a folder")?,
                ))
            }
            other => dirs.push(PathBuf::from(other)),
        }
    }
    let lib = LensLibrary::load(&data)?;
    let model = BokehNet::load_bundled(&data)?;
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
            anamorphic: 100.0,
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
        let t = Instant::now();
        let classical = LensPlanes::build(&plan, &image, focus, &|| false)?;
        let classical_ms = t.elapsed().as_secs_f64() * 1e3;
        let planar = PlanarRgbRef::new(w, h, &plane)?;
        let (map, tm) = model.map_timed(&planar)?;
        let mut rows = Vec::new();
        for (&k, &tt) in ks.iter().flat_map(|k| ts.iter().map(move |t| (k, t))) {
            let t = Instant::now();
            let planes = LensPlanes::build_with_map(
                &plan,
                &image,
                focus,
                Some(&input(&map, k, tt)),
                &|| false,
            )?;
            let ms = t.elapsed().as_secs_f64() * 1e3;
            let mut s = stats(&planes, ms);
            s["k"] = json!(k);
            s["t"] = json!(tt);
            let valid = map
                .b
                .iter()
                .zip(&map.c)
                .filter(|&(&b, &c)| {
                    let beta = -0.25 * libm::log(f64::from(c).max(1e-6));
                    f64::from(b) * libm::exp(-k * beta) >= tt * ym_lens::consts::B_MIN
                })
                .count();
            s["valid_cells"] = json!(valid as f64 / map.b.len().max(1) as f64);
            if let (Some(out), Some(b)) = (&write_maps, planes.bokeh())
                && (k - 1.61).abs() < 1e-9
                && (tt - 2.0).abs() < 1e-9
            {
                let o = out.join(&name);
                std::fs::create_dir_all(&o)?;
                write_f32(&o.join("blur_b.f32"), b.blur())?;
                write_f32(&o.join("conf.f32"), b.confidence())?;
                let balls: Vec<_> = b
                    .balls()
                    .iter()
                    .map(|x| json!({"x": x.x, "y": x.y, "radius": x.radius, "supported": x.supported}))
                    .collect();
                std::fs::write(o.join("balls.json"), serde_json::to_string(&balls)?)?;
                std::fs::write(
                    o.join("size.json"),
                    serde_json::to_string(
                        &json!({"size": b.size(), "model_cells": [map.width, map.height]}),
                    )?,
                )?;
                write_f32(&o.join("model_b.f32"), &map.b)?;
                write_f32(&o.join("model_c.f32"), &map.c)?;
            }
            rows.push(s);
        }
        println!(
            "{}",
            json!({
                "photo": name,
                "plane": [w, h],
                "model_ms": {"reduce": tm.reduce_ms, "encode": tm.encode_ms, "forward": tm.forward_ms, "convert": tm.convert_ms},
                "classical": stats(&classical, classical_ms),
                "model": rows,
            })
        );
    }
    Ok(())
}
