//! Runs the bokeh map network on dumped lens planes and says what the lens's planes make of its map, next to the
//! classical detector. A developer tool for local review (never shipped).
//!
//! ```text
//! cargo run -p ym-core --example bokeh_gate -- --data DIR [--rules 0.7:0.9,0.5:0.9] [--passes both] \
//!     [--write-maps OUT] <dump-dir>...
//! ```
//!
//! A dump folder holds `lens_plane.f32` and `lens_plane.json` as `bokeh_probe` reads them. `DIR` is a data folder
//! with `models/bokeh/` and `lenses/`. One JSON line per photo goes to stdout: the network's timings, and for the
//! classical detector and each rule (`blurred:strong`, the probabilities that call a cell blurred and count it
//! for "the photo has bokeh"; the lens uses `0.7:0.9`, the others are tried by remapping the probability so that
//! the lens's own thresholds fall there) and pass (`plain`, `mirrored`) whether the planes are active, their edge
//! count, blurred fraction, balls and the share of the map above zero. With `--write-maps` the lens's own rule's
//! map and the network's raw `b`, `c` and `p` go to `OUT/<name>/` as little-endian f32 (`model_b.f32`,
//! `model_c.f32`, `model_p.f32`, `blur_b.f32`, `conf.f32`, `balls.json`, `size.json`) for the mirrored pass.

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

/// The map as the lens takes it, with the probability remapped (piecewise linear, monotone) so that `blurred` falls
/// at the lens's 0.7 and `strong` at its 0.9.
fn input(map: &BokehMap, blurred: f64, strong: f64) -> BlurMapInput {
    let remap = |p: f64| {
        if p < blurred {
            0.7 * p / blurred
        } else if p < strong {
            0.7 + 0.2 * (p - blurred) / (strong - blurred).max(1e-9)
        } else {
            0.9 + 0.1 * (p - strong) / (1.0 - strong).max(1e-9)
        }
    };
    BlurMapInput {
        width: map.width,
        height: map.height,
        plane_width: map.plane_width,
        plane_height: map.plane_height,
        b: map.b.clone(),
        p: map.p.iter().map(|&p| remap(f64::from(p)) as f32).collect(),
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
    let mut rules: Vec<(f64, f64)> = vec![(0.7, 0.9)];
    let mut passes = vec![false, true];
    let mut write_maps: Option<PathBuf> = None;
    let mut dirs: Vec<PathBuf> = Vec::new();
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--data" => data = PathBuf::from(it.next().ok_or("--data needs a folder")?),
            "--rules" => {
                rules = it
                    .next()
                    .ok_or("--rules needs a list")?
                    .split(',')
                    .map(|r| {
                        let (a, b) = r.split_once(':').ok_or("a rule is blurred:strong")?;
                        Ok::<_, Box<dyn std::error::Error>>((a.parse()?, b.parse()?))
                    })
                    .collect::<Result<_, _>>()?;
            }
            "--passes" => {
                passes = match it
                    .next()
                    .ok_or("--passes needs plain, mirrored or both")?
                    .as_str()
                {
                    "plain" => vec![false],
                    "mirrored" => vec![true],
                    _ => vec![false, true],
                };
            }
            "--write-maps" => {
                write_maps = Some(PathBuf::from(
                    it.next().ok_or("--write-maps needs a folder")?,
                ));
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
        let mut rows = Vec::new();
        let mut timing = json!({});
        for &mirrored in &passes {
            let (map, tm) = model.map_with(&planar, mirrored)?;
            timing[if mirrored { "mirrored" } else { "plain" }] = json!({
                "reduce": tm.reduce_ms, "encode": tm.encode_ms, "forward": tm.forward_ms,
                "mirrored": tm.mirrored_ms, "convert": tm.convert_ms,
            });
            for &(blurred, strong) in &rules {
                let t = Instant::now();
                let planes = LensPlanes::build_with_map(
                    &plan,
                    &image,
                    focus,
                    Some(&input(&map, blurred, strong)),
                    &|| false,
                )?;
                let ms = t.elapsed().as_secs_f64() * 1e3;
                let mut s = stats(&planes, ms);
                s["mirrored"] = json!(mirrored);
                s["blurred_p"] = json!(blurred);
                s["strong_p"] = json!(strong);
                s["share_blurred"] = json!(
                    map.p.iter().filter(|&&p| f64::from(p) >= blurred).count() as f64
                        / map.p.len().max(1) as f64
                );
                s["share_strong"] = json!(
                    map.p.iter().filter(|&&p| f64::from(p) >= strong).count() as f64
                        / map.p.len().max(1) as f64
                );
                if let (Some(out), Some(b)) = (&write_maps, planes.bokeh())
                    && mirrored
                    && (blurred, strong) == (0.7, 0.9)
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
                    write_f32(&o.join("model_p.f32"), &map.p)?;
                }
                rows.push(s);
            }
        }
        println!(
            "{}",
            json!({
                "photo": name,
                "plane": [w, h],
                "model_ms": timing,
                "classical": stats(&classical, classical_ms),
                "model": rows,
            })
        );
    }
    Ok(())
}
