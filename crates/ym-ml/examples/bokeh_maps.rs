//! Runs the bokeh map network (`ym_ml::bokeh_net`) on the lens planes dumped by
//! `crates/ym-core/examples/lens_sheet.rs --dump-planes` and writes its maps beside the classical
//! detector's, in the same false-colour scales, so the two can be compared side by side.
//!
//! ```text
//! cargo run -p ym-ml --example bokeh_maps -- --sheets <lens-sheets dir> [options]
//!
//!   --sheets DIR    folder of `<stem>/lens_plane.f32` + `lens_plane.json` (+ `bokeh.json`, `blur_b.png`)
//!   --weights PATH  a model folder (`data/models/bokeh`, with SHA256SUMS: checked) or a .safetensors file
//!                   (default: data/models/bokeh when it holds weights, else testdata/bokeh-smoke)
//!   --out DIR       write the results under DIR/<stem>/ instead of beside the planes
//!   --only STEM     only this photo (repeatable)
//!   --limit N       at most N photos
//! ```
//!
//! Per photo it writes, in `<stem>/`:
//!
//! * `model_b.png`: the network's `b` as the detector's `blur_b.png` shows its own (0 to 0.03
//!   half-diagonals, false colour), with the detector's caution rule applied so that "nothing" reads as
//!   nothing: `b = 0` (dark grey) where `c < 0.3` or `b < b_min = 0.004`;
//! * `model_b_raw.png`: the same without the rule (the network's `b` as it is);
//! * `model_c.png`: the confidence, 0 to 1, as `confidence_c.png`;
//! * `model.json`: the numbers (timings, the share of the frame at `b >= 2 b_min` with `c >= 0.5`,
//!   the same for the detector when `bokeh.json` is there).
//!
//! The maps are written at the size of the detector's own (`bokeh.json`'s `size`, else the plane's),
//! by bilinear sampling of the network's map at those cells' centres. The network is not told the
//! camera's focus point; the cross, when `bokeh.json` has one, only marks where the camera focused.
//!
//! Like the lens sheet, the CPU kernels are much faster with `CARGO_PROFILE_DEV_OPT_LEVEL=3`.

use std::error::Error;
use std::fs::{self, File};
use std::io::BufWriter;
use std::path::{Path, PathBuf};

use serde_json::{Value, json};
use ym_ml::bokeh_net::{BokehNet, PlanarRgbRef, WEIGHTS_FILE};

type Fallible<T> = Result<T, Box<dyn Error>>;

/// The smallest blur that matters, in half-diagonals (`docs/21` §4).
const B_MIN: f32 = 0.004;
/// The detector's rule: `b = 0` where the confidence is below this.
const C_MIN: f32 = 0.3;

struct Args {
    sheets: PathBuf,
    weights: Option<PathBuf>,
    out: Option<PathBuf>,
    only: Vec<String>,
    limit: usize,
}

fn parse_args() -> Fallible<Args> {
    let mut args = Args {
        sheets: PathBuf::new(),
        weights: None,
        out: None,
        only: Vec::new(),
        limit: usize::MAX,
    };
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        let mut value = |name: &str| it.next().ok_or_else(|| format!("{name} needs a value"));
        match a.as_str() {
            "--sheets" => args.sheets = value("--sheets")?.into(),
            "--weights" => args.weights = Some(value("--weights")?.into()),
            "--out" => args.out = Some(value("--out")?.into()),
            "--only" => args.only.push(value("--only")?),
            "--limit" => args.limit = value("--limit")?.parse()?,
            other => return Err(format!("unknown option {other}; see the file's header").into()),
        }
    }
    if args.sheets.as_os_str().is_empty() {
        return Err("--sheets <dir> is required; see the file's header".into());
    }
    Ok(args)
}

fn repo() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn load_net(weights: Option<&Path>) -> Fallible<(BokehNet, String)> {
    let default_dir = if repo().join("data/models/bokeh").join(WEIGHTS_FILE).exists() {
        repo().join("data/models/bokeh")
    } else {
        repo().join("testdata/bokeh-smoke")
    };
    let path = weights.map_or(default_dir, Path::to_path_buf);
    if path.is_dir() {
        let net = if path.join("SHA256SUMS").exists() {
            BokehNet::load(&path)?
        } else {
            BokehNet::from_file(&path.join(WEIGHTS_FILE))?
        };
        Ok((net, path.display().to_string()))
    } else {
        Ok((BokehNet::from_file(&path)?, path.display().to_string()))
    }
}

fn read_f32(path: &Path) -> Fallible<Vec<f32>> {
    let bytes = fs::read(path)?;
    if bytes.len() % 4 != 0 {
        return Err(format!("{} is not a multiple of 4 bytes", path.display()).into());
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect())
}

// The false-colour writers are `lens_sheet.rs`'s, so the maps are comparable.

fn write_png_rgb(path: &Path, w: usize, h: usize, rgb: &[u8]) -> Fallible<()> {
    let mut enc = png::Encoder::new(BufWriter::new(File::create(path)?), w as u32, h as u32);
    enc.set_color(png::ColorType::Rgb);
    enc.set_depth(png::BitDepth::Eight);
    let mut writer = enc.write_header()?;
    writer.write_image_data(rgb)?;
    Ok(())
}

/// A turbo-like ramp for 0 to 1 (0 is dark grey so that "nothing" reads as nothing).
fn ramp(t: f32) -> [u8; 3] {
    const STOPS: [[f32; 3]; 6] = [
        [0.10, 0.10, 0.45],
        [0.05, 0.45, 0.95],
        [0.10, 0.85, 0.75],
        [0.55, 0.95, 0.20],
        [0.98, 0.75, 0.10],
        [0.90, 0.10, 0.10],
    ];
    let t = t.clamp(0.0, 1.0) * (STOPS.len() - 1) as f32;
    let i = (t.floor() as usize).min(STOPS.len() - 2);
    let f = t - i as f32;
    let mut out = [0u8; 3];
    for c in 0..3 {
        out[c] = ((STOPS[i][c] * (1.0 - f) + STOPS[i + 1][c] * f) * 255.0 + 0.5) as u8;
    }
    out
}

/// A false-colour PNG of a map: values from 0 to `top`; exact zeros are dark grey; the focus point
/// (a fraction of the frame) is marked with a white cross.
fn write_false_colour(
    path: &Path,
    w: usize,
    h: usize,
    values: &[f32],
    top: f32,
    focus: Option<(f64, f64)>,
) -> Fallible<()> {
    let mut rgb = vec![0u8; w * h * 3];
    for (i, v) in values.iter().enumerate() {
        let c = if *v <= 0.0 {
            [28, 28, 28]
        } else {
            ramp(*v / top)
        };
        rgb[i * 3..i * 3 + 3].copy_from_slice(&c);
    }
    if let Some((fx, fy)) = focus {
        let (cx, cy) = ((fx * w as f64) as isize, (fy * h as f64) as isize);
        for d in -8isize..=8 {
            for (x, y) in [(cx + d, cy), (cx, cy + d)] {
                if x >= 0 && y >= 0 && (x as usize) < w && (y as usize) < h {
                    let o = (y as usize * w + x as usize) * 3;
                    rgb[o..o + 3].copy_from_slice(&[255, 255, 255]);
                }
            }
        }
    }
    write_png_rgb(path, w, h, &rgb)
}

/// The share of cells with `b >= 2 b_min` and `c >= 0.5` (the spec's "out of focus with confidence").
fn blurred_share(b: &[f32], c: &[f32]) -> f64 {
    let n = b
        .iter()
        .zip(c)
        .filter(|&(&b, &c)| b >= 2.0 * B_MIN && c >= 0.5)
        .count();
    n as f64 / b.len().max(1) as f64
}

fn process(net: &BokehNet, dir: &Path, out: &Path) -> Fallible<Value> {
    let meta: Value = serde_json::from_str(&fs::read_to_string(dir.join("lens_plane.json"))?)?;
    let dim = |k: &str| -> Fallible<usize> {
        Ok(meta[k]
            .as_u64()
            .ok_or_else(|| format!("lens_plane.json has no {k}"))? as usize)
    };
    let (w, h) = (dim("width")?, dim("height")?);
    let data = read_f32(&dir.join("lens_plane.f32"))?;
    let plane = PlanarRgbRef::new(w, h, &data)?;

    let (map, timings) = net.map_timed(&plane)?;

    // The detector's maps, when they are there: their size, the focus point and its own numbers.
    let detector: Option<Value> = fs::read_to_string(dir.join("bokeh.json"))
        .ok()
        .and_then(|t| serde_json::from_str(&t).ok());
    let size = detector
        .as_ref()
        .and_then(|d| {
            Some((
                d["bokeh"]["size"][0].as_u64()?,
                d["bokeh"]["size"][1].as_u64()?,
            ))
        })
        .map_or((w, h), |(a, b)| (a as usize, b as usize));
    let focus = detector.as_ref().and_then(|d| {
        Some((
            d["focus_fraction"][0].as_f64()?,
            d["focus_fraction"][1].as_f64()?,
        ))
    });
    let (b_raw, c) = map.resample(size.0, size.1);
    let b: Vec<f32> = b_raw
        .iter()
        .zip(&c)
        .map(|(&b, &c)| if c < C_MIN || b < B_MIN { 0.0 } else { b })
        .collect();

    fs::create_dir_all(out)?;
    write_false_colour(&out.join("model_b.png"), size.0, size.1, &b, 0.03, focus)?;
    write_false_colour(
        &out.join("model_b_raw.png"),
        size.0,
        size.1,
        &b_raw,
        0.03,
        focus,
    )?;
    write_false_colour(&out.join("model_c.png"), size.0, size.1, &c, 1.0, focus)?;

    let share = blurred_share(&b, &c);
    let detector_share = detector
        .as_ref()
        .and_then(|d| d["bokeh"]["blurred_fraction"].as_f64());
    let mean_c = c.iter().map(|&v| f64::from(v)).sum::<f64>() / c.len().max(1) as f64;
    let max_b = b.iter().copied().fold(0.0f32, f32::max);
    let info = json!({
        "plane": [w, h],
        "reduced_plane": [map.plane_width, map.plane_height],
        "map_cells": [map.width, map.height],
        "written_at": [size.0, size.1],
        "half_diagonal_px_of_reduced_plane": map.half_diagonal_px,
        "blurred_share_model": share,
        "blurred_share_detector": detector_share,
        "mean_confidence": mean_c,
        "max_b": max_b,
        "ms": {
            "reduce": timings.reduce_ms, "encode": timings.encode_ms,
            "network": timings.forward_ms, "convert": timings.convert_ms,
        },
        "note": "model_b.png: b = 0 where c < 0.3 or b < 0.004; model_b_raw.png: no rule; share = cells with b >= 2 b_min and c >= 0.5",
    });
    fs::write(out.join("model.json"), serde_json::to_string_pretty(&info)?)?;
    Ok(info)
}

fn main() -> Fallible<()> {
    let args = parse_args()?;
    let t = std::time::Instant::now();
    let (net, label) = load_net(args.weights.as_deref())?;
    println!(
        "weights: {label} (SHA-256 {}, {} parameters), loaded in {:.0} ms; rayon threads: {}",
        net.sha256(),
        net.parameters(),
        t.elapsed().as_secs_f64() * 1e3,
        rayon::current_num_threads()
    );
    let mut stems: Vec<PathBuf> = fs::read_dir(&args.sheets)?
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| p.join("lens_plane.f32").exists() && p.join("lens_plane.json").exists())
        .filter(|p| {
            args.only.is_empty()
                || p.file_name()
                    .is_some_and(|n| args.only.iter().any(|o| n.to_string_lossy() == o.as_str()))
        })
        .collect();
    stems.sort();
    stems.truncate(args.limit);
    println!("{} photos with a lens plane", stems.len());
    println!(
        "{:<34} {:>11} {:>9} {:>9} {:>8} {:>8}",
        "photo", "plane", "model %", "detect %", "mean c", "ms"
    );
    let mut failed = 0;
    for dir in &stems {
        let stem = dir
            .file_name()
            .map_or_else(String::new, |n| n.to_string_lossy().into_owned());
        let out = args
            .out
            .as_ref()
            .map_or_else(|| dir.clone(), |o| o.join(&stem));
        match process(&net, dir, &out) {
            Ok(info) => {
                let ms = &info["ms"];
                let total: f64 = ["reduce", "encode", "network", "convert"]
                    .iter()
                    .filter_map(|k| ms[k].as_f64())
                    .sum();
                println!(
                    "{:<34} {:>5}x{:<5} {:>9.2} {:>9} {:>8.3} {:>8.0}",
                    stem.chars().take(34).collect::<String>(),
                    info["plane"][0],
                    info["plane"][1],
                    100.0 * info["blurred_share_model"].as_f64().unwrap_or(0.0),
                    info["blurred_share_detector"]
                        .as_f64()
                        .map_or_else(|| "-".to_owned(), |v| format!("{:.2}", 100.0 * v)),
                    info["mean_confidence"].as_f64().unwrap_or(0.0),
                    total
                );
            }
            Err(e) => {
                failed += 1;
                println!("{stem}: failed: {e}");
            }
        }
    }
    if failed > 0 {
        return Err(format!("{failed} photos failed").into());
    }
    Ok(())
}
