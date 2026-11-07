//! Runs the Depth Anything V2-Small runtime on a folder of pictures and writes the maps (the Rust half of the DIODE ordering
//! gate of D-06, `docs/algorithms/06-masks.md` section 8.8; `training/depth/eval_diode.py` drives it in CI).
//!
//! ```text
//! cargo run --release -p ym-ml --example depth_maps -- IN_DIR OUT_DIR [DATA_ROOT]
//! ```
//!
//! Every `IN_DIR/<name>.npy` is a little-endian `f32` array `H x W x 3` (RGB, the 8-bit picture divided by 255, so in `[0, 1]`).
//! For each one, in name order, it does what the app does (`prepare_input`: the upstream size held to the 1036 px cap, the cubic
//! resize, the ImageNet normalisation; then `DepthAnything::infer`) and writes `OUT_DIR/<name>.depth.npy`, an `f32` array
//! `h x w` at the size of the network input, and prints one JSON line per file:
//! `{"name":"..","picture":[W,H],"network":[w,h],"ms":..}` (`ms` covers the preparation and the network, not the file reads).
//! `DATA_ROOT` is the folder that holds `models/depth/` (default `$YELLOWMOON_DATA`, else `data`). The `.npy` format is read and
//! written here by hand (version 1 to 3 headers, C order, `<f4`); there is no other dependency. Evaluation tooling only: the
//! app and `ym` never run this.

use std::path::{Path, PathBuf};
use std::time::Instant;

use ym_ml::depth::{DepthAnything, prepare_input};

/// Reads an `f32` `.npy` file; returns its shape and values.
fn read_npy(path: &Path) -> Result<(Vec<usize>, Vec<f32>), String> {
    let b = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let bad = |what: &str| format!("{}: {what}", path.display());
    if b.len() < 12 || &b[..6] != b"\x93NUMPY" {
        return Err(bad("not an .npy file"));
    }
    let (hlen, start) = match b[6] {
        1 => (usize::from(u16::from_le_bytes([b[8], b[9]])), 10),
        2 | 3 => (u32::from_le_bytes([b[8], b[9], b[10], b[11]]) as usize, 12),
        _ => return Err(bad("unknown .npy version")),
    };
    let header = b
        .get(start..start + hlen)
        .and_then(|h| std::str::from_utf8(h).ok())
        .ok_or_else(|| bad("bad header"))?;
    if !header.contains("'<f4'") || !header.contains("'fortran_order': False") {
        return Err(bad("expected little-endian f32 in C order"));
    }
    let shape: Vec<usize> = header
        .split("'shape':")
        .nth(1)
        .and_then(|s| s.split(['(', ')']).nth(1))
        .ok_or_else(|| bad("no shape"))?
        .split(',')
        .filter_map(|t| t.trim().parse().ok())
        .collect();
    let body = &b[start + hlen..];
    if body.len() != 4 * shape.iter().product::<usize>() {
        return Err(bad("the data does not match the shape"));
    }
    let data = body
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    Ok((shape, data))
}

/// Writes a 2-D `f32` `.npy` file (version 1 header, padded so the data starts on a multiple of 64 bytes).
fn write_npy(path: &Path, rows: usize, cols: usize, data: &[f32]) -> Result<(), String> {
    let mut header =
        format!("{{'descr': '<f4', 'fortran_order': False, 'shape': ({rows}, {cols}), }}");
    while (10 + header.len() + 1) % 64 != 0 {
        header.push(' ');
    }
    header.push('\n');
    let mut out = Vec::with_capacity(10 + header.len() + 4 * data.len());
    out.extend_from_slice(b"\x93NUMPY\x01\x00");
    // The header is a few dozen bytes, far below the version 1 limit of 65535.
    out.extend_from_slice(&(header.len() as u16).to_le_bytes());
    out.extend_from_slice(header.as_bytes());
    for v in data {
        out.extend_from_slice(&v.to_le_bytes());
    }
    std::fs::write(path, out).map_err(|e| format!("{}: {e}", path.display()))
}

fn run() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() < 2 || args.len() > 3 {
        return Err("usage: depth_maps IN_DIR OUT_DIR [DATA_ROOT]".into());
    }
    let (input, output) = (PathBuf::from(&args[0]), PathBuf::from(&args[1]));
    let root = args
        .get(2)
        .cloned()
        .or_else(|| std::env::var("YELLOWMOON_DATA").ok())
        .unwrap_or_else(|| "data".into());
    let model =
        DepthAnything::load_bundled(Path::new(&root)).map_err(|e| format!("weights: {e}"))?;
    std::fs::create_dir_all(&output).map_err(|e| format!("{}: {e}", output.display()))?;
    let mut files: Vec<PathBuf> = std::fs::read_dir(&input)
        .map_err(|e| format!("{}: {e}", input.display()))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "npy"))
        .collect();
    files.sort();
    for path in files {
        let name = path
            .file_stem()
            .and_then(|s| s.to_str())
            .ok_or("bad file name")?
            .to_owned();
        let (shape, rgb) = read_npy(&path)?;
        let [h, w, 3] = shape[..] else {
            return Err(format!("{name}: expected H x W x 3, found {shape:?}"));
        };
        let t = Instant::now();
        let prepared = prepare_input(&rgb, w, h).map_err(|e| format!("{name}: {e}"))?;
        let map = model
            .infer(&prepared.data, prepared.width, prepared.height)
            .map_err(|e| format!("{name}: {e}"))?;
        let ms = t.elapsed().as_secs_f64() * 1000.0;
        write_npy(
            &output.join(format!("{name}.depth.npy")),
            map.height,
            map.width,
            &map.data,
        )?;
        println!(
            "{{\"name\":\"{name}\",\"picture\":[{w},{h}],\"network\":[{},{}],\"ms\":{ms:.1}}}",
            map.width, map.height
        );
    }
    Ok(())
}

fn main() {
    if let Err(e) = run() {
        eprintln!("depth_maps: {e}");
        std::process::exit(1);
    }
}
