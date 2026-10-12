//! M0-08 benchmark harness: time the `jdd` network per available backend (`docs/07-ml-models.md`
//! section Runtime, decision rule: pick the faster by 20 percent, ties go to burn).
//!
//! ```text
//! make -C training/jdd bench-weights          # seeded random weights + ONNX + PyTorch reference tile
//! cargo run --release -p ym-ml --example bench -- --dir testdata/ml/random-512 [options]
//! cargo run --release -p ym-ml --features burn,ort --example bench -- --dir testdata/ml/random-512
//!
//!   --dir DIR           folder from `export.py --random` (jdd.safetensors, jdd.onnx, io.safetensors)
//!   --backends LIST     comma separated subset of cpu,burn,ort (default: every compiled-in backend)
//!   --reps N            timed tile runs after one warm-up run (default 3)
//!   --frame-mp MP       also run a full frame of MP megapixels tile by tile (default 0 = skip)
//!   --extrapolate       with --frame-mp: estimate the full frame from the tile time instead of running it
//!   --core N            tile core in half-resolution px for cpu and burn (default 512; ort uses its ONNX shape)
//!   --no-cpu-diff       skip the comparison with the CPU runtime (saves one CPU-runtime tile run per backend)
//!   --json FILE         write the results as JSON
//! ```
//!
//! Every backend runs in its own child process so that the peak memory (`ru_maxrss`) belongs to that
//! backend alone. Reported per backend: load time, time per tile (the exported tile: core 512 plus halo,
//! half-resolution pixels), full-frame time for a 50 MP frame (8192 x 6144), peak memory and the maximum
//! absolute difference to the PyTorch output of the same tile (and to the CPU runtime).
//!
//! **Numbers measured on a Linux machine (CPU, or Mesa lavapipe for burn) are not decision-grade.** The
//! decision needs an M1 and an M3 Mac (HUMAN M0-H4); this harness is what they run.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

use serde_json::{Value, json};
use ym_ml::cpu_runtime::{CpuJdd, read_float_tensors};
use ym_ml::model::{ExecContext, IN_CHANNELS, TileInputs, TileModel, TileOutput, TileShape};
use ym_ml::tiles::{Frame, run_tiled_with_progress, tile_count};

const FRAME_50MP: (usize, usize) = (8192, 6144); // full-resolution pixels (50.3 MP)

struct Args {
    dir: PathBuf,
    backends: Vec<String>,
    reps: usize,
    frame_mp: f64,
    extrapolate: bool,
    cpu_diff: bool,
    core: Option<usize>,
    json: Option<PathBuf>,
    child: Option<String>,
}

fn parse_args() -> Result<Args, String> {
    let mut a = Args {
        dir: PathBuf::from("testdata/ml/random-512"),
        backends: ym_ml::backends::available()
            .into_iter()
            .map(String::from)
            .collect(),
        reps: 3,
        frame_mp: 0.0,
        extrapolate: false,
        cpu_diff: true,
        core: None,
        json: None,
        child: None,
    };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        let mut val = |name: &str| it.next().ok_or(format!("{name} needs a value"));
        match arg.as_str() {
            "--dir" => a.dir = PathBuf::from(val("--dir")?),
            "--backends" => {
                a.backends = val("--backends")?.split(',').map(str::to_string).collect()
            }
            "--reps" => a.reps = val("--reps")?.parse().map_err(|e| format!("--reps: {e}"))?,
            "--frame-mp" => {
                a.frame_mp = val("--frame-mp")?
                    .parse()
                    .map_err(|e| format!("--frame-mp: {e}"))?
            }
            "--extrapolate" => a.extrapolate = true,
            "--no-cpu-diff" => a.cpu_diff = false,
            "--core" => a.core = Some(val("--core")?.parse().map_err(|e| format!("--core: {e}"))?),
            "--json" => a.json = Some(PathBuf::from(val("--json")?)),
            "--child" => a.child = Some(val("--child")?),
            other => {
                return Err(format!(
                    "unknown argument {other}; see the file header for usage"
                ));
            }
        }
    }
    Ok(a)
}

/// Peak resident set size of this process in bytes.
fn peak_rss_bytes() -> u64 {
    // SAFETY: `getrusage` only writes into the zero-initialised struct we pass.
    let mut ru: libc::rusage = unsafe { std::mem::zeroed() };
    let rc = unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut ru) };
    if rc != 0 {
        return 0;
    }
    let max = ru.ru_maxrss as u64;
    if cfg!(target_os = "macos") {
        max
    } else {
        max * 1024
    }
}

/// The docs/07 decision rule: ort wins only if it is at least 20 percent faster (its time at most 0.8 of
/// burn's); burn is faster by 20 percent or more, or the two tie, so burn is picked in every other case.
fn decide(burn_s: f64, ort_s: f64) -> &'static str {
    if ort_s <= 0.8 * burn_s { "ort" } else { "burn" }
}

/// The 50 MP frame time (measured, or tile time x tile count) of a backend result, if it has one.
fn frame_seconds(result: &Value) -> Option<f64> {
    let direct = result["frame"]["seconds"].as_f64();
    direct
        .or_else(|| result["frame_50mp_s_extrapolated"].as_f64())
        .filter(|s| s.is_finite() && *s > 0.0)
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.total_cmp(b));
    v[v.len() / 2]
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| f64::from((x - y).abs()))
        .fold(0.0, f64::max)
}

/// The PyTorch reference tile written by `export.py` (`io.safetensors`: `x`, `rgb`), if present.
fn load_reference(dir: &Path) -> Option<(TileInputs, Vec<f32>)> {
    let bytes = std::fs::read(dir.join("io.safetensors")).ok()?;
    let t = read_float_tensors(&bytes).ok()?;
    let (xs, x) = t.get("x")?.clone();
    let (_, y) = t.get("rgb")?.clone();
    let shape = TileShape {
        height: xs[2],
        width: xs[3],
    };
    Some((TileInputs::new(shape, x).ok()?, y))
}

fn synthetic_tile(shape: TileShape) -> TileInputs {
    let mut t = TileInputs::zeros(shape);
    let mut s = 0x9E3779B97F4A7C15u64;
    let mut next = move || {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((s >> 40) as f32) / (1u64 << 24) as f32
    };
    for c in 0..IN_CHANNELS {
        let scale = [0.5, 0.5, 0.5, 0.5, 0.05, 0.01, 1.0][c];
        for v in t.plane_mut(c) {
            *v = next() * scale;
        }
    }
    t
}

fn build_model(
    backend: &str,
    dir: &Path,
    core: Option<usize>,
) -> Result<(Box<dyn TileModel>, TileShape), String> {
    let weights = std::fs::read(dir.join("jdd.safetensors"))
        .map_err(|e| format!("{}: {e}", dir.join("jdd.safetensors").display()))?;
    match backend {
        "cpu" => {
            let mut m = CpuJdd::from_bytes(&weights, None).map_err(|e| e.to_string())?;
            if let Some(c) = core {
                m = m.with_core_half_px(c);
            }
            let shape = m.input_shape();
            Ok((Box::new(m), shape))
        }
        #[cfg(feature = "burn")]
        "burn" => {
            let m = ym_ml::backends::burn_backend::BurnJdd::from_bytes(&weights)
                .map_err(|e| e.to_string())?;
            let shape = m.input_shape();
            Ok((Box::new(m), shape))
        }
        #[cfg(feature = "ort")]
        "ort" => {
            let cpu = CpuJdd::from_bytes(&weights, None).map_err(|e| e.to_string())?;
            let shape = cpu.input_shape();
            let onnx = dir.join("jdd.onnx");
            let m = match std::env::var_os("ORT_DYLIB_PATH") {
                Some(lib) => ym_ml::backends::ort_backend::OrtJdd::with_library(
                    lib,
                    onnx,
                    shape,
                    cpu.halo_half_px(),
                ),
                None => ym_ml::backends::ort_backend::OrtJdd::new(onnx, shape, cpu.halo_half_px()),
            }
            .map_err(|e| e.to_string())?;
            Ok((Box::new(m), shape))
        }
        other => Err(format!(
            "backend `{other}` is not compiled in (features burn, ort)"
        )),
    }
}

/// Run one backend and return its result as JSON (child mode).
fn run_backend(backend: &str, args: &Args) -> Result<Value, String> {
    let t0 = Instant::now();
    let (model, shape) = build_model(backend, &args.dir, args.core)?;
    let load_s = t0.elapsed().as_secs_f64();
    let reference = load_reference(&args.dir);
    let input = reference
        .as_ref()
        .map_or_else(|| synthetic_tile(shape), |(x, _)| x.clone());
    if input.shape() != shape {
        return Err(format!(
            "reference tile is {} but the model expects {shape}",
            input.shape()
        ));
    }
    let ctx = ExecContext::cpu();
    let mut out = TileOutput::zeros(shape);
    model
        .run(&input, &mut out, &ctx)
        .map_err(|e| e.to_string())?; // warm-up (shader compilation, allocations)
    let mut times = Vec::new();
    for _ in 0..args.reps.max(1) {
        let t = Instant::now();
        model
            .run(&input, &mut out, &ctx)
            .map_err(|e| e.to_string())?;
        times.push(t.elapsed().as_secs_f64() * 1e3);
    }
    let tile_peak = peak_rss_bytes();
    let diff_torch = reference
        .as_ref()
        .map(|(_, y)| max_abs_diff(out.as_slice(), y));
    let diff_cpu = if backend == "cpu" {
        Some(0.0)
    } else if !args.cpu_diff {
        None
    } else {
        let (cpu, _) = build_model("cpu", &args.dir, args.core)?;
        let mut o = TileOutput::zeros(shape);
        cpu.run(&input, &mut o, &ctx).map_err(|e| e.to_string())?;
        Some(max_abs_diff(out.as_slice(), o.as_slice()))
    };
    let halo = model.halo_half_px();
    let tiles_50mp =
        tile_count(FRAME_50MP.0 / 2, FRAME_50MP.1 / 2, shape, halo).map_err(|e| e.to_string())?;
    let tile_ms = median(&mut times.clone());
    let mut result = json!({
        "backend": backend,
        "tile_input": [shape.height, shape.width],
        "halo_half_px": halo,
        "load_s": load_s,
        "tile_ms_median": tile_ms,
        "tile_ms_min": times.iter().copied().fold(f64::INFINITY, f64::min),
        "peak_rss_mb_tile": tile_peak as f64 / 1e6,
        "max_abs_diff_vs_pytorch": diff_torch,
        "max_abs_diff_vs_cpu": diff_cpu,
        "tiles_per_50mp": tiles_50mp,
        "frame_50mp_s_extrapolated": tile_ms * tiles_50mp as f64 / 1e3,
    });
    if args.frame_mp > 0.0 {
        let (fw, fh) = frame_dims(args.frame_mp);
        let tiles = tile_count(fw, fh, shape, halo).map_err(|e| e.to_string())?;
        result["frame"] = json!({"width_half_px": fw, "height_half_px": fh, "tiles": tiles, "megapixels": args.frame_mp});
        if args.extrapolate {
            result["frame"]["seconds"] = json!(tile_ms * tiles as f64 / 1e3);
            result["frame"]["measured"] = json!(false);
        } else {
            let mut frame = Frame::zeros(fw, fh);
            let mut s = 12345u64;
            for v in frame.as_mut_slice() {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                *v = ((s >> 40) as f32) / (1u64 << 24) as f32 * 0.5;
            }
            let t = Instant::now();
            run_tiled_with_progress(model.as_ref(), &frame, &ctx, |done, total| {
                if done % 8 == 0 || done == total {
                    eprintln!(
                        "  [{backend}] tile {done}/{total} ({:.0}s)",
                        t.elapsed().as_secs_f64()
                    );
                }
                true
            })
            .map_err(|e| e.to_string())?;
            result["frame"]["seconds"] = json!(t.elapsed().as_secs_f64());
            result["frame"]["measured"] = json!(true);
        }
        result["peak_rss_mb_frame"] = json!(peak_rss_bytes() as f64 / 1e6);
    }
    Ok(result)
}

/// Half-resolution frame size for `mp` megapixels at 3:2.
fn frame_dims(mp: f64) -> (usize, usize) {
    let full_w = ((mp * 1e6 * 1.5).sqrt() / 8.0).round() as usize * 8;
    let full_h = ((mp * 1e6 / full_w as f64) / 8.0).round() as usize * 8;
    (full_w / 2, full_h / 2)
}

fn main() {
    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(2);
        }
    };
    if let Some(backend) = &args.child {
        match run_backend(backend, &args) {
            Ok(v) => println!("{v}"),
            Err(e) => {
                println!("{}", json!({"backend": backend, "error": e}));
            }
        }
        return;
    }
    let exe = std::env::current_exe().expect("own path");
    let mut results = Vec::new();
    for backend in &args.backends {
        eprintln!("== {backend}");
        let mut cmd = Command::new(&exe);
        cmd.args(["--child", backend, "--dir"])
            .arg(&args.dir)
            .args(["--reps", &args.reps.to_string()]);
        if args.frame_mp > 0.0 {
            cmd.args(["--frame-mp", &args.frame_mp.to_string()]);
        }
        if args.extrapolate {
            cmd.arg("--extrapolate");
        }
        if !args.cpu_diff {
            cmd.arg("--no-cpu-diff");
        }
        if let Some(c) = args.core {
            cmd.args(["--core", &c.to_string()]);
        }
        let out = cmd
            .stderr(std::process::Stdio::inherit())
            .output()
            .expect("spawn child");
        let line = String::from_utf8_lossy(&out.stdout);
        match line
            .lines()
            .last()
            .and_then(|l| serde_json::from_str::<Value>(l).ok())
        {
            Some(v) => results.push(v),
            None => results.push(
                json!({"backend": backend, "error": format!("child failed: {}", out.status)}),
            ),
        }
    }
    println!(
        "\nLinux CPU / lavapipe, not decision-grade (the M0-08 decision needs M1 and M3 numbers, HUMAN M0-H4)"
    );
    println!(
        "{:<6} {:>9} {:>13} {:>16} {:>12} {:>10} {:>12} {:>12}",
        "back",
        "load s",
        "tile ms (med)",
        "50 MP frame s",
        "peak MB",
        "tiles",
        "diff torch",
        "diff cpu"
    );
    for r in &results {
        if let Some(e) = r.get("error") {
            println!("{:<6} error: {e}", r["backend"].as_str().unwrap_or("?"));
            continue;
        }
        let frame_s = r["frame"]["seconds"]
            .as_f64()
            .unwrap_or_else(|| r["frame_50mp_s_extrapolated"].as_f64().unwrap_or(f64::NAN));
        let measured = r["frame"]["measured"].as_bool().unwrap_or(false);
        let peak = r["peak_rss_mb_frame"]
            .as_f64()
            .unwrap_or_else(|| r["peak_rss_mb_tile"].as_f64().unwrap_or(f64::NAN));
        println!(
            "{:<6} {:>9.2} {:>13.1} {:>13.1}{} {:>12.0} {:>10} {:>12} {:>12}",
            r["backend"].as_str().unwrap_or("?"),
            r["load_s"].as_f64().unwrap_or(f64::NAN),
            r["tile_ms_median"].as_f64().unwrap_or(f64::NAN),
            frame_s,
            if measured { "  " } else { " *" },
            peak,
            r["tiles_per_50mp"],
            r["max_abs_diff_vs_pytorch"]
                .as_f64()
                .map_or("-".to_string(), |d| format!("{d:.2e}")),
            r["max_abs_diff_vs_cpu"]
                .as_f64()
                .map_or("-".to_string(), |d| format!("{d:.2e}")),
        );
    }
    println!(
        "* extrapolated: tile time x tile count; frame = 8192 x 6144 px, tiles of core 512 plus halo (half-res)"
    );
    let time_of = |name: &str| {
        results
            .iter()
            .find(|r| r["backend"] == name && r.get("error").is_none())
            .and_then(frame_seconds)
    };
    match (time_of("burn"), time_of("ort")) {
        (Some(burn), Some(ort)) => println!(
            "docs/07 rule (50 MP frame time, ort must be >= 20% faster, ties go to burn): {} (burn {burn:.1} s, ort {ort:.1} s)",
            decide(burn, ort)
        ),
        _ => println!(
            "docs/07 rule: no decision, it needs a burn and an ort result from the same machine (features burn,ort)"
        ),
    }
    if let Some(path) = &args.json {
        let doc = json!({"note": "Linux CPU / lavapipe, not decision-grade", "results": results});
        std::fs::write(path, serde_json::to_string_pretty(&doc).expect("json"))
            .expect("write json");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decision_rule_of_docs_07() {
        assert_eq!(decide(10.0, 7.9), "ort"); // ort at least 20% faster
        assert_eq!(decide(10.0, 8.0), "ort"); // exactly 20%
        assert_eq!(decide(10.0, 8.1), "burn"); // within the tie band
        assert_eq!(decide(10.0, 10.0), "burn"); // tie
        assert_eq!(decide(10.0, 15.0), "burn"); // burn faster
    }

    #[test]
    fn frame_seconds_prefers_the_measured_frame() {
        let measured = json!({"frame": {"seconds": 12.5}, "frame_50mp_s_extrapolated": 99.0});
        assert_eq!(frame_seconds(&measured), Some(12.5));
        let tile_only = json!({"frame_50mp_s_extrapolated": 99.0});
        assert_eq!(frame_seconds(&tile_only), Some(99.0));
        assert_eq!(frame_seconds(&json!({"backend": "burn"})), None);
    }
}
