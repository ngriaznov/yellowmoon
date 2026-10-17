//! CPU time of every demosaic method on a 50 MP frame, single-thread and parallel (the timing of
//! the demosaic gate, `docs/05` R6).
//!
//! ```text
//! CARGO_INCREMENTAL=0 cargo run -p ym-gpu --example demosaic_timing \
//!     --config 'profile.dev.package.ym-gpu.opt-level=3' \
//!     --config 'profile.dev.package.ym-gpu.debug-assertions=false' \
//!     --config 'profile.dev.package.ym-gpu.overflow-checks=false' \
//!     --config 'profile.dev.package.ym-gpu.codegen-units=16' [-- <megapixels>] [--out DIR]
//! ```
//!
//! The four `--config` flags make the `ym-gpu` crate compile like the release profile (opt-level 3,
//! no debug assertions, no overflow checks, 16 codegen units; only the thin LTO is missing), without
//! a second profile of the whole dependency tree: the dependencies already build at opt-level 3 in
//! the workspace's dev profile. Without them the numbers are those of the workspace's dev profile
//! (opt-level 1) and mean little for shipping code.
//!
//! The frame is `<megapixels>` MP (default 50, 8658 x 5772 photosites for 50, a 3:2 frame with even sides, RGGB): a 512 x 512 tile of
//! the procedural natural scene repeated over the frame at 0.4 of white, with Gaussian noise of
//! sigma 0.002 on top. The adaptive ratio demosaic gets gains `[2.0, 1.0, 1.6]` and the nominal
//! low-noise model of spec 01 section 6.3 (which keeps its shadow matching off, `Auto`); a second
//! row forces shadow matching on, the configuration a noisy photo gets. Every number is the best
//! of `REPEATS` runs: wall seconds, and CPU seconds of the whole process (user + system from
//! `/proc/self/stat`, 10 ms resolution, Linux). On a machine shared with other jobs the wall time
//! of the parallel runs is inflated; the CPU seconds are the dependable number. The load average
//! at the start and the end is printed for that reason.
//!
//! "1 thread" is a rayon pool of one thread running the same banded code as the pipeline;
//! "parallel" is the default pool (every hardware thread). The output of each run is dropped
//! before the next (a 50 MP result is 600 MB), and the outputs of the 1-thread and the parallel
//! run of one method are compared: the banded runners are thread-count independent, so they must
//! be bit-identical.

use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::time::Instant;

use serde_json::json;
use ym_gpu::cpu::demosaic::adaptive_ratio::{self, Params, ShadowMatch};
use ym_gpu::cpu::demosaic::{self, DemosaicInputs, Method};
use ym_gpu::cpu::synth;
use ym_gpu::cpu::{Cfa, Mosaic, NoiseDn, RgbImage};
use ym_noise::model::NoiseModel;

/// How many times each measurement is repeated (the best run is reported).
const REPEATS: usize = 3;

/// User + system CPU seconds of this process so far (Linux), 10 ms resolution.
fn cpu_seconds() -> Option<f64> {
    let stat = std::fs::read_to_string("/proc/self/stat").ok()?;
    // The command name may contain spaces; the fields after it start at ") ".
    let rest = &stat[stat.rfind(") ")? + 2..];
    let fields: Vec<&str> = rest.split_whitespace().collect();
    // utime and stime are fields 14 and 15 of the whole line: indices 11 and 12 after the name.
    let utime: f64 = fields.get(11)?.parse().ok()?;
    let stime: f64 = fields.get(12)?.parse().ok()?;
    Some((utime + stime) / 100.0)
}

fn load_average() -> String {
    std::fs::read_to_string("/proc/loadavg")
        .ok()
        .map(|s| s.split_whitespace().take(3).collect::<Vec<_>>().join(" "))
        .unwrap_or_else(|| "n/a".to_owned())
}

fn cpu_model() -> String {
    std::fs::read_to_string("/proc/cpuinfo")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("model name"))
                .and_then(|l| l.split(':').nth(1))
                .map(|m| m.trim().to_owned())
        })
        .unwrap_or_else(|| "unknown CPU".to_owned())
}

/// Best wall and best CPU seconds over [`REPEATS`] runs, and the last result.
fn best_of<T>(mut f: impl FnMut() -> T) -> (f64, f64, T) {
    let (mut wall, mut cpu) = (f64::INFINITY, f64::INFINITY);
    let mut last: Option<T> = None;
    for _ in 0..REPEATS {
        // Drop the previous result before the next run starts: 600 MB each at 50 MP.
        drop(last.take());
        let (c0, t0) = (cpu_seconds(), Instant::now());
        let out = f();
        wall = wall.min(t0.elapsed().as_secs_f64());
        if let Some(c) = c0.zip(cpu_seconds()).map(|(a, b)| b - a) {
            cpu = cpu.min(c);
        }
        last = Some(out);
    }
    (wall, cpu, last.expect("at least one run"))
}

fn pool(threads: usize) -> rayon::ThreadPool {
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .expect("thread pool")
}

/// One row of the table: a label and how to run the method.
struct Row<'a> {
    label: &'static str,
    run: Box<dyn Fn() -> RgbImage + Send + Sync + 'a>,
}

fn main() {
    let mut mp_arg = 50.0f64;
    let mut out_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata/demosaic_gate");
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        if a == "--out" {
            out_dir = PathBuf::from(args.next().expect("--out needs a folder"));
        } else if let Ok(v) = a.parse() {
            mp_arg = v;
        } else {
            eprintln!("usage: demosaic_timing [<megapixels>] [--out DIR] (got {a})");
            std::process::exit(2);
        }
    }
    // 3:2 frame with even sides.
    let h = ((mp_arg * 1e6 / 1.5).sqrt() as usize) & !1;
    let w = (h * 3 / 2) & !1;
    let mp = (w * h) as f64 / 1e6;
    let (tw, th) = (512usize, 512usize);
    let tile = synth::mosaic_from_rgb(&synth::natural(tw, th, 3), Cfa::RGGB);
    let mut m = Mosaic::new(w, h, Cfa::RGGB);
    for y in 0..h {
        for x in 0..w {
            m.data[y * w + x] = 0.4 * tile.data[(y % th) * tw + x % tw];
        }
    }
    synth::add_noise(&mut m.data, 0.002, 9);

    let nominal = NoiseDn::from_model(&NoiseModel {
        a: 2e-5,
        b: 1e-8,
        delta: [0.0; 4],
        sigma_row: 0.0,
        q: 1.0 / 65535.0,
    });
    let inputs = DemosaicInputs {
        gains: [2.0, 1.0, 1.6],
        noise: nominal,
        ca: None,
    };
    let matching = Params {
        shadow_match: ShadowMatch::On,
        ..Params::default()
    };
    let default_inputs = DemosaicInputs::default();

    let rows: Vec<Row<'_>> = vec![
        Row {
            label: "bilinear",
            run: Box::new(|| demosaic::demosaic_with(&m, Method::Bilinear, &default_inputs)),
        },
        Row {
            label: "menon",
            run: Box::new(|| demosaic::demosaic_with(&m, Method::Menon, &default_inputs)),
        },
        Row {
            label: "lmmse",
            run: Box::new(|| demosaic::demosaic_with(&m, Method::Lmmse, &default_inputs)),
        },
        Row {
            label: "adaptive_ratio (Auto, matching off)",
            run: Box::new(|| demosaic::demosaic_with(&m, Method::AdaptiveRatio, &inputs)),
        },
        Row {
            label: "adaptive_ratio (shadow matching on)",
            run: Box::new(|| {
                adaptive_ratio::demosaic_balanced(&m, &inputs, &matching).expect("valid inputs")
            }),
        },
        Row {
            label: "log_guided",
            run: Box::new(|| demosaic::demosaic_with(&m, Method::LogGuided, &inputs)),
        },
    ];

    let threads = rayon::current_num_threads();
    let one = pool(1);
    let mut text = String::new();
    let mut say = |s: String| {
        println!("{s}");
        text.push_str(&s);
        text.push('\n');
    };
    say(format!(
        "{w} x {h} = {mp:.1} MP, RGGB; {}; {threads} hardware threads; load average at start: {}",
        cpu_model(),
        load_average()
    ));
    say(String::new());
    // Measure every row first, then print the table (the relative column needs Menon's number).
    struct Measured {
        label: &'static str,
        wall_1: f64,
        cpu_1: f64,
        wall_n: f64,
        cpu_n: f64,
    }
    let mut measured = Vec::new();
    for row in &rows {
        let (wall_1, cpu_1, out1) = best_of(|| one.install(|| (row.run)()));
        let (wall_n, cpu_n, outn) = best_of(|| (row.run)());
        assert!(
            out1 == outn,
            "{}: 1 thread and {threads} threads must be bit-identical",
            row.label
        );
        drop((out1, outn));
        eprintln!("  measured {}", row.label);
        measured.push(Measured {
            label: row.label,
            wall_1,
            cpu_1,
            wall_n,
            cpu_n,
        });
    }
    let menon_cpu = measured
        .iter()
        .find(|r| r.label == "menon")
        .map_or(f64::NAN, |r| r.cpu_1);
    say(format!(
        "| method | 1 thread: wall s | 1 thread: CPU s | {threads} threads: wall s | {threads} threads: CPU s | CPU s / MP (1 thread) | x menon (CPU, 1 thread) |"
    ));
    say("|---|---|---|---|---|---|---|".to_owned());
    let mut json_rows = Vec::new();
    for r in &measured {
        let mut line = String::new();
        let _ = write!(
            line,
            "| {} | {:.2} | {:.2} | {:.2} | {:.2} | {:.3} | {:.2} |",
            r.label,
            r.wall_1,
            r.cpu_1,
            r.wall_n,
            r.cpu_n,
            r.cpu_1 / mp,
            r.cpu_1 / menon_cpu
        );
        say(line);
        json_rows.push(json!({
            "method": r.label,
            "single_thread": { "wall_s": r.wall_1, "cpu_s": r.cpu_1 },
            "parallel": { "threads": threads, "wall_s": r.wall_n, "cpu_s": r.cpu_n },
            "cpu_s_per_mp_single_thread": r.cpu_1 / mp,
            "cpu_vs_menon_single_thread": r.cpu_1 / menon_cpu,
        }));
    }
    say(String::new());
    say(format!(
        "load average at the end: {}. Best of {REPEATS} runs; CPU seconds are the dependable number on a shared machine.",
        load_average()
    ));
    let json = json!({
        "frame": { "width": w, "height": h, "megapixels": mp },
        "cpu": cpu_model(),
        "hardware_threads": threads,
        "repeats": REPEATS,
        "rows": json_rows,
    });
    std::fs::create_dir_all(&out_dir).expect("create the output folder");
    std::fs::write(out_dir.join("timing.md"), &text).expect("write timing.md");
    std::fs::write(
        out_dir.join("timing.json"),
        serde_json::to_string_pretty(&json).expect("json") + "\n",
    )
    .expect("write timing.json");
}
