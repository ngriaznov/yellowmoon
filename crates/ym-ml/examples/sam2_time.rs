//! Times SAM 2.1 small on this machine (`docs/07-ml-models.md`, N-17): the weights load, the encoder (per stage) and the
//! decoder per click, on a synthetic picture (the cost does not depend on the content).
//!
//! ```text
//! tools/fetch/sam2.sh
//! cargo run --release -p ym-ml --example sam2_time -- [--data data] [--threads N] [--reps 3] [--clicks 20]
//! ```
//!
//! Prints one line per measurement; the best and the median of the repetitions, with the machine's load average, because
//! the numbers of a loaded machine mean little.

use std::path::PathBuf;
use std::time::Instant;

use ym_ml::sam2::{Picture, Prompt, Sam2};

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

fn load_average() -> String {
    std::fs::read_to_string("/proc/loadavg").map_or_else(
        |_| "unknown".into(),
        |s| s.split_whitespace().take(3).collect::<Vec<_>>().join(" "),
    )
}

/// CPU seconds this process has used so far (user and system, all threads), from `/proc/self/stat` at 100 ticks a second:
/// unlike the wall time, the work an encoder run costs does not grow when another program takes some of the cores.
fn cpu_seconds() -> f64 {
    std::fs::read_to_string("/proc/self/stat")
        .ok()
        .and_then(|s| {
            let after_name = s.rsplit_once(')')?.1.to_owned();
            let f: Vec<&str> = after_name.split_whitespace().collect();
            Some((f.get(11)?.parse::<f64>().ok()? + f.get(12)?.parse::<f64>().ok()?) / 100.0)
        })
        .unwrap_or(0.0)
}

fn peak_rss_mb() -> f64 {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("VmHWM:"))
                .and_then(|l| l.split_whitespace().nth(1)?.parse::<f64>().ok())
        })
        .map_or(0.0, |kb| kb / 1024.0)
}

fn main() {
    let mut data = PathBuf::from("data");
    let (mut threads, mut reps, mut clicks) = (0usize, 3usize, 20usize);
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        let mut value = || {
            args.next()
                .expect("a value")
                .parse::<usize>()
                .expect("a number")
        };
        match a.as_str() {
            "--data" => data = PathBuf::from(args.next().expect("a path")),
            "--threads" => threads = value(),
            "--reps" => reps = value(),
            "--clicks" => clicks = value(),
            other => panic!("unknown argument {other}"),
        }
    }
    if threads > 0 {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build_global()
            .expect("pool");
    }
    println!(
        "threads {}, load average (1, 5, 15 min) {}",
        rayon::current_num_threads(),
        load_average()
    );

    let t = Instant::now();
    let model = Sam2::load_bundled(&data).expect("weights (tools/fetch/sam2.sh)");
    println!(
        "load (read, SHA-256, widen, pack): {:.0} ms",
        t.elapsed().as_secs_f64() * 1e3
    );

    // A 1024 x 683 picture of smooth colour and some texture.
    let (w, h) = (1024usize, 683usize);
    let mut rgb = vec![0u8; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            let n = ((x * 7 + y * 13) % 31) as f32;
            rgb[(y * w + x) * 3] = (x * 255 / w) as u8;
            rgb[(y * w + x) * 3 + 1] = (y * 255 / h) as u8;
            rgb[(y * w + x) * 3 + 2] =
                (128.0 + 30.0 * ((x as f32 / 90.0).sin() + (y as f32 / 70.0).cos()) + n) as u8;
        }
    }
    let picture = Picture::new(w, h, &rgb).expect("picture");

    let mut total = Vec::new();
    let mut cpu = Vec::new();
    let mut stages = Vec::new();
    let mut features = None;
    for rep in 0..=reps {
        let (t, c) = (Instant::now(), cpu_seconds());
        let (f, s) = model.encode_timed(&picture).expect("encode");
        let ms = t.elapsed().as_secs_f64() * 1e3;
        let c = cpu_seconds() - c;
        if rep == 0 {
            println!("encoder, first run (warm-up): {ms:.0} ms");
        } else {
            total.push(ms);
            cpu.push(c);
            stages.push((s.preprocess_ms, s.trunk_ms, s.neck_ms));
        }
        features = Some(f);
    }
    let features = features.expect("features");
    let best = total.iter().copied().fold(f64::INFINITY, f64::min);
    let i = total.iter().position(|&v| v == best).unwrap_or(0);
    println!(
        "encoder, {} runs: best {best:.0} ms (preprocess {:.0}, trunk {:.0}, neck {:.0}), median {:.0} ms",
        total.len(),
        stages[i].0,
        stages[i].1,
        stages[i].2,
        median(&mut total.clone())
    );
    println!(
        "encoder CPU time (all threads, the work done): least {:.1} s, median {:.1} s per run",
        cpu.iter().copied().fold(f64::INFINITY, f64::min),
        median(&mut cpu.clone())
    );

    let prompts = [
        ("one click", Prompt::click(0.5, 0.5)),
        (
            "click + negative click",
            Prompt::click(0.5, 0.5).with_point(0.2, 0.2, false),
        ),
        ("box", Prompt::rect(0.3, 0.3, 0.7, 0.7)),
    ];
    for (name, prompt) in &prompts {
        let mut times = Vec::new();
        for _ in 0..=clicks {
            let t = Instant::now();
            let d = model.decode_raw(&features, prompt).expect("decode");
            std::hint::black_box(&d);
            times.push(t.elapsed().as_secs_f64() * 1e3);
        }
        times.remove(0); // warm-up
        let best = times.iter().copied().fold(f64::INFINITY, f64::min);
        println!(
            "decoder, {name}: best {best:.1} ms, median {:.1} ms ({} runs)",
            median(&mut times),
            times.len()
        );
    }
    println!(
        "features {:.1} MB, peak resident memory {:.0} MB, load average after {}",
        features.bytes() as f64 / 1e6,
        peak_rss_mb(),
        load_average()
    );
}
