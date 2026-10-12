//! Times the decoders on 50 MP synthetic files with different thread counts.
//!
//! ```text
//! cargo run -p ym-raw --features testgen --release --example decode_bench -- /tmp/bench
//! ```
//!
//! The first run writes one 8192 x 6144 file per variant into the folder (it takes a while: the
//! test encoders are single-threaded), later runs reuse them. Each decode reads the file from
//! disk (warm page cache) and is repeated; the best and the median wall time are printed, for 1, 2
//! and 4 rayon threads, and the process CPU time of the best-looking run (user plus system, from
//! `/proc/self/stat`, Linux only): on a machine that is busy with other jobs the CPU seconds are
//! the dependable number. The numbers are for this machine; the targets in `docs/04` are for an
//! Apple M1.

use std::path::{Path, PathBuf};
use std::time::Instant;

use ym_raw::decode::decode;
use ym_raw::layout::ChromaSubsampling;
use ym_raw::testgen::{
    ArwBuilder, CrawEncoder, LosslessL, LosslessMs, Packed12, gradient_mosaic, gradient_rgb,
};

/// Process CPU seconds (user plus system) so far, or 0 where `/proc` is not available.
fn cpu_seconds() -> f64 {
    let Ok(stat) = std::fs::read_to_string("/proc/self/stat") else {
        return 0.0;
    };
    // The command name may contain spaces; the fields after the closing parenthesis are fixed.
    let after = stat.rsplit_once(')').map_or("", |(_, rest)| rest);
    let fields: Vec<&str> = after.split_whitespace().collect();
    let ticks = |i: usize| {
        fields
            .get(i)
            .and_then(|v| v.parse::<f64>().ok())
            .unwrap_or(0.0)
    };
    // utime and stime are fields 14 and 15 of the line, 11 and 12 after the name.
    (ticks(11) + ticks(12)) / 100.0
}

const W: u32 = 8192;
const H: u32 = 6144;

fn ensure(dir: &Path, name: &str, make: impl FnOnce() -> Vec<u8>) -> PathBuf {
    let path = dir.join(name);
    if !path.exists() {
        let started = Instant::now();
        std::fs::write(&path, make()).expect("write bench file");
        eprintln!(
            "wrote {} in {:.1} s",
            path.display(),
            started.elapsed().as_secs_f64()
        );
    }
    path
}

fn main() {
    let dir = PathBuf::from(
        std::env::args()
            .nth(1)
            .unwrap_or_else(|| "/tmp/ym-bench".into()),
    );
    std::fs::create_dir_all(&dir).expect("create the folder");
    let mosaic14 = gradient_mosaic(W, H, 512, 16000, 1);
    let mosaic12 = gradient_mosaic(W, H, 128, 4000, 2);
    let encoder = CrawEncoder::new();
    let files: Vec<(&str, PathBuf)> = vec![
        (
            "uncompressed 14-bit",
            ensure(&dir, "unc14.ARW", || {
                ArwBuilder::new("ILCE-7RM4", W, H).build(&mosaic14)
            }),
        ),
        (
            "packed 12-bit",
            ensure(&dir, "packed12.ARW", || {
                ArwBuilder::new("DSLR-A900", W, H)
                    .payload(Packed12)
                    .build(&mosaic12)
            }),
        ),
        (
            "cRAW (with qstep)",
            ensure(&dir, "craw.ARW", || {
                let m = gradient_mosaic(W, H, 512, encoder.max_value(), 3);
                ArwBuilder::new("ILCE-7M3", W, H)
                    .craw(encoder.clone())
                    .build(&m)
            }),
        ),
        (
            "lossless L",
            ensure(&dir, "lossless_l.ARW", || {
                ArwBuilder::new("ILCE-1", W, H)
                    .payload(LosslessL::new())
                    .build(&mosaic14)
            }),
        ),
        (
            "lossless M/S 4:2:0",
            ensure(&dir, "lossless_ms.ARW", || {
                let rgb = gradient_rgb(W, H, 1100, 9000, 4);
                let payload =
                    LosslessMs::new(ChromaSubsampling::Yuv420, rgb, W, H, [2.0, 1.0, 1.6]);
                ArwBuilder::new("ILCE-7RM5", W, H)
                    .lossless_ms(payload)
                    .build(&[])
            }),
        ),
    ];
    println!("| variant | file MB | 1 thread: wall ms (median), CPU s | 2 threads | 4 threads |");
    println!("|---|---|---|---|---|");
    for (name, path) in &files {
        let size = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0) as f64 / 1e6;
        let mut cells = Vec::new();
        for threads in [1usize, 2, 4] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .expect("thread pool");
            let mut runs: Vec<(f64, f64)> = (0..5)
                .map(|_| {
                    let cpu_before = cpu_seconds();
                    let started = Instant::now();
                    let image = pool.install(|| decode(path)).expect("decode");
                    std::hint::black_box(&image);
                    (
                        started.elapsed().as_secs_f64() * 1000.0,
                        cpu_seconds() - cpu_before,
                    )
                })
                .collect();
            runs.sort_by(|a, b| a.0.total_cmp(&b.0));
            cells.push(format!(
                "{:.0} (median {:.0}), {:.2} s CPU",
                runs[0].0, runs[2].0, runs[0].1
            ));
        }
        println!(
            "| {name} | {size:.1} | {} | {} | {} |",
            cells[0], cells[1], cells[2]
        );
    }
}
