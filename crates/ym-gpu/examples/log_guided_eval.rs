//! RA-06 evaluation of the log-guided demosaic (`docs/algorithms/02-log-guided-demosaic.md`
//! section 6.4): timings against bilinear, Menon and LMMSE, and PSNR on the synthetic set and on
//! Kodak and McMaster.
//!
//! ```text
//! cargo run --release -p ym-gpu --example log_guided_eval -- timing   # 1/4, 1/2 and full resolution
//! cargo run --release -p ym-gpu --example log_guided_eval -- timing quarter  # the 1/4 cases only
//! cargo run --release -p ym-gpu --example log_guided_eval -- psnr     # synthetic set, Kodak, McMaster
//! RAYON_NUM_THREADS=1 cargo run --release -p ym-gpu --example log_guided_eval -- timing
//! ```
//!
//! Timings are best-of-N wall time and CPU seconds of the process (from `/proc/self/stat`, Linux
//! only; the dependable number on a shared machine). PSNR follows `docs/05` R6: sRGB images or
//! procedural scenes taken as linear light, mosaicked as RGGB, a 10 px border excluded, at signal
//! scales 1, 0.1 and 0.01 with the peak scaled along (so a scale-invariant method gives one
//! number). The log-guided method runs with a zero-noise model, so `p = 1` (spec section 6.4).

use std::hint::black_box;
use std::path::{Path, PathBuf};
use std::time::Instant;

use ym_gpu::cpu::demosaic::log_guided::{self, LogGuidedParams};
use ym_gpu::cpu::demosaic::{self, Method};
use ym_gpu::cpu::filters::{self, BAND_ROWS};
use ym_gpu::cpu::metrics::psnr;
use ym_gpu::cpu::synth;
use ym_gpu::cpu::{Cfa, Mosaic, NoiseDn, RgbImage};

/// The methods compared: the three classical ones and the log-guided float form.
const METHODS: [Method; 4] = [
    Method::Bilinear,
    Method::Menon,
    Method::Lmmse,
    Method::LogGuided,
];

fn main() {
    let mode = std::env::args().nth(1).unwrap_or_else(|| "timing".into());
    match mode.as_str() {
        "timing" => timing(std::env::args().nth(2).as_deref() == Some("quarter")),
        "psnr" => psnr_tables(),
        other => {
            eprintln!("usage: log_guided_eval timing|psnr (got {other})");
            std::process::exit(2);
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Timing
// ---------------------------------------------------------------------------------------------

/// User plus system CPU seconds of this process (all threads), 100 ticks per second.
fn process_cpu_seconds() -> f64 {
    let stat = std::fs::read_to_string("/proc/self/stat").unwrap_or_default();
    let after = stat.rsplit_once(')').map_or("", |(_, rest)| rest);
    let fields: Vec<&str> = after.split_whitespace().collect();
    let ticks = |i: usize| {
        fields
            .get(i)
            .and_then(|f| f.parse::<f64>().ok())
            .unwrap_or(0.0)
    };
    (ticks(11) + ticks(12)) / 100.0
}

struct Timing {
    wall_ms: f64,
    cpu_ms: f64,
}

/// Runs `f` once to warm up, then repeatedly for at least `min_seconds` (and at least 3 times):
/// the best wall time and the mean CPU time per run.
fn measure(min_seconds: f64, mut f: impl FnMut() -> RgbImage) -> Timing {
    black_box(f());
    let (mut best, mut runs) = (f64::INFINITY, 0u32);
    let cpu0 = process_cpu_seconds();
    let start = Instant::now();
    while runs < 3 || start.elapsed().as_secs_f64() < min_seconds {
        let t = Instant::now();
        black_box(f());
        best = best.min(t.elapsed().as_secs_f64());
        runs += 1;
    }
    Timing {
        wall_ms: best * 1e3,
        cpu_ms: (process_cpu_seconds() - cpu0) * 1e3 / f64::from(runs),
    }
}

fn timing(quarter_only: bool) {
    let threads = rayon::current_num_threads();
    println!("rayon threads: {threads}");
    // (label, width, height): 24 MP is 6000 x 4000 and 61 MP is 9504 x 6336.
    let cases = [
        ("24 MP at 1/4 resolution", 1500, 1000),
        ("61 MP at 1/4 resolution", 2376, 1584),
        ("24 MP at 1/2 resolution", 3000, 2000),
        ("61 MP at 1/2 resolution", 4752, 3168),
        ("24 MP at full resolution", 6000, 4000),
    ];
    let count = if quarter_only { 2 } else { cases.len() };
    for (label, w, h) in cases.into_iter().take(count) {
        let m = synth::noise_mosaic(w, h, Cfa::RGGB, 3);
        let mp = (w * h) as f64 / 1e6;
        println!("\n== {label}: {w} x {h} ({mp:.2} MP)");
        println!(
            "{:<22} {:>10} {:>10} {:>9} {:>9} {:>9}",
            "method", "wall ms", "cpu ms", "MP/s", "x bilinear", "x menon"
        );
        let mut rows: Vec<(String, Timing)> = Vec::new();
        for method in METHODS {
            // Menon and LMMSE at full resolution of 24 MP need a few GB of planes: fine here.
            let t = measure(1.0, || demosaic::demosaic(&m, method));
            rows.push((method.name().to_owned(), t));
        }
        // The fixed-point form through the same banded runner, with the frame's headroom.
        let params = LogGuidedParams::for_mosaic(&m);
        let t = measure(1.0, || {
            filters::run_mosaic_bands(&m, log_guided::HALO_PX, BAND_ROWS, |band| {
                log_guided::demosaic_fixed(
                    band,
                    NoiseDn::ZERO,
                    params.guide_detail,
                    params.headroom_log2,
                )
            })
        });
        rows.push(("log_guided (fixed)".to_owned(), t));
        let bilinear = rows[0].1.cpu_ms;
        let menon = rows[1].1.cpu_ms;
        for (name, t) in &rows {
            println!(
                "{name:<22} {:>10.1} {:>10.1} {:>9.1} {:>9.2} {:>9.2}",
                t.wall_ms,
                t.cpu_ms,
                mp / (t.wall_ms / 1e3),
                t.cpu_ms / bilinear,
                t.cpu_ms / menon,
            );
        }
    }
}

// ---------------------------------------------------------------------------------------------
// PSNR
// ---------------------------------------------------------------------------------------------

fn load_png(path: &Path) -> RgbImage {
    let file = std::fs::File::open(path).expect("open png");
    let mut reader = png::Decoder::new(std::io::BufReader::new(file))
        .read_info()
        .expect("png info");
    let mut buf = vec![0; reader.output_buffer_size().expect("png size")];
    let info = reader.next_frame(&mut buf).expect("png frame");
    assert_eq!(info.bit_depth, png::BitDepth::Eight);
    let channels = info.color_type.samples();
    let rgb: Vec<f32> = buf[..info.width as usize * info.height as usize * channels]
        .chunks(channels)
        .flat_map(|p| {
            if channels >= 3 {
                [p[0], p[1], p[2]]
            } else {
                [p[0], p[0], p[0]]
            }
        })
        .map(|v| f32::from(v) / 255.0)
        .collect();
    RgbImage::from_interleaved(info.width as usize, info.height as usize, &rgb).expect("image")
}

fn load_tiff(path: &Path) -> RgbImage {
    let file = std::fs::File::open(path).expect("open tiff");
    let mut decoder =
        tiff::decoder::Decoder::new(std::io::BufReader::new(file)).expect("tiff decoder");
    let (w, h) = decoder.dimensions().expect("tiff dims");
    let data = match decoder.read_image().expect("tiff data") {
        tiff::decoder::DecodingResult::U8(v) => v,
        other => panic!("unexpected tiff sample type {other:?}"),
    };
    let rgb: Vec<f32> = data.iter().map(|&v| f32::from(v) / 255.0).collect();
    RgbImage::from_interleaved(w as usize, h as usize, &rgb).expect("image")
}

fn load_set(dir: &Path) -> Vec<(String, RgbImage)> {
    let mut paths: Vec<PathBuf> = match std::fs::read_dir(dir) {
        Ok(rd) => rd.filter_map(|e| e.ok().map(|e| e.path())).collect(),
        Err(_) => return Vec::new(),
    };
    paths.sort();
    paths
        .into_iter()
        .filter_map(|p| {
            let name = p.file_name()?.to_string_lossy().to_string();
            match p.extension()?.to_str()? {
                "png" => Some((name, load_png(&p))),
                "tif" | "tiff" => Some((name, load_tiff(&p))),
                _ => None,
            }
        })
        .collect()
}

/// `img` times `scale`, mosaicked as RGGB, demosaiced by each of [`METHODS`], PSNR with the peak
/// scaled along and a 10 px border excluded.
fn psnr_row(img: &RgbImage, scale: f32) -> Vec<f64> {
    let mut truth = img.clone();
    for v in &mut truth.data {
        *v *= scale;
    }
    let m: Mosaic = synth::mosaic_from_rgb(&truth, Cfa::RGGB);
    METHODS
        .iter()
        .map(|&method| {
            psnr(
                &demosaic::demosaic(&m, method),
                &truth,
                10,
                f64::from(scale),
            )
        })
        .collect()
}

fn header() -> String {
    let mut line = format!("{:<18}", "image");
    for method in METHODS {
        line.push_str(&format!(" {:>10}", method.name()));
    }
    line.push_str(&format!(" {:>10}", "lg - menon"));
    line
}

fn line(name: &str, row: &[f64]) -> String {
    let mut s = format!("{name:<18}");
    for v in row {
        s.push_str(&format!(" {v:>10.2}"));
    }
    s.push_str(&format!(" {:>10.2}", row[3] - row[1]));
    s
}

fn table(title: &str, images: &[(String, RgbImage)], scales: &[f32], per_image: bool) {
    println!("\n== {title} ({} images)", images.len());
    for &scale in scales {
        println!("-- signal scale {scale}");
        println!("{}", header());
        let mut mean = vec![0.0; METHODS.len()];
        for (name, img) in images {
            let row = psnr_row(img, scale);
            if per_image {
                println!("{}", line(name, &row));
            }
            for (m, v) in mean.iter_mut().zip(&row) {
                *m += v / images.len() as f64;
            }
        }
        println!("{}", line("mean", &mean));
    }
}

fn psnr_tables() {
    let synthetic: Vec<(String, RgbImage)> = vec![
        ("zone plate".to_owned(), synth::zone_plate(192, 192)),
        ("edges".to_owned(), synth::edges(192, 192)),
        ("colour patches".to_owned(), synth::color_patches(192, 192)),
        ("fine text".to_owned(), synth::fine_text(192, 192, 1)),
        ("natural 1".to_owned(), synth::natural(192, 192, 1)),
        ("natural 2".to_owned(), synth::natural(192, 192, 2)),
        ("natural 3".to_owned(), synth::natural(192, 192, 3)),
    ];
    table(
        "synthetic set of tests/cpu_demosaic.rs",
        &synthetic,
        &[1.0, 0.1, 0.01],
        true,
    );
    let root = std::env::var_os("YM_TESTDATA")
        .map(PathBuf::from)
        .unwrap_or_else(|| Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata"))
        .join("demosaic");
    for set in ["kodak", "mcmaster"] {
        let images = load_set(&root.join(set));
        if images.is_empty() {
            println!(
                "\n== {set}: no images in {} (skipped)",
                root.join(set).display()
            );
            continue;
        }
        table(set, &images, &[1.0, 0.1, 0.01], false);
    }
}
