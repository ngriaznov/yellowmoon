//! Timing of the metadata probe and the preview extraction on a synthetic 50 MP file
//! (`docs/04-raw-decoding.md` section Performance targets: probe under 20 ms, preview under
//! 30 ms).
//!
//! The timed tests are `#[ignore]`d because wall-clock limits do not belong in a normal test run
//! on a shared machine. Run them with
//!
//! ```text
//! cargo test -p ym-raw --release --test perf -- --ignored --nocapture
//! ```
//!
//! The non-ignored test checks that the 50 MP file is handled at all and that the probe stays
//! far from reading the pixel data.

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use ym_raw::metadata::{probe, read_preview};
use ym_raw::testgen::ArwBuilder;

/// 8192 x 6144 is 50.3 MP; the 14-bit uncompressed payload is about 100 MB.
const WIDTH: u32 = 8192;
const HEIGHT: u32 = 6144;

fn write_50mp(dir: &Path) -> PathBuf {
    let path = dir.join("50mp.ARW");
    let mosaic = vec![1024u16; WIDTH as usize * HEIGHT as usize];
    ArwBuilder::new("ILCE-7RM4", WIDTH, HEIGHT)
        // A realistic full-size preview (1616 x 1080, at least 300 kB) next to the thumbnail.
        .preview(Some((1616, 1080)))
        .preview_min_bytes(300_000)
        .write_to(&path, &mosaic)
        .expect("write the 50 MP file");
    path
}

fn median(mut samples: Vec<Duration>) -> Duration {
    samples.sort();
    samples[samples.len() / 2]
}

fn time<T>(runs: usize, mut f: impl FnMut() -> T) -> (Duration, Duration) {
    // One warm-up run, then the median and the slowest of `runs`.
    let _ = f();
    let mut samples = Vec::with_capacity(runs);
    for _ in 0..runs {
        let start = Instant::now();
        let _ = std::hint::black_box(f());
        samples.push(start.elapsed());
    }
    let worst = *samples.iter().max().expect("at least one run");
    (median(samples), worst)
}

#[test]
fn the_probe_of_a_50_megapixel_file_does_not_read_the_pixels() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = write_50mp(dir.path());
    let size = std::fs::metadata(&path).expect("metadata").len();
    assert!(
        size > 100_000_000,
        "the file should be about 100 MB, is {size}"
    );
    let meta = probe(&path).expect("probe");
    assert_eq!((meta.width, meta.height), (WIDTH, HEIGHT));
    // Even a debug build with a cold cache reads a few hundred kilobytes, not 100 MB: a very
    // loose bound that still fails if the probe starts reading the pixel data.
    let (median, _) = time(5, || probe(&path).expect("probe"));
    assert!(median < Duration::from_millis(500), "probe took {median:?}");
}

#[test]
#[ignore = "wall-clock timing; run with --release and --ignored (see the module docs)"]
fn probe_of_a_50_megapixel_file_takes_under_20_ms() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = write_50mp(dir.path());
    let (median, worst) = time(50, || probe(&path).expect("probe"));
    println!("probe 50 MP: median {median:?}, worst {worst:?}");
    assert!(
        median < Duration::from_millis(20),
        "probe median {median:?}"
    );
}

#[test]
#[ignore = "wall-clock timing; run with --release and --ignored (see the module docs)"]
fn preview_extraction_takes_under_30_ms() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = write_50mp(dir.path());
    let jpeg = read_preview(&path).expect("probe").expect("preview");
    println!("preview size: {} bytes", jpeg.len());
    let (median, worst) = time(50, || read_preview(&path).expect("preview"));
    println!("preview 50 MP file: median {median:?}, worst {worst:?}");
    assert!(
        median < Duration::from_millis(30),
        "preview median {median:?}"
    );
}
