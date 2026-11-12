//! The size of a filmstrip thumbnail's PNG (`docs/26` H-11, finding P3-2): the numbers behind the
//! "about N KB" of `docs/01` §Thumbnails and tiles by address and behind the shell's byte budget.
//!
//! Two sets are measured and printed with `--nocapture`:
//!
//! * synthetic scenes of 960 x 640 pixels (so the thumbnail is the full 240 x 160), flat to noisy;
//! * the owner's photographs in `testdata/readme-photos/src` (`tools/fetch/owner-photos.sh`), when
//!   they are there. CI has them; a machine without them skips that part and says so.
//!
//! ```text
//! YELLOWMOON_DATA=data cargo test -p ym-core --release --test thumbnail_sizes -- --nocapture
//! ```
//!
//! (`YELLOWMOON_DATA` because a release build does not find the data folder by itself; a debug
//! build is too slow for the 100-megapixel files.)
//!
//! Asserts only what must hold on any machine: every thumbnail is a PNG, at most 240 px on the long
//! edge, and under [`CEILING_BYTES`], the largest a 240 x 160 RGB PNG can reasonably be. The shell's
//! budget (`THUMBNAIL_BUDGET_BYTES`) is chosen from the printed numbers.
mod common;

use std::path::{Path, PathBuf};
use std::time::Duration;

use common::*;
use ym_core::pipeline::{PhotoSource, Renderer};
use ym_core::session::{OpenOptions, Session};
use ym_gpu::queue::CancelToken;

/// No thumbnail of 240 x 160 pixels, even of white noise, is larger than its raw size (115,200
/// bytes) plus the PNG framing; this is a little above it.
const CEILING_BYTES: usize = 120_000;

fn owner_photos() -> Vec<PathBuf> {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata/readme-photos/src");
    let mut files: Vec<PathBuf> = std::fs::read_dir(dir)
        .map(|d| {
            d.filter_map(Result::ok)
                .map(|e| e.path())
                .filter(|p| p.extension().is_some_and(|x| x.eq_ignore_ascii_case("arw")))
                .collect()
        })
        .unwrap_or_default();
    files.sort();
    files
}

/// Develops one file and returns its thumbnail's PNG.
fn thumbnail_of(renderer: &Renderer, path: &Path) -> Vec<u8> {
    let session = Session::new(base_env());
    let out = session.open(&[path.to_path_buf()], OpenOptions::default());
    assert!(
        session.wait_idle(Duration::from_secs(300)),
        "{path:?} developed"
    );
    let src = PhotoSource::from_session(&session, out.added[0]).expect("source");
    renderer
        .thumbnail_png(&src, &CancelToken::new())
        .expect("thumbnail")
}

/// Width and height from the PNG header.
fn png_size(png: &[u8]) -> (u32, u32) {
    assert_eq!(&png[1..4], b"PNG");
    let be = |at: usize| u32::from_be_bytes([png[at], png[at + 1], png[at + 2], png[at + 3]]);
    (be(16), be(20))
}

fn summary(label: &str, sizes: &mut [usize]) {
    sizes.sort_unstable();
    let n = sizes.len();
    let mean = sizes.iter().sum::<usize>() / n.max(1);
    eprintln!(
        "{label}: {n} thumbnails, min {} B, median {} B, mean {mean} B, max {} B",
        sizes[0],
        sizes[n / 2],
        sizes[n - 1]
    );
}

#[test]
fn thumbnail_pngs_are_measured() {
    let renderer = Renderer::neutral_only().expect("render thread");
    let mut synthetic = Vec::new();
    for (name, layout, a, b) in [
        ("flat", Layout::Uniform, 0.0, 0.0),
        ("ramp", Layout::Ramp, 0.0, 0.0),
        ("ramp, noise", Layout::Ramp, 2e-4, 1e-5),
        ("patchwork", Layout::Patchwork, 0.0, 0.0),
        ("patchwork, noise", Layout::Patchwork, 2e-4, 1e-5),
        ("patchwork, heavy noise", Layout::Patchwork, 2e-3, 1e-4),
    ] {
        let scene = Scene::new(960, 640, light("sony-ilce-7rm4", 5200.0), layout, 0.4)
            .noise(a, b)
            .seed(3);
        let dir = tempfile::tempdir().expect("dir");
        let path = scene.write(dir.path(), "DSC00001.ARW", "ILCE-7RM4");
        let png = thumbnail_of(&renderer, &path);
        let (w, h) = png_size(&png);
        eprintln!("synthetic {name:<24} {w} x {h}: {} B", png.len());
        assert!(w.max(h) <= 240 && png.len() < CEILING_BYTES);
        synthetic.push(png.len());
    }
    summary("synthetic", &mut synthetic);

    let files = owner_photos();
    if files.is_empty() {
        eprintln!("owner photos: testdata/readme-photos/src is absent, skipped");
        return;
    }
    let mut real = Vec::new();
    for path in &files {
        let png = thumbnail_of(&renderer, path);
        let (w, h) = png_size(&png);
        eprintln!(
            "owner photo {:<14} {w} x {h}: {} B",
            path.file_name()
                .map_or_else(String::new, |n| n.to_string_lossy().into_owned()),
            png.len()
        );
        assert!(w.max(h) <= 240 && png.len() < CEILING_BYTES);
        real.push(png.len());
    }
    summary("owner photos", &mut real);
}
