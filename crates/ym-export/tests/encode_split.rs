//! Where the time of an export goes on the CPU: the encode split behind the export tail on the device
//! (`docs/09-export.md`, "Where the time goes", `docs/17-gpu.md` section 10).
//!
//! The test is ignored by default (it needs a release build and a quiet machine). Run it with
//!
//! ```text
//! cargo test -p ym-export --release --test encode_split -- --ignored --nocapture
//! ```
//!
//! It times, on a 24 MP (6000 x 4000) display-linear scene of the shared test generator, the stages
//! the device takes over and the ones that stay on the host:
//!
//! * **device stages** (the CPU reference for each): the Lanczos-3 resize (half the long edge), the
//!   output sharpening at 100 %, and stage C13 (the transfer function, the dither and the quantiser)
//!   to 8 and to 16 bit;
//! * **host stages**: what a writer does with rows it already has: the JPEG DCT and Huffman coder,
//!   the PNG deflate, and the TIFF LZW and Zip compressors (8 and 16 bit), and the uncompressed TIFF
//!   (the copy alone).
//!
//! The encoders are timed on rows served from memory (a [`BandSource`] that copies precomputed
//! codes), so the figure is the writer without the C13 conversion; the C13 rows are timed apart.
//! Nothing is asserted but that the files are not empty: the numbers go to the log (and to
//! `docs/09-export.md` by hand).

mod common;

use std::io::Cursor;
use std::time::Instant;

use ym_color::encode::OutputSpace;
use ym_export::exif::ExifTables;
use ym_export::jpeg::{JpegMetadata, write_jpeg_bands};
use ym_export::pixels::{BandSource, ExportHooks, ImageBands, encode_rows_u8, encode_rows_u16};
use ym_export::png_writer::{PngMetadata, write_png_bands};
use ym_export::resize::{resize_lanczos3, sharpen_output};
use ym_export::tiff_writer::{TiffMetadata, write_tiff_bands};
use ym_export::{
    BitDepth, ColorSpace, ExportError, ImageRef, JpegQuality, PixelEncoding, TiffCompression,
};

const W: u32 = 6000;
const H: u32 = 4000;

/// Rows served from memory: the writer without the C13 conversion in front of it.
struct Replay<'a> {
    w: u32,
    h: u32,
    u8s: &'a [u8],
    u16s: &'a [u16],
}

impl BandSource for Replay<'_> {
    fn width(&self) -> u32 {
        self.w
    }

    fn height(&self) -> u32 {
        self.h
    }

    fn rows_u8(&mut self, y0: usize, rows: usize, out: &mut Vec<u8>) -> Result<(), ExportError> {
        let row = self.w as usize * 3;
        out.clear();
        out.extend_from_slice(&self.u8s[y0 * row..(y0 + rows) * row]);
        Ok(())
    }

    fn rows_u16(&mut self, y0: usize, rows: usize, out: &mut Vec<u16>) -> Result<(), ExportError> {
        let row = self.w as usize * 3;
        out.clear();
        out.extend_from_slice(&self.u16s[y0 * row..(y0 + rows) * row]);
        Ok(())
    }
}

/// Runs `f` three times and returns the best time in seconds (the first run warms the caches and
/// the thread pool) and the last result.
fn best_of<T>(mut f: impl FnMut() -> T) -> (f64, T) {
    let mut best = f64::INFINITY;
    let mut last = None;
    for _ in 0..3 {
        let start = Instant::now();
        let value = f();
        best = best.min(start.elapsed().as_secs_f64());
        last = Some(value);
    }
    (best, last.expect("three runs"))
}

fn line(label: &str, seconds: f64, megapixels: f64, extra: &str) {
    eprintln!(
        "{label:<38} {:>8.0} ms {:>8.1} MP/s {extra}",
        seconds * 1e3,
        megapixels / seconds
    );
}

#[test]
#[ignore = "a timing run: cargo test -p ym-export --release --test encode_split -- --ignored --nocapture"]
fn where_the_time_of_an_export_goes() {
    let scene = common::display_scene(W, H);
    let image = ImageRef::new(W, H, &scene, PixelEncoding::DisplayLinear).expect("image");
    let mp = f64::from(W) * f64::from(H) / 1e6;
    let space = OutputSpace::Srgb;
    let hooks = ExportHooks::new(None, None);
    eprintln!(
        "\nencode split on {} threads, {W} x {H} ({mp:.0} MP), display-linear scene of tests/common",
        std::thread::available_parallelism().map_or(1, usize::from)
    );

    // The stages the device takes over.
    let (t, resized) =
        best_of(|| resize_lanczos3(&image, W / 2, H / 2, space, &hooks).expect("resize"));
    line(
        "resize Lanczos-3 to half the long edge",
        t,
        mp,
        &format!("({} x {} out)", resized.width, resized.height),
    );
    let (t, sharpened) = best_of(|| {
        sharpen_output(&image, 100.0, 1.0, space, &hooks)
            .expect("sharpen")
            .expect("a stage")
    });
    line("output sharpening 100 %", t, mp, "");
    assert_eq!(sharpened.rgb.len(), scene.len());

    let mut u8s = Vec::new();
    let (t, ()) = best_of(|| encode_rows_u8(&image, space, 0, H as usize, &mut u8s));
    line("C13 8 bit, dithered, f64", t, mp, "");
    let mut u16s = Vec::new();
    let (t, ()) = best_of(|| encode_rows_u16(&image, space, 0, H as usize, &mut u16s));
    line("C13 16 bit, f64", t, mp, "");
    // Through the BandSource the writers use (band by band, as the writers pull).
    let (t, ()) = best_of(|| {
        let mut bands = ImageBands::new(image, space);
        let (mut buf, rows) = (Vec::new(), 128);
        let mut y = 0;
        while y < H as usize {
            let n = rows.min(H as usize - y);
            bands.rows_u8(y, n, &mut buf).expect("rows");
            y += n;
        }
    });
    line("C13 8 bit through ImageBands, 128 rows", t, mp, "");

    // The stages that stay on the host.
    let exif = ExifTables::default();
    let mut total_lines = Vec::new();
    let mut run = |label: &str, f: &mut dyn FnMut() -> usize| {
        let (t, bytes) = best_of(&mut *f);
        assert!(bytes > 0, "{label}: an empty file");
        line(label, t, mp, &format!("({:.1} MB)", bytes as f64 / 1e6));
        total_lines.push((label.to_owned(), t));
    };
    let replay = |u8s: &'static [u8], u16s: &'static [u16]| Replay {
        w: W,
        h: H,
        u8s,
        u16s,
    };
    // The buffers live for the rest of the process: the bands borrow them.
    let u8s: &'static [u8] = Vec::leak(u8s);
    let u16s: &'static [u16] = Vec::leak(u16s);
    run("JPEG q92 (DCT + Huffman)", &mut || {
        let mut out = Vec::new();
        write_jpeg_bands(
            &mut out,
            &mut replay(u8s, u16s),
            JpegQuality::new(92).expect("quality"),
            &JpegMetadata {
                exif_app1: None,
                xmp: None,
                icc: None,
            },
            &hooks,
        )
        .expect("jpeg");
        out.len()
    });
    for (bits, name) in [(BitDepth::Eight, "8 bit"), (BitDepth::Sixteen, "16 bit")] {
        run(&format!("PNG {name} (deflate)"), &mut || {
            let mut out = Vec::new();
            write_png_bands(
                &mut out,
                &mut replay(u8s, u16s),
                ColorSpace::Srgb,
                bits,
                &PngMetadata {
                    exif: None,
                    xmp: None,
                    icc: None,
                },
                &hooks,
            )
            .expect("png");
            out.len()
        });
        for (compression, cname) in [
            (TiffCompression::None, "uncompressed"),
            (TiffCompression::Lzw, "LZW"),
            (TiffCompression::Zip, "Zip"),
        ] {
            run(&format!("TIFF {name} {cname}"), &mut || {
                let mut out = Cursor::new(Vec::new());
                write_tiff_bands(
                    &mut out,
                    &mut replay(u8s, u16s),
                    bits,
                    compression,
                    &TiffMetadata {
                        exif: &exif,
                        xmp: None,
                        icc: None,
                    },
                    &hooks,
                )
                .expect("tiff");
                out.into_inner().len()
            });
        }
    }
    eprintln!();
}
