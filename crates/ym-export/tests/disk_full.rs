//! A full disk during encoding gives the same advice in every format (`AGENTS.md`: errors say
//! what went wrong and what to do next), and a DNG failure says "DNG", not "TIFF".
//!
//! The encoders are fed a sink that fails with "no space left on device" after a given number of
//! bytes, so every encoder write path is hit: the PNG stream, the TIFF strips and directories, the
//! DNG raw strips and the JPEG body.

mod common;

use std::io::{self, Seek, SeekFrom, Write};

use ym_export::dng::{BaselineExposure, DngProfile};
use ym_export::request::encode_to_writer;
use ym_export::*;

/// A seekable in-memory sink that refuses to grow beyond `limit` bytes.
struct FullDisk {
    data: Vec<u8>,
    pos: usize,
    limit: usize,
}

impl FullDisk {
    fn new(limit: usize) -> Self {
        Self {
            data: Vec::new(),
            pos: 0,
            limit,
        }
    }
}

fn enospc() -> io::Error {
    io::Error::from_raw_os_error(28)
}

impl Write for FullDisk {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        if self.pos + buf.len() > self.limit {
            return Err(enospc());
        }
        if self.data.len() < self.pos + buf.len() {
            self.data.resize(self.pos + buf.len(), 0);
        }
        self.data[self.pos..self.pos + buf.len()].copy_from_slice(buf);
        self.pos += buf.len();
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

impl Seek for FullDisk {
    fn seek(&mut self, from: SeekFrom) -> io::Result<u64> {
        let target = match from {
            SeekFrom::Start(n) => n as i64,
            SeekFrom::Current(n) => self.pos as i64 + n,
            SeekFrom::End(n) => self.data.len() as i64 + n,
        };
        if target < 0 {
            return Err(io::Error::from(io::ErrorKind::InvalidInput));
        }
        self.pos = target as usize;
        Ok(self.pos as u64)
    }
}

fn linear_request<'a>(
    image: &'a [f32],
    w: u32,
    h: u32,
    format: ExportFormat,
    meta: &'a Metadata,
) -> ExportRequest<'a> {
    ExportRequest {
        pixels: ImageRef::new(w, h, image, PixelEncoding::DisplayLinear).unwrap(),
        format,
        color_space: ColorSpace::Srgb,
        size: Size::Full,
        metadata: meta,
        remove_location: false,
        dng: None,
    }
}

/// Encodes with `limit` bytes of room and returns the error text, or `None` when it all fits.
fn failure_text(req: &ExportRequest<'_>, limit: usize) -> Option<String> {
    let mut sink = FullDisk::new(limit);
    match encode_to_writer(req, &mut sink, &ExportHooks::none()) {
        Ok(_) => None,
        Err(e) => Some(e.to_string()),
    }
}

fn assert_advises_freeing_space(text: &str, what: &str) {
    assert!(
        text.contains("free space"),
        "{what}: the message gives no next step: {text}"
    );
    assert!(
        text.contains("No space left on device"),
        "{what}: the message hides the cause: {text}"
    );
}

#[test]
fn png_and_tiff_name_the_next_step_when_the_disk_is_full() {
    let meta = common::metadata();
    let image = common::display_scene(96, 64);
    let formats = [
        (
            "PNG 8",
            ExportFormat::Png {
                bits: BitDepth::Eight,
            },
        ),
        (
            "PNG 16",
            ExportFormat::Png {
                bits: BitDepth::Sixteen,
            },
        ),
        (
            "TIFF",
            ExportFormat::Tiff {
                bits: BitDepth::Sixteen,
                compression: TiffCompression::None,
            },
        ),
        (
            "TIFF lzw",
            ExportFormat::Tiff {
                bits: BitDepth::Eight,
                compression: TiffCompression::Lzw,
            },
        ),
    ];
    for (name, format) in formats {
        let req = linear_request(&image, 96, 64, format, &meta);
        // A sink that is full from the first byte, and one that fills up part way through.
        let whole = {
            let mut sink = FullDisk::new(usize::MAX);
            encode_to_writer(&req, &mut sink, &ExportHooks::none()).unwrap();
            sink.data.len()
        };
        for limit in [0, 40, whole / 3, whole - 1] {
            let text = failure_text(&req, limit)
                .unwrap_or_else(|| panic!("{name}: {limit} of {whole} bytes should not be enough"));
            assert_advises_freeing_space(&text, &format!("{name}, room for {limit} bytes"));
        }
    }
}

#[test]
fn jpeg_names_the_next_step_when_the_disk_is_full() {
    let meta = common::metadata();
    let image = common::display_scene(96, 64);
    let req = linear_request(
        &image,
        96,
        64,
        ExportFormat::Jpeg {
            quality: JpegQuality::default(),
        },
        &meta,
    );
    let text = failure_text(&req, 100).expect("a JPEG does not fit into 100 bytes");
    assert_advises_freeing_space(&text, "JPEG");
}

#[test]
fn a_dng_failure_says_dng_and_names_the_next_step() {
    let meta = common::metadata();
    let (w, h) = (96_u32, 64_u32);
    let raw = common::awg4_scene(w, h);
    let preview = common::display_scene(w, h);
    let profile = DngProfile::identity();
    let req = ExportRequest {
        pixels: ImageRef::new(w, h, &raw, PixelEncoding::SceneLinearAwg4).unwrap(),
        format: ExportFormat::Dng,
        color_space: ColorSpace::Srgb,
        size: Size::Full,
        metadata: &meta,
        remove_location: false,
        dng: Some(DngInputs {
            preview: ImageRef::new(w, h, &preview, PixelEncoding::DisplayLinear).unwrap(),
            orientation: 1,
            crop: None,
            baseline_exposure: BaselineExposure::default(),
            profile: &profile,
        }),
    };
    let whole = {
        let mut sink = FullDisk::new(usize::MAX);
        encode_to_writer(&req, &mut sink, &ExportHooks::none()).unwrap();
        sink.data.len()
    };
    for limit in [0, 60, whole / 4, whole / 2, whole - 1] {
        let text = failure_text(&req, limit)
            .unwrap_or_else(|| panic!("room for {limit} of {whole} bytes should not be enough"));
        assert_advises_freeing_space(&text, &format!("DNG, room for {limit} bytes"));
        assert!(
            !text.contains("TIFF"),
            "a DNG failure must not talk about a TIFF: {text}"
        );
        assert!(text.contains("DNG"), "{text}");
    }
}

#[test]
fn other_encoder_failures_keep_their_own_wording() {
    // An I/O error is not the only thing an encoder can say: a bad value still reads as an encoding
    // failure of that format, not as a disk problem.
    let e = ExportError::encode("PNG", "the XMP packet isn't valid UTF-8");
    let text = e.to_string();
    assert!(text.starts_with("Encoding the PNG failed"), "{text}");
    assert!(!text.contains("free space"), "{text}");
}
