//! Times every format on a large synthetic frame (default 6000 x 4000):
//! `cargo run --release -p ym-export --example time_exports -- [width height]`.

#[path = "../tests/common/mod.rs"]
mod common;

use std::time::Instant;

use ym_export::dng::{BaselineExposure, DngProfile};
use ym_export::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let w: u32 = args.next().map_or(Ok(6000), |s| s.parse())?;
    let h: u32 = args.next().map_or(Ok(4000), |s| s.parse())?;
    let display = common::display_scene(w, h);
    let awg4 = common::awg4_scene(w, h);
    let meta = common::metadata();
    println!("{w} x {h} ({:.1} MP)", f64::from(w) * f64::from(h) / 1e6);
    let cases: Vec<(&str, ExportFormat, Size)> = vec![
        (
            "jpeg q92",
            ExportFormat::Jpeg {
                quality: JpegQuality::new(92)?,
            },
            Size::Full,
        ),
        (
            "jpeg q75",
            ExportFormat::Jpeg {
                quality: JpegQuality::new(75)?,
            },
            Size::Full,
        ),
        (
            "jpeg q92, long edge 3000",
            ExportFormat::Jpeg {
                quality: JpegQuality::new(92)?,
            },
            Size::LongEdge { px: 3000 },
        ),
        (
            "tiff 16 none",
            ExportFormat::Tiff {
                bits: BitDepth::Sixteen,
                compression: TiffCompression::None,
            },
            Size::Full,
        ),
        (
            "tiff 16 lzw",
            ExportFormat::Tiff {
                bits: BitDepth::Sixteen,
                compression: TiffCompression::Lzw,
            },
            Size::Full,
        ),
        (
            "tiff 16 zip",
            ExportFormat::Tiff {
                bits: BitDepth::Sixteen,
                compression: TiffCompression::Zip,
            },
            Size::Full,
        ),
        (
            "tiff 8 zip",
            ExportFormat::Tiff {
                bits: BitDepth::Eight,
                compression: TiffCompression::Zip,
            },
            Size::Full,
        ),
        (
            "png 16",
            ExportFormat::Png {
                bits: BitDepth::Sixteen,
            },
            Size::Full,
        ),
        (
            "png 8",
            ExportFormat::Png {
                bits: BitDepth::Eight,
            },
            Size::Full,
        ),
    ];
    for (name, format, size) in cases {
        let req = ExportRequest {
            pixels: ImageRef::new(w, h, &display, PixelEncoding::DisplayLinear)?,
            format,
            color_space: ColorSpace::DisplayP3,
            size,
            metadata: &meta,
            remove_location: false,
            dng: None,
        };
        let start = Instant::now();
        let (bytes, _) = encode_to_vec(&req, &ExportHooks::none())?;
        let est = estimate_bytes(w, h, format, size);
        println!(
            "{name:<26} {:>6.2} s  {:>7.1} MB (estimate {:>7.1} MB)",
            start.elapsed().as_secs_f64(),
            bytes.len() as f64 / 1e6,
            est as f64 / 1e6
        );
    }
    let profile = DngProfile::identity();
    let req = ExportRequest {
        pixels: ImageRef::new(w, h, &awg4, PixelEncoding::SceneLinearAwg4)?,
        format: ExportFormat::Dng,
        color_space: ColorSpace::Srgb,
        size: Size::Full,
        metadata: &meta,
        remove_location: false,
        dng: Some(DngInputs {
            preview: ImageRef::new(w, h, &display, PixelEncoding::DisplayLinear)?,
            orientation: 1,
            crop: None,
            baseline_exposure: BaselineExposure::default(),
            profile: &profile,
        }),
    };
    let start = Instant::now();
    let (bytes, summary) = encode_to_vec(&req, &ExportHooks::none())?;
    println!(
        "{:<26} {:>6.2} s  {:>7.1} MB (estimate {:>7.1} MB)  x_white {}",
        "dng",
        start.elapsed().as_secs_f64(),
        bytes.len() as f64 / 1e6,
        estimate_bytes(w, h, ExportFormat::Dng, Size::Full) as f64 / 1e6,
        summary.dng.map_or(0.0, |d| d.x_white)
    );
    Ok(())
}
