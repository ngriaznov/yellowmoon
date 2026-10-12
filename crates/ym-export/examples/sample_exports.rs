//! Writes one sample of every format and colour space from a deterministic synthetic scene, so
//! the files can be opened in other tools:
//! `cargo run -p ym-export --example sample_exports -- <output folder> [width height]`.

#[path = "../tests/common/mod.rs"]
mod common;

use std::path::PathBuf;

use ym_export::dng::{BaselineExposure, DngProfile};
use ym_export::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let dir = PathBuf::from(
        args.next()
            .ok_or("usage: sample_exports <folder> [width height]")?,
    );
    let w: u32 = args.next().map_or(Ok(320), |s| s.parse())?;
    let h: u32 = args.next().map_or(Ok(240), |s| s.parse())?;
    std::fs::create_dir_all(&dir)?;
    let meta = common::metadata();
    let display = common::display_scene(w, h);
    let awg4 = common::awg4_scene(w, h);

    let formats = [
        ExportFormat::Jpeg {
            quality: JpegQuality::new(92)?,
        },
        ExportFormat::Jpeg {
            quality: JpegQuality::new(75)?,
        },
        ExportFormat::Tiff {
            bits: BitDepth::Sixteen,
            compression: TiffCompression::Lzw,
        },
        ExportFormat::Tiff {
            bits: BitDepth::Eight,
            compression: TiffCompression::Zip,
        },
        ExportFormat::Png {
            bits: BitDepth::Sixteen,
        },
        ExportFormat::Png {
            bits: BitDepth::Eight,
        },
    ];
    for (n, format) in formats.into_iter().enumerate() {
        for space in ColorSpace::ALL {
            let pixels = ImageRef::new(w, h, &display, PixelEncoding::DisplayLinear)?;
            let req = ExportRequest {
                pixels,
                format,
                color_space: space,
                size: Size::Full,
                metadata: &meta,
                remove_location: false,
                dng: None,
            };
            let dest = ExportDestination {
                dir: dir.clone(),
                file_stem: format!("sample{n}_{space:?}"),
                if_exists: IfExists::Replace,
                source_path: None,
            };
            let out = write_export(&req, &dest, &ExportHooks::none())?;
            println!("{}", out.path().display());
        }
    }
    let raw = ImageRef::new(w, h, &awg4, PixelEncoding::SceneLinearAwg4)?;
    let preview = ImageRef::new(w, h, &display, PixelEncoding::DisplayLinear)?;
    let profile = DngProfile::identity();
    let req = ExportRequest {
        pixels: raw,
        format: ExportFormat::Dng,
        color_space: ColorSpace::Srgb,
        size: Size::Full,
        metadata: &meta,
        remove_location: false,
        dng: Some(DngInputs {
            preview,
            orientation: 1,
            crop: None,
            baseline_exposure: BaselineExposure::default(),
            profile: &profile,
        }),
    };
    let dest = ExportDestination {
        dir: dir.clone(),
        file_stem: "sample_dng".into(),
        if_exists: IfExists::Replace,
        source_path: None,
    };
    let out = write_export(&req, &dest, &ExportHooks::none())?;
    println!("{} {:?}", out.path().display(), out.dng);
    Ok(())
}
