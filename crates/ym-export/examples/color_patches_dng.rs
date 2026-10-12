//! Writes a DNG made of flat scene-linear AWG4 patches, for rendering with Adobe's `dng_validate`.
//! `cargo run -p ym-export --example color_patches_dng -- <out.dng>`

#[path = "../tests/common/mod.rs"]
mod common;

use ym_export::dng::{BaselineExposure, DngProfile};
use ym_export::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let path = args
        .next()
        .ok_or("usage: color_patches_dng <out.dng> [look encoding 0|1] [value divisions]")?;
    let encoding: u32 = args.next().map_or(Ok(1), |s| s.parse())?;
    let value_divisions: u32 = args.next().map_or(Ok(2), |s| s.parse())?;
    // 4 x 4 patches of 8 x 8 pixels; the largest value is below 1 so x_white stays 1 and BaselineExposure is 0.
    let patches: [[f32; 3]; 16] = [
        [0.18, 0.18, 0.18],
        [0.5, 0.5, 0.5],
        [0.02, 0.02, 0.02],
        [0.9, 0.9, 0.9],
        [0.4, 0.1, 0.05],
        [0.05, 0.4, 0.1],
        [0.1, 0.05, 0.4],
        [0.3, 0.3, 0.05],
        [0.05, 0.3, 0.3],
        [0.3, 0.05, 0.3],
        [0.6, 0.3, 0.15],
        [0.15, 0.3, 0.6],
        [0.25, 0.2, 0.1],
        [0.1, 0.2, 0.25],
        [0.7, 0.5, 0.4],
        [0.4, 0.5, 0.7],
    ];
    let (w, h) = (32_u32, 32_u32);
    let mut raw = Vec::new();
    for y in 0..h {
        for x in 0..w {
            raw.extend_from_slice(&patches[((y / 8) * 4 + x / 8) as usize]);
        }
    }
    let preview = common::display_scene(w, h);
    let meta = Metadata::default();
    let mut profile = DngProfile::identity();
    let entries = (2 * 2 * value_divisions) as usize;
    profile.look_table = ym_export::dng::LookTable {
        dims: [2, 2, value_divisions],
        data: (0..entries).flat_map(|_| [0.0, 1.0, 1.0]).collect(),
        encoding,
    };
    let req = ExportRequest {
        pixels: ImageRef::new(w, h, &raw, PixelEncoding::SceneLinearAwg4)?,
        format: ExportFormat::Dng,
        color_space: ColorSpace::Srgb,
        size: Size::Full,
        metadata: &meta,
        remove_location: false,
        dng: Some(DngInputs {
            preview: ImageRef::new(w, h, &preview, PixelEncoding::DisplayLinear)?,
            orientation: 1,
            crop: None,
            baseline_exposure: BaselineExposure::Fixed(0.0),
            profile: &profile,
        }),
    };
    let (bytes, summary) = encode_to_vec(&req, &ExportHooks::none())?;
    std::fs::write(&path, bytes)?;
    println!("{path} {:?}", summary.dng);
    for p in patches {
        println!("{} {} {}", p[0], p[1], p[2]);
    }
    Ok(())
}
