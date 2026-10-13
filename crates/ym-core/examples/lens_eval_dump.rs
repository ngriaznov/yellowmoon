//! Dumps the camera-RGB images of the raw stage group at the points the real-file lens evaluation
//! needs (`tools/lens/real_eval.py`, `docs/05-raw-processing.md` section R9): R1 and R4, the
//! whole-frame demosaic of the chosen method with no lens correction, and R9 on top of it with
//! and without the CA correction. A developer tool for local evaluation (real files, never
//! committed); not part of the product.
//!
//! ```text
//! cargo run -p ym-core --example lens_eval_dump -- <file.ARW> <out-dir> [adaptive|menon|lmmse|bilinear] [--ca-scale A]
//! ```
//!
//! Writes `<out-dir>/<stem>.<method>.{demosaic,r9,r9noca}.f32`: `u32` width, `u32` height, then the
//! red, green and blue planes as little-endian `f32`. The balance `g0` is the as-shot one (SR2
//! 0x7313) and the noise model a flat guess: only the CA the stages leave matters here. With
//! `--ca-scale A` every CA knot is multiplied by `A` before R9 (a sensitivity check of the table's
//! amplitude).

use std::io::Write;
use std::path::PathBuf;
use std::sync::Arc;

use ym_core::pipeline::raw::{FrameData, RawParams, RawPlan};
use ym_gpu::cpu::demosaic::{self, DemosaicInputs, Method};
use ym_gpu::cpu::detail::NoiseReduction;
use ym_gpu::cpu::lens::{LensGeometry, LensSwitches, LensTags};
use ym_gpu::cpu::sharpen::SharpenParams;
use ym_gpu::cpu::{NoiseDn, Rect, RgbImage, gains, normalize, resample};
use ym_noise::model::NoiseModel;

fn write_f32(path: &PathBuf, img: &RgbImage) {
    let mut f = std::io::BufWriter::new(std::fs::File::create(path).expect("create output"));
    f.write_all(&(img.width as u32).to_le_bytes()).expect("write");
    f.write_all(&(img.height as u32).to_le_bytes()).expect("write");
    f.write_all(bytemuck::cast_slice(&img.data)).expect("write");
}

fn main() {
    let mut args = std::env::args().skip(1);
    let file = PathBuf::from(args.next().expect("a .ARW file"));
    let out_dir = PathBuf::from(args.next().expect("an output directory"));
    let mut method_name = "adaptive".to_owned();
    let mut ca_scale = 1.0f64;
    let rest: Vec<String> = args.collect();
    let mut i = 0;
    while i < rest.len() {
        if rest[i] == "--ca-scale" {
            ca_scale = rest[i + 1].parse().expect("a number");
            i += 2;
        } else {
            method_name.clone_from(&rest[i]);
            i += 1;
        }
    }
    let method = match method_name.as_str() {
        "adaptive" => Method::AdaptiveRatio,
        other => Method::from_name(other).expect("a demosaic method"),
    };
    std::fs::create_dir_all(&out_dir).expect("output directory");
    let stem = file.file_stem().unwrap_or_default().to_string_lossy();

    let raw = Arc::new(ym_raw::decode::decode(&file).expect("decode"));
    let wb = raw.wb_as_shot.expect("as-shot white balance");
    let g0 = gains::g0_from_wb_levels(wb).expect("gains");
    let noise = NoiseModel {
        a: 2.0e-5,
        b: 1.0e-8,
        delta: [0.0; 4],
        sigma_row: 0.0,
        q: 1.0 / 65535.0,
    };
    let params = RawParams {
        g0,
        neural_demosaic: false,
        highlight_recovery: false,
        lens_corrections: true,
        chromatic_aberration: true,
        noise_reduction: NoiseReduction {
            luminance: 0.0,
            color: 0.0,
            detail: 0.0,
        },
        sharpen: SharpenParams {
            amount: 0.0,
            radius_px: 0.8,
        },
        iso: 100.0,
        noise,
        diffraction: None,
    };
    let frame = FrameData::new(&raw).expect("frame");
    let plan = RawPlan::new(&raw, &frame, &params).expect("plan");
    let (fw, fh) = (frame.width, frame.height);

    let (mut m, _) = normalize::normalize(frame.data(), fw, fh, frame.cfa, &plan.levels, None)
        .expect("normalize");
    gains::apply_gains(&mut m, g0);
    let img = demosaic::demosaic_whole_with(
        &m,
        method,
        &DemosaicInputs {
            gains: g0,
            noise: NoiseDn::from_model(&params.noise),
            ca: None,
        },
    );
    // Undo the gains so that every plane has the scale of the scene again (the demosaic output of
    // the chain is balanced; the measurement only needs a common scale per plane).
    write_f32(&out_dir.join(format!("{stem}.{method_name}.demosaic.f32")), &img);

    let tags = LensTags {
        ca: plan
            .tags
            .ca
            .iter()
            .map(|&c| (f64::from(c) * ca_scale).round() as i16)
            .collect(),
        ..plan.tags.clone()
    };
    let crop = plan.lens.as_ref().map(LensGeometry::crop);
    if let Some(crop) = crop {
        let geo = LensGeometry::new(fw, fh, &tags, LensSwitches::default())
            .expect("lens")
            .with_crop(crop);
        let r9 = resample::resample_lens(&img, &geo, Rect::new(0, 0, crop.w, crop.h)).expect("R9");
        write_f32(&out_dir.join(format!("{stem}.{method_name}.r9.f32")), &r9);
        let no_ca = LensGeometry::new(
            fw,
            fh,
            &tags,
            LensSwitches {
                lens_corrections: true,
                chromatic_aberration: false,
            },
        )
        .expect("lens")
        .with_crop(crop);
        let r9n =
            resample::resample_lens(&img, &no_ca, Rect::new(0, 0, crop.w, crop.h)).expect("R9");
        write_f32(&out_dir.join(format!("{stem}.{method_name}.r9noca.f32")), &r9n);
        println!(
            "{stem}: frame {fw}x{fh}, crop {}x{} at {},{}",
            crop.w, crop.h, crop.x, crop.y
        );
    } else {
        println!("{stem}: frame {fw}x{fh}, no lens geometry");
    }
}
