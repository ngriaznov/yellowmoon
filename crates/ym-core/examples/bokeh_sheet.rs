//! Bokeh review frames: renders each raw file with and without Anamorphic 100 (Default lens, Flares 0, Edges 0, so
//! L3 does not run) at the fit view and at 100 % windows, to judge the bokeh reshaping (`docs/21` section 4, L1).
//! A developer tool for local review (real photos, never committed); it is not shipped.
//!
//! ```text
//! cargo run -p ym-core --example bokeh_sheet -- [options] <file.ARW>...
//!
//!   --out DIR          output folder (default ./bokeh-sheets)
//!   --long-edge N      long edge of the fit frames in pixels (default 1500)
//!   --quality Q        JPEG quality, 50 to 100 (default 90)
//!   --only a,b         render only these variants: off, a100, a50 (default off,a100)
//!   --crop NAME=cx,cy,w,h  also render a 100 % window of w x h pixels centred at (cx, cy), fractions of the
//!                      final image (repeatable); written as <stem>/<NAME>_<variant>.jpg
//!   --no-fit           skip the fit frames
//!   --resume           skip the frames whose .jpg exists
//!   --dump-planes      also write the lens plane (lens_plane.f32 / .json) and bokeh.json
//! ```
//!
//! Each photo is opened the way the app opens it (`Core::open`: the shipped data folder, the Signature look as the
//! default, every setting automatic, no thumbnails, preferences in a scratch folder under the output folder). The
//! frame is Display P3; it is converted to sRGB with clipping before it is written.

use std::collections::HashMap;
use std::fs::{self, File, OpenOptions};
use std::io::{BufWriter, Write};
use std::panic::AssertUnwindSafe;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use rayon::prelude::*;
use serde_json::json;
use ym_core::app::{Core, CoreOptions};
use ym_core::pipeline::raw::{
    FrameData, FullChain, RawParams, RawPlan, diffraction_params, noise_model_is_usable,
    reduced_bayer, reduced_rgb, sigma_plane, usable_noise_model,
};
use ym_core::pipeline::{
    FrameOptions, PhotoSource, RenderedFrame, ViewportMode, ViewportRect, ViewportState,
};
use ym_core::session::OpenOptions as SessionOpenOptions;
use ym_core::settings::{LensKind, LensSettings};
use ym_export::ExportError;
use ym_export::jpeg::{JpegMetadata, write_jpeg_bands};
use ym_export::options::JpegQuality;
use ym_export::pixels::{BandSource, ExportHooks};
use ym_gpu::cpu::detail::NoiseReduction;
use ym_gpu::cpu::sharpen::SharpenParams;
use ym_gpu::queue::CancelToken;
use ym_lens::{
    Axis, BlurMapInput, CameraResponse, FrameInfo, ImageRef, LensAmounts, LensLibrary, LensPlan,
    LensPlanes,
};
use ym_ml::bokeh_net::{BokehNet, MIRRORED_PASS, PlanarRgbRef};
use ym_raw::makernote::SonyFocus;

type Fallible<T> = Result<T, Box<dyn std::error::Error>>;

/// A view to render: the file name, the viewport and the window size of a 100 % crop.
type ViewItem = (String, ViewportState, Option<(usize, usize)>);

/// One lens setting to render.
#[derive(Clone)]
struct Variant {
    /// File stem of the output and name in the JSON.
    name: &'static str,
    /// The lens settings (all three amounts 0 is no lens).
    lens: LensSettings,
}

fn ls(kind: LensKind, flares: f64, edges: f64, anamorphic: f64) -> LensSettings {
    LensSettings {
        kind,
        flares,
        edges,
        anamorphic,
    }
}

/// The variants: no lens, and Anamorphic 100 and 50 on the Default lens with Flares 0 and Edges 0 (L3 does not run
/// without Flares; L1 reshapes the bokeh, L2's anamorphic terms and L4's bend act alone).
fn variants(_lib: &LensLibrary) -> Fallible<Vec<Variant>> {
    Ok(vec![
        Variant {
            name: "off",
            lens: LensSettings::default(),
        },
        Variant {
            name: "a100",
            lens: ls(LensKind::Default, 0.0, 0.0, 100.0),
        },
        Variant {
            name: "a50",
            lens: ls(LensKind::Default, 0.0, 0.0, 50.0),
        },
    ])
}

/// The order the variants are rendered in.
const RENDER_ORDER: [&str; 3] = ["off", "a100", "a50"];

/// A 100 % window: a name, its centre as fractions of the final image and its size in pixels.
#[derive(Clone)]
struct Crop {
    name: String,
    centre: [f64; 2],
    size: (usize, usize),
}

struct Args {
    out: PathBuf,
    long_edge: usize,
    quality: u8,
    only: Option<Vec<String>>,
    resume: bool,
    dump_planes: bool,
    light_planes: bool,
    fit: bool,
    crops: Vec<Crop>,
    progress: Option<PathBuf>,
    files: Vec<PathBuf>,
}

fn parse_args() -> Fallible<Args> {
    let mut a = Args {
        out: PathBuf::from("bokeh-sheets"),
        long_edge: 1500,
        quality: 90,
        only: Some(vec!["off".to_owned(), "a100".to_owned()]),
        resume: false,
        dump_planes: false,
        light_planes: false,
        fit: true,
        crops: Vec::new(),
        progress: None,
        files: Vec::new(),
    };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        let mut value = |name: &str| it.next().ok_or_else(|| format!("{name} needs a value"));
        match arg.as_str() {
            "--out" => a.out = PathBuf::from(value("--out")?),
            "--long-edge" => a.long_edge = value("--long-edge")?.parse()?,
            "--quality" => a.quality = value("--quality")?.parse()?,
            "--only" => {
                a.only = Some(
                    value("--only")?
                        .split(',')
                        .map(str::to_owned)
                        .collect::<Vec<_>>(),
                );
            }
            "--progress" => a.progress = Some(PathBuf::from(value("--progress")?)),
            "--resume" => a.resume = true,
            "--dump-planes" => a.dump_planes = true,
            "--light-planes" => a.light_planes = true,
            "--no-fit" => a.fit = false,
            "--crop" => {
                let v = value("--crop")?;
                let (name, rest) = v.split_once('=').ok_or("--crop NAME=cx,cy,w,h")?;
                let n: Vec<f64> = rest.split(',').map(str::parse).collect::<Result<_, _>>()?;
                if n.len() != 4 {
                    return Err("--crop NAME=cx,cy,w,h".into());
                }
                a.crops.push(Crop {
                    name: name.to_owned(),
                    centre: [n[0], n[1]],
                    size: (n[2] as usize, n[3] as usize),
                });
            }
            "-h" | "--help" => {
                println!("see the documentation at the top of examples/lens_sheet.rs");
                std::process::exit(0);
            }
            other if other.starts_with("--") => {
                return Err(format!("unknown option {other}").into());
            }
            file => a.files.push(PathBuf::from(file)),
        }
    }
    if a.files.is_empty() {
        return Err("give at least one .ARW file".into());
    }
    Ok(a)
}

fn log(progress: &Path, line: &str) {
    if let Ok(mut f) = OpenOptions::new().create(true).append(true).open(progress) {
        let _ = f.write_all(format!("{line}\n").as_bytes());
    }
    eprintln!("{line}");
}

fn now() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs());
    format!(
        "{:02}:{:02}:{:02}",
        (secs / 3600) % 24,
        (secs / 60) % 60,
        secs % 60
    )
}

// -------------------------------------------------------------------------------------------------
// Output: Display P3 frame to sRGB JPEG
// -------------------------------------------------------------------------------------------------

/// 8-bit interleaved RGB rows for the JPEG writer.
struct Rgb8<'a> {
    width: u32,
    height: u32,
    rgb: &'a [u8],
}

impl BandSource for Rgb8<'_> {
    fn width(&self) -> u32 {
        self.width
    }
    fn height(&self) -> u32 {
        self.height
    }
    fn rows_u8(&mut self, y0: usize, rows: usize, out: &mut Vec<u8>) -> Result<(), ExportError> {
        out.clear();
        let row = self.width as usize * 3;
        let (a, b) = (y0 * row, (y0 + rows) * row);
        let slice = self
            .rgb
            .get(a..b)
            .ok_or_else(|| ExportError::invalid("rows outside the picture"))?;
        out.extend_from_slice(slice);
        Ok(())
    }
    fn rows_u16(
        &mut self,
        _y0: usize,
        _rows: usize,
        _out: &mut Vec<u16>,
    ) -> Result<(), ExportError> {
        Err(ExportError::invalid("8-bit rows only"))
    }
}

fn srgb_decode(v: f32) -> f32 {
    if v <= 0.040_45 {
        v / 12.92
    } else {
        ((v + 0.055) / 1.055).powf(2.4)
    }
}

fn srgb_encode(v: f32) -> f32 {
    let v = v.clamp(0.0, 1.0);
    if v <= 0.003_130_8 {
        v * 12.92
    } else {
        1.055 * v.powf(1.0 / 2.4) - 0.055
    }
}

/// The part of a Display P3 viewport frame that holds the picture (the cell is the photo's aspect, so
/// at most a pixel of letterbox), converted to sRGB with the gamut clipped. Returns width, height and
/// interleaved RGB.
fn frame_to_srgb(frame: &RenderedFrame, image: Option<(usize, usize)>) -> (usize, usize, Vec<u8>) {
    let (shown_w, shown_h) = match image {
        Some(image) => {
            let fit = frame
                .report
                .zoom
                .map_or(1.0, |z| z.fit_zoom)
                .max(f64::MIN_POSITIVE);
            (
                ((image.0 as f64 * fit).floor() as usize).clamp(1, frame.width),
                ((image.1 as f64 * fit).floor() as usize).clamp(1, frame.height),
            )
        }
        None => (frame.width, frame.height),
    };
    let x0 = (frame.width - shown_w) / 2;
    let y0 = (frame.height - shown_h) / 2;
    // Display P3 to sRGB, linear: inverse(sRGB to XYZ) * (P3 to XYZ), both D65.
    const P3_TO_XYZ: [[f32; 3]; 3] = [
        [0.486_570_95, 0.265_667_7, 0.198_217_29],
        [0.228_974_56, 0.691_738_5, 0.079_286_91],
        [0.0, 0.045_113_38, 1.043_944_4],
    ];
    const XYZ_TO_SRGB: [[f32; 3]; 3] = [
        [3.240_454_2, -1.537_138_5, -0.498_531_4],
        [-0.969_266, 1.876_010_8, 0.041_556],
        [0.055_643_4, -0.204_025_9, 1.057_225_2],
    ];
    let mut m = [[0f32; 3]; 3];
    for (i, row) in m.iter_mut().enumerate() {
        for (j, v) in row.iter_mut().enumerate() {
            *v = (0..3).map(|k| XYZ_TO_SRGB[i][k] * P3_TO_XYZ[k][j]).sum();
        }
    }
    let lut: Vec<f32> = (0..256).map(|v| srgb_decode(v as f32 / 255.0)).collect();
    let mut rgb = vec![0u8; shown_w * shown_h * 3];
    rgb.par_chunks_mut(shown_w * 3)
        .enumerate()
        .for_each(|(y, row)| {
            let src = &frame.rgba[((y0 + y) * frame.width + x0) * 4..];
            for x in 0..shown_w {
                let p = [
                    lut[usize::from(src[x * 4])],
                    lut[usize::from(src[x * 4 + 1])],
                    lut[usize::from(src[x * 4 + 2])],
                ];
                for c in 0..3 {
                    let lin = m[c][0] * p[0] + m[c][1] * p[1] + m[c][2] * p[2];
                    row[x * 3 + c] = (srgb_encode(lin) * 255.0 + 0.5) as u8;
                }
            }
        });
    (shown_w, shown_h, rgb)
}

fn write_jpeg_file(path: &Path, w: usize, h: usize, rgb: &[u8], quality: u8) -> Fallible<()> {
    let mut src = Rgb8 {
        width: w as u32,
        height: h as u32,
        rgb,
    };
    let out = BufWriter::new(File::create(path)?);
    write_jpeg_bands(
        out,
        &mut src,
        JpegQuality::new(quality)?,
        &JpegMetadata {
            exif_app1: None,
            xmp: None,
            icc: None,
        },
        &ExportHooks::none(),
    )?;
    Ok(())
}

// -------------------------------------------------------------------------------------------------
// Plane dumps
// -------------------------------------------------------------------------------------------------

fn write_f32(path: &Path, data: &[f32]) -> Fallible<()> {
    let mut f = BufWriter::new(File::create(path)?);
    for v in data {
        f.write_all(&v.to_le_bytes())?;
    }
    f.flush()?;
    Ok(())
}

fn write_png_rgb(path: &Path, w: usize, h: usize, rgb: &[u8]) -> Fallible<()> {
    let mut enc = png::Encoder::new(BufWriter::new(File::create(path)?), w as u32, h as u32);
    enc.set_color(png::ColorType::Rgb);
    enc.set_depth(png::BitDepth::Eight);
    let mut writer = enc.write_header()?;
    writer.write_image_data(rgb)?;
    Ok(())
}

/// A turbo-like ramp for 0 to 1 (0 is dark grey so that "nothing" reads as nothing).
fn ramp(t: f32) -> [u8; 3] {
    const STOPS: [[f32; 3]; 6] = [
        [0.10, 0.10, 0.45],
        [0.05, 0.45, 0.95],
        [0.10, 0.85, 0.75],
        [0.55, 0.95, 0.20],
        [0.98, 0.75, 0.10],
        [0.90, 0.10, 0.10],
    ];
    let t = t.clamp(0.0, 1.0) * (STOPS.len() - 1) as f32;
    let i = (t.floor() as usize).min(STOPS.len() - 2);
    let f = t - i as f32;
    let mut out = [0u8; 3];
    for c in 0..3 {
        out[c] = ((STOPS[i][c] * (1.0 - f) + STOPS[i + 1][c] * f) * 255.0 + 0.5) as u8;
    }
    out
}

/// A false-colour PNG of a map: values from 0 to `top`; exact zeros are dark grey; the focus point
/// (a fraction of the frame) is marked with a white cross.
fn write_false_colour(
    path: &Path,
    w: usize,
    h: usize,
    values: &[f32],
    top: f32,
    focus: Option<(f64, f64)>,
) -> Fallible<()> {
    let mut rgb = vec![0u8; w * h * 3];
    for (i, v) in values.iter().enumerate() {
        let c = if *v <= 0.0 {
            [28, 28, 28]
        } else {
            ramp(*v / top)
        };
        rgb[i * 3..i * 3 + 3].copy_from_slice(&c);
    }
    if let Some((fx, fy)) = focus {
        let (cx, cy) = ((fx * w as f64) as isize, (fy * h as f64) as isize);
        for d in -8isize..=8 {
            for (x, y) in [(cx + d, cy), (cx, cy + d)] {
                if x >= 0 && y >= 0 && (x as usize) < w && (y as usize) < h {
                    let o = (y as usize * w + x as usize) * 3;
                    rgb[o..o + 3].copy_from_slice(&[255, 255, 255]);
                }
            }
        }
    }
    write_png_rgb(path, w, h, &rgb)
}

/// The camera's focus point as fractions of the auto-cropped frame, in the buffer's orientation (the
/// same arithmetic as the render graph's).
fn focus_fraction(
    focus: Option<SonyFocus>,
    sensor: (usize, usize),
    crop: Option<ym_gpu::cpu::Rect>,
) -> Option<(f64, f64)> {
    let focus = focus?;
    let (frame_w, frame_h) = (f64::from(focus.frame[0]), f64::from(focus.frame[1]));
    if frame_w <= 0.0 || frame_h <= 0.0 {
        return None;
    }
    let x = f64::from(focus.point[0]) / frame_w * sensor.0 as f64;
    let y = f64::from(focus.point[1]) / frame_h * sensor.1 as f64;
    let (x, y, w, h) = match crop {
        Some(c) => (x - c.x as f64, y - c.y as f64, c.w as f64, c.h as f64),
        None => (x, y, sensor.0 as f64, sensor.1 as f64),
    };
    if w <= 0.0 || h <= 0.0 {
        return None;
    }
    Some(((x / w).clamp(0.0, 1.0), (y / h).clamp(0.0, 1.0)))
}

/// The raw-group numbers the render graph reads from the settings and the estimates (`Job::params`).
fn raw_params(src: &PhotoSource) -> Fallible<RawParams> {
    let s = &src.settings;
    let e = &src.estimates;
    let g0 = e.initial_gains()?;
    let nr = s
        .noise_reduction
        .manual()
        .copied()
        .unwrap_or(e.noise_reduction);
    let sh = s.sharpen.manual().copied().unwrap_or(e.sharpen);
    let iso = src
        .raw
        .meta
        .iso
        .filter(|i| *i > 0)
        .map_or(100.0, |i| i as f32);
    if !noise_model_is_usable(&e.noise.model) {
        eprintln!("the noise model of the photo is not usable; zero noise");
    }
    let diffraction = if src.capture_method == ym_core::pipeline::raw::CaptureMethod::Diffraction {
        diffraction_params(
            &src.raw.meta,
            src.sensor.as_ref(),
            src.raw.data.width,
            src.raw.data.height,
            src.raw.is_reduced(),
        )
    } else {
        None
    };
    Ok(RawParams {
        g0: [g0[0] as f32, g0[1] as f32, g0[2] as f32],
        neural_demosaic: s.decode.neural_demosaic,
        highlight_recovery: s.decode.highlight_recovery,
        lens_corrections: s.decode.lens_corrections,
        chromatic_aberration: s.decode.chromatic_aberration,
        vignetting: true,
        noise_reduction: NoiseReduction {
            luminance: nr.luminance as f32,
            color: nr.color as f32,
            detail: nr.detail as f32,
        },
        sharpen: SharpenParams {
            amount: sh.amount as f32,
            radius_px: sh.radius_px as f32,
        },
        iso,
        noise: usable_noise_model(&e.noise.model),
        diffraction,
    })
}

/// Runs the bokeh network on the lens plane, writes its raw map (`model_b.f32`, `model_c.f32`, `model_p.f32`) and
/// the planes built from it (the lens's own decision: blurred at `P >= 0.7`, the photo has bokeh by the cells at
/// `P >= 0.9`), with the time each took.
fn model_dump(
    model: &BokehNet,
    lens_plan: &LensPlan,
    image: &ImageRef,
    focus: Option<(f64, f64)>,
    dir: &Path,
    light: bool,
) -> Fallible<serde_json::Value> {
    let planar = PlanarRgbRef::new(image.width, image.height, image.data)?;
    let (map, timings) = model.map_timed(&planar)?;
    if !light {
        write_f32(&dir.join("model_b.f32"), &map.b)?;
        write_f32(&dir.join("model_c.f32"), &map.c)?;
        write_f32(&dir.join("model_p.f32"), &map.p)?;
    }
    let input = BlurMapInput {
        width: map.width,
        height: map.height,
        plane_width: map.plane_width,
        plane_height: map.plane_height,
        b: map.b.clone(),
        p: map.p.clone(),
    };
    let t = Instant::now();
    let planes = LensPlanes::build_with_map(lens_plan, image, focus, Some(&input), &|| false)?;
    let planes_ms = t.elapsed().as_secs_f64() * 1000.0;
    let report = planes.report();
    let mut info = json!({
        "cells": [map.width, map.height],
        "plane": [map.plane_width, map.plane_height],
        "reduce_ms": timings.reduce_ms,
        "encode_ms": timings.encode_ms,
        "forward_ms": timings.forward_ms,
        "mirrored_ms": timings.mirrored_ms,
        "convert_ms": timings.convert_ms,
        "planes_ms": planes_ms,
        "mirrored_pass": MIRRORED_PASS,
        "reshaped": report.reshaped,
        "reason": report.reason.map(|r| format!("{r:?}")),
        "share_p_ge_0_7": map.p.iter().filter(|p| **p >= 0.7).count() as f64 / map.p.len().max(1) as f64,
        "share_p_ge_0_9": map.p.iter().filter(|p| **p >= 0.9).count() as f64 / map.p.len().max(1) as f64,
    });
    if let Some(b) = planes.bokeh() {
        let (bw, bh) = b.size();
        let nonzero = b.blur().iter().filter(|v| **v > 0.0).count();
        info["active"] = json!(b.active());
        info["edge_count"] = json!(b.edge_count());
        info["blurred_fraction"] = json!(b.blurred_fraction());
        info["balls"] = json!(b.balls().len());
        info["nonzero_fraction"] = json!(nonzero as f64 / (bw * bh) as f64);
        let balls: Vec<_> = b
            .balls()
            .iter()
            .map(|x| json!({"x": x.x, "y": x.y, "radius": x.radius, "supported": x.supported}))
            .collect();
        fs::write(dir.join("model_balls.json"), serde_json::to_string(&balls)?)?;
        if !light {
            write_f32(&dir.join("model_blur_b.f32"), b.blur())?;
            write_f32(&dir.join("model_confidence_c.f32"), b.confidence())?;
        }
        write_false_colour(&dir.join("model_blur_b.png"), bw, bh, b.blur(), 0.03, focus)?;
        info["size"] = json!([bw, bh]);
    }
    Ok(info)
}

/// Dumps the lens plane, the blur map and the confidence of one photo into `dir`.
fn dump_planes(
    src: &PhotoSource,
    lib: &Arc<LensLibrary>,
    data_root: &Path,
    dir: &Path,
    light: bool,
    crops: &[Crop],
) -> Fallible<serde_json::Value> {
    let cancel = CancelToken::new();
    let params = raw_params(src)?;
    let frame = FrameData::new(&src.raw)?;
    let plan = RawPlan::new(&src.raw, &frame, &params)?;
    let geometry =
        ym_core::pipeline::geometry::Geometry::new(src.exif_orientation(), &src.settings);
    let axis = if geometry.swaps_sides() {
        Axis::Y
    } else {
        Axis::X
    };
    let f_number = src.raw.meta.f_number.filter(|f| f.is_finite() && *f > 0.0);
    let focus = focus_fraction(
        src.raw.meta.focus,
        (plan.width, plan.height),
        plan.lens
            .as_ref()
            .map(ym_gpu::cpu::lens::LensGeometry::crop),
    );
    let camera = src
        .profile
        .load_curves(data_root)
        .ok()
        .and_then(|c| CameraResponse::from_curves(&c).ok())
        .unwrap_or_else(CameraResponse::neutral);
    let amounts = LensAmounts::from(ls(LensKind::Default, 60.0, 40.0, 100.0));
    let lens_plan = LensPlan::new(
        lib,
        &amounts,
        FrameInfo {
            width: plan.out_w,
            height: plan.out_h,
            axis,
            f_number,
            clip: params.g0,
        },
        &camera,
    )?;
    let k = lens_plan.lens_plane_scale();
    if k < 2 {
        return Err("the frame is too small for a lens plane at k_L >= 2".into());
    }
    let t = Instant::now();
    let image = if frame.reduced {
        reduced_rgb(&frame, &plan, &params, k, &cancel)?
    } else {
        let sigma = sigma_plane(&frame, &plan, &params, &cancel)?;
        reduced_bayer(&frame, &plan, &params, &sigma, k, &cancel)?
    };
    let group_ms = t.elapsed().as_secs_f64() * 1000.0;
    let (w, h) = (image.width, image.height);
    if !light {
        write_f32(&dir.join("lens_plane.f32"), &image.data)?;
    }
    let half_diagonal = 0.5 * (w as f64).hypot(h as f64);
    fs::write(
        dir.join("lens_plane.json"),
        serde_json::to_string_pretty(&json!({
            "width": w,
            "height": h,
            "channels": "planar R, G, B; camera RGB, scene linear, balanced with g0; little-endian f32",
            "k_L": k,
            "frame_width": plan.out_w,
            "frame_height": plan.out_h,
            "half_diagonal_px": half_diagonal,
            "g0": params.g0,
            "f_number": f_number,
            "axis": format!("{axis:?}"),
        }))?,
    )?;
    let t = Instant::now();
    let image_ref = ImageRef {
        width: w,
        height: h,
        x0: 0,
        y0: 0,
        frame_width: w,
        frame_height: h,
        data: &image.data,
    };
    let planes = LensPlanes::build(&lens_plan, &image_ref, focus, &|| false)?;
    let planes_ms = t.elapsed().as_secs_f64() * 1000.0;
    // The learned map, when the data folder has the model: its raw outputs and the planes made from it.
    let model_info = match BokehNet::load_bundled(data_root) {
        Ok(model) => Some(model_dump(
            &model, &lens_plan, &image_ref, focus, dir, light,
        )?),
        Err(_) => None,
    };
    // The 100 % windows' inputs: the raw group at full resolution over each window grown by L1's halo,
    // for `bokeh_probe --crop` (the lens plane and the plan are the ones above).
    if !crops.is_empty() && !frame.reduced {
        let sigma = sigma_plane(&frame, &plan, &params, &cancel)?;
        let chain = FullChain::new(&frame, &plan, &params, &sigma, &cancel)?;
        let halo = ym_lens::reshape_halo(&lens_plan, &planes, 1);
        for c in crops {
            let (fw, fh) = (plan.out_w, plan.out_h);
            let (cw, ch) = (c.size.0.min(fw), c.size.1.min(fh));
            let x0 = ((c.centre[0] * fw as f64) as isize - cw as isize / 2)
                .clamp(0, (fw - cw) as isize) as usize;
            let y0 = ((c.centre[1] * fh as f64) as isize - ch as isize / 2)
                .clamp(0, (fh - ch) as isize) as usize;
            let gx0 = x0.saturating_sub(halo);
            let gy0 = y0.saturating_sub(halo);
            let gx1 = (x0 + cw + halo).min(fw);
            let gy1 = (y0 + ch + halo).min(fh);
            let tile = chain.run_tile(
                &frame,
                ym_gpu::cpu::Rect {
                    x: gx0,
                    y: gy0,
                    w: gx1 - gx0,
                    h: gy1 - gy0,
                },
            )?;
            write_f32(&dir.join(format!("{}_l1in.f32", c.name)), &tile.data)?;
            fs::write(
                dir.join(format!("{}_l1in.json", c.name)),
                serde_json::to_string_pretty(&json!({
                    "x0": gx0, "y0": gy0, "width": gx1 - gx0, "height": gy1 - gy0,
                    "frame_width": fw, "frame_height": fh,
                    "want": {"x": x0, "y": y0, "w": cw, "h": ch}, "halo": halo,
                }))?,
            )?;
        }
    }
    let report = planes.report();
    let mut info = json!({
        "focus_fraction": focus,
        "focus_raw": src.raw.meta.focus,
        "reshaped": report.reshaped,
        "reason": report.reason.map(|r| format!("{r:?}")),
        "group_ms": group_ms,
        "planes_ms": planes_ms,
    });
    if let Some(m) = model_info {
        info["model"] = m;
    }
    if let Some(b) = planes.bokeh() {
        let (bw, bh) = b.size();
        if !light {
            write_f32(&dir.join("blur_b.f32"), b.blur())?;
            write_f32(&dir.join("confidence_c.f32"), b.confidence())?;
        }
        write_false_colour(&dir.join("blur_b.png"), bw, bh, b.blur(), 0.03, focus)?;
        write_false_colour(
            &dir.join("confidence_c.png"),
            bw,
            bh,
            b.confidence(),
            1.0,
            focus,
        )?;
        let balls: Vec<_> = b
            .balls()
            .iter()
            .map(|ball| {
                json!({
                    "x": ball.x, "y": ball.y, "radius": ball.radius,
                    "take_out_radius": ball.take_out_radius, "energy": ball.energy,
                    "peak": ball.peak, "supported": ball.supported,
                })
            })
            .collect();
        let nonzero = b.blur().iter().filter(|v| **v > 0.0).count();
        info["bokeh"] = json!({
            "size": [bw, bh],
            "reduce": b.reduce(),
            "cells_per_side_px_of_lens_plane": b.reduce(),
            "active": b.active(),
            "edge_count": b.edge_count(),
            "blurred_fraction": b.blurred_fraction(),
            "max_blur": b.max_blur(),
            "nonzero_cells": nonzero,
            "nonzero_fraction": nonzero as f64 / (bw * bh) as f64,
            "balls": balls,
            "map": "blur_b.f32 and confidence_c.f32: row-major little-endian f32 of `size`; b in half-diagonals of the frame",
        });
    }
    fs::write(dir.join("bokeh.json"), serde_json::to_string_pretty(&info)?)?;
    Ok(info)
}

// -------------------------------------------------------------------------------------------------
// One photo
// -------------------------------------------------------------------------------------------------

struct Context<'a> {
    core: &'a Core,
    lib: &'a Arc<LensLibrary>,
    variants: &'a [Variant],
    args: &'a Args,
}

fn render_photo(cx: &Context<'_>, file: &Path, stem: &str, dir: &Path) -> Fallible<usize> {
    let session = &cx.core.session;
    let t_open = Instant::now();
    let outcome = session.open(&[file.to_path_buf()], SessionOpenOptions::default());
    let id = *outcome
        .added
        .first()
        .ok_or("the session did not take the file (not an ARW, or already open)")?;
    if !session.wait_idle(Duration::from_secs(600)) {
        return Err("the session did not finish developing the photo in 10 minutes".into());
    }
    let result = (|| -> Fallible<usize> {
        let view = session
            .view(id)
            .map_err(|e| format!("the photo was skipped: {e}"))?;
        let raw = session.raw_image(id)?;
        let base = PhotoSource::from_view(&view, raw);
        let open_ms = t_open.elapsed().as_secs_f64() * 1000.0;
        let cancel = CancelToken::new();
        let renderer = &cx.core.renderer;
        let image = renderer.final_size(&base, &cancel)?;
        let long = image.0.max(image.1) as f64;
        let s = (cx.args.long_edge as f64 / long).min(1.0);
        let (cw, ch) = (
            (image.0 as f64 * s).round().max(1.0),
            (image.1 as f64 * s).round().max(1.0),
        );
        let state = ViewportState::new(ViewportRect::cell(cw, ch, 1.0));
        fs::create_dir_all(dir)?;
        let mut lines = BufWriter::new(
            OpenOptions::new()
                .create(true)
                .append(true)
                .open(dir.join("variants.jsonl"))?,
        );
        writeln!(
            lines,
            "{}",
            json!({
                "photo": stem, "event": "open", "open_ms": open_ms,
                "camera": format!("{:?}", view.camera.info()),
                "image": [image.0, image.1], "cell": [cw, ch],
                "look": format!("{:?}", base.settings.look),
                "f_number": base.raw.meta.f_number, "iso": base.raw.meta.iso,
                "focus": base.raw.meta.focus,
            })
        )?;
        lines.flush()?;
        if cx.args.dump_planes && !(cx.args.resume && dir.join("bokeh.json").exists()) {
            let t = Instant::now();
            match dump_planes(
                &base,
                cx.lib,
                cx.core.data_root(),
                dir,
                cx.args.light_planes,
                &cx.args.crops,
            ) {
                Ok(info) => {
                    writeln!(
                        lines,
                        "{}",
                        json!({"photo": stem, "event": "planes", "ms": t.elapsed().as_secs_f64() * 1000.0, "info": info})
                    )?;
                }
                Err(e) => {
                    writeln!(
                        lines,
                        "{}",
                        json!({"photo": stem, "event": "planes_failed", "error": e.to_string()})
                    )?;
                }
            }
            lines.flush()?;
        }
        let mut done = 0usize;
        let by_name: HashMap<&str, &Variant> = cx.variants.iter().map(|v| (v.name, v)).collect();
        // The views: the fit frame, then the 100 % windows.
        let mut views: Vec<ViewItem> = Vec::new();
        if cx.args.fit {
            views.push(("fit".to_owned(), state.clone(), Some(image)));
        }
        for c in &cx.args.crops {
            let mut st =
                ViewportState::new(ViewportRect::cell(c.size.0 as f64, c.size.1 as f64, 1.0));
            st.mode = ViewportMode::Percent { value: 100.0 };
            st.centre = c.centre;
            views.push((c.name.clone(), st, None));
        }
        for (view_name, view_state, fit_of) in &views {
            for name in RENDER_ORDER {
                let Some(v) = by_name.get(name) else { continue };
                if let Some(only) = &cx.args.only
                    && !only.iter().any(|o| o == name)
                {
                    continue;
                }
                let file = if view_name == "fit" {
                    format!("{}.jpg", v.name)
                } else {
                    format!("{view_name}_{}.jpg", v.name)
                };
                let target = dir.join(file);
                if cx.args.resume && target.exists() {
                    continue;
                }
                let src = base.with_settings({
                    let mut st = base.settings.clone();
                    st.lens = v.lens;
                    st
                });
                let t = Instant::now();
                let frame =
                    renderer.viewport_frame(&src, view_state, &FrameOptions::default(), &cancel)?;
                let wall_ms = t.elapsed().as_secs_f64() * 1000.0;
                let (w, h, rgb) = frame_to_srgb(&frame, *fit_of);
                write_jpeg_file(&target, w, h, &rgb, cx.args.quality)?;
                let report = &frame.report;
                writeln!(
                    lines,
                    "{}",
                    json!({
                        "photo": stem,
                        "view": view_name,
                        "variant": v.name,
                        "anamorphic": v.lens.anamorphic,
                        "width": w, "height": h,
                        "render_ms": wall_ms,
                        "scale": report.scale,
                        "lens_ms": report.timings.lens_ms,
                        "raw_ms": report.timings.raw_ms,
                        "lens_planes_built": report.runs.lens_planes,
                        "reshaped": report.lens.map(|l| l.reshaped),
                        "reason": report.lens.and_then(|l| l.reason).map(|r| format!("{r:?}")),
                        "notes": report.notes,
                    })
                )?;
                lines.flush()?;
                done += 1;
            }
        }
        Ok(done)
    })();
    cx.core.remove_photos(&[id]);
    result
}

fn main() -> Fallible<()> {
    let args = parse_args()?;
    fs::create_dir_all(&args.out)?;
    let progress = args
        .progress
        .clone()
        .unwrap_or_else(|| args.out.join("progress.txt"));
    // Preferences and looks in a scratch folder: the tool must not read or write the person's.
    let mut options = CoreOptions::from_environment()?;
    options.app_root = args.out.join(".app-home");
    options.thumbnails = false;
    fs::create_dir_all(&options.app_root)?;
    let core = Core::open(options)?;
    let lib = Arc::new(LensLibrary::load(core.data_root())?);
    let variants = variants(&lib)?;
    let cx = Context {
        core: &core,
        lib: &lib,
        variants: &variants,
        args: &args,
    };
    let total = args.files.len();
    let (mut ok, mut failed) = (0usize, 0usize);
    for (i, file) in args.files.iter().enumerate() {
        let stem = file
            .file_stem()
            .map_or_else(|| "photo".to_owned(), |s| s.to_string_lossy().into_owned());
        let dir = args.out.join(&stem);
        let wanted = |n: &str| args.only.as_ref().is_none_or(|o| o.iter().any(|x| x == n));
        let names = |view: &str, v: &str| {
            if view == "fit" {
                format!("{v}.jpg")
            } else {
                format!("{view}_{v}.jpg")
            }
        };
        let all_done = variants.iter().filter(|v| wanted(v.name)).all(|v| {
            (!args.fit || dir.join(names("fit", v.name)).exists())
                && args
                    .crops
                    .iter()
                    .all(|c| dir.join(names(&c.name, v.name)).exists())
        }) && (!args.dump_planes || dir.join("bokeh.json").exists());
        if args.resume && all_done {
            log(
                &progress,
                &format!("{} SKIP {stem} ({}/{total}) already done", now(), i + 1),
            );
            continue;
        }
        log(
            &progress,
            &format!("{} START {stem} ({}/{total})", now(), i + 1),
        );
        let t = Instant::now();
        let outcome =
            std::panic::catch_unwind(AssertUnwindSafe(|| render_photo(&cx, file, &stem, &dir)));
        match outcome {
            Ok(Ok(n)) => {
                ok += 1;
                log(
                    &progress,
                    &format!(
                        "{} DONE {stem} ({n} variants, {:.0} s)",
                        now(),
                        t.elapsed().as_secs_f64()
                    ),
                );
            }
            Ok(Err(e)) => {
                failed += 1;
                log(&progress, &format!("{} FAIL {stem}: {e}", now()));
            }
            Err(_) => {
                failed += 1;
                log(&progress, &format!("{} FAIL {stem}: panic", now()));
            }
        }
    }
    log(
        &progress,
        &format!("{} END {ok} ok, {failed} failed", now()),
    );
    Ok(())
}
