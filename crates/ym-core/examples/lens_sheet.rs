//! Lens contact-sheet frames: renders each raw file at the lens variants of the first visual review
//! of "lens character" (`docs/21`) and, optionally, dumps the planes the lens works from. A developer
//! tool for local review (real photos, never committed); it is not shipped and not part of the
//! product.
//!
//! ```text
//! cargo run -p ym-core --example lens_sheet -- [options] <file.ARW>...
//!
//!   --out DIR          output folder (default ./lens-sheets)
//!   --long-edge N      long edge of the fit frames in pixels (default 1500)
//!   --quality Q        JPEG quality, 50 to 100 (default 90)
//!   --only a,b,c       render only these variants (names below, and the custom ones)
//!   --variant N=K,F,E,A  add a variant named N: lens K (`default`, `vintage`, ...) with Flares F, Edges E
//!                      and Anamorphic A; repeat for several (rendered after the built-in ones)
//!   --resume           skip the variants whose .jpg exists (and the plane dump when bokeh.json exists)
//!   --dump-planes      also write the lens plane, the blur map b and the confidence c
//!   --light-planes     with --dump-planes: no .f32 files (saves disk), only the json and the PNGs
//!   --progress FILE    append "START/DONE/FAIL" lines here (default <out>/progress.txt)
//! ```
//!
//! Each photo is opened the way the app opens it (`Core::open`: the shipped data folder, the Signature
//! look as the default, every setting automatic, no thumbnails, preferences in a scratch folder under
//! the output folder), then rendered through `Renderer::viewport_frame` at Fit in a cell of the
//! photo's own aspect with the long edge at `--long-edge`: the reduced-group render the fit view makes,
//! not a full-resolution export. The frame is Display P3 (the viewport's space); it is converted to
//! sRGB with clipping before it is written.
//!
//! Writes `<out>/<stem>/<variant>.jpg` and appends one JSON line per variant to
//! `<out>/<stem>/variants.jsonl` (render time, the lens report, the stage timings). Variants, as
//! `kind flares edges anamorphic`:
//!
//! | name | lens |
//! |---|---|
//! | `off` | no lens |
//! | `default` | default 60 40 0 |
//! | `classic`, `vintage`, `petzval`, `triplet`, `sonnar`, `detuned` | the lens with its `start` amounts, 0 |
//! | `anamorphic` | default 60 40 100 |
//! | `vintage-max` | vintage 100 100 100 |
//!
//! The variants are rendered in an order that builds the lens planes once per lens (the renderer keeps
//! one set of planes per photo, and a plan with Anamorphic needs the blur map).
//!
//! Speed: the CPU lens at the fit view costs time in proportion to the photo's bright light sources
//! (L3 draws every ghost, streak and starburst of up to 256 source cells at every pixel they reach).
//! A photo without such lights takes 2 to 10 s per variant, one with 20 cells 3 to 10 minutes, one that
//! reaches the 256 cells more than that, so build this example with
//! `CARGO_PROFILE_DEV_OPT_LEVEL=3 CARGO_PROFILE_DEV_DEBUG=false CARGO_INCREMENTAL=0` and copy the binary
//! out of `target/` (point `YELLOWMOON_DATA` at a copy of `data/` to keep a run on one lens data version).
//!
//! With `--dump-planes` it writes, per photo, in `<out>/<stem>/`: `lens_plane.f32` (the raw group at
//! `k_L` as the lens receives it: camera RGB, scene linear, white-balanced with g0, planar R then G
//! then B, row-major, little-endian f32, no header) with `lens_plane.json` (width, height, `k_L`, the
//! half-diagonal in pixels of the plane); `blur_b.f32` and `confidence_c.f32` (the classical bokeh
//! detector's maps, row-major, little-endian f32, `b` in half-diagonals) with `bokeh.json` (size,
//! the detector's counters, the balls, the focus point); and the false-colour `blur_b.png` (0 to 0.03
//! half-diagonals) and `confidence_c.png` (0 to 1). The planes are built here from the same public
//! pieces the renderer uses (`reduced_bayer`, `LensPlanes::build` with Default and Anamorphic 100, so
//! the plan needs the blur map); they are not the renderer's own copy.

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
    FrameData, RawParams, RawPlan, diffraction_params, noise_model_is_usable, reduced_bayer,
    reduced_rgb, sigma_plane, usable_noise_model,
};
use ym_core::pipeline::{FrameOptions, PhotoSource, RenderedFrame, ViewportRect, ViewportState};
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
    Axis, CameraResponse, FrameInfo, ImageRef, LensAmounts, LensLibrary, LensPlan, LensPlanes,
};
use ym_raw::makernote::SonyFocus;

type Fallible<T> = Result<T, Box<dyn std::error::Error>>;

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

/// The ten variants, with the start amounts of the lenses read from the library. Listed in the
/// order of the sheet.
fn variants(lib: &LensLibrary) -> Fallible<Vec<Variant>> {
    let start = |kind: LensKind| -> Fallible<(f64, f64)> {
        lib.lenses()
            .iter()
            .find(|i| i.kind == kind)
            .map(|i| (i.start_flares, i.start_edges))
            .ok_or_else(|| format!("the lens library has no {kind}").into())
    };
    let mut out = vec![
        Variant {
            name: "off",
            lens: LensSettings::default(),
        },
        Variant {
            name: "default",
            lens: ls(LensKind::Default, 60.0, 40.0, 0.0),
        },
    ];
    for (name, kind) in [
        ("classic", LensKind::Classic),
        ("vintage", LensKind::Vintage),
        ("petzval", LensKind::Petzval),
        ("triplet", LensKind::Triplet),
        ("sonnar", LensKind::Sonnar),
        ("detuned", LensKind::Detuned),
    ] {
        let (f, e) = start(kind)?;
        out.push(Variant {
            name,
            lens: ls(kind, f, e, 0.0),
        });
    }
    out.push(Variant {
        name: "anamorphic",
        lens: ls(LensKind::Default, 60.0, 40.0, 100.0),
    });
    out.push(Variant {
        name: "vintage-max",
        lens: ls(LensKind::Vintage, 100.0, 100.0, 100.0),
    });
    Ok(out)
}

/// The order the variants are rendered in: planes are made once per lens (`anamorphic` first among the
/// Default ones so its blur map serves `default`, `vintage-max` before `vintage`).
const RENDER_ORDER: [&str; 10] = [
    "off",
    "anamorphic",
    "default",
    "vintage-max",
    "vintage",
    "classic",
    "petzval",
    "triplet",
    "sonnar",
    "detuned",
];

struct Args {
    out: PathBuf,
    long_edge: usize,
    quality: u8,
    only: Option<Vec<String>>,
    resume: bool,
    dump_planes: bool,
    light_planes: bool,
    progress: Option<PathBuf>,
    custom: Vec<Variant>,
    files: Vec<PathBuf>,
}

fn parse_args() -> Fallible<Args> {
    let mut a = Args {
        out: PathBuf::from("lens-sheets"),
        long_edge: 1500,
        quality: 90,
        only: None,
        resume: false,
        dump_planes: false,
        light_planes: false,
        progress: None,
        custom: Vec::new(),
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
            "--variant" => {
                let spec = value("--variant")?;
                let (name, rest) = spec
                    .split_once('=')
                    .ok_or("--variant wants NAME=KIND,FLARES,EDGES,ANAMORPHIC")?;
                let p: Vec<&str> = rest.split(',').collect();
                if p.len() != 4 {
                    return Err("--variant wants NAME=KIND,FLARES,EDGES,ANAMORPHIC".into());
                }
                a.custom.push(Variant {
                    // A few names for the length of the process.
                    name: Box::leak(name.to_owned().into_boxed_str()),
                    lens: ls(p[0].parse()?, p[1].parse()?, p[2].parse()?, p[3].parse()?),
                });
            }
            "--resume" => a.resume = true,
            "--dump-planes" => a.dump_planes = true,
            "--light-planes" => a.light_planes = true,
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
fn frame_to_srgb(frame: &RenderedFrame, image: (usize, usize)) -> (usize, usize, Vec<u8>) {
    let fit = frame
        .report
        .zoom
        .map_or(1.0, |z| z.fit_zoom)
        .max(f64::MIN_POSITIVE);
    let shown_w = ((image.0 as f64 * fit).floor() as usize).clamp(1, frame.width);
    let shown_h = ((image.1 as f64 * fit).floor() as usize).clamp(1, frame.height);
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

/// Dumps the lens plane, the blur map and the confidence of one photo into `dir`.
fn dump_planes(
    src: &PhotoSource,
    lib: &Arc<LensLibrary>,
    data_root: &Path,
    dir: &Path,
    light: bool,
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
    let report = planes.report();
    let mut info = json!({
        "focus_fraction": focus,
        "focus_raw": src.raw.meta.focus,
        "reshaped": report.reshaped,
        "reason": report.reason.map(|r| format!("{r:?}")),
        "group_ms": group_ms,
        "planes_ms": planes_ms,
    });
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
        let order: Vec<&str> = RENDER_ORDER
            .iter()
            .copied()
            .chain(cx.args.custom.iter().map(|v| v.name))
            .collect();
        for name in order {
            let Some(v) = by_name.get(name) else { continue };
            if let Some(only) = &cx.args.only
                && !only.iter().any(|o| o == name)
            {
                continue;
            }
            let target = dir.join(format!("{}.jpg", v.name));
            if cx.args.resume && target.exists() {
                continue;
            }
            let src = base.with_settings({
                let mut st = base.settings.clone();
                st.lens = v.lens;
                st
            });
            let t = Instant::now();
            let frame = renderer.viewport_frame(&src, &state, &FrameOptions::default(), &cancel)?;
            let wall_ms = t.elapsed().as_secs_f64() * 1000.0;
            let (w, h, rgb) = frame_to_srgb(&frame, image);
            write_jpeg_file(&target, w, h, &rgb, cx.args.quality)?;
            let report = &frame.report;
            writeln!(
                lines,
                "{}",
                json!({
                    "photo": stem,
                    "variant": v.name,
                    "kind": v.lens.kind.as_str(),
                    "flares": v.lens.flares,
                    "edges": v.lens.edges,
                    "anamorphic": v.lens.anamorphic,
                    "width": w, "height": h,
                    "render_ms": wall_ms,
                    "report_ms": report.ms,
                    "scale": report.scale,
                    "lens_ms": report.timings.lens_ms,
                    "raw_ms": report.timings.raw_ms,
                    "pass_ms": report.timings.pass_ms,
                    "lens_planes_built": report.runs.lens_planes,
                    "lens_groups_built": report.runs.lens_groups,
                    "reshaped": report.lens.map(|l| l.reshaped),
                    "reason": report.lens.and_then(|l| l.reason).map(|r| format!("{r:?}")),
                    "notes": report.notes,
                })
            )?;
            lines.flush()?;
            done += 1;
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
    let mut variants = variants(&lib)?;
    variants.extend(args.custom.iter().cloned());
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
        let all_done = variants
            .iter()
            .filter(|v| wanted(v.name))
            .all(|v| dir.join(format!("{}.jpg", v.name)).exists())
            && (!args.dump_planes || dir.join("bokeh.json").exists());
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
