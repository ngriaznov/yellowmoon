//! The export tail on the device (Mesa lavapipe in CI) through `export_photo`.
//!
//! With [`GpuStages::EXPORT_TAIL`] a JPEG, TIFF or PNG export never makes the host frame of display-
//! linear f32: the device samples, colours, resizes, sharpens and encodes the picture, and the
//! writer pulls the encoded rows. What is checked here:
//!
//! * **The tail is the CPU's pixels.** An export with the tail equals the same export with the raw
//!   chain alone on the device (the CPU samples, colours, resizes, sharpens and encodes) within one
//!   8-bit code, for JPEG, PNG and TIFF, 8 and 16 bit, full size and a long edge, sharpened or not,
//!   a straightened crop with paper in its corners included; and the device export equals the CPU
//!   export within the golden metric.
//! * **No host frame.** The host ledger holds the working set of a tile, not the picture.
//! * **Determinism.** The same bytes twice, through `export_photo` and through the export queue.
//! * **Cancel, progress, the look.** A cancel between bands removes the temporary file and hands
//!   every device buffer back; progress never goes back; a look that changes while the picture
//!   renders is not written under its old name.
//! * **Fall back** (`--features fault-injection`): a device error makes the whole photo again on the
//!   CPU, equal to the CPU export bit for bit, with the note and `device = cpu`; a strict policy
//!   fails the photo (kind `gpu`) and leaves no file.
//! * The DNG's colour stage runs on the device too and agrees with the CPU's.
//!
//! Skipped, loudly, when no adapter opens, unless `YM_REQUIRE_GPU=1` turns that into a failure.
mod common;

use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use common::*;
use ym_color::consts::linalg;
use ym_color::deltae::{d65_white_xyz, delta_e_2000, percentile, xyz_to_lab};
use ym_color::encode::srgb_decode;
use ym_core::app::gpu::GpuStages;
use ym_core::export::{
    ExportContext, ExportControl, ExportError, ExportParams, ExportPhoto, ExportReport,
    export_photo,
};
use ym_core::pipeline::{
    ComputeDevice, FullImage, FullRequest, FullTarget, LookProblem, LookProvider, NeutralOnly,
    PhotoSource, RenderOptions, Renderer, ResolvedLook,
};
use ym_core::session::{OpenOptions, PhotoId, Session};
use ym_core::settings::{Crop, CropAspect, NormRect, RolloffPatch, SettingsPatch};
use ym_export::{BitDepth, ExportOptions, FormatKind, IfExists, Size, TiffCompression};
use ym_gpu::device::{Gpu, GpuConfig, gpu_required_by_env};
use ym_gpu::queue::{CancelToken, JobSpec, Priority};
use ym_looks::library::LookId;

const MODEL: &str = "ILCE-7RM4";

fn gpu() -> Option<Arc<Gpu>> {
    // Poison mode: every pool buffer starts as NaN, so a stage that reads what it did not write
    // fails the comparison instead of passing by luck.
    match Gpu::start(GpuConfig {
        poison: true,
        ..GpuConfig::default()
    }) {
        Ok(g) => Some(Arc::new(g)),
        Err(e) => {
            assert!(
                !gpu_required_by_env(),
                "YM_REQUIRE_GPU is set but no GPU adapter could be opened: {e}"
            );
            let _ = writeln!(
                std::io::stderr(),
                "\n* GPU TESTS SKIPPED in gpu_export: {e}\n* (set WGPU_BACKEND=vulkan and install mesa-vulkan-drivers)\n"
            );
            None
        }
    }
}

/// A line for the log and, on GitHub Actions, an annotation (the numbers the Apple runners measure).
fn ci_note(text: &str) {
    let mut err = std::io::stderr();
    let _ = writeln!(err, "[export tail] {text}");
    if std::env::var_os("GITHUB_ACTIONS").is_some() {
        let _ = writeln!(err, "::notice title=Export tail::{text}");
    }
}

fn renderer(gpu: Option<Arc<Gpu>>, stages: GpuStages, strict: bool) -> Renderer {
    Renderer::new(
        Arc::new(NeutralOnly),
        RenderOptions {
            gpu,
            gpu_stages: stages,
            gpu_strict: strict,
            // 256 px tiles: a 900 x 600 photo is twelve tiles, in three rows, and the slabs of the
            // tail are crossed by tile rows.
            full_tile_px: 256,
            ..RenderOptions::default()
        },
    )
    .expect("render thread")
}

fn context(home: &Path, r: Renderer) -> ExportContext {
    ExportContext::new(r).with_home(Some(home.to_path_buf()))
}

/// A photo of 900 x 600 with edges, flat patches and noise, as the session develops it.
struct Shot {
    d: Developed,
    /// The same photo in a session of its own that is shared, for the export queue.
    shared: Arc<Session>,
    shared_id: PhotoId,
    _shared_dir: tempfile::TempDir,
}

impl Shot {
    fn new() -> Self {
        Self::sized(900, 600)
    }

    fn sized(w: u32, h: u32) -> Self {
        let white = light("sony-ilce-7rm4", 4500.0);
        let scene = Scene::new(w, h, white, Layout::Patchwork, 0.5).noise(2e-4, 2e-6);
        let dir = tempfile::tempdir().expect("dir");
        scene.write(dir.path(), "DSC00001.ARW", MODEL);
        let shared = Arc::new(Session::new(base_env()));
        let out = shared.open(&[dir.path().to_path_buf()], OpenOptions::default());
        assert_eq!(out.added.len(), 1);
        settle(&shared);
        Self {
            d: Developed::new(&scene, MODEL),
            shared_id: out.added[0],
            shared,
            _shared_dir: dir,
        }
    }

    fn edit(&self, patch: SettingsPatch) {
        self.d
            .session
            .update_settings(self.d.id, &patch)
            .expect("settings");
        self.shared
            .update_settings(self.shared_id, &patch)
            .expect("settings");
    }

    fn photo(&self) -> ExportPhoto {
        ExportPhoto::from_session(&self.d.session, self.d.id).expect("export photo")
    }
}

fn options(out: &Path, format: FormatKind) -> ExportOptions {
    ExportOptions {
        format,
        destination: Some(out.to_path_buf()),
        if_exists: IfExists::Replace,
        ..ExportOptions::default()
    }
}

fn png(out: &Path, bits: BitDepth) -> ExportOptions {
    ExportOptions {
        bit_depth: bits,
        ..options(out, FormatKind::Png)
    }
}

fn run(
    ctx: &ExportContext,
    photo: &ExportPhoto,
    opts: ExportOptions,
) -> Result<ExportReport, ExportError> {
    export_photo(ctx, photo, &ExportParams::new(opts), &ExportControl::none())
}

fn export(ctx: &ExportContext, photo: &ExportPhoto, opts: ExportOptions) -> ExportReport {
    run(ctx, photo, opts).expect("export")
}

fn read(path: &Path) -> Vec<u8> {
    std::fs::read(path).expect("read the export")
}

/// The codes of a PNG: width, height, bytes per sample, samples (u16 for 16 bit, widened u8 else).
fn decode_png(bytes: &[u8]) -> (usize, usize, bool, Vec<u32>) {
    let decoder = png::Decoder::new(std::io::Cursor::new(bytes.to_vec()));
    let mut reader = decoder.read_info().expect("png header");
    let mut buf = vec![0; reader.output_buffer_size().expect("size")];
    let info = reader.next_frame(&mut buf).expect("png frame");
    let sixteen = info.bit_depth == png::BitDepth::Sixteen;
    let data = &buf[..info.buffer_size()];
    let codes = if sixteen {
        data.chunks_exact(2)
            .map(|c| u32::from(u16::from_be_bytes([c[0], c[1]])))
            .collect()
    } else {
        data.iter().map(|&b| u32::from(b)).collect()
    };
    (info.width as usize, info.height as usize, sixteen, codes)
}

fn decode_jpeg(bytes: &[u8]) -> (usize, usize, Vec<u32>) {
    let mut d = zune_jpeg::JpegDecoder::new(std::io::Cursor::new(bytes));
    let pixels = d.decode().expect("jpeg");
    let info = d.info().expect("info");
    (
        info.width as usize,
        info.height as usize,
        pixels.iter().map(|&b| u32::from(b)).collect(),
    )
}

struct CodeStats {
    n: usize,
    exact: usize,
    max: u32,
    over_one: usize,
    over_eight: usize,
}

fn stats(a: &[u32], b: &[u32]) -> CodeStats {
    assert_eq!(a.len(), b.len());
    let mut s = CodeStats {
        n: a.len(),
        exact: 0,
        max: 0,
        over_one: 0,
        over_eight: 0,
    };
    for (&x, &y) in a.iter().zip(b) {
        let d = x.abs_diff(y);
        s.exact += usize::from(d == 0);
        s.over_one += usize::from(d > 1);
        s.over_eight += usize::from(d > 8);
        s.max = s.max.max(d);
    }
    s
}

/// CIEDE2000 on a 1/4 downsample (a box filter in linear light, the samples read as sRGB), the
/// comparison of the golden runner: median, 99th percentile, maximum.
fn golden(a: &[u32], b: &[u32], w: usize, h: usize, max_code: u32) -> (f64, f64, f64) {
    let to_xyz = ym_color::gamut::matrices().rec709_to_xyz;
    let white = d65_white_xyz();
    let table: Vec<f64> = (0..=max_code)
        .map(|v| srgb_decode(f64::from(v) / f64::from(max_code)))
        .collect();
    let lab = |px: &[u32]| -> Vec<[f64; 3]> {
        let (bw, bh) = (w / 4, h / 4);
        let mut out = Vec::with_capacity(bw * bh);
        for by in 0..bh {
            for bx in 0..bw {
                let mut sum = [0.0f64; 3];
                for dy in 0..4 {
                    for dx in 0..4 {
                        let i = ((by * 4 + dy) * w + bx * 4 + dx) * 3;
                        for (c, s) in sum.iter_mut().enumerate() {
                            *s += table[px[i + c] as usize];
                        }
                    }
                }
                out.push(xyz_to_lab(
                    linalg::mul_vec(&to_xyz, sum.map(|v| v / 16.0)),
                    white,
                ));
            }
        }
        out
    };
    let d: Vec<f64> = lab(a)
        .iter()
        .zip(&lab(b))
        .map(|(x, y)| delta_e_2000(*x, *y))
        .collect();
    (
        percentile(&d, 50.0).expect("median"),
        percentile(&d, 99.0).expect("p99"),
        d.iter().copied().fold(0.0, f64::max),
    )
}

/// The bytes in use in the device's pool, the colour tables (resident on purpose) let go.
fn pool_in_use(gpu: &Gpu) -> u64 {
    gpu.run(JobSpec::new(Priority::Viewport), |ctx, _| {
        ym_gpu::kernels::color::release_tables(ctx);
        ctx.pool().trim();
        Ok(ctx.pool().stats().in_use_bytes)
    })
    .expect("pool stats")
}

const RAW: GpuStages = GpuStages::RAW_CHAIN;
const RAW_AND_TAIL: GpuStages = GpuStages::RAW_CHAIN.union(GpuStages::EXPORT_TAIL);

/// (name, sharpening %, size, crop and straighten).
fn variants() -> Vec<(&'static str, f64, Size, bool)> {
    vec![
        ("full size", 100.0, Size::Full, false),
        ("full size, not sharpened", 0.0, Size::Full, false),
        ("long edge 500", 100.0, Size::LongEdge { px: 500 }, false),
        (
            "long edge 331, not sharpened",
            0.0,
            Size::LongEdge { px: 331 },
            false,
        ),
        (
            "straightened crop, long edge 600",
            100.0,
            Size::LongEdge { px: 600 },
            true,
        ),
    ]
}

fn apply_variant(shot: &Shot, pct: f64, straighten: bool) {
    let crop = straighten.then_some(Crop {
        rect: NormRect {
            x: 0.04,
            y: 0.06,
            width: 0.9,
            height: 0.88,
        },
        angle_deg: 5.0,
        aspect: CropAspect::Free,
    });
    shot.edit(SettingsPatch {
        output_sharpening: Some(pct),
        crop: Some(crop),
        ..SettingsPatch::default()
    });
}

/// What [`the_tail_writes_the_cpu_pixels_in_every_format`] allows of the sharpened and the JPEG
/// exports, for one kind of adapter. Each value is twice the first measured one (nothing is loosened
/// without Mykyta): software on Mesa lavapipe, hardware on the Apple runners (from the fifteen
/// cases that run 38694506557 printed on macos-14 and macos-15, identical to the digit).
struct TailBounds {
    /// Sharpened 16 bit: the share of samples more than 1 code off, more than 8 codes off, and the
    /// largest difference in codes.
    over_one_16: f64,
    over_eight_16: f64,
    max_16: u32,
    /// The decoded JPEG: the largest difference, the share more than 1 code off, the exact share.
    jpeg_max: u32,
    jpeg_over_one: f64,
    jpeg_exact: f64,
    /// Sharpened 8 bit: the largest difference, the share more than 1 code off (with a floor of
    /// codes, for a picture too small for a share), the exact share.
    sharp8_max: u32,
    sharp8_over_one: f64,
    sharp8_floor: f64,
    sharp8_exact: f64,
}

/// First measured on lavapipe: 354 of 1.62 M 16-bit samples over 1 code and 81 over 8 (at most 89
/// off), the JPEGs 280 over 1 (1.7e-4) at most 5 off, and no sharpened 8-bit sample more than one off.
const SOFTWARE: TailBounds = TailBounds {
    over_one_16: 4.4e-4,
    over_eight_16: 1.0e-4,
    max_16: 180,
    jpeg_max: 10,
    jpeg_over_one: 3.5e-4,
    jpeg_exact: 0.99,
    sharp8_max: 1,
    sharp8_over_one: 0.0,
    sharp8_floor: 0.0,
    sharp8_exact: 0.99,
};

/// First measured on Metal: sharpened 8 bit 6 of 1.62 M codes over 1, at most 2, 0.99988 exact;
/// sharpened 16 bit 2.55e-4 over 1, 9.7e-5 over 8, at most 551; JPEG at most 5 codes, 2.12e-4 over 1
/// (long edge 500), 0.99829 exact (straightened crop). The limiter's rank flips are the cause, not the
/// sharpening kernel, which gives lavapipe's bits on Metal (`docs/algorithms/04` section 10).
const HARDWARE: TailBounds = TailBounds {
    over_one_16: 5.1e-4,
    over_eight_16: 1.94e-4,
    max_16: 1102,
    jpeg_max: 10,
    jpeg_over_one: 4.25e-4,
    jpeg_exact: 0.9965,
    sharp8_max: 4,
    sharp8_over_one: 7.4e-6,
    sharp8_floor: 2.0,
    sharp8_exact: 0.9997,
};

#[test]
fn the_tail_writes_the_cpu_pixels_in_every_format() {
    let Some(gpu) = gpu() else { return };
    let shot = Shot::new();
    let home = tempfile::tempdir().expect("home");
    let raw_only = context(home.path(), renderer(Some(Arc::clone(&gpu)), RAW, false));
    let with_tail = context(
        home.path(),
        renderer(Some(Arc::clone(&gpu)), RAW_AND_TAIL, false),
    );
    // Software (lavapipe) holds the bounds first measured there. A hardware adapter has its own
    // column for the sharpened outputs only: the raw-chain-only export runs the colour chain on the
    // CPU in f64, the tail on the device in f32, and the output-sharpening limiter decides on a rank
    // of luma values, so last-bit differences flip a few pixels (`docs/algorithms/04` section 10).
    // Metal (4c5939f, run 38694506557, macos-14 and macos-15 alike) printed all fifteen cases; the
    // hardware column is twice the worst of them, see [`HARDWARE`]: the sharpened 8-bit
    // exports 6 of 1.62 M codes over 1, at most 2, 0.99988 exact; the sharpened 16-bit exports
    // 413 / 157 over 1 / over 8 of 1.62 M (2.55e-4 / 9.7e-5), at most 551; the JPEGs at most 5 codes
    // and 2.1e-4 over 1, 0.99829 exact. The sharpening kernel alone matches lavapipe's bits there.
    let bounds = if gpu.info().is_software {
        &SOFTWARE
    } else {
        &HARDWARE
    };
    let mut failures: Vec<String> = Vec::new();
    let mut check = |ok: bool, what: String| {
        if !ok {
            failures.push(what);
        }
    };
    let (mut worst8, mut min_exact8) = (0u32, 1.0f64);
    for (name, pct, size, straighten) in variants() {
        apply_variant(&shot, pct, straighten);
        let photo = shot.photo();
        for (label, mk) in [
            (
                "PNG 8",
                (|out: &Path| png(out, BitDepth::Eight)) as fn(&Path) -> ExportOptions,
            ),
            ("PNG 16", |out: &Path| png(out, BitDepth::Sixteen)),
            ("JPEG", |out: &Path| options(out, FormatKind::Jpeg)),
        ] {
            let (a, b) = (
                tempfile::tempdir().expect("a"),
                tempfile::tempdir().expect("b"),
            );
            let (mut oa, mut ob) = (mk(a.path()), mk(b.path()));
            oa.size = size;
            ob.size = size;
            let ra = export(&raw_only, &photo, oa);
            let rb = export(&with_tail, &photo, ob);
            assert_eq!(
                rb.device,
                ComputeDevice::Gpu,
                "{name} {label}: {:?}",
                rb.notes
            );
            assert!(
                !rb.notes.iter().any(|n| n.contains("graphics card")),
                "{name} {label}: {:?}",
                rb.notes
            );
            assert_eq!(
                (ra.width, ra.height),
                (rb.width, rb.height),
                "{name} {label}"
            );
            assert_eq!(ra.rendered, rb.rendered, "{name} {label}");
            let (ba, bb) = (read(&ra.path), read(&rb.path));
            let (ca, cb, w, h, sixteen) = if label == "JPEG" {
                let (w, h, pa) = decode_jpeg(&ba);
                let (_, _, pb) = decode_jpeg(&bb);
                (pa, pb, w, h, false)
            } else {
                let (w, h, sixteen, pa) = decode_png(&ba);
                let (_, _, _, pb) = decode_png(&bb);
                (pa, pb, w, h, sixteen)
            };
            assert_eq!((w as u32, h as u32), (rb.width, rb.height));
            let s = stats(&ca, &cb);
            ci_note(&format!(
                "{name}, {label}: {} codes, {:.5} exact, {} over 1, {} over 8, max {}",
                s.n,
                s.exact as f64 / s.n as f64,
                s.over_one,
                s.over_eight,
                s.max
            ));
            let n = s.n as f64;
            let exact = s.exact as f64 / n;
            if sixteen && pct == 0.0 {
                // The starting bound for the 16-bit C13: within 8 codes. First measured: 1 code.
                check(s.max <= 2, format!("{name} {label}: max {}", s.max));
            } else if sixteen {
                // Sharpening amplifies the f32 against f64 colour difference at the few pixels
                // where its halo limiter decides on a rank of luma values (`gpu_tail.rs`). First
                // measured: 354 samples over 1 code (2.2e-4), 81 over 8 (5.0e-5), at most 89 codes
                // off, in 1.62 M samples; the bounds are twice that (hardware: see above).
                check(
                    s.over_one as f64 <= bounds.over_one_16 * n
                        && s.over_eight as f64 <= bounds.over_eight_16 * n
                        && s.max <= bounds.max_16,
                    format!(
                        "{name} {label}: {} over 1, {} over 8, max {}",
                        s.over_one, s.over_eight, s.max
                    ),
                );
            } else if label == "JPEG" {
                // The DCT and the quantiser turn a one-code input difference into a step of the
                // quantiser in one coefficient: a few codes over the block. First measured: 280
                // samples over 1 code (1.7e-4) and 5 codes at most; the bounds are twice that.
                check(
                    s.max <= bounds.jpeg_max
                        && s.over_one as f64 <= bounds.jpeg_over_one * n
                        && exact >= bounds.jpeg_exact,
                    format!(
                        "{name} {label}: {} over 1, max {}, {exact} exact",
                        s.over_one, s.max
                    ),
                );
            } else if pct > 0.0 {
                // A rank flip moves a dithered 8-bit code by more than one (see above); on a software
                // adapter none did (the C13 bound of `docs/17-gpu.md` section 12: within one code,
                // 99 % exact).
                check(
                    s.max <= bounds.sharp8_max
                        && s.over_one as f64
                            <= (bounds.sharp8_over_one * n).max(bounds.sharp8_floor)
                        && exact >= bounds.sharp8_exact,
                    format!(
                        "{name} {label}: {} over 1, max {}, {exact} exact",
                        s.over_one, s.max
                    ),
                );
                worst8 = worst8.max(s.max);
                min_exact8 = min_exact8.min(exact);
            } else {
                // The C13 bound of `docs/17-gpu.md` section 12: within one code, 99 % of the codes
                // exact.
                check(s.max <= 1, format!("{name} {label}: max {}", s.max));
                check(exact >= 0.99, format!("{name} {label}: {exact} exact"));
                worst8 = worst8.max(s.max);
                min_exact8 = min_exact8.min(exact);
            }
        }
    }
    ci_note(&format!(
        "tail against the raw-chain-only export: worst 8-bit difference {worst8}, smallest exact share {min_exact8:.5}"
    ));
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn a_tiff_through_the_tail_is_the_png_of_the_same_codes() {
    let Some(gpu) = gpu() else { return };
    let shot = Shot::new();
    apply_variant(&shot, 100.0, false);
    let photo = shot.photo();
    let home = tempfile::tempdir().expect("home");
    let ctx = context(
        home.path(),
        renderer(Some(Arc::clone(&gpu)), RAW_AND_TAIL, false),
    );
    let out = tempfile::tempdir().expect("out");
    let mut codes_of = Vec::new();
    for (bits, compression) in [
        (BitDepth::Sixteen, TiffCompression::None),
        (BitDepth::Sixteen, TiffCompression::Lzw),
        (BitDepth::Eight, TiffCompression::Zip),
    ] {
        let o = ExportOptions {
            bit_depth: bits,
            tiff_compression: compression,
            size: Size::LongEdge { px: 640 },
            ..options(out.path(), FormatKind::Tiff)
        };
        let r = export(&ctx, &photo, o);
        assert_eq!(r.device, ComputeDevice::Gpu);
        let bytes = read(&r.path);
        let t = common::tiffmini::Tiff::new(&bytes).expect("a TIFF");
        let ifd = t.ifd0();
        assert_eq!(t.uint(ifd, 256), Some(r.width), "{compression:?}");
        assert_eq!(t.uint(ifd, 257), Some(r.height), "{compression:?}");
        if compression == TiffCompression::None {
            let (_, _, samples) = t.uncompressed_u16_rgb(ifd);
            codes_of.push(samples);
        }
    }
    // The PNG of the same depth holds the same codes as the uncompressed TIFF.
    let p = export(
        &ctx,
        &photo,
        ExportOptions {
            size: Size::LongEdge { px: 640 },
            ..png(out.path(), BitDepth::Sixteen)
        },
    );
    let (_, _, _, png_codes) = decode_png(&read(&p.path));
    let tiff_codes: Vec<u32> = codes_of[0].iter().map(|&v| u32::from(v)).collect();
    assert!(png_codes == tiff_codes, "TIFF and PNG codes differ");
}

#[test]
fn device_and_cpu_exports_agree_within_the_golden_metric() {
    let Some(gpu) = gpu() else { return };
    let shot = Shot::new();
    let home = tempfile::tempdir().expect("home");
    let cpu = context(home.path(), renderer(None, GpuStages::NONE, false));
    let dev = context(home.path(), renderer(Some(gpu), GpuStages::ALL, false));
    // (name, options, decoder: file bytes to (width, height, codes, largest code)).
    type Decoded = (usize, usize, Vec<u32>, u32);
    type Format = (
        &'static str,
        fn(&Path) -> ExportOptions,
        fn(&[u8]) -> Decoded,
    );
    let formats: [Format; 4] = [
        (
            "PNG 8",
            |o| png(o, BitDepth::Eight),
            |b| {
                let (w, h, _, c) = decode_png(b);
                (w, h, c, 255)
            },
        ),
        (
            "PNG 16",
            |o| png(o, BitDepth::Sixteen),
            |b| {
                let (w, h, _, c) = decode_png(b);
                (w, h, c, 65535)
            },
        ),
        (
            "JPEG",
            |o| options(o, FormatKind::Jpeg),
            |b| {
                let (w, h, c) = decode_jpeg(b);
                (w, h, c, 255)
            },
        ),
        (
            "TIFF 16",
            |o| ExportOptions {
                bit_depth: BitDepth::Sixteen,
                tiff_compression: TiffCompression::None,
                ..options(o, FormatKind::Tiff)
            },
            |b| {
                let t = common::tiffmini::Tiff::new(b).expect("a TIFF");
                let (w, h, samples) = t.uncompressed_u16_rgb(t.ifd0());
                (w, h, samples.into_iter().map(u32::from).collect(), 65535)
            },
        ),
    ];
    for (i, (name, pct, size, straighten)) in variants().into_iter().enumerate() {
        apply_variant(&shot, pct, straighten);
        let photo = shot.photo();
        for (label, make, decode) in formats {
            // Every variant in 8-bit PNG; the other formats on three of them (full size, a long
            // edge, the straightened crop), to keep the run short.
            if label != "PNG 8" && ![0, 2, 4].contains(&i) {
                continue;
            }
            let (a, b) = (
                tempfile::tempdir().expect("a"),
                tempfile::tempdir().expect("b"),
            );
            let (mut oa, mut ob) = (make(a.path()), make(b.path()));
            oa.size = size;
            ob.size = size;
            let (ra, rb) = (export(&cpu, &photo, oa), export(&dev, &photo, ob));
            assert_eq!(ra.device, ComputeDevice::Cpu);
            assert_eq!(
                rb.device,
                ComputeDevice::Gpu,
                "{name} {label}: {:?}",
                rb.notes
            );
            let (w, h, ca, top) = decode(&read(&ra.path));
            let (_, _, cb, _) = decode(&read(&rb.path));
            let (median, p99, max) = golden(&ca, &cb, w, h, top);
            let s = stats(&ca, &cb);
            ci_note(&format!(
                "{name}, {label}: golden median {median:.4} p99 {p99:.4} max {max:.4}; codes max {} over one {}",
                s.max, s.over_one
            ));
            // The first proposal of gates (docs/17-gpu.md section 12): 0.02 / 0.05 / 0.5.
            // JPEG: the median and the 99th percentile are those, the maximum is not. The DCT and
            // the quantiser turn a one-code difference of the 8-bit input (the PNG 8 of the same
            // variant is within 0.065 of the CPU's) into up to 5 codes in a block, and the ΔE of a
            // dark block moves with them: first measured maxima 1.51 (full size), 0.58 (long edge
            // 500), 0.98 (straightened crop). The bound is twice the largest, and the proposed 0.5 is
            // reported as not met for JPEG (`docs/09`, "As built (export tail on the device)";
            // Questions for Mykyta).
            let max_gate = if label == "JPEG" { 3.0 } else { 0.5 };
            assert!(
                median <= 0.02 && p99 <= 0.05 && max <= max_gate,
                "{name} {label}: {median} {p99} {max}"
            );
        }
    }
}

#[test]
fn the_picture_is_never_a_host_frame() {
    let Some(gpu) = gpu() else { return };
    let shot = Shot::new();
    apply_variant(&shot, 100.0, false);
    let photo = shot.photo();
    let home = tempfile::tempdir().expect("home");
    let out = tempfile::tempdir().expect("out");
    let (ra, rb) = (
        renderer(Some(Arc::clone(&gpu)), RAW, false),
        renderer(Some(Arc::clone(&gpu)), RAW_AND_TAIL, false),
    );
    let (ca, cb) = (context(home.path(), ra), context(home.path(), rb));
    let report_a = export(&ca, &photo, png(out.path(), BitDepth::Sixteen));
    let report_b = export(&cb, &photo, png(out.path(), BitDepth::Sixteen));
    let frame = (report_a.rendered.0 as u64) * u64::from(report_a.rendered.1) * 12;
    let (peak_a, peak_b) = (ca.renderer().ledger().peak(), cb.renderer().ledger().peak());
    ci_note(&format!(
        "host ledger peak: {peak_a} bytes with the CPU frame, {peak_b} with the tail ({frame} bytes of display-linear frame)"
    ));
    // The reservation of the CPU path is the frame plus the working set; the tail's is the working
    // set alone.
    assert_eq!(peak_a - peak_b, frame, "{peak_a} against {peak_b}");
    assert_eq!(report_a.rendered, report_b.rendered);
    assert_eq!(ca.renderer().ledger().used(), 0);
    assert_eq!(cb.renderer().ledger().used(), 0);
}

#[test]
fn the_same_bytes_every_time_and_through_the_queue() {
    use ym_core::export_queue::{ExportEvent, ExportJob, ExportQueue, ExportSink};
    use ym_core::session::ApplyScope;
    let Some(gpu) = gpu() else { return };
    let shot = Shot::new();
    apply_variant(&shot, 100.0, false);
    let photo = shot.photo();
    let home = tempfile::tempdir().expect("home");
    let ctx = Arc::new(context(
        home.path(),
        renderer(Some(Arc::clone(&gpu)), GpuStages::ALL, false),
    ));
    for (label, opts) in [
        ("JPEG", options(Path::new("."), FormatKind::Jpeg)),
        (
            "TIFF 16",
            ExportOptions {
                bit_depth: BitDepth::Sixteen,
                tiff_compression: TiffCompression::Lzw,
                ..options(Path::new("."), FormatKind::Tiff)
            },
        ),
        ("PNG 16", png(Path::new("."), BitDepth::Sixteen)),
    ] {
        let mut files = Vec::new();
        for _ in 0..2 {
            let dir = tempfile::tempdir().expect("dir");
            let mut o = opts.clone();
            o.destination = Some(dir.path().to_path_buf());
            o.size = Size::LongEdge { px: 700 };
            let r = export(&ctx, &photo, o);
            assert_eq!(r.device, ComputeDevice::Gpu, "{label}");
            files.push(read(&r.path));
        }
        assert!(files[0] == files[1], "{label}: two exports differ");
        // The export queue: one job, the same options.
        let (sink, rx) = ExportSink::channel();
        let queue = ExportQueue::new(Arc::clone(&ctx), sink);
        let dir = tempfile::tempdir().expect("dir");
        let mut o = opts.clone();
        o.destination = Some(dir.path().to_path_buf());
        o.size = Size::LongEdge { px: 700 };
        queue
            .start(
                &shot.shared,
                &ExportJob {
                    scope: ApplyScope::This,
                    options: o,
                },
                false,
            )
            .expect("queued");
        let mut written = None;
        while let Ok(event) = rx.recv_timeout(std::time::Duration::from_secs(120)) {
            match event {
                ExportEvent::File(done) => written = Some(done.outcome.expect("exported")),
                ExportEvent::Busy(false) => break,
                _ => {}
            }
        }
        let report = written.expect("the queue exported the photo");
        assert_eq!(report.device, ComputeDevice::Gpu, "{label}");
        assert!(
            read(&report.path) == files[0],
            "{label}: the queue's bytes differ"
        );
    }
}

#[test]
fn progress_never_goes_back_and_a_cancel_between_bands_leaves_nothing() {
    let Some(gpu) = gpu() else { return };
    // 320 x 3072: three bands of 1024 rows (the tail's band is about a megapixel), six chunks of
    // the writers, in twenty-four tiles.
    let shot = Shot::sized(320, 3072);
    apply_variant(&shot, 100.0, false);
    let photo = shot.photo();
    let home = tempfile::tempdir().expect("home");
    let ctx = context(
        home.path(),
        renderer(Some(Arc::clone(&gpu)), GpuStages::ALL, false),
    );
    let baseline = pool_in_use(&gpu);
    // Progress.
    let out = tempfile::tempdir().expect("out");
    let seen = Mutex::new(Vec::new());
    let note = |f: f32| seen.lock().expect("lock").push(f);
    let r = export_photo(
        &ctx,
        &photo,
        &ExportParams::new(png(out.path(), BitDepth::Sixteen)),
        &ExportControl {
            cancel: CancelToken::new(),
            progress: Some(&note),
        },
    )
    .expect("export");
    assert_eq!(r.device, ComputeDevice::Gpu);
    let seen = seen.into_inner().expect("lock");
    assert!(seen.len() >= 3, "{seen:?}");
    assert!(seen.windows(2).all(|w| w[0] <= w[1]), "monotone: {seen:?}");
    assert!((seen.last().copied().unwrap_or(0.0) - 1.0).abs() < 1e-6);
    assert_eq!(
        pool_in_use(&gpu),
        baseline,
        "a finished export hands everything back"
    );
    // A cancel in the middle of the writing.
    for format in [FormatKind::Png, FormatKind::Tiff, FormatKind::Jpeg] {
        let out = tempfile::tempdir().expect("out");
        let token = CancelToken::new();
        let cancel_at = |f: f32| {
            if f >= 0.45 {
                token.cancel();
            }
        };
        let mut o = options(out.path(), format);
        o.bit_depth = BitDepth::Sixteen;
        let err = export_photo(
            &ctx,
            &photo,
            &ExportParams::new(o),
            &ExportControl {
                cancel: token.clone(),
                progress: Some(&cancel_at),
            },
        )
        .expect_err("cancelled");
        assert!(err.is_cancelled(), "{format:?}: {err}");
        assert!(
            names_in(out.path()).is_empty(),
            "{format:?}: a file was left"
        );
        assert_eq!(
            pool_in_use(&gpu),
            baseline,
            "{format:?}: the device buffers of a cancelled export are handed back"
        );
        assert_eq!(ctx.renderer().ledger().used(), 0);
    }
}

/// A look whose file changes while the photo renders: the first answer is `v1`, later ones `v2`.
struct ChangingLook {
    calls: AtomicUsize,
}

impl LookProvider for ChangingLook {
    fn look(&self, id: &LookId) -> Result<ResolvedLook, LookProblem> {
        let n = self.calls.fetch_add(1, Ordering::SeqCst);
        NeutralOnly.look(&LookId::builtin("neutral")).map(|mut l| {
            l.fingerprint = format!("{id}-{}", if n == 0 { "v1" } else { "v2" });
            l
        })
    }

    fn display_transform(
        &self,
        id: &LookId,
    ) -> Result<(ym_color::display::DisplayTransform, String), LookProblem> {
        NeutralOnly.display_transform(id)
    }
}

#[test]
fn a_look_that_changes_while_the_picture_renders_is_not_written() {
    let Some(gpu) = gpu() else { return };
    let shot = Shot::new();
    let r = Renderer::new(
        Arc::new(ChangingLook {
            calls: AtomicUsize::new(0),
        }),
        RenderOptions {
            gpu: Some(gpu),
            gpu_stages: GpuStages::ALL,
            full_tile_px: 256,
            ..RenderOptions::default()
        },
    )
    .expect("render thread");
    let home = tempfile::tempdir().expect("home");
    let ctx = context(home.path(), r);
    let out = tempfile::tempdir().expect("out");
    let err = run(&ctx, &shot.photo(), png(out.path(), BitDepth::Sixteen))
        .expect_err("the look changed under the render");
    assert!(matches!(err, ExportError::Look(_)), "{err}");
    assert!(names_in(out.path()).is_empty(), "no file");
}

/// A look of our own that is shaped like the built-in Signature: a luma curve baked into a 33^3 cube
/// in LogC4, then a CDL saturation of 1.2. Every id resolves to it.
struct CubeLook;

impl LookProvider for CubeLook {
    fn look(&self, id: &LookId) -> Result<ResolvedLook, LookProblem> {
        let lut = ym_color::lut::Lut3d::bake(33, [0.0; 3], [1.0; 3], |code| {
            let y = 0.2126 * code[0] + 0.7152 * code[1] + 0.0722 * code[2];
            // An S curve about the middle of the code range, steeper than the identity.
            let curved = y + 0.35 * (y - 0.5) * (1.0 - (2.0 * (y - 0.5)).abs());
            let k = if y > 1e-6 { curved.max(0.0) / y } else { 1.0 };
            code.map(|c| (c * k).clamp(0.0, 1.0))
        })
        .expect("a cube");
        let transform = ym_color::look::LookTransform::new(
            ym_color::look::LookInput::LogC4,
            Some(ym_color::cdl::Cdl::new([1.0; 3], [0.0; 3], [1.0; 3], 1.2).expect("cdl")),
            Some(ym_color::lut::Lut::from_cube(lut)),
        )
        .expect("a look");
        Ok(ResolvedLook {
            transform: Arc::new(transform),
            fingerprint: format!("{id}"),
        })
    }

    fn display_transform(
        &self,
        id: &LookId,
    ) -> Result<(ym_color::display::DisplayTransform, String), LookProblem> {
        NeutralOnly.display_transform(id)
    }
}

/// The picture the user sees is the export's picture: with a look that is a cube and a CDL (the
/// shape of the default look), a look intensity below 100, a white offset (the white lift's manual
/// part, C12 step 1b), the highlight range at 1 (the camera's clip on the ALEXA's, C7), contrast,
/// saturation and a curve edit, the tail's export is the CPU export. These are
/// stages of the colour chain, which the tail runs from the very chain object the CPU path uses; the
/// test is the end-to-end proof, with the persistent tables of the device (the look's cube is
/// uploaded once for the twelve tiles of the picture, not per tile).
#[test]
fn a_look_a_white_offset_the_range_and_a_curve_come_out_of_the_tail_as_on_the_cpu() {
    let Some(gpu) = gpu() else { return };
    let shot = Shot::new();
    let home = tempfile::tempdir().expect("home");
    // The photo with its default look (Neutral here), as a reference for "the look does something".
    let neutral_dir = tempfile::tempdir().expect("dir");
    let neutral = {
        let ctx = context(home.path(), renderer(None, GpuStages::NONE, false));
        let r = export(
            &ctx,
            &shot.photo(),
            png(neutral_dir.path(), BitDepth::Sixteen),
        );
        decode_png(&read(&r.path)).3
    };
    shot.edit(SettingsPatch {
        look: Some(LookId::parse("user:cube").expect("look id")),
        look_intensity: Some(70.0),
        white_offset_ev: Some(0.4),
        rolloff: Some(RolloffPatch {
            highlights: Some(1.0),
            low_light: None,
        }),
        contrast: Some(ym_core::settings::PercentSetting::manual_value(15.0)),
        saturation: Some(ym_core::settings::PercentSetting::manual_value(10.0)),
        output_sharpening: Some(0.0),
        ..SettingsPatch::default()
    });
    let photo = shot.photo();
    let make = |gpu: Option<Arc<Gpu>>, stages: GpuStages| {
        context(
            home.path(),
            Renderer::new(
                Arc::new(CubeLook),
                RenderOptions {
                    gpu,
                    gpu_stages: stages,
                    full_tile_px: 256,
                    ..RenderOptions::default()
                },
            )
            .expect("render thread"),
        )
    };
    let cpu = make(None, GpuStages::NONE);
    let raw_only = make(Some(Arc::clone(&gpu)), RAW);
    let tail = make(Some(Arc::clone(&gpu)), RAW_AND_TAIL);
    for (label, opts) in [
        (
            "PNG 8",
            (|o: &Path| png(o, BitDepth::Eight)) as fn(&Path) -> ExportOptions,
        ),
        ("PNG 16", |o: &Path| png(o, BitDepth::Sixteen)),
    ] {
        let (a, b, c) = (
            tempfile::tempdir().expect("a"),
            tempfile::tempdir().expect("b"),
            tempfile::tempdir().expect("c"),
        );
        let (ra, rb, rc) = (
            export(&cpu, &photo, opts(a.path())),
            export(&raw_only, &photo, opts(b.path())),
            export(&tail, &photo, opts(c.path())),
        );
        assert_eq!(rc.device, ComputeDevice::Gpu, "{label}: {:?}", rc.notes);
        let (_, _, sixteen, want_cpu) = decode_png(&read(&ra.path));
        let (_, _, _, want_raw) = decode_png(&read(&rb.path));
        let (w, h, _, got) = decode_png(&read(&rc.path));
        let s_tail = stats(&want_raw, &got);
        let s_cpu = stats(&want_cpu, &got);
        ci_note(&format!(
            "{label}, look + white offset + highlight range + curve: tail against the CPU colour pass {} over 1, max {}; against the CPU export {} over 1, max {}",
            s_tail.over_one, s_tail.max, s_cpu.over_one, s_cpu.max
        ));
        if sixteen {
            // The starting C13 bound for 16 bit is 8 codes; first measured on lavapipe is below it.
            assert!(s_tail.max <= 8, "{label}: max {}", s_tail.max);
            // The look is in the picture: Neutral is far from it.
            assert!(
                stats(&neutral, &want_cpu).max > 1000,
                "the look moved the CPU picture by only {} codes",
                stats(&neutral, &want_cpu).max
            );
        } else {
            assert!(s_tail.max <= 1, "{label}: max {}", s_tail.max);
            assert!(
                s_tail.exact as f64 >= 0.99 * s_tail.n as f64,
                "{label}: only {} of {} exact",
                s_tail.exact,
                s_tail.n
            );
        }
        let top = if sixteen { 65535 } else { 255 };
        let (median, p99, max) = golden(&want_cpu, &got, w, h, top);
        assert!(
            median <= 0.02 && p99 <= 0.05 && max <= 0.5,
            "{label}: {median} {p99} {max}"
        );
    }
}

/// A look provider that, the first time the export worker asks it for a look, writes `target`: the
/// file of somebody else that appears after the export decided there was none and before it has
/// pulled a row.
struct FileAppears {
    test_thread: std::thread::ThreadId,
    target: PathBuf,
    created: AtomicUsize,
}

impl LookProvider for FileAppears {
    fn look(&self, id: &LookId) -> Result<ResolvedLook, LookProblem> {
        if std::thread::current().id() != self.test_thread
            && self.created.fetch_add(1, Ordering::SeqCst) == 0
        {
            std::fs::write(&self.target, b"somebody else's file").expect("the other file");
        }
        NeutralOnly.look(&LookId::builtin("neutral")).map(|mut l| {
            l.fingerprint = format!("{id}");
            l
        })
    }

    fn display_transform(
        &self,
        id: &LookId,
    ) -> Result<(ym_color::display::DisplayTransform, String), LookProblem> {
        NeutralOnly.display_transform(id)
    }
}

/// "If the file exists: skip", and the file appears while the device starts the render: the photo is
/// skipped (nothing is written, the other file is left alone), not reported as cancelled.
#[test]
fn a_file_that_appears_before_the_first_row_is_skipped_not_cancelled() {
    let Some(gpu) = gpu() else { return };
    let shot = Shot::new();
    let photo = shot.photo();
    let home = tempfile::tempdir().expect("home");
    let out = tempfile::tempdir().expect("out");
    // The name the export would use.
    let ctx = context(
        home.path(),
        renderer(Some(Arc::clone(&gpu)), GpuStages::ALL, false),
    );
    let first = export(&ctx, &photo, png(out.path(), BitDepth::Eight));
    let target = first.path.clone();
    std::fs::remove_file(&target).expect("remove");
    let provider = Arc::new(FileAppears {
        test_thread: std::thread::current().id(),
        target: target.clone(),
        created: AtomicUsize::new(0),
    });
    let ctx = context(
        home.path(),
        Renderer::new(
            provider.clone(),
            RenderOptions {
                gpu: Some(Arc::clone(&gpu)),
                gpu_stages: GpuStages::ALL,
                full_tile_px: 256,
                ..RenderOptions::default()
            },
        )
        .expect("render thread"),
    );
    let mut o = png(out.path(), BitDepth::Eight);
    o.if_exists = IfExists::Skip;
    let r = run(&ctx, &photo, o).expect("a skipped photo is not an error");
    assert!(
        provider.created.load(Ordering::SeqCst) > 0,
        "the file appeared"
    );
    assert!(r.skipped, "{r:?}");
    assert_eq!(r.bytes, 0);
    assert_eq!(read(&target), b"somebody else's file", "left alone");
    assert_eq!(names_in(out.path()).len(), 1, "no temporary file");
}

fn dng_image(r: &Renderer, src: &PhotoSource, target: FullTarget) -> FullImage {
    r.render_full(src, &FullRequest { target }, &CancelToken::new())
        .expect("render")
}

#[test]
fn the_colour_stage_of_a_dng_runs_on_the_device_and_agrees_with_the_cpu() {
    let Some(gpu) = gpu() else { return };
    let shot = Shot::new();
    let src = shot.d.source();
    let raw_only = renderer(Some(Arc::clone(&gpu)), RAW, false);
    let with_tail = renderer(Some(Arc::clone(&gpu)), RAW_AND_TAIL, false);
    for target in [FullTarget::DngLinearAwg4, FullTarget::DngPreview] {
        let a = dng_image(&raw_only, &src, target);
        let b = dng_image(&with_tail, &src, target);
        assert_eq!(
            b.report.device,
            ComputeDevice::Gpu,
            "{target:?}: {:?}",
            b.report.notes
        );
        assert_ne!(a.report.device, ComputeDevice::Gpu, "{target:?}");
        assert_eq!((a.width, a.height), (b.width, b.height));
        let mut worst = 0.0f64;
        for (&x, &y) in a.rgb.iter().zip(&b.rgb) {
            assert!(y.is_finite(), "{target:?}: a sample is not finite");
            worst = worst.max(f64::from((x - y).abs()) / f64::from(x.abs().max(1.0)));
        }
        ci_note(&format!(
            "DNG colour stage ({target:?}) on the device against the CPU: worst relative difference {worst:e}"
        ));
        // First measured: 2.3e-6 (linear AWG4) and 3.5e-6 (preview); the bound is twice the larger.
        assert!(worst <= 7e-6, "{target:?}: {worst:e}");
    }
}

/// A picture of 8-bit codes as a `BandSource`: what a JPEG writer would be handed after C13.
struct Codes8 {
    width: u32,
    height: u32,
    rgb: Vec<u8>,
}

impl ym_export::pixels::BandSource for Codes8 {
    fn width(&self) -> u32 {
        self.width
    }

    fn height(&self) -> u32 {
        self.height
    }

    fn rows_u8(
        &mut self,
        y0: usize,
        rows: usize,
        out: &mut Vec<u8>,
    ) -> Result<(), ym_export::ExportError> {
        let n = self.width as usize * 3;
        out.clear();
        out.extend_from_slice(&self.rgb[y0 * n..(y0 + rows) * n]);
        Ok(())
    }

    fn rows_u16(
        &mut self,
        _: usize,
        _: usize,
        _: &mut Vec<u16>,
    ) -> Result<(), ym_export::ExportError> {
        Err(ym_export::ExportError::Unsupported("8-bit codes".into()))
    }
}

/// The decoded pixels of the JPEG that the writer makes of `codes` (8-bit RGB, rows of `width`).
fn jpeg_of_codes(codes: &[u32], width: usize, height: usize) -> Vec<u32> {
    use ym_export::jpeg::{JpegMetadata, write_jpeg_bands};
    let mut bands = Codes8 {
        width: width as u32,
        height: height as u32,
        rgb: codes.iter().map(|&c| c as u8).collect(),
    };
    let quality = match ExportOptions::default().format_spec() {
        ym_export::ExportFormat::Jpeg { quality } => quality,
        other => panic!("the default format is not a JPEG: {other:?}"),
    };
    let mut file = Vec::new();
    write_jpeg_bands(
        &mut file,
        &mut bands,
        quality,
        &JpegMetadata {
            exif_app1: None,
            xmp: None,
            icc: None,
        },
        &ym_export::ExportHooks::none(),
    )
    .expect("a JPEG of the codes");
    decode_jpeg(&file).2
}

/// The JPEG gap to the CPU export (maximum ΔE 1.5 against the proposed 0.5, `docs/09`) is the quantiser
/// and nothing else: the JPEG of the device export is, pixel for pixel, the JPEG of the 8-bit codes
/// of the device's PNG, and the CPU's the JPEG of the CPU's codes. So the picture the device makes
/// differs from the CPU's by the one-code differences of the PNG 8 (maximum ΔE 0.06), and the DCT
/// and the quantiser turn those into the gap.
#[test]
fn the_jpeg_of_an_export_is_the_jpeg_of_its_8_bit_codes() {
    let Some(gpu) = gpu() else { return };
    let shot = Shot::new();
    let home = tempfile::tempdir().expect("home");
    let cpu = context(home.path(), renderer(None, GpuStages::NONE, false));
    let dev = context(home.path(), renderer(Some(gpu), GpuStages::ALL, false));
    for (i, (name, pct, size, straighten)) in variants().into_iter().enumerate() {
        if ![0, 2, 4].contains(&i) {
            continue;
        }
        apply_variant(&shot, pct, straighten);
        let photo = shot.photo();
        let mut gaps = Vec::new();
        for (who, ctx) in [("CPU", &cpu), ("device", &dev)] {
            let dir = tempfile::tempdir().expect("dir");
            let mut o = png(dir.path(), BitDepth::Eight);
            o.size = size;
            let r = export(ctx, &photo, o);
            let (w, h, _, codes) = decode_png(&read(&r.path));
            let mut o = options(dir.path(), FormatKind::Jpeg);
            o.size = size;
            let j = export(ctx, &photo, o);
            let (jw, jh, from_file) = decode_jpeg(&read(&j.path));
            assert_eq!((jw, jh), (w, h), "{name} {who}");
            let from_codes = jpeg_of_codes(&codes, w, h);
            assert!(
                from_file == from_codes,
                "{name} {who}: the JPEG is not the JPEG of the PNG 8's codes ({} samples differ)",
                from_file
                    .iter()
                    .zip(&from_codes)
                    .filter(|(a, b)| a != b)
                    .count()
            );
            gaps.push((codes, from_file, w, h));
        }
        let (cpu_codes, cpu_jpeg, w, h) = &gaps[0];
        let (dev_codes, dev_jpeg, _, _) = &gaps[1];
        let (png_median, png_p99, png_max) = golden(cpu_codes, dev_codes, *w, *h, 255);
        let (jpeg_median, jpeg_p99, jpeg_max) = golden(cpu_jpeg, dev_jpeg, *w, *h, 255);
        ci_note(&format!(
            "{name}: the PNG 8 differs by median {png_median:.4} p99 {png_p99:.4} max {png_max:.4}; the JPEG of the same codes by median {jpeg_median:.4} p99 {jpeg_p99:.4} max {jpeg_max:.4}"
        ));
        assert!(png_max <= 0.5 && jpeg_median <= 0.02 && jpeg_p99 <= 0.05);
    }
}

/// The estimate reads the numbers of the path the renderer takes (the tail on the device, or the CPU
/// reference with the raw chain wherever it runs), so an export must teach the set of that path:
/// an export whose raw chain ran on the device and whose colour pass and writer ran on the CPU (what
/// `Auto` does on a Mac) is the CPU reference's, and the estimate that reads it must learn from it.
#[test]
fn an_export_teaches_the_numbers_of_the_path_the_estimate_reads() {
    let Some(gpu) = gpu() else { return };
    let shot = Shot::new();
    let photo = shot.photo();
    let home = tempfile::tempdir().expect("home");
    let prior = ym_core::export::RENDER_S_PER_MP_PRIOR;
    for (name, stages, tail) in [
        ("the raw chain alone on the device", RAW, false),
        ("the export tail", RAW_AND_TAIL, true),
    ] {
        let ctx = context(home.path(), renderer(Some(Arc::clone(&gpu)), stages, false));
        let out = tempfile::tempdir().expect("out");
        let r = export(&ctx, &photo, png(out.path(), BitDepth::Eight));
        assert_eq!(r.device, ComputeDevice::Gpu, "{name}: {:?}", r.notes);
        let speed = ctx.speed();
        assert!(speed.is_measured(), "{name}");
        // The numbers of the path that made the file moved, the other path's did not.
        let (taken, other) = if tail {
            (ComputeDevice::Gpu, ComputeDevice::Cpu)
        } else {
            (ComputeDevice::Cpu, ComputeDevice::Gpu)
        };
        assert!(
            (speed.render_s_per_mp(other) - prior).abs() < 1e-12
                && speed.combined_s_per_mp(other, FormatKind::Png).is_none(),
            "{name}: the other path's numbers moved to {}",
            speed.render_s_per_mp(other)
        );
        // The CPU reference learns its render speed; the tail learns the time of the whole export of
        // that format (the writer pulls while the device renders), and only of that format.
        let learned = if tail {
            speed.combined_s_per_mp(taken, FormatKind::Png).is_some()
                && speed.combined_s_per_mp(taken, FormatKind::Jpeg).is_none()
                && speed.combined_s_per_mp(taken, FormatKind::Tiff).is_none()
        } else {
            (speed.render_s_per_mp(taken) - prior).abs() > 1e-9
        };
        assert!(
            learned,
            "{name}: the path that made the file learned nothing ({})",
            speed.render_s_per_mp(taken)
        );
    }
}

/// Resets the kernel's peak resident-set mark of this process (Linux), so that a later read shows
/// the peak of the work in between.
fn reset_peak_rss() {
    let _ = std::fs::write("/proc/self/clear_refs", "5");
}

/// (resident set now, peak resident set since the last reset) in MiB, on Linux.
fn rss_mib() -> Option<(u64, u64)> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    let kib = |key: &str| -> Option<u64> {
        status
            .lines()
            .find(|l| l.starts_with(key))?
            .split_whitespace()
            .nth(1)?
            .parse()
            .ok()
    };
    Some((kib("VmRSS:")? >> 10, kib("VmHWM:")? >> 10))
}

/// 50 MP (8688 x 5792) through the export queue's code path: the tail never holds the picture on the
/// host, a DNG still does. Records the ledger's peak (the reservation the export makes) and the
/// process's resident set (what the operating system saw) for each. Ignored: it writes a 100 MB raw
/// file, and renders 50 MP three times (minutes on lavapipe); the Metal CI runs it
/// (`.github/workflows/mac-gpu.yml`).
#[test]
#[ignore = "50 MP: run with --release -- --ignored (the Metal CI runs it)"]
fn fifty_megapixels_leave_no_host_frame_and_a_dng_keeps_its_own() {
    let Some(gpu) = gpu() else { return };
    let white = light("sony-ilce-7rm4", 4500.0);
    let dir = tempfile::tempdir().expect("dir");
    {
        let scene = Scene::new(8688, 5792, white, Layout::Patchwork, 0.5).noise(2e-4, 2e-6);
        scene.write(dir.path(), "DSC00001.ARW", MODEL);
    }
    let session = Session::new(base_env());
    let opened = session.open(&[dir.path().to_path_buf()], OpenOptions::default());
    assert_eq!(opened.added.len(), 1);
    settle(&session);
    let photo = ExportPhoto::from_session(&session, opened.added[0]).expect("export photo");
    let home = tempfile::tempdir().expect("home");
    let frame = 8688u64 * 5792 * 12;
    let mut results = Vec::new();
    for (label, stages, options) in [
        (
            "PNG 16, tail",
            GpuStages::ALL,
            png(Path::new("."), BitDepth::Sixteen),
        ),
        (
            "PNG 16, host frame",
            RAW,
            png(Path::new("."), BitDepth::Sixteen),
        ),
        (
            "JPEG, long edge 4000, tail",
            GpuStages::ALL,
            ExportOptions {
                size: Size::LongEdge { px: 4000 },
                ..options(Path::new("."), FormatKind::Jpeg)
            },
        ),
        (
            "DNG",
            GpuStages::ALL,
            options(Path::new("."), FormatKind::Dng),
        ),
    ] {
        let ctx = context(home.path(), renderer(Some(Arc::clone(&gpu)), stages, false));
        let out = tempfile::tempdir().expect("out");
        let mut o = options;
        o.destination = Some(out.path().to_path_buf());
        reset_peak_rss();
        let before = rss_mib();
        let started = std::time::Instant::now();
        let r = export(&ctx, &photo, o);
        let seconds = started.elapsed().as_secs_f64();
        let after = rss_mib();
        let peak = ctx.renderer().ledger().peak();
        ci_note(&format!(
            "50 MP {label}: {seconds:.0} s, {:?} ({:?}); ledger peak {} MiB (the display-linear frame is {} MiB); resident set {:?} MiB before, peak {:?} MiB",
            r.device,
            r.notes,
            peak >> 20,
            frame >> 20,
            before.map(|b| b.0),
            after.map(|a| a.1),
        ));
        assert_eq!(ctx.renderer().ledger().used(), 0);
        results.push((label, r.device, peak));
    }
    // The tail's reservation is the host path's less the frame.
    assert_eq!(results[0].1, ComputeDevice::Gpu);
    assert_eq!(
        results[1].2 - results[0].2,
        frame,
        "{} against {}",
        results[1].2,
        results[0].2
    );
    // The tail holds well under one frame; the DNG's host frames are still there.
    assert!(
        results[0].2 < frame,
        "tail peak {} of a {frame} B frame",
        results[0].2
    );
    assert!(results[3].2 >= frame, "the DNG keeps its host frame");
}

#[cfg(feature = "fault-injection")]
mod faults {
    use super::*;
    use ym_color::display::OutputTarget;
    use ym_gpu::device::GpuFault;

    /// The CPU export and the files of a device export that failed over to the CPU.
    struct Pair {
        cpu_bytes: Vec<u8>,
    }

    fn cpu_export(shot: &Shot, home: &Path, format: FormatKind) -> Pair {
        let ctx = context(home, renderer(None, GpuStages::NONE, false));
        let out = tempfile::tempdir().expect("out");
        let mut o = options(out.path(), format);
        o.bit_depth = BitDepth::Sixteen;
        o.size = Size::LongEdge { px: 600 };
        let r = export(&ctx, &shot.photo(), o);
        Pair {
            cpu_bytes: read(&r.path),
        }
    }

    #[test]
    fn a_device_error_makes_the_whole_photo_again_on_the_cpu_bit_for_bit() {
        let Some(gpu) = gpu() else { return };
        let shot = Shot::new();
        apply_variant(&shot, 100.0, false);
        let home = tempfile::tempdir().expect("home");
        let pairs = [
            (
                FormatKind::Png,
                cpu_export(&shot, home.path(), FormatKind::Png),
            ),
            (
                FormatKind::Tiff,
                cpu_export(&shot, home.path(), FormatKind::Tiff),
            ),
        ];
        // `stages`: the tail alone (the raw chain is on the CPU, the tail fails at its first job)
        // and everything (the raw chain fails first).
        for (stages_name, stages) in [
            ("the tail alone", GpuStages::EXPORT_TAIL),
            ("every stage", GpuStages::ALL),
        ] {
            for fault in [GpuFault::Shader, GpuFault::OutOfBudget, GpuFault::Lost] {
                for (format, pair) in &pairs {
                    // A device that is lost stays lost: each case gets a device of its own.
                    let own = if matches!(fault, GpuFault::Lost) {
                        Arc::new(Gpu::start(GpuConfig::default()).expect("device"))
                    } else {
                        Arc::clone(&gpu)
                    };
                    let ctx = context(home.path(), renderer(Some(Arc::clone(&own)), stages, false));
                    let baseline = pool_in_use(&own);
                    own.set_fault(Some(fault));
                    let out = tempfile::tempdir().expect("out");
                    let mut o = options(out.path(), *format);
                    o.bit_depth = BitDepth::Sixteen;
                    o.size = Size::LongEdge { px: 600 };
                    let seen = Mutex::new(Vec::new());
                    let note = |f: f32| seen.lock().expect("lock").push(f);
                    let r = export_photo(
                        &ctx,
                        &shot.photo(),
                        &ExportParams::new(o),
                        &ExportControl {
                            cancel: CancelToken::new(),
                            progress: Some(&note),
                        },
                    );
                    own.set_fault(None);
                    let r = r.unwrap_or_else(|e| panic!("{stages_name} {fault:?} {format:?}: {e}"));
                    assert_eq!(r.device, ComputeDevice::Cpu, "{stages_name} {fault:?}");
                    assert!(
                        r.notes.iter().any(|n| n.contains(
                            "The graphics card could not render this photo, so the CPU did"
                        )),
                        "{stages_name} {fault:?}: {:?}",
                        r.notes
                    );
                    assert!(
                        read(&r.path) == pair.cpu_bytes,
                        "{stages_name} {fault:?} {format:?}: the fallback is not the CPU export"
                    );
                    assert_eq!(names_in(out.path()).len(), 1, "one file, no temporary left");
                    let seen = seen.into_inner().expect("lock");
                    assert!(seen.windows(2).all(|w| w[0] <= w[1]), "monotone: {seen:?}");
                    if !matches!(fault, GpuFault::Lost) {
                        assert_eq!(pool_in_use(&own), baseline, "{stages_name} {fault:?}");
                    }
                }
            }
        }
    }

    /// How a device fails once the writer has taken rows.
    #[derive(Clone, Copy, Debug)]
    enum Part {
        /// The `n`th tile of the raw chain fails.
        Tile(u64),
        /// The device is lost at the first job after the writer has taken this share of the rows.
        LostAfter(f32),
    }

    #[test]
    fn a_device_that_fails_part_way_is_made_again_on_the_cpu_bit_for_bit() {
        let Some(gpu) = gpu() else { return };
        // 96 x 8192 in 256 px tiles is thirty-two tile rows; sharpened, the picture comes out in
        // eight bands of 1024 rows, and the band channel and the writer's own buffer hold five of
        // them at the most. A writer that stalls in its progress callback (below) therefore has the
        // render stopped on a full channel with work left, whatever the speed of the device, when it
        // takes the first third of the rows: the fault always finds a job to fail, after the
        // writer has taken rows and compressed them.
        let shot = Shot::sized(96, 8192);
        apply_variant(&shot, 100.0, false);
        let home = tempfile::tempdir().expect("home");
        drop(gpu);
        let sixteen = |out: &Path, format: FormatKind| {
            let mut o = options(out, format);
            o.bit_depth = BitDepth::Sixteen;
            o
        };
        for format in [FormatKind::Png, FormatKind::Tiff, FormatKind::Jpeg] {
            let cpu_bytes = {
                let ctx = context(home.path(), renderer(None, GpuStages::NONE, false));
                let out = tempfile::tempdir().expect("out");
                read(&export(&ctx, &shot.photo(), sixteen(out.path(), format)).path)
            };
            // The JPEG writer reports half of its progress while it collects the rows.
            let rows_to_progress = if format == FormatKind::Jpeg { 0.5 } else { 1.0 };
            for (stages_name, stages, part) in [
                ("every stage", GpuStages::ALL, Part::Tile(20)),
                ("every stage", GpuStages::ALL, Part::LostAfter(0.2)),
                (
                    "the tail alone",
                    GpuStages::EXPORT_TAIL,
                    Part::LostAfter(0.2),
                ),
            ] {
                // The TIFF writer takes a picture this narrow in one batch (16 strips of 910 rows)
                // and reports its progress when all of it is written: no progress to fail after.
                if format == FormatKind::Tiff && matches!(part, Part::LostAfter(_)) {
                    continue;
                }
                let own = Arc::new(
                    Gpu::start(GpuConfig {
                        poison: true,
                        ..GpuConfig::default()
                    })
                    .expect("device"),
                );
                let ctx = context(home.path(), renderer(Some(Arc::clone(&own)), stages, false));
                let baseline = pool_in_use(&own);
                if let Part::Tile(n) = part {
                    own.set_fault(Some(GpuFault::FailTile(n)));
                }
                let out = tempfile::tempdir().expect("out");
                let seen = Mutex::new(Vec::new());
                let note = |f: f32| {
                    seen.lock().expect("lock").push(f);
                    if let Part::LostAfter(share) = part {
                        // A slow writer: the render runs ahead until the channel is full.
                        std::thread::sleep(std::time::Duration::from_millis(15));
                        if f >= share * rows_to_progress {
                            own.set_fault(Some(GpuFault::Lost));
                        }
                    }
                };
                let r = export_photo(
                    &ctx,
                    &shot.photo(),
                    &ExportParams::new(sixteen(out.path(), format)),
                    &ExportControl {
                        cancel: CancelToken::new(),
                        progress: Some(&note),
                    },
                );
                own.set_fault(None);
                let what = format!("{format:?}, {stages_name}, {part:?}");
                let r = r.unwrap_or_else(|e| panic!("{what}: {e}"));
                assert_eq!(r.device, ComputeDevice::Cpu, "{what}: {:?}", r.notes);
                assert!(
                    r.notes.iter().any(|n| n
                        .contains("The graphics card could not render this photo, so the CPU did")),
                    "{what}: {:?}",
                    r.notes
                );
                assert!(
                    read(&r.path) == cpu_bytes,
                    "{what}: the fallback is not the CPU export"
                );
                assert_eq!(
                    names_in(out.path()).len(),
                    1,
                    "{what}: one file, no temporary"
                );
                let seen = seen.into_inner().expect("lock");
                assert!(
                    seen.windows(2).all(|w| w[0] <= w[1]),
                    "{what}: monotone {seen:?}"
                );
                if let Part::LostAfter(share) = part {
                    assert!(
                        seen.iter().any(|&f| f >= share * rows_to_progress),
                        "{what}: the writer never got as far as the fault: {seen:?}"
                    );
                } else {
                    // The device is usable again and everything was handed back.
                    assert_eq!(pool_in_use(&own), baseline, "{what}");
                }
                assert_eq!(ctx.renderer().ledger().used(), 0, "{what}");
            }
        }
    }

    #[test]
    fn a_strict_policy_fails_the_photo_and_leaves_no_file() {
        let Some(gpu) = gpu() else { return };
        let shot = Shot::new();
        let home = tempfile::tempdir().expect("home");
        let ctx = context(
            home.path(),
            renderer(Some(Arc::clone(&gpu)), GpuStages::ALL, true),
        );
        gpu.set_fault(Some(GpuFault::Shader));
        let out = tempfile::tempdir().expect("out");
        let err = run(&ctx, &shot.photo(), png(out.path(), BitDepth::Sixteen));
        gpu.set_fault(None);
        let err = err.expect_err("strict");
        assert_eq!(err.kind(), "gpu", "{err}");
        assert!(names_in(out.path()).is_empty());
    }

    #[test]
    fn a_dng_whose_colour_stage_fails_on_the_device_is_made_again_on_the_cpu() {
        let Some(gpu) = gpu() else { return };
        let shot = Shot::new();
        let src = shot.d.source();
        let cpu = renderer(None, GpuStages::NONE, false);
        let want = dng_image(&cpu, &src, FullTarget::DngLinearAwg4);
        let dev = renderer(Some(Arc::clone(&gpu)), GpuStages::EXPORT_TAIL, false);
        gpu.set_fault(Some(GpuFault::Shader));
        let got = dng_image(&dev, &src, FullTarget::DngLinearAwg4);
        gpu.set_fault(None);
        assert_eq!(got.report.device, ComputeDevice::Cpu);
        assert!(
            got.report.notes.iter().any(
                |n| n.contains("The graphics card could not render this photo, so the CPU did")
            ),
            "{:?}",
            got.report.notes
        );
        assert!(got.rgb == want.rgb, "the fallback is the CPU render");
        // A strict policy fails it.
        let strict = renderer(Some(Arc::clone(&gpu)), GpuStages::EXPORT_TAIL, true);
        gpu.set_fault(Some(GpuFault::Shader));
        let err = strict.render_full(
            &src,
            &FullRequest {
                target: FullTarget::DngLinearAwg4,
            },
            &CancelToken::new(),
        );
        gpu.set_fault(None);
        assert!(
            matches!(err, Err(ym_core::pipeline::RenderError::Gpu(_))),
            "{err:?}"
        );
    }

    /// A device of its own, with nothing compiled yet: a pipeline that an earlier test compiled
    /// would not meet the injected shader fault.
    fn fresh_device() -> Arc<Gpu> {
        Arc::new(
            Gpu::start(GpuConfig {
                poison: true,
                ..GpuConfig::default()
            })
            .expect("device"),
        )
    }

    /// `docs/17-gpu.md` section 9: a shader of the tail that does not compile is a property of the machine,
    /// not of the photo, so the tail is off for the rest of the process. The photo that met it was
    /// made again on the CPU (the fault test above); the next one must not try the shader again.
    #[test]
    fn a_shader_that_fails_in_the_tail_switches_the_tail_off_for_the_process() {
        let Some(_) = gpu() else { return };
        let shot = Shot::new();
        apply_variant(&shot, 100.0, false);
        let home = tempfile::tempdir().expect("home");
        let own = fresh_device();
        let sixteen = |out: &Path| {
            let mut o = options(out, FormatKind::Png);
            o.bit_depth = BitDepth::Sixteen;
            o.size = Size::LongEdge { px: 600 };
            o
        };
        let cpu_bytes = {
            let out = tempfile::tempdir().expect("out");
            let ctx = context(home.path(), renderer(None, GpuStages::NONE, false));
            read(&export(&ctx, &shot.photo(), sixteen(out.path())).path)
        };
        // The raw chain is on the CPU, so the only shader that can fail is the tail's.
        let ctx = context(
            home.path(),
            renderer(Some(Arc::clone(&own)), GpuStages::EXPORT_TAIL, false),
        );
        assert!(ctx.renderer().export_tail_enabled());
        assert!(!ctx.renderer().export_tail_disabled());
        own.set_fault(Some(GpuFault::Shader));
        let out = tempfile::tempdir().expect("out");
        let first = export(&ctx, &shot.photo(), sixteen(out.path()));
        own.set_fault(None);
        assert_eq!(first.device, ComputeDevice::Cpu);
        assert!(
            first.notes.iter().any(
                |n| n.contains("The graphics card could not render this photo, so the CPU did")
            ),
            "{:?}",
            first.notes
        );
        assert!(
            read(&first.path) == cpu_bytes,
            "the fallback is the CPU export"
        );
        assert!(ctx.renderer().export_tail_disabled(), "the tail is off now");
        assert!(!ctx.renderer().export_tail_enabled());
        // The fault is gone and the shader would compile, but the tail stays off: the second photo
        // is a plain CPU export, the device is not asked, and no note says the card failed again.
        let asked = own.counters().submissions;
        let out2 = tempfile::tempdir().expect("out");
        let second = export(&ctx, &shot.photo(), sixteen(out2.path()));
        assert_eq!(second.device, ComputeDevice::Cpu);
        assert!(
            !second.notes.iter().any(|n| n.contains("graphics card")),
            "{:?}",
            second.notes
        );
        assert!(read(&second.path) == cpu_bytes);
        assert_eq!(
            own.counters().submissions,
            asked,
            "the device was not asked"
        );
        // The flag is the renderer's: another renderer on the same (healthy) device runs the tail.
        let again = context(
            home.path(),
            renderer(Some(Arc::clone(&own)), GpuStages::EXPORT_TAIL, false),
        );
        let out3 = tempfile::tempdir().expect("out");
        let third = export(&again, &shot.photo(), sixteen(out3.path()));
        assert_eq!(third.device, ComputeDevice::Gpu, "{:?}", third.notes);

        // A strict policy never turns the switched-off tail into a quiet CPU export.
        let own = fresh_device();
        let strict = context(
            home.path(),
            renderer(Some(Arc::clone(&own)), GpuStages::EXPORT_TAIL, true),
        );
        own.set_fault(Some(GpuFault::Shader));
        let out4 = tempfile::tempdir().expect("out");
        let err = run(&strict, &shot.photo(), sixteen(out4.path())).expect_err("strict");
        own.set_fault(None);
        assert_eq!(err.kind(), "gpu", "{err}");
        assert!(strict.renderer().export_tail_disabled());
        let out5 = tempfile::tempdir().expect("out");
        let err = run(&strict, &shot.photo(), sixteen(out5.path())).expect_err("refused");
        assert_eq!(err.kind(), "gpu", "{err}");
        assert!(err.to_string().contains("earlier error"), "{err}");
        assert!(names_in(out4.path()).is_empty() && names_in(out5.path()).is_empty());
    }

    /// The photo's graph forgets everything cached for a photo whose fingerprint changed (a second
    /// session, a file replaced), the raw words on the device included. The same pixels under another
    /// fingerprint are how a test takes the words away from an export between two of its tiles
    /// without touching the export itself (`Renderer::final_size` makes a request that adopts it and
    /// uploads nothing).
    ///
    /// The words go, and the next upload of them (the export's next tile, which finds none) is the
    /// one that fails when `fail_upload` is set. The fault is armed **before** the words are taken,
    /// never after: when the progress report comes from a thread other than the render's (the
    /// writer's, in an export of the tail), the render keeps tiling while this runs, and the two
    /// requests below take tens of milliseconds, longer than a tile on a quick device. A tile that
    /// found the words gone in that gap would upload them again with nothing armed, and the fault
    /// armed afterwards would never meet an upload (the export would finish on the device). Armed
    /// first, the first upload that follows the loss fails, whichever tile makes it.
    fn take_the_raw_words_away(renderer: &Renderer, src: &PhotoSource, fail_upload: Option<&Gpu>) {
        if let Some(gpu) = fail_upload {
            gpu.set_fault(Some(GpuFault::FailUpload(1)));
        }
        let mut other = src.clone();
        other.fingerprint ^= 1;
        let cancel = CancelToken::new();
        renderer
            .final_size(&other, &cancel)
            .expect("a request for the same photo under another fingerprint");
        // And back under its own: the graph now holds nothing, and is for this photo again, so the
        // export's next tile uploads the words once and keeps them.
        renderer.final_size(src, &cancel).expect("the photo again");
    }

    /// The guard of `Job::raw_route`: tiles of an export have run the raw front on the device, and
    /// then the photo's words are gone and cannot be put back (`upload@1`: no room). The rest of the
    /// file must not be made behind the CPU's front, nor on a device that has just failed to hold the
    /// photo: the export starts again, wholly on the CPU, and says why. Without the guard the tiles
    /// that follow run R6..R8 on the device and the file is neither the device's nor the CPU's.
    #[test]
    fn an_export_whose_raw_words_cannot_be_uploaded_again_is_made_again_on_the_cpu() {
        let Some(gpu) = gpu() else { return };
        drop(gpu);
        let shot = Shot::new();
        let src = shot.d.source();
        let cpu = renderer(None, GpuStages::NONE, false);
        let want = dng_image(&cpu, &src, FullTarget::DisplayLinear(OutputTarget::Srgb));
        let stages = GpuStages::RAW_CHAIN.union(GpuStages::RAW_FRONT);
        for (name, fail) in [("the upload fails", true), ("the upload works", false)] {
            let own = fresh_device();
            let dev = renderer(Some(Arc::clone(&own)), stages, false);
            let baseline = pool_in_use(&own);
            let tiles = Arc::new(AtomicUsize::new(0));
            let progress: Arc<dyn Fn(f32) + Send + Sync> = {
                let (tiles, own, dev, src) = (
                    Arc::clone(&tiles),
                    Arc::clone(&own),
                    dev.clone(),
                    src.clone(),
                );
                Arc::new(move |_| {
                    // After the third of the twelve tiles (they run in raster order), on the render's
                    // own thread: nothing can be between this report and the next tile.
                    if tiles.fetch_add(1, Ordering::SeqCst) + 1 == 3 {
                        take_the_raw_words_away(&dev, &src, fail.then_some(&*own));
                    }
                })
            };
            let got = dev
                .spawn_full_reporting(
                    src.clone(),
                    FullRequest {
                        target: FullTarget::DisplayLinear(OutputTarget::Srgb),
                    },
                    progress,
                )
                .wait()
                .expect("the job resolved")
                .expect("the render");
            own.set_fault(None);
            let notes = &got.report.notes;
            let note = "The graphics card could not render this photo, so the CPU did";
            if fail {
                assert!(
                    got.rgb == want.rgb,
                    "{name}: the picture is not the CPU reference"
                );
                assert!(notes.iter().any(|n| n.contains(note)), "{name}: {notes:?}");
                assert!(
                    got.report
                        .ranges
                        .iter()
                        .all(|r| r.device == ComputeDevice::Cpu),
                    "{name}: {:?}",
                    got.report.ranges
                );
            } else {
                // The control: the words are uploaded again and the export finishes on the device.
                assert!(!notes.iter().any(|n| n.contains(note)), "{name}: {notes:?}");
                assert!(
                    !got.report.ranges.is_empty()
                        && got
                            .report
                            .ranges
                            .iter()
                            .all(|r| r.device == ComputeDevice::Gpu),
                    "{name}: {:?}",
                    got.report.ranges
                );
                assert_eq!(got.report.runs.raw_uploads, 2, "the words went up twice");
            }
            assert!(tiles.load(Ordering::SeqCst) >= 12, "{name}");
            assert_eq!(pool_in_use(&own), baseline, "{name}: the buffers are back");
            assert_eq!(dev.ledger().used(), 0, "{name}");
        }
    }

    /// The same guard through an export of the tail: the writer takes a share of the rows, the
    /// words go, the upload fails; the whole file is the CPU export, byte for byte, with the note.
    ///
    /// The words are taken from the writer's thread (its progress report) while the render goes on
    /// on its own, and how far the render has got when the report comes is no matter of the speed of
    /// the device: the channel of bands holds it back. The picture is thirty-two tile rows and eight
    /// bands of 1024 rows. The first report at or above 0.2 is the one after row 2048: the writer
    /// has taken bands 0 and 1 and is inside the callback, so it takes no more until the callback
    /// returns. The channel holds three bands (2 to 4) and the render blocks sending the fifth (band
    /// 5 needs the rows up to 6144 and the sharpening's halo: twenty-five tile rows at the most), so
    /// at least seven tile rows are left, on a device as quick as Metal and on a slow one alike.
    /// Each of them starts by asking for the photo's words, and the first upload after they went
    /// fails (the fault is armed before they go, see `take_the_raw_words_away`).
    #[test]
    fn a_tail_export_whose_raw_words_cannot_be_uploaded_again_is_the_cpu_export() {
        let Some(gpu) = gpu() else { return };
        drop(gpu);
        let shot = Shot::sized(96, 8192);
        apply_variant(&shot, 100.0, false);
        let src = shot.d.source();
        let home = tempfile::tempdir().expect("home");
        let sixteen = |out: &Path| {
            let mut o = options(out, FormatKind::Png);
            o.bit_depth = BitDepth::Sixteen;
            o
        };
        let cpu_bytes = {
            let ctx = context(home.path(), renderer(None, GpuStages::NONE, false));
            let out = tempfile::tempdir().expect("out");
            read(&export(&ctx, &shot.photo(), sixteen(out.path())).path)
        };
        let own = fresh_device();
        let ctx = context(
            home.path(),
            renderer(Some(Arc::clone(&own)), GpuStages::ALL, false),
        );
        let baseline = pool_in_use(&own);
        let out = tempfile::tempdir().expect("out");
        let seen = Mutex::new(Vec::new());
        let taken = AtomicUsize::new(0);
        let note = |f: f32| {
            seen.lock().expect("lock").push(f);
            if f >= 0.2 && taken.fetch_add(1, Ordering::SeqCst) == 0 {
                take_the_raw_words_away(ctx.renderer(), &src, Some(&own));
            }
        };
        let r = export_photo(
            &ctx,
            &shot.photo(),
            &ExportParams::new(sixteen(out.path())),
            &ExportControl {
                cancel: CancelToken::new(),
                progress: Some(&note),
            },
        );
        own.set_fault(None);
        let r = r.expect("the export");
        assert_eq!(r.device, ComputeDevice::Cpu, "{:?}", r.notes);
        assert!(
            r.notes.iter().any(
                |n| n.contains("The graphics card could not render this photo, so the CPU did")
            ),
            "{:?}",
            r.notes
        );
        assert!(
            read(&r.path) == cpu_bytes,
            "the fallback is not the CPU export"
        );
        assert_eq!(names_in(out.path()).len(), 1, "one file, no temporary");
        let seen = seen.into_inner().expect("lock");
        assert!(seen.windows(2).all(|w| w[0] <= w[1]), "monotone: {seen:?}");
        assert!(
            taken.load(Ordering::SeqCst) > 0,
            "the words were taken away"
        );
        assert_eq!(pool_in_use(&own), baseline);
        assert_eq!(ctx.renderer().ledger().used(), 0);
    }

    /// The DNG's colour stage runs on the tail's kernels: a shader that fails there switches the
    /// tail off as well, and the next DNG render makes its colour stage on the CPU without trying.
    #[test]
    fn a_shader_that_fails_in_a_dngs_colour_stage_switches_the_tail_off_too() {
        let Some(_) = gpu() else { return };
        let shot = Shot::new();
        let src = shot.d.source();
        let cpu = renderer(None, GpuStages::NONE, false);
        let want = dng_image(&cpu, &src, FullTarget::DngLinearAwg4);
        let own = fresh_device();
        let dev = renderer(Some(Arc::clone(&own)), GpuStages::EXPORT_TAIL, false);
        own.set_fault(Some(GpuFault::Shader));
        let first = dng_image(&dev, &src, FullTarget::DngLinearAwg4);
        own.set_fault(None);
        assert_eq!(first.report.device, ComputeDevice::Cpu);
        assert!(first.rgb == want.rgb, "the fallback is the CPU render");
        assert!(dev.export_tail_disabled());
        let asked = own.counters().submissions;
        let second = dng_image(&dev, &src, FullTarget::DngLinearAwg4);
        assert_eq!(second.report.device, ComputeDevice::Cpu);
        assert!(
            !second
                .report
                .notes
                .iter()
                .any(|n| n.contains("graphics card")),
            "{:?}",
            second.report.notes
        );
        assert!(second.rgb == want.rgb);
        assert_eq!(
            own.counters().submissions,
            asked,
            "the device was not asked"
        );
    }
}

#[test]
fn nothing_but_the_picture_is_written() {
    let Some(gpu) = gpu() else { return };
    let shot = Shot::new();
    let home = tempfile::tempdir().expect("home");
    let ctx = context(home.path(), renderer(Some(gpu), GpuStages::ALL, false));
    let out = tempfile::tempdir().expect("out");
    let r = export(&ctx, &shot.photo(), png(out.path(), BitDepth::Eight));
    assert_eq!(names_in(out.path()).len(), 1);
    assert_eq!(r.path.parent(), Some(out.path()));
}
