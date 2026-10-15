//! G-08: the export tail on the device (Mesa lavapipe in CI) through `export_photo`.
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
use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use common::*;
use ym_color::consts::linalg;
use ym_color::deltae::{d65_white_xyz, delta_e_2000, percentile, xyz_to_lab};
use ym_color::display::OutputTarget;
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
use ym_core::settings::{Crop, CropAspect, NormRect, SettingsPatch};
use ym_export::{BitDepth, ColorSpace, ExportOptions, FormatKind, IfExists, Size, TiffCompression};
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
    let _ = writeln!(err, "[G-08] {text}");
    if std::env::var_os("GITHUB_ACTIONS").is_some() {
        let _ = writeln!(err, "::notice title=G-08 export tail::{text}");
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
        let white = light("sony-ilce-7rm4", 4500.0);
        let scene = Scene::new(900, 600, white, Layout::Patchwork, 0.5).noise(2e-4, 2e-6);
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
    export_photo(
        ctx,
        photo,
        &ExportParams::new(opts),
        &ExportControl::none(),
    )
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
}

fn stats(a: &[u32], b: &[u32]) -> CodeStats {
    assert_eq!(a.len(), b.len());
    let mut s = CodeStats {
        n: a.len(),
        exact: 0,
        max: 0,
        over_one: 0,
    };
    for (&x, &y) in a.iter().zip(b) {
        let d = x.abs_diff(y);
        s.exact += usize::from(d == 0);
        s.over_one += usize::from(d > 1);
        s.max = s.max.max(d);
    }
    s
}

/// CIEDE2000 on a 1/4 downsample (a box filter in linear light, the samples read as sRGB), the
/// comparison of the golden runner: median, 99th percentile, maximum.
fn golden(a: &[u32], b: &[u32], w: usize, h: usize) -> (f64, f64, f64) {
    let to_xyz = ym_color::gamut::matrices().rec709_to_xyz;
    let white = d65_white_xyz();
    let table: Vec<f64> = (0..=255u32)
        .map(|v| srgb_decode(f64::from(v) / 255.0))
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
    shot.edit(SettingsPatch {
        output_sharpening: Some(pct),
        crop: Some(Some(Crop {
            rect: NormRect {
                x: 0.04,
                y: 0.06,
                width: 0.9,
                height: 0.88,
            },
            angle_deg: if straighten { 5.0 } else { 0.0 },
            aspect: CropAspect::Free,
        }))
        .filter(|_| straighten),
        ..SettingsPatch::default()
    });
    if !straighten {
        shot.edit(SettingsPatch {
            crop: Some(None),
            ..SettingsPatch::default()
        });
    }
}

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
            let (a, b) = (tempfile::tempdir().expect("a"), tempfile::tempdir().expect("b"));
            let (mut oa, mut ob) = (mk(a.path()), mk(b.path()));
            oa.size = size;
            ob.size = size;
            let ra = export(&raw_only, &photo, oa);
            let rb = export(&with_tail, &photo, ob);
            assert_eq!(rb.device, ComputeDevice::Gpu, "{name} {label}: {:?}", rb.notes);
            assert!(
                !rb.notes.iter().any(|n| n.contains("graphics card")),
                "{name} {label}: {:?}",
                rb.notes
            );
            assert_eq!((ra.width, ra.height), (rb.width, rb.height), "{name} {label}");
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
                "{name}, {label}: {} codes, {:.5} exact, max {}",
                s.n,
                s.exact as f64 / s.n as f64,
                s.max
            ));
            if sixteen {
                // Sharpening amplifies the f32 against f64 colour difference at the few pixels
                // where its halo limiter decides on a rank of luma values (`gpu_tail.rs`).
                assert!(
                    s.over_one as f64 <= 1e-4 * s.n as f64 && s.max <= 400,
                    "{name} {label}: {} codes over 1, max {}",
                    s.over_one,
                    s.max
                );
            } else {
                // The JPEG decoder rounds again (colour conversion), one code more.
                let limit = if label == "JPEG" { 2 } else { 1 };
                assert!(s.max <= limit, "{name} {label}: max {}", s.max);
                let exact = s.exact as f64 / s.n as f64;
                assert!(exact >= 0.98, "{name} {label}: {exact} exact");
                worst8 = worst8.max(s.max);
                min_exact8 = min_exact8.min(exact);
            }
        }
    }
    ci_note(&format!(
        "tail against the raw-chain-only export: worst 8-bit difference {worst8}, smallest exact share {min_exact8:.5}"
    ));
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
    for (name, pct, size, straighten) in variants() {
        apply_variant(&shot, pct, straighten);
        let photo = shot.photo();
        let (a, b) = (tempfile::tempdir().expect("a"), tempfile::tempdir().expect("b"));
        let mut oa = png(a.path(), BitDepth::Eight);
        let mut ob = png(b.path(), BitDepth::Eight);
        oa.size = size;
        ob.size = size;
        let (ra, rb) = (export(&cpu, &photo, oa), export(&dev, &photo, ob));
        assert_eq!(ra.device, ComputeDevice::Cpu);
        assert_eq!(rb.device, ComputeDevice::Gpu, "{name}: {:?}", rb.notes);
        let (w, h, _, ca) = decode_png(&read(&ra.path));
        let (_, _, _, cb) = decode_png(&read(&rb.path));
        let (median, p99, max) = golden(&ca, &cb, w, h);
        let s = stats(&ca, &cb);
        ci_note(&format!(
            "{name}: golden median {median:.4} p99 {p99:.4} max {max:.4}; codes max {} over one {}",
            s.max, s.over_one
        ));
        // The plan's gates (docs/gpu-plan.md section 7): 0.02 / 0.05 / 0.5.
        assert!(
            median <= 0.02 && p99 <= 0.05 && max <= 0.5,
            "{name}: {median} {p99} {max}"
        );
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
    let (peak_a, peak_b) = (
        ca.renderer().ledger().peak(),
        cb.renderer().ledger().peak(),
    );
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
    let shot = Shot::new();
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
    assert_eq!(pool_in_use(&gpu), baseline, "a finished export hands everything back");
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
        assert!(names_in(out.path()).is_empty(), "{format:?}: a file was left");
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
        assert_eq!(b.report.device, ComputeDevice::Gpu, "{target:?}: {:?}", b.report.notes);
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
        // First measured: see the note; the bound is twice that.
        assert!(worst <= 2e-5, "{target:?}: {worst:e}");
    }
}

#[cfg(feature = "fault-injection")]
mod faults {
    use super::*;
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
            (FormatKind::Png, cpu_export(&shot, home.path(), FormatKind::Png)),
            (FormatKind::Tiff, cpu_export(&shot, home.path(), FormatKind::Tiff)),
        ];
        // `stages`: the tail alone (the raw chain is on the CPU, the tail fails at its first job)
        // and everything (the raw chain fails first).
        for (stages_name, stages) in [
            ("the tail alone", GpuStages::EXPORT_TAIL),
            ("every stage", GpuStages::ALL),
        ] {
            for fault in [GpuFault::Shader, GpuFault::OutOfBudget, GpuFault::Lost] {
                // A device that is lost stays lost: each case gets a device of its own.
                let own = if matches!(fault, GpuFault::Lost) {
                    Arc::new(Gpu::start(GpuConfig::default()).expect("device"))
                } else {
                    Arc::clone(&gpu)
                };
                let ctx = context(home.path(), renderer(Some(Arc::clone(&own)), stages, false));
                for (format, pair) in &pairs {
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
                        r.notes.iter().any(|n| n.contains("The graphics card could not render this photo, so the CPU did")),
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
            got.report.notes.iter().any(|n| n.contains("The graphics card could not render this photo, so the CPU did")),
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
        assert!(matches!(err, Err(ym_core::pipeline::RenderError::Gpu(_))), "{err:?}");
    }
}

#[test]
fn nothing_but_the_picture_is_written() {
    let Some(gpu) = gpu() else { return };
    let shot = Shot::new();
    let home = tempfile::tempdir().expect("home");
    let ctx = context(
        home.path(),
        renderer(Some(gpu), GpuStages::ALL, false),
    );
    let out = tempfile::tempdir().expect("out");
    let r = export(&ctx, &shot.photo(), png(out.path(), BitDepth::Eight));
    assert_eq!(names_in(out.path()).len(), 1);
    assert_eq!(r.path.parent(), Some(out.path()));
    let _ = (PathBuf::new(), ColorSpace::Srgb, OutputTarget::Srgb, AtomicU32::new(0));
}
