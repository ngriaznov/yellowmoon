//! App and CLI parity with the raw chain on the device (`docs/17-gpu.md` section 12): the same
//! job exported through the app's path (`Core::export_start`, the device attached the way the
//! application attaches it) and through the `ym` binary with `YELLOWMOON_GPU=on` gives
//! byte-identical files, whatever the number of photos in flight.
//!
//! The device is Mesa lavapipe in CI (`WGPU_BACKEND=vulkan`, `YM_REQUIRE_GPU=1`); without an adapter
//! the tests skip with a notice, unless `YM_REQUIRE_GPU=1` turns that into a failure. A GPU export is
//! not the CPU export (the kernels agree with the CPU within their tolerances, not bit for bit,
//! `docs/05`), so the CPU path is not compared here: `parity.rs` does that with `--cpu` semantics.
//! Apple Auto is off on both sides (no advisor in the app's core, `--apple-auto off` for the
//! binary), so a Mac with Apple Intelligence compares the same pictures (`docs/19` section 7).

mod common;

use std::path::{Path, PathBuf};
use std::time::Duration;

use common::{Sandbox, data_dir, names_in};
use ym_core::app::gpu::{GpuPolicy, GpuSlot, open_gpu};
use ym_core::app::{Core, CoreOptions};
use ym_core::export_queue::ExportJob;
use ym_core::scheduler::MemoryBudget;
use ym_core::session::{ApplyScope, OpenOptions};
use ym_export::{BitDepth, ExportOptions, FormatKind};
use ym_gpu::device::gpu_required_by_env;

/// Whether the slot holds a device; says so, loudly, when it does not (and fails when the environment
/// demands one).
fn slot_has_device(slot: &GpuSlot) -> bool {
    if slot.gpu().is_some() {
        return true;
    }
    assert!(
        !gpu_required_by_env(),
        "YM_REQUIRE_GPU is set but no device opened: {:?}",
        slot.missing_device_reason()
    );
    eprintln!("\n* GPU TESTS SKIPPED in parity_gpu: no adapter\n");
    false
}

/// Whether a device opens under the `On` policy.
fn device_opens() -> bool {
    slot_has_device(&open_gpu(GpuPolicy::On, MemoryBudget::detect()))
}

/// The app's side: a core over its own app folder with the device attached.
struct App {
    core: Core,
    slot: GpuSlot,
    _home: tempfile::TempDir,
}

impl App {
    /// Opens `files` with the device of the `On` policy attached, or `None` (with a notice) when
    /// there is no adapter.
    fn open(files: &[&Path]) -> Option<Self> {
        let slot = open_gpu(GpuPolicy::On, MemoryBudget::detect());
        if !slot_has_device(&slot) {
            return None;
        }
        let home = tempfile::tempdir().expect("tempdir");
        let mut options = CoreOptions::new(data_dir(), home.path().to_path_buf());
        // What the CLI does: no thumbnail renders competing with the export.
        options.thumbnails = false;
        let core = Core::open(options).expect("core");
        core.attach_gpu(&slot);
        let paths: Vec<PathBuf> = files.iter().map(|p| p.to_path_buf()).collect();
        let outcome = core.session.open(&paths, OpenOptions::default());
        assert_eq!(outcome.added.len(), files.len(), "{outcome:?}");
        assert!(
            core.session.wait_idle(Duration::from_secs(120)),
            "the photos develop"
        );
        Some(Self {
            core,
            slot,
            _home: home,
        })
    }

    /// Exports every photo with `options` to `dir` and returns the files, in job order.
    fn export(&self, options: ExportOptions, dir: &Path) -> Vec<PathBuf> {
        let job = ExportJob {
            scope: ApplyScope::All,
            options: ExportOptions {
                destination: Some(dir.to_path_buf()),
                ..options
            },
        };
        let id = self.core.export_start(&job).expect("export starts");
        assert!(
            self.core.exports.wait_idle(Duration::from_secs(600)),
            "the export finishes"
        );
        let summary = self.core.export_summary(id).expect("summary");
        assert_eq!(summary.failed, 0, "{:?}", summary.errors);
        summary.files
    }
}

fn bytes(path: &Path) -> Vec<u8> {
    std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

/// Runs the binary on `inputs` with the device switched on, `flags` extra, writing to `dir`.
fn cli_export(sb: &Sandbox, inputs: &[&Path], flags: &[&str], dir: &Path) -> Vec<PathBuf> {
    let out = sb
        .ym()
        .env("YELLOWMOON_GPU", "on")
        .args(["render", "--apple-auto", "off"])
        .args(inputs)
        .args(flags)
        .arg("--out")
        .arg(dir)
        .assert()
        .success();
    String::from_utf8_lossy(&out.get_output().stdout)
        .lines()
        .map(PathBuf::from)
        .collect()
}

fn assert_same_files(a: &[PathBuf], b: &[PathBuf], what: &str) {
    assert_eq!(a.len(), b.len(), "{what}: the same number of files");
    assert!(!a.is_empty(), "{what}");
    for (x, y) in a.iter().zip(b) {
        assert_eq!(x.file_name(), y.file_name(), "{what}: the same file name");
        let (bx, by) = (bytes(x), bytes(y));
        assert!(!bx.is_empty(), "{what}");
        assert!(
            bx == by,
            "{what}: {} ({} bytes) and {} ({} bytes) differ",
            x.display(),
            bx.len(),
            y.display(),
            by.len()
        );
    }
}

/// A photo of several export tiles (three wide, two high), so that an export is several device
/// jobs and the tile seams are part of the comparison.
#[test]
fn a_jpeg_of_several_tiles_is_byte_identical_through_the_app_path_and_the_binary_on_the_device() {
    let sb = Sandbox::new();
    let arw = sb.arw_sized("big.ARW", 2100, 1100, 9);
    let Some(app) = App::open(&[&arw]) else {
        return;
    };
    let device = app.slot.gpu().expect("a device").clone();
    let before = device.counters();
    let from_app = app.export(ExportOptions::default(), &sb.root().join("app"));
    let used = device.counters().since(before);
    assert!(
        used.dispatches > 0,
        "the app's export used the device: {used:?}"
    );
    let from_cli = cli_export(&sb, &[&arw], &["--format", "jpeg"], &sb.root().join("cli"));
    assert_same_files(&from_app, &from_cli, "jpeg on the device");
    assert_eq!(names_in(&sb.root().join("app")).len(), 1);
    assert_eq!(names_in(&sb.root().join("cli")).len(), 1);
}

#[test]
fn the_binary_gives_the_same_bytes_twice_and_with_one_and_several_photos_in_flight() {
    let sb = Sandbox::new();
    sb.arw_sized("a.ARW", 2100, 1100, 9);
    sb.arw_sized("b.ARW", 1300, 900, 5);
    if !device_opens() {
        return;
    }
    let png = ["--format", "png", "--bits", "16"];
    let run = |name: &str, jobs: &str| {
        let mut flags: Vec<&str> = png.to_vec();
        flags.extend(["--jobs", jobs]);
        cli_export(&sb, &[&sb.photos()], &flags, &sb.root().join(name))
    };
    let first = run("first", "1");
    let second = run("second", "1");
    let several = run("several", "4");
    assert_eq!(first.len(), 2, "{first:?}");
    assert_same_files(&first, &second, "twice");
    assert_same_files(&first, &several, "--jobs 1 and --jobs 4");
}

/// The comparison above would also pass if the binary had quietly rendered on the CPU (a JPEG hides
/// the kernels' last digits), so the binary's own report is checked: the pictures were made on the
/// device and no fall-back note was written.
#[test]
fn the_binary_really_renders_the_raw_chain_on_the_device_under_the_on_policy() {
    let sb = Sandbox::new();
    let arw = sb.arw_sized("a.ARW", 1300, 900, 5);
    if !device_opens() {
        return;
    }
    let out = sb
        .ym()
        .env("YELLOWMOON_GPU", "on")
        .args(["render", "--json", "--quiet", "--apple-auto", "off"])
        .arg(&arw)
        .arg("--out")
        .arg(sb.root().join("json"))
        .assert()
        .success();
    let text = String::from_utf8(out.get_output().stdout.clone()).expect("utf-8");
    let v: serde_json::Value = serde_json::from_str(text.trim()).expect("one JSON object");
    assert_eq!(v["device"], "gpu", "{text}");
    let notes = v["notes"].to_string();
    assert!(
        !notes.contains("graphics card"),
        "the binary fell back to the CPU: {notes}"
    );
}

/// A 16-bit TIFF keeps the pixels the JPEG quantises: the two front ends agree on every one.
#[test]
fn a_sixteen_bit_tiff_is_byte_identical_through_the_app_path_and_the_binary_on_the_device() {
    let sb = Sandbox::new();
    let arw = sb.arw_sized("big.ARW", 1500, 1100, 3);
    let Some(app) = App::open(&[&arw]) else {
        return;
    };
    let tiff = ExportOptions {
        format: FormatKind::Tiff,
        bit_depth: BitDepth::Sixteen,
        ..ExportOptions::default()
    };
    let from_app = app.export(tiff, &sb.root().join("app"));
    let from_cli = cli_export(
        &sb,
        &[&arw],
        &["--format", "tiff", "--bits", "16", "--compression", "lzw"],
        &sb.root().join("cli"),
    );
    assert_same_files(&from_app, &from_cli, "tiff-16 on the device");
}

/// A DNG (no R8 capture sharpening, `docs/09` "What a DNG carries"; the raw chain on the device): the
/// two front ends agree on every byte, the XMP, the preview and the 16-bit pixels included.
#[test]
fn a_dng_is_byte_identical_through_the_app_path_and_the_binary_on_the_device() {
    let sb = Sandbox::new();
    let arw = sb.arw_sized("big.ARW", 1500, 1100, 3);
    let Some(app) = App::open(&[&arw]) else {
        return;
    };
    let device = app.slot.gpu().expect("a device").clone();
    let before = device.counters();
    let dng = ExportOptions {
        format: FormatKind::Dng,
        ..ExportOptions::default()
    };
    let from_app = app.export(dng, &sb.root().join("app"));
    let used = device.counters().since(before);
    assert!(
        used.dispatches > 0,
        "the app's export used the device: {used:?}"
    );
    let from_cli = cli_export(&sb, &[&arw], &["--format", "dng"], &sb.root().join("cli"));
    assert_same_files(&from_app, &from_cli, "dng on the device");
}

/// The renders of a DNG run with R8 capture sharpening off and the other formats with it on, in one session and on
/// one device (the device caches what it builds by the chain's parameters, the capture sharpening among them): a TIFF
/// exported after a DNG is the TIFF exported before it, and a DNG exported after a TIFF is the DNG the binary writes
/// alone. Nothing of R8 leaks from one into the other (`docs/09`, "What a DNG carries").
#[test]
fn a_dng_and_a_tiff_exported_one_after_the_other_on_the_device_do_not_share_r8() {
    let sb = Sandbox::new();
    let arw = sb.arw_sized("big.ARW", 1500, 1100, 3);
    let Some(app) = App::open(&[&arw]) else {
        return;
    };
    let tiff = || ExportOptions {
        format: FormatKind::Tiff,
        bit_depth: BitDepth::Sixteen,
        ..ExportOptions::default()
    };
    let dng = || ExportOptions {
        format: FormatKind::Dng,
        ..ExportOptions::default()
    };
    let tiff_first = app.export(tiff(), &sb.root().join("tiff1"));
    let dng_after_tiff = app.export(dng(), &sb.root().join("dng1"));
    let tiff_after_dng = app.export(tiff(), &sb.root().join("tiff2"));
    let dng_again = app.export(dng(), &sb.root().join("dng2"));
    assert_same_files(
        &tiff_first,
        &tiff_after_dng,
        "a TIFF before and after a DNG",
    );
    assert_same_files(&dng_after_tiff, &dng_again, "a DNG before and after a TIFF");
    let dng_alone = cli_export(&sb, &[&arw], &["--format", "dng"], &sb.root().join("cli"));
    assert_same_files(
        &dng_after_tiff,
        &dng_alone,
        "a DNG after a TIFF and a DNG alone",
    );
}
