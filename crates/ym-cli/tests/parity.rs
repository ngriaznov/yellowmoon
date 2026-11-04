//! App and CLI parity (`docs/09`, `docs/13`, M3-14): the same job exported through the app's path
//! (`ym_core::app::Core::export_start`, what the Tauri commands call) and through the `ym` binary
//! gives byte-identical files, for every format and option.
//!
//! The app side runs in this process; the CLI side is a separate process. Each side has its own
//! app folder, its own renderer and its own thread layout, so the test also stands for "different
//! machines of the same build".
//!
//! Two legs (`docs/17-gpu.md` section 12). The CPU leg gives the binary `--cpu` and the app no
//! device: both are the CPU reference whatever graphics card the machine has (without the flag `ym`
//! would run `auto`, which on a Mac puts every stage on the device while this process's core has none,
//! and the comparison would be CPU against GPU). The GPU leg opens the device the way the app does
//! (`open_gpu(On)`, attached to the core) and starts the binary with `YELLOWMOON_GPU=on`: the same
//! eleven cases, byte for byte. `parity_gpu.rs` goes further on the device (tiles, thread layouts, a
//! DNG after a TIFF); the GPU leg here is the whole matrix of formats. Without an adapter the GPU leg
//! skips with a notice, unless `YM_REQUIRE_GPU=1` makes that a failure.
//!
//! Auto providers (`docs/19` section 7, `docs/22` section 14.4). The app's side here has no advisor
//! (Default Auto) and the binary is given `--auto-provider yellowmoon`, so a Mac with Apple
//! Intelligence compares today's bytes too. The Apple Auto leg gives both sides the test advisor
//! (`FakeAdvisor` in the core, the binary `YELLOWMOON_TEST_APPLE_AUTO=fake` and no option, so the
//! app's setting decides as in the app): the same answers, the same files. The Claude Code leg gives
//! both sides the in-process fake of the program (`FakeProvider` in the core, the binary
//! `YELLOWMOON_TEST_AUTO_PROVIDERS=claude=available` and `--auto-provider claude`): the same answers,
//! the same files, which record `ym:Auto` `claude`.

mod common;

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use common::{Sandbox, data_dir, names_in};
use ym_core::app::gpu::{GpuPolicy, GpuSlot, open_gpu};
use ym_core::app::{Core, CoreOptions};
use ym_core::apple_auto::FakeAdvisor;
use ym_core::auto_provider::{AutoChoice, AutoProvider, FakeProvider, ProviderAdvisors};
use ym_core::export_queue::ExportJob;
use ym_core::scheduler::MemoryBudget;
use ym_core::session::{ApplyScope, OpenOptions, PhotoId};
use ym_core::settings::{AutoOr, DevelopSettings, SettingsPatch};
use ym_core::settings::{NodeId, NodePatch};
use ym_export::{BitDepth, ColorSpace, ExportOptions, FormatKind, Size, TiffCompression};
use ym_gpu::cpu::demosaic::{DEFAULT_METHOD, Method};
use ym_gpu::device::gpu_required_by_env;

/// What the app's core is given to grade with.
#[derive(Clone, Copy)]
enum Auto {
    /// No advisor: Default Auto.
    None,
    /// Apple Intelligence's test advisor, the saved choice deciding.
    AppleFake,
    /// The in-process fake of Claude Code, named for the run.
    ClaudeFake,
}

/// How Subject and People are made on both sides of a parity leg (`docs/23` section 4.5).
#[derive(Clone, Copy, PartialEq, Eq)]
enum Vision {
    /// The deterministic fake, on every platform: `FakeVision` for the app, `YELLOWMOON_TEST_VISION=fake` for the binary.
    Fake,
    /// This computer's own Vision framework on both sides (a Mac; `ym_apple::system_vision()` for the app,
    /// `YELLOWMOON_TEST_VISION=system` for the binary).
    System,
}

impl Vision {
    /// The environment of the binary.
    fn env(self) -> (&'static str, &'static str) {
        match self {
            Self::Fake => common::FAKE_VISION,
            Self::System => common::SYSTEM_VISION,
        }
    }

    /// The provider of the app's core.
    fn provider(self) -> Arc<dyn ym_core::vision::VisionProvider> {
        match self {
            Self::Fake => Arc::new(ym_core::vision::FakeVision::new()),
            Self::System => Arc::new(ym_apple::AppleVision),
        }
    }
}

/// The app's side: a core over its own app folder with one photo open.
struct App {
    core: Core,
    photo: PhotoId,
    /// The device of the GPU leg, kept open as long as the core.
    _slot: Option<GpuSlot>,
    _home: tempfile::TempDir,
}

impl App {
    fn open(files: &[&Path]) -> Self {
        Self::open_with(files, DEFAULT_METHOD)
    }

    /// Like [`App::open`] with the R6 demosaic method of the full-resolution renders set the way
    /// the app's code would (`RenderOptions::demosaic`, an internal switch with no UI).
    fn open_with(files: &[&Path], method: Method) -> Self {
        Self::open_on(files, method, None)
    }

    /// Like [`App::open`] with Apple Auto's test advisor, as the app has it with
    /// `YELLOWMOON_TEST_APPLE_AUTO=fake`: the `apple_auto` preference (on in a new app folder)
    /// decides, and the photos are answered before they count as developed.
    fn open_with_fake_advisor(files: &[&Path]) -> Self {
        Self::open_full(files, DEFAULT_METHOD, None, Auto::AppleFake)
    }

    /// Like [`App::open`] with the in-process fake of Claude Code, as the app has it with
    /// `YELLOWMOON_TEST_AUTO_PROVIDERS=claude=available` and the run naming the provider.
    fn open_with_fake_claude(files: &[&Path]) -> Self {
        Self::open_full(files, DEFAULT_METHOD, None, Auto::ClaudeFake)
    }

    /// The GPU leg's app: the device of the `On` policy attached to the core the way the application
    /// attaches it. `None` (with a notice) when no adapter opens; a failure when `YM_REQUIRE_GPU` is set.
    fn open_on_device(files: &[&Path]) -> Option<Self> {
        let slot = open_gpu(GpuPolicy::On, MemoryBudget::detect());
        if slot.gpu().is_none() {
            assert!(
                !gpu_required_by_env(),
                "YM_REQUIRE_GPU is set but no device opened: {:?}",
                slot.missing_device_reason()
            );
            eprintln!("\n* GPU LEG SKIPPED in parity: no adapter\n");
            return None;
        }
        Some(Self::open_on(files, DEFAULT_METHOD, Some(slot)))
    }

    fn open_on(files: &[&Path], method: Method, slot: Option<GpuSlot>) -> Self {
        Self::open_full(files, method, slot, Auto::None)
    }

    /// Like [`App::open`] over another data folder (one with synthetic lenses, `Sandbox::data_with_lenses`).
    fn open_in(files: &[&Path], data: &Path) -> Self {
        Self::open_full_in(data, files, DEFAULT_METHOD, None, Auto::None)
    }

    fn open_full(files: &[&Path], method: Method, slot: Option<GpuSlot>, auto: Auto) -> Self {
        Self::open_full_in(&data_dir(), files, method, slot, auto)
    }

    /// The app of a leg with Subject and People made by `vision` (the GPU leg's device, when it has one, attached).
    fn open_with_vision(files: &[&Path], slot: Option<GpuSlot>, vision: Vision) -> Self {
        Self::open_complete(
            &data_dir(),
            files,
            DEFAULT_METHOD,
            slot,
            Auto::None,
            Some(vision),
        )
    }

    fn open_full_in(
        data: &Path,
        files: &[&Path],
        method: Method,
        slot: Option<GpuSlot>,
        auto: Auto,
    ) -> Self {
        Self::open_complete(data, files, method, slot, auto, None)
    }

    fn open_complete(
        data: &Path,
        files: &[&Path],
        method: Method,
        slot: Option<GpuSlot>,
        auto: Auto,
        vision: Option<Vision>,
    ) -> Self {
        let home = tempfile::tempdir().expect("tempdir");
        let mut options = CoreOptions::new(data.to_path_buf(), home.path().to_path_buf());
        // What the CLI does: no thumbnail renders competing with the export.
        options.thumbnails = false;
        options.render.demosaic = method;
        options.render.vision = vision.map(|v| ym_core::vision::VisionMasks::new(v.provider()));
        match auto {
            Auto::None => {}
            Auto::AppleFake => {
                options.advisors =
                    ProviderAdvisors::default().with(AutoProvider::Apple, Arc::new(FakeAdvisor));
            }
            Auto::ClaudeFake => {
                options.advisors = ProviderAdvisors::default().with(
                    AutoProvider::Claude,
                    Arc::new(FakeProvider::available(AutoProvider::Claude).with_details(
                        ym_core::auto_provider::AdvisorDetails {
                            tool_version: Some("0.0.0-test".to_owned()),
                            ..Default::default()
                        },
                    )),
                );
                // Naming the provider is the consent for the run, as `--auto-provider claude` is.
                options.auto_choice = AutoChoice::Provider(AutoProvider::Claude);
            }
        }
        let core = Core::open(options).expect("core");
        if let Some(slot) = &slot {
            core.attach_gpu(slot);
        }
        let paths: Vec<PathBuf> = files.iter().map(|p| p.to_path_buf()).collect();
        let outcome = core.session.open(&paths, OpenOptions::default());
        assert_eq!(outcome.added.len(), files.len(), "{outcome:?}");
        assert!(
            core.session.wait_idle(Duration::from_secs(120)),
            "the photos develop"
        );
        let photo = outcome.added[0];
        Self {
            core,
            photo,
            _slot: slot,
            _home: home,
        }
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
            self.core.exports.wait_idle(Duration::from_secs(300)),
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

/// Runs the binary on `inputs` with `flags`, writing to `dir`.
fn cli_export(sb: &Sandbox, inputs: &[&Path], flags: &[&str], dir: &Path) -> Vec<PathBuf> {
    cli_export_env(sb, inputs, flags, dir, &[])
}

/// [`cli_export`] with extra environment variables for the child process.
fn cli_export_env(
    sb: &Sandbox,
    inputs: &[&Path],
    flags: &[&str],
    dir: &Path,
    env: &[(&str, &str)],
) -> Vec<PathBuf> {
    // The CPU leg asks for the CPU reference by name; a test that names the policy itself
    // (`YELLOWMOON_GPU=on`) is the GPU leg and is not given `--cpu`, which would beat it.
    let device: &[&str] = if env.iter().any(|(k, _)| *k == "YELLOWMOON_GPU") {
        &[]
    } else {
        &["--cpu"]
    };
    // Default Auto, as on the app's side, unless the test gives both sides a provider.
    let apple_auto: &[&str] = if env
        .iter()
        .any(|(k, _)| *k == common::FAKE_APPLE_AUTO.0 || *k == common::AUTO_PROVIDERS_ENV)
    {
        &[]
    } else {
        &["--auto-provider", "yellowmoon"]
    };
    let out = sb
        .ym()
        .envs(env.iter().copied())
        .arg("render")
        .args(device)
        .args(apple_auto)
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

fn assert_same_files(app: &[PathBuf], cli: &[PathBuf], what: &str) {
    assert_eq!(app.len(), cli.len(), "{what}: the same number of files");
    assert!(!app.is_empty(), "{what}");
    for (a, c) in app.iter().zip(cli) {
        assert_eq!(a.file_name(), c.file_name(), "{what}: the same file name");
        let (ba, bc) = (bytes(a), bytes(c));
        assert!(!ba.is_empty(), "{what}");
        assert!(
            ba == bc,
            "{what}: {} ({} bytes) and {} ({} bytes) differ",
            a.display(),
            ba.len(),
            c.display(),
            bc.len()
        );
    }
}

/// One format with the options of the app and the flags of the CLI that mean the same.
struct Case {
    name: &'static str,
    options: ExportOptions,
    flags: &'static [&'static str],
}

fn cases() -> Vec<Case> {
    let tiff = |bits: BitDepth, compression: TiffCompression| ExportOptions {
        format: FormatKind::Tiff,
        bit_depth: bits,
        tiff_compression: compression,
        ..ExportOptions::default()
    };
    let png = |bits: BitDepth| ExportOptions {
        format: FormatKind::Png,
        bit_depth: bits,
        ..ExportOptions::default()
    };
    vec![
        Case {
            name: "jpeg",
            options: ExportOptions::default(),
            flags: &["--format", "jpeg"],
        },
        Case {
            name: "jpeg-q70-p3",
            options: ExportOptions {
                jpeg_quality: ym_export::JpegQuality::new(70).expect("quality"),
                color_space: ColorSpace::DisplayP3,
                ..ExportOptions::default()
            },
            flags: &[
                "--format",
                "jpeg",
                "--quality",
                "70",
                "--colour-space",
                "p3",
            ],
        },
        Case {
            name: "tiff-8-lzw",
            options: tiff(BitDepth::Eight, TiffCompression::Lzw),
            flags: &["--format", "tiff", "--bits", "8", "--compression", "lzw"],
        },
        Case {
            name: "tiff-8-zip",
            options: tiff(BitDepth::Eight, TiffCompression::Zip),
            flags: &["--format", "tiff", "--bits", "8", "--compression", "zip"],
        },
        Case {
            name: "tiff-16-lzw",
            options: tiff(BitDepth::Sixteen, TiffCompression::Lzw),
            flags: &["--format", "tiff", "--bits", "16", "--compression", "lzw"],
        },
        Case {
            name: "tiff-16-zip",
            options: tiff(BitDepth::Sixteen, TiffCompression::Zip),
            flags: &["--format", "tiff", "--bits", "16", "--compression", "zip"],
        },
        Case {
            name: "tiff-16-none",
            options: tiff(BitDepth::Sixteen, TiffCompression::None),
            flags: &["--format", "tiff", "--bits", "16", "--compression", "none"],
        },
        Case {
            name: "png-8",
            options: png(BitDepth::Eight),
            flags: &["--format", "png", "--bits", "8"],
        },
        Case {
            name: "png-16",
            options: png(BitDepth::Sixteen),
            flags: &["--format", "png", "--bits", "16"],
        },
        Case {
            name: "png-16-rec2020-long-edge",
            options: ExportOptions {
                color_space: ColorSpace::Rec2020,
                size: Size::LongEdge { px: 100 },
                ..png(BitDepth::Sixteen)
            },
            flags: &[
                "--format",
                "png",
                "--bits",
                "16",
                "--colour-space",
                "rec2020",
                "--long-edge",
                "100",
            ],
        },
        Case {
            name: "dng",
            options: ExportOptions {
                format: FormatKind::Dng,
                ..ExportOptions::default()
            },
            flags: &["--format", "dng"],
        },
    ]
}

#[test]
fn every_format_is_byte_identical_through_the_app_path_and_the_binary() {
    let sb = Sandbox::new();
    let arw = sb.arw("DSC00001.ARW");
    let app = App::open(&[&arw]);
    for case in cases() {
        let app_dir = sb.root().join("app").join(case.name);
        let cli_dir = sb.root().join("cli").join(case.name);
        let from_app = app.export(case.options.clone(), &app_dir);
        let from_cli = cli_export(&sb, &[&arw], case.flags, &cli_dir);
        assert_same_files(&from_app, &from_cli, case.name);
        // Nothing but the file in either folder.
        assert_eq!(names_in(&app_dir).len(), 1, "{}", case.name);
        assert_eq!(names_in(&cli_dir).len(), 1, "{}", case.name);
    }
}

/// The GPU leg: the same eleven cases with the device attached to the app's core and `YELLOWMOON_GPU=on` for the binary.
#[test]
fn every_format_is_byte_identical_through_the_app_path_and_the_binary_on_the_device() {
    let sb = Sandbox::new();
    let arw = sb.arw("DSC00001.ARW");
    let Some(app) = App::open_on_device(&[&arw]) else {
        return;
    };
    for case in cases() {
        let app_dir = sb.root().join("app").join(case.name);
        let cli_dir = sb.root().join("cli").join(case.name);
        let from_app = app.export(case.options.clone(), &app_dir);
        let from_cli = cli_export_env(
            &sb,
            &[&arw],
            case.flags,
            &cli_dir,
            &[("YELLOWMOON_GPU", "on")],
        );
        assert_same_files(&from_app, &from_cli, case.name);
        assert_eq!(names_in(&app_dir).len(), 1, "{}", case.name);
        assert_eq!(names_in(&cli_dir).len(), 1, "{}", case.name);
    }
}

/// The Apple Auto leg (`docs/19` section 7): the app's core with the test advisor and the binary with
/// `YELLOWMOON_TEST_APPLE_AUTO=fake` ask the same question about the same picture, get the same
/// answer and write the same bytes, for a finished picture, a 16-bit TIFF and a DNG, one photo and a
/// folder of two (a burst shares its answers). The files record Apple Auto (`ym:Auto` `apple`), so
/// this is not today's picture with another name.
#[test]
fn apple_auto_gives_the_same_files_through_the_app_path_and_the_binary() {
    let sb = Sandbox::new();
    let a = sb.arw("a.ARW");
    let b = sb.arw_sized("b.ARW", 128, 96, 7);
    let app = App::open_with_fake_advisor(&[&a, &b]);
    assert!(app.core.auto_status().active, "the app's Apple Auto acts");
    assert!(
        app.core.session.apple_auto_asks() >= 2,
        "both photos were asked"
    );
    let fake = [common::FAKE_APPLE_AUTO];
    for case in cases()
        .into_iter()
        .filter(|c| matches!(c.name, "jpeg" | "tiff-16-lzw" | "dng"))
    {
        let app_dir = sb.root().join("app").join(case.name);
        let cli_dir = sb.root().join("cli").join(case.name);
        let from_app = app.export(case.options.clone(), &app_dir);
        let from_cli = cli_export_env(&sb, &[&sb.photos()], case.flags, &cli_dir, &fake);
        assert_same_files(&from_app, &from_cli, &format!("Apple Auto, {}", case.name));
        for file in &from_cli {
            let text = String::from_utf8_lossy(&bytes(file)).into_owned();
            assert!(
                text.contains("<ym:Auto>apple</ym:Auto>") && text.contains("<ym:AutoAdvisor>"),
                "{} records Apple Auto",
                file.display()
            );
        }
    }
    // And the answer changed the picture: the same photo without Apple Auto is another file.
    let plain = cli_export(&sb, &[&a], &["--format", "jpeg"], &sb.root().join("plain"));
    let with = sb.root().join("cli").join("jpeg").join("a_signature.jpg");
    assert_ne!(bytes(&plain[0]), bytes(&with));
}

/// The Claude Code leg (`docs/22` section 14.4): the app's core with the in-process fake of the program
/// and the binary with `YELLOWMOON_TEST_AUTO_PROVIDERS=claude=available` and `--auto-provider claude`
/// ask the same question about the same picture, get the same answer and write the same bytes, for a
/// finished picture, a 16-bit TIFF and a DNG, one photo and a folder of two. The files record
/// `ym:Auto` `claude` and a `ym:AutoAdvisor` that says the answer is not reproducible.
#[test]
fn claude_code_gives_the_same_files_through_the_app_path_and_the_binary() {
    let sb = Sandbox::new();
    let a = sb.arw("a.ARW");
    let b = sb.arw_sized("b.ARW", 128, 96, 7);
    let app = App::open_with_fake_claude(&[&a, &b]);
    let status = app.core.auto_status();
    assert!(status.active, "the app's Claude Code acts");
    assert_eq!(status.effective, AutoProvider::Claude);
    assert!(
        app.core.session.apple_auto_asks() >= 2,
        "both photos were asked"
    );
    let hook = [(common::AUTO_PROVIDERS_ENV, "claude=available")];
    for case in cases()
        .into_iter()
        .filter(|c| matches!(c.name, "jpeg" | "tiff-16-lzw" | "dng"))
    {
        let app_dir = sb.root().join("app").join(case.name);
        let cli_dir = sb.root().join("cli").join(case.name);
        let from_app = app.export(case.options.clone(), &app_dir);
        let mut flags = vec!["--auto-provider", "claude"];
        flags.extend_from_slice(case.flags);
        let from_cli = cli_export_env(&sb, &[&sb.photos()], &flags, &cli_dir, &hook);
        assert_same_files(&from_app, &from_cli, &format!("Claude Code, {}", case.name));
        for file in &from_cli {
            let text = String::from_utf8_lossy(&bytes(file)).into_owned();
            assert!(
                text.contains("<ym:Auto>claude</ym:Auto>")
                    && text.contains("<ym:AutoAdvisor>")
                    && text.contains("&quot;provider&quot;:&quot;claude&quot;")
                    && text.contains("&quot;reproducible&quot;:false"),
                "{} records Claude Code",
                file.display()
            );
        }
    }
}

/// Both sides see the same policy result: with `--cpu` neither side touches a device, so the CPU leg is the CPU
/// reference even where `auto` would pick a graphics card.
#[test]
fn the_cpu_leg_asks_for_no_device_on_either_side() {
    let sb = Sandbox::new();
    let arw = sb.arw("DSC00001.ARW");
    let out = sb
        .ym()
        .env("YELLOWMOON_GPU", "on")
        .args([
            "render",
            "--cpu",
            "--json",
            "--quiet",
            "--auto-provider",
            "yellowmoon",
        ])
        .arg(&arw)
        .arg("--out")
        .arg(sb.root().join("json"))
        .assert()
        .success();
    let text = String::from_utf8(out.get_output().stdout.clone()).expect("utf-8");
    let v: serde_json::Value = serde_json::from_str(text.trim()).expect("one JSON object");
    assert_eq!(v["device"], "cpu", "--cpu beats YELLOWMOON_GPU=on: {text}");
    let app = App::open(&[&arw]);
    assert!(
        app.core.renderer.options().gpu.is_none(),
        "the app side of the CPU leg has no device"
    );
}

#[test]
fn a_folder_of_photos_with_the_name_template_and_seq_is_identical_too() {
    let sb = Sandbox::new();
    let a = sb.arw("a.ARW");
    let b = sb.arw_sized("b.ARW", 128, 96, 7);
    let app = App::open(&[&a, &b]);
    let options = ExportOptions {
        name_template: "{seq}-{name}-{camera}".to_owned(),
        ..ExportOptions::default()
    };
    let from_app = app.export(options, &sb.root().join("app"));
    let from_cli = cli_export(
        &sb,
        &[&sb.photos()],
        &["--name", "{seq}-{name}-{camera}"],
        &sb.root().join("cli"),
    );
    assert_same_files(&from_app, &from_cli, "folder");
    let names = names_in(&sb.root().join("cli"));
    assert_eq!(names, ["001-a-sony-a7r-iv.jpg", "002-b-sony-a7r-iv.jpg"]);
}

/// The synthetic photos of the other cases are one export tile (1024 px) each. This one is three
/// tiles wide and two high, with a lens crop and a different number of decode threads on the CLI
/// side: the tile seams and the thread layout are part of the bit-identity contract
/// (`docs/01`, "the app and the CLI must both keep the defaults").
#[test]
fn a_photo_of_several_export_tiles_is_identical_whatever_the_thread_layout() {
    let sb = Sandbox::new();
    let arw = sb.arw_sized("big.ARW", 2100, 1100, 9);
    let app = App::open(&[&arw]);
    let png = ExportOptions {
        format: FormatKind::Png,
        bit_depth: BitDepth::Sixteen,
        ..ExportOptions::default()
    };
    let from_app = app.export(png, &sb.root().join("app"));
    let from_cli = cli_export(
        &sb,
        &[&arw],
        &["--format", "png", "--bits", "16", "--jobs", "1"],
        &sb.root().join("cli"),
    );
    assert_same_files(&from_app, &from_cli, "several tiles");
    let again = cli_export(
        &sb,
        &[&arw],
        &["--format", "png", "--bits", "16", "--jobs", "3"],
        &sb.root().join("again"),
    );
    assert_same_files(&from_cli, &again, "jobs 1 and 3");
}

#[test]
fn settings_given_as_options_are_the_settings_of_the_app() {
    let sb = Sandbox::new();
    let arw = sb.arw("DSC00001.ARW");
    let app = App::open(&[&arw]);
    let patch: SettingsPatch = serde_json::from_value(common::field_patch(serde_json::json!({
        "look_intensity": 80.0,
        "exposure": { "kind": "manual", "ev": 0.5 },
        "contrast": 20.0,
        "saturation": -10.0,
        "white_balance": { "kind": "manual", "kelvin": 5200.0, "tint": 3.0 },
        "sharpen": { "kind": "manual", "amount": 90.0, "radius_px": 1.0 },
        "noise_reduction": { "kind": "manual", "luminance": 10.0, "color": 20.0, "detail": 30.0 },
        "rolloff": { "highlights": 0.3, "low_light": 0.6 },
        "decode": { "chromatic_aberration": false },
        "highlights": 15.0,
        "black": -5.0
    })))
    .expect("patch");
    app.core
        .session
        .update_settings(app.photo, &patch)
        .expect("settings");
    let png = ExportOptions {
        format: FormatKind::Png,
        bit_depth: BitDepth::Sixteen,
        ..ExportOptions::default()
    };
    let from_app = app.export(png, &sb.root().join("app"));
    let from_cli = cli_export(
        &sb,
        &[&arw],
        &[
            "--format",
            "png",
            "--intensity",
            "80",
            "--exposure",
            "0.5",
            "--contrast",
            "20",
            "--saturation",
            "-10",
            "--wb",
            "5200,3",
            "--sharpen",
            "90,1.0",
            "--nr",
            "10,20,30",
            "--rolloff",
            "0.3,0.6",
            "--no-ca",
            "--highlights",
            "15",
            "--black",
            "-5",
        ],
        &sb.root().join("cli"),
    );
    assert_same_files(&from_app, &from_cli, "settings as options");
    // And the options really changed the picture: the same photo with its defaults differs.
    let plain = cli_export(&sb, &[&arw], &["--format", "png"], &sb.root().join("plain"));
    assert_ne!(bytes(&plain[0]), bytes(&from_cli[0]));
}

/// Output sharpening (Detail › Sharpening › Output) is part of the bit-identity contract: the app
/// and the CLI write the same bytes with a non-default amount and with none, at full size and
/// after a resize, whatever the thread layout. The default (100) is what every other case of this
/// file already runs.
#[test]
fn output_sharpening_is_byte_identical_through_the_app_path_and_the_binary() {
    let sb = Sandbox::new();
    let arw = sb.arw_sized("big.ARW", 1300, 900, 5);
    let app = App::open(&[&arw]);
    for (amount, long_edge) in [
        (35.0_f64, None),
        (0.0, None),
        (150.0, Some(500_u32)),
        (0.0, Some(500)),
    ] {
        app.core
            .session
            .update_settings(
                app.photo,
                &SettingsPatch::default().and(
                    NodeId::Output,
                    NodePatch {
                        output_sharpening: Some(amount),
                        ..NodePatch::default()
                    },
                ),
            )
            .expect("settings");
        let png = ExportOptions {
            format: FormatKind::Png,
            bit_depth: BitDepth::Sixteen,
            size: long_edge.map_or(Size::Full, |px| Size::LongEdge { px }),
            ..ExportOptions::default()
        };
        let name = format!("{amount}-{long_edge:?}");
        let from_app = app.export(png, &sb.root().join("app").join(&name));
        let amount_text = amount.to_string();
        let edge_text = long_edge.map(|px| px.to_string());
        let mut flags = vec![
            "--format",
            "png",
            "--bits",
            "16",
            "--output-sharpening",
            &amount_text,
        ];
        if let Some(px) = &edge_text {
            flags.extend(["--long-edge", px]);
        }
        let mut one = flags.clone();
        one.extend(["--jobs", "1"]);
        let from_cli = cli_export(&sb, &[&arw], &one, &sb.root().join("cli").join(&name));
        assert_same_files(&from_app, &from_cli, &format!("output sharpening {name}"));
        let mut four = flags;
        four.extend(["--jobs", "4"]);
        let again = cli_export(&sb, &[&arw], &four, &sb.root().join("again").join(&name));
        assert_same_files(&from_cli, &again, &format!("jobs 1 and 4, {name}"));
    }
    // And the amount changes the picture: the same photo at 0 and at 35 differ.
    let at = |name: &str| bytes(&sb.root().join("cli").join(name).join("big_signature.png"));
    assert_ne!(at("35-None"), at("0-None"));
    // A DNG is the same bytes whatever the amount, apart from the settings the XMP records.
    let dng = |amount: &str, dir: &str| {
        cli_export(
            &sb,
            &[&arw],
            &["--format", "dng", "--output-sharpening", amount],
            &sb.root().join(dir),
        )
    };
    let (d100, d200) = (dng("100", "dng100"), dng("200", "dng200"));
    let (a, b) = (bytes(&d100[0]), bytes(&d200[0]));
    assert_eq!(a.len(), b.len(), "the XMP text has the same length");
    let differing = a.iter().zip(&b).filter(|(x, y)| x != y).count();
    assert!(
        differing < 8,
        "only the recorded number differs, not the pixels ({differing} bytes)"
    );
}

#[test]
fn a_settings_file_is_the_same_settings_as_the_apps() {
    let sb = Sandbox::new();
    let arw = sb.arw("DSC00001.ARW");
    let app = App::open(&[&arw]);
    // A settings document with curves, colour balance and a crop: things no option can say.
    let doc = common::field_doc(serde_json::json!({
        "exposure": { "kind": "manual", "ev": -0.25 },
        "color_balance": { "shadows": [0.1, -0.1], "midtones": [0.0, 0.0], "highlights": [-0.2, 0.15] },
        "curve": { "luma": [[0.0, 0.0], [0.25, 0.2], [0.75, 0.8], [1.0, 1.0]] },
        "crop": { "rect": { "x": 0.1, "y": 0.1, "width": 0.6, "height": 0.7 }, "angle_deg": 2.0, "aspect": "free" },
        "orientation": 90
    }));
    let (settings, clamped) =
        DevelopSettings::from_value(doc.clone(), &app.core.looks.effective_default())
            .expect("settings");
    assert!(clamped.is_empty(), "{clamped:?}");
    app.core
        .session
        .replace_settings(app.photo, settings)
        .expect("replace");
    let tiff = ExportOptions {
        format: FormatKind::Tiff,
        bit_depth: BitDepth::Eight,
        ..ExportOptions::default()
    };
    let from_app = app.export(tiff, &sb.root().join("app"));
    let file = sb.root().join("settings.json");
    std::fs::write(&file, doc.to_string()).expect("write");
    let from_cli = cli_export(
        &sb,
        &[&arw],
        &[
            "--format",
            "tiff",
            "--bits",
            "8",
            "--settings",
            file.to_str().expect("utf-8"),
        ],
        &sb.root().join("cli"),
    );
    assert_same_files(&from_app, &from_cli, "settings file");
}

/// A hand-written graph of three correctors beside Main, in the form of `docs/10`: a trim in the
/// first, a curve and a saturation in the second, a bypassed node and a grade in the third, and a dead
/// branch (node 7 reads Main and goes nowhere).
fn three_correctors() -> serde_json::Value {
    let picture = |from: serde_json::Value, to: serde_json::Value| {
        serde_json::json!({
            "from": { "node": from, "port": "picture" },
            "to": { "node": to, "port": "picture" }
        })
    };
    serde_json::json!({ "nodes": {
        "items": [
            { "id": "input", "exposure": { "kind": "manual", "ev": -0.25 } },
            { "id": 1, "grade": { "contrast": { "kind": "manual", "value": 10 } } },
            { "id": 2, "kind": "corrector", "pos": [1, 0], "name": "Warm",
              "trim": { "exposure_ev": 0.4, "temperature": 25, "tint": -8 } },
            { "id": 3, "kind": "corrector", "pos": [2, 0], "name": "Curve",
              "grade": { "curve": { "luma": [[0, 0], [0.25, 0.18], [0.6, 0.66], [1, 1]] },
                         "saturation": { "kind": "manual", "value": -15 } } },
            { "id": 4, "kind": "corrector", "pos": [3, 0], "bypass": true,
              "trim": { "exposure_ev": 3 } },
            { "id": 5, "kind": "corrector", "pos": [4, 0], "trim": { "exposure_ev": -0.2 },
              "grade": { "highlights": { "kind": "manual", "value": -20 } } },
            { "id": 7, "kind": "corrector", "pos": [1, 1], "trim": { "exposure_ev": -2 } }
        ],
        "links": [
            picture("input".into(), 1.into()),
            picture(1.into(), 2.into()),
            picture(2.into(), 3.into()),
            picture(3.into(), 4.into()),
            picture(4.into(), 5.into()),
            picture(5.into(), "output".into()),
            picture(1.into(), 7.into())
        ]
    } })
}

/// A hand-written graph of masked nodes, in the form of `docs/10` (N-10, N-13): node 2 has a Linear and a Radial
/// subtracted from it, node 3 a Colour range and a Brightness range (with the node's Invert off), node 4 reads
/// Main and goes nowhere with its picture but has a Radial, and its key output feeds node 5's key input (node 5
/// has a Linear subtracted of its own), and node 6 has a Brush row with Grow, Feather and Refine edges (a raster
/// mask, drawn into a plane and put through the raster pass). Node 4's mask, which feeds node 5's key input, is
/// a Radial added to a Background row (the photo's blur map cut at a level, the same raster pass after it).
fn masked_nodes() -> serde_json::Value {
    let picture = |from: serde_json::Value, to: serde_json::Value| {
        serde_json::json!({
            "from": { "node": from, "port": "picture" },
            "to": { "node": to, "port": "picture" }
        })
    };
    serde_json::json!({ "nodes": {
        "items": [
            { "id": 2, "kind": "corrector", "pos": [1, 0], "trim": { "exposure_ev": 0.6 },
              "masks": { "invert": false, "items": [
                  { "kind": "linear", "mode": "add", "center": [0.5, 0.45], "angle_deg": 20, "feather": 0.3 },
                  { "kind": "radial", "mode": "subtract", "center": [0.6, 0.5], "radii": [0.15, 0.1],
                    "angle_deg": -30, "feather": 0.4, "opacity": 0.8 }
              ] } },
            { "id": 3, "kind": "corrector", "pos": [2, 0],
              "grade": { "saturation": { "kind": "manual", "value": -40 },
                         "contrast": { "kind": "manual", "value": 25 } },
              "masks": { "invert": false, "items": [
                  { "kind": "color_range", "mode": "add", "center": [0, 0], "width": 0.12, "softness": 0.6 },
                  { "kind": "brightness_range", "mode": "add", "low": 0.15, "high": 0.8, "softness": 0.4 }
              ] } },
            { "id": 4, "kind": "corrector", "pos": [1, 1], "trim": { "exposure_ev": 1 },
              "masks": { "invert": false, "items": [
                  { "kind": "radial", "mode": "add", "center": [0.4, 0.5], "radii": [0.3, 0.2],
                    "angle_deg": 10, "feather": 0.5 },
                  { "kind": "background", "mode": "add", "level": 0.8, "softness": 0.4,
                    "feather": 0.15, "grow": 0.1, "refine": 0.5 }
              ] } },
            { "id": 5, "kind": "corrector", "pos": [3, 0], "trim": { "temperature": 40 },
              "grade": { "saturation": { "kind": "manual", "value": 35 } },
              "masks": { "invert": false, "items": [
                  { "kind": "linear", "mode": "subtract", "center": [0.5, 0.7], "angle_deg": -15, "feather": 0.25 }
              ] } },
            { "id": 6, "kind": "corrector", "pos": [4, 0], "trim": { "exposure_ev": -0.3 },
              "masks": { "invert": false, "items": [
                  { "kind": "brush", "mode": "add", "feather": 0.25, "grow": 0.3, "refine": 0.5,
                    "strokes": [
                        { "size": 0.14, "hardness": 0.4, "erase": false,
                          "points": [[0.25, 0.4], [0.5, 0.55], [0.75, 0.45]] },
                        { "size": 0.05, "hardness": 1, "erase": true, "points": [[0.5, 0.55]] }
                    ] }
              ] } }
        ],
        "links": [
            picture("input".into(), 1.into()),
            picture(1.into(), 2.into()),
            picture(2.into(), 3.into()),
            picture(3.into(), 5.into()),
            picture(5.into(), 6.into()),
            picture(6.into(), "output".into()),
            picture(1.into(), 4.into()),
            { "from": { "node": 4, "port": "key" }, "to": { "node": 5, "port": "key" } }
        ]
    } })
}

/// The same document without its masks and its key links: what the masks are compared with.
fn without_masks(doc: &serde_json::Value) -> serde_json::Value {
    let mut doc = doc.clone();
    if let Some(items) = doc["nodes"]["items"].as_array_mut() {
        for item in items {
            if let Some(o) = item.as_object_mut() {
                o.remove("masks");
            }
        }
    }
    if let Some(links) = doc["nodes"]["links"].as_array_mut() {
        links.retain(|l| l["from"]["port"] != "key");
    }
    doc
}

/// What `three_correctors` is exported as through the app, then through the binary with the same
/// document as a `--settings` file: the same bytes, a picture that is not the default graph's, and
/// (with a device) a report that says nothing about nodes: the device runs them (N-04). `device` runs
/// the binary with `YELLOWMOON_GPU=on` and gives the app's core the device.
fn three_correctors_leg(device: bool) {
    graph_leg("three correctors", three_correctors(), 6, device);
}

/// A graph of nodes exported through the app and through the binary: the same bytes in every format, a
/// picture that is not the default graph's, and with masks a picture that is not the graph without them.
fn graph_leg(what: &str, doc: serde_json::Value, correctors: usize, device: bool) {
    graph_leg_with(what, doc, correctors, device, |_| {});
}

/// [`graph_leg`] that also hands the app, once it holds the graph, to `check` (a test that has to ask the core something about the
/// photo before the files are compared).
fn graph_leg_with(
    what: &str,
    doc: serde_json::Value,
    correctors: usize,
    device: bool,
    check: impl FnOnce(&App),
) {
    graph_leg_full(what, doc, correctors, device, None, true, check);
}

/// [`graph_leg_with`] with Subject and People made by `vision` on both sides (`None`: neither side can make them), and
/// `acts` saying whether the masks must change a pixel: a leg that asks this computer's own Vision about a synthetic gradient
/// may be told "nothing found", which is the same file as no row.
fn graph_leg_full(
    what: &str,
    doc: serde_json::Value,
    correctors: usize,
    device: bool,
    vision: Option<Vision>,
    acts: bool,
    check: impl FnOnce(&App),
) {
    let sb = Sandbox::new();
    let arw = sb.arw("DSC00001.ARW");
    let app = if device {
        let slot = open_gpu(GpuPolicy::On, MemoryBudget::detect());
        if slot.gpu().is_none() {
            assert!(
                !gpu_required_by_env(),
                "YM_REQUIRE_GPU is set but no device opened: {:?}",
                slot.missing_device_reason()
            );
            eprintln!("\n* GPU LEG SKIPPED in parity: no adapter\n");
            return;
        }
        match vision {
            Some(v) => App::open_with_vision(&[&arw], Some(slot), v),
            None => App::open_on(&[&arw], DEFAULT_METHOD, Some(slot)),
        }
    } else {
        match vision {
            Some(v) => App::open_with_vision(&[&arw], None, v),
            None => App::open(&[&arw]),
        }
    };
    let (settings, clamped) =
        DevelopSettings::from_value(doc.clone(), &app.core.looks.effective_default())
            .expect("settings");
    assert!(clamped.is_empty(), "{clamped:?}");
    assert_eq!(settings.nodes.corrector_count(), correctors);
    let (unmasked, _) =
        DevelopSettings::from_value(without_masks(&doc), &app.core.looks.effective_default())
            .expect("settings without masks");
    let file = sb.root().join("nodes.json");
    std::fs::write(&file, doc.to_string()).expect("write");
    let mut env_pairs: Vec<(&str, &str)> = Vec::new();
    if device {
        env_pairs.push(("YELLOWMOON_GPU", "on"));
    }
    if let Some(v) = vision {
        env_pairs.push(v.env());
    }
    let env: &[(&str, &str)] = &env_pairs;
    let default_graph = app.export(
        ExportOptions {
            format: FormatKind::Tiff,
            bit_depth: BitDepth::Sixteen,
            ..ExportOptions::default()
        },
        &sb.root().join("app-default"),
    );
    let without = if doc.to_string().contains("\"masks\"") {
        app.core
            .session
            .replace_settings(app.photo, unmasked)
            .expect("replace");
        Some(app.export(
            ExportOptions {
                format: FormatKind::Tiff,
                bit_depth: BitDepth::Sixteen,
                ..ExportOptions::default()
            },
            &sb.root().join("app-unmasked"),
        ))
    } else {
        None
    };
    app.core
        .session
        .replace_settings(app.photo, settings)
        .expect("replace");
    check(&app);
    for (name, options, flags) in [
        ("jpeg", ExportOptions::default(), vec!["--format", "jpeg"]),
        (
            "tiff16",
            ExportOptions {
                format: FormatKind::Tiff,
                bit_depth: BitDepth::Sixteen,
                ..ExportOptions::default()
            },
            vec!["--format", "tiff", "--bits", "16"],
        ),
        (
            "dng",
            ExportOptions {
                format: FormatKind::Dng,
                ..ExportOptions::default()
            },
            vec!["--format", "dng"],
        ),
    ] {
        let from_app = app.export(options, &sb.root().join(format!("app-{name}")));
        let mut args = flags;
        args.extend(["--settings", file.to_str().expect("utf-8")]);
        let from_cli = cli_export_env(
            &sb,
            &[&arw],
            &args,
            &sb.root().join(format!("cli-{name}")),
            env,
        );
        assert_same_files(&from_app, &from_cli, &format!("{what}, {name}"));
        if name == "tiff16" {
            // The nodes did something: not the default graph's picture.
            assert!(
                bytes(&from_app[0]) != bytes(&default_graph[0]),
                "the nodes changed no pixel"
            );
            // And the masks did something: not the graph without them.
            if let Some(without) = &without {
                if acts {
                    assert!(
                        bytes(&from_app[0]) != bytes(&without[0]),
                        "the masks changed no pixel"
                    );
                } else {
                    eprintln!(
                        "[vision] {what}: the masks {} a pixel",
                        if bytes(&from_app[0]) == bytes(&without[0]) {
                            "changed no"
                        } else {
                            "changed"
                        }
                    );
                }
            }
        }
    }
    if device {
        // The binary's report says nothing about the nodes: the device rendered them.
        let out = sb
            .ym()
            .envs(env.iter().copied())
            .args([
                "render",
                "--json",
                "--format",
                "jpeg",
                "--auto-provider",
                "yellowmoon",
            ])
            .arg(&arw)
            .args(["--settings", file.to_str().expect("utf-8")])
            .arg("--out")
            .arg(sb.root().join("cli-json"))
            .assert()
            .success();
        let line = String::from_utf8_lossy(&out.get_output().stdout).into_owned();
        let v: serde_json::Value = serde_json::from_str(line.trim_end()).expect("JSON");
        let notes: Vec<&str> = v["notes"]
            .as_array()
            .expect("notes")
            .iter()
            .filter_map(|n| n.as_str())
            .collect();
        assert!(
            !notes.iter().any(|n| n.to_lowercase().contains("node")),
            "{notes:?}"
        );
    }
}

#[test]
fn three_correctors_are_the_same_files_through_the_app_path_and_the_binary() {
    three_correctors_leg(false);
}

/// Masks (N-10, N-13): a Linear and a Radial, a Colour range and a Brightness range, a node that only makes a mask
/// whose key feeds another (with a Background row cut from the photo's blur map, N-15), and a Brush row with Grow,
/// Feather and Refine edges: the app and the binary write the same bytes.
#[test]
fn masked_nodes_are_the_same_files_through_the_app_path_and_the_binary() {
    graph_leg("masked nodes", masked_nodes(), 6, false);
}

/// A Film node (`docs/24` section 4.5) with a mask, halation and grain on, after Main (`docs/24` section 8, *App
/// against `ym`*).
fn film_node() -> serde_json::Value {
    serde_json::json!({ "nodes": { "items": [
        { "id": 2, "kind": "film", "pos": [1, 0], "film": { "stock": "kodak_portra_400", "exposure_ev": 0.3 },
          "print": { "lights": { "y": 4 } },
          "halation": { "amount": 200, "size": 110 }, "grain": { "amount": 150, "seed": 5 }, "strength": 90,
          "masks": { "invert": false, "items": [
              { "kind": "linear", "center": [0.5, 0.5], "angle_deg": 20, "feather": 0.4 } ] } }
    ], "links": [
        { "from": { "node": "input", "port": "picture" }, "to": { "node": 1, "port": "picture" } },
        { "from": { "node": 1, "port": "picture" }, "to": { "node": 2, "port": "picture" } },
        { "from": { "node": 2, "port": "picture" }, "to": { "node": "output", "port": "picture" } }
    ] } })
}

/// The Film node's pictures are the app's and the binary's alike, in JPEG, TIFF 16 and DNG (`graph_leg`) and in PNG 16:
/// the CPU makes them on both sides, whatever the device (`docs/24` section 3.9; F-08b moves the passes to it).
#[test]
fn a_film_node_is_the_same_files_through_the_app_path_and_the_binary() {
    graph_leg("a film node", film_node(), 1, false);
    let sb = Sandbox::new();
    let arw = sb.arw("DSC00001.ARW");
    let app = App::open(&[&arw]);
    let (settings, clamped) =
        DevelopSettings::from_value(film_node(), &app.core.looks.effective_default())
            .expect("settings");
    assert!(clamped.is_empty(), "{clamped:?}");
    app.core
        .session
        .replace_settings(app.photo, settings)
        .expect("replace");
    let file = sb.root().join("film.json");
    std::fs::write(&file, film_node().to_string()).expect("write");
    let options = ExportOptions {
        format: FormatKind::Png,
        bit_depth: BitDepth::Sixteen,
        ..ExportOptions::default()
    };
    let from_app = app.export(options.clone(), &sb.root().join("app-png"));
    let from_cli = cli_export(
        &sb,
        &[&arw],
        &["--format", "png", "--bits", "16", "--settings", file.to_str().expect("utf-8")],
        &sb.root().join("cli-png"),
    );
    assert_same_files(&from_app, &from_cli, "a film node, png16");
    // The file says which film data made it, and a repeat export is the same file.
    let xmp = String::from_utf8_lossy(&bytes(&from_app[0])).into_owned();
    assert!(xmp.contains("<ym:FilmData>kodak_portra_400 on kodak_portra_endura; profiles 0.3.2; engine "));
    let again = app.export(options, &sb.root().join("app-png-again"));
    assert_eq!(bytes(&again[0]), bytes(&from_app[0]), "a repeat export is the same file");
}

/// Whether this machine has SAM's weights (`data/models/sam2`, `tools/fetch/sam2.sh`): the Object tests need them and say so and pass without
/// them, unless `YM_REQUIRE_SAM2` makes that a failure (CI's other jobs fetch none).
fn sam_weights() -> bool {
    let there = ym_core::pipeline::ObjectModel::probe(&data_dir()).is_some_and(|m| m.sam().is_ok());
    if !there {
        assert!(
            std::env::var_os("YM_REQUIRE_SAM2").is_none(),
            "YM_REQUIRE_SAM2 is set and data/models/sam2 is missing"
        );
        eprintln!("\n* OBJECT TESTS SKIPPED in parity: no SAM weights (tools/fetch/sam2.sh)\n");
    }
    there
}

/// Node 2 with this Object row (an exposure trim of a stop, so the mask shows in the picture), the graph of `docs/10`.
fn object_doc(row: serde_json::Value) -> serde_json::Value {
    node_2_doc(vec![row])
}

/// Node 2 with these mask rows (an exposure trim of a stop, so the masks show in the picture).
fn node_2_doc(rows: Vec<serde_json::Value>) -> serde_json::Value {
    serde_json::json!({ "nodes": { "items": [
        { "id": 2, "kind": "corrector", "pos": [1, 0], "trim": { "exposure_ev": 1.0 },
          "masks": { "invert": false, "items": rows } }
    ], "links": [
        { "from": { "node": "input", "port": "picture" }, "to": { "node": 1, "port": "picture" } },
        { "from": { "node": 1, "port": "picture" }, "to": { "node": 2, "port": "picture" } },
        { "from": { "node": 2, "port": "picture" }, "to": { "node": "output", "port": "picture" } }
    ] } })
}

/// The Object row of the parity tests: a click on the photo, a click taking a part away, Grow, Feather and Refine edges, and the box that
/// holds them. On the synthetic gradient the model finds an object at these places (the test asks the core before it compares anything:
/// a row that found nothing would be the same file as no row).
fn object_row() -> serde_json::Value {
    serde_json::json!({
        "kind": "object", "mode": "add", "grow": 0.1, "feather": 0.2, "refine": 0.5,
        "points": [[0.5, 0.5, true], [0.85, 0.85, false]], "box": [0.1, 0.1, 0.9, 0.9]
    })
}

/// The app's click is the same row `ym` reads: asks the core of the app (`mask_find`) for the row's own prompt and says what it found.
fn object_is_found(app: &App) {
    use ym_core::pipeline::{MaskFound, ObjectPrompt};
    let found = app
        .core
        .mask_find(
            app.photo,
            2,
            0,
            ObjectPrompt::Point {
                x: 0.5,
                y: 0.5,
                positive: true,
            },
            &ym_gpu::queue::CancelToken::new(),
        )
        .expect("mask_find");
    assert!(
        matches!(found, MaskFound::Found { .. }),
        "the model finds an object at the middle of the synthetic photo: {found:?}"
    );
}

/// Object masks (N-18): an Object row with points, a negative point, a box, Grow, Feather and Refine edges is the same files through the app
/// path (which has found the object with a click, `mask_find`, and holds the encoder's features of the viewer) and through the binary (which
/// makes its own picture and features at export time from the stored points): the CPU leg.
#[test]
fn an_object_mask_is_the_same_files_through_the_app_path_and_the_binary() {
    if !sam_weights() {
        return;
    }
    graph_leg_with(
        "an object mask",
        object_doc(object_row()),
        2,
        false,
        object_is_found,
    );
}

/// The device leg of the Object mask.
#[test]
fn an_object_mask_is_the_same_files_through_the_app_path_and_the_binary_on_the_device() {
    if !sam_weights() {
        return;
    }
    graph_leg_with(
        "an object mask",
        object_doc(object_row()),
        2,
        true,
        object_is_found,
    );
}

/// The device leg of the masks.
/// A Subject row with Grow, Feather and Refine edges, and a People row taken away from it: the rows of the Vision legs.
fn vision_rows() -> Vec<serde_json::Value> {
    vec![
        serde_json::json!({ "kind": "subject", "mode": "add", "grow": 0.1, "feather": 0.2, "refine": 0.5 }),
        serde_json::json!({ "kind": "people", "mode": "subtract", "feather": 0.1, "refine": 0.3 }),
    ]
}

/// Subject and People masks (N-19) with the deterministic fake on both sides (`FakeVision` for the app, `YELLOWMOON_TEST_VISION=fake`
/// for the binary): the same files in JPEG, 16-bit TIFF and DNG, the rows change pixels, and the app, which makes its own masks at export
/// time as `ym` does, agrees with it. The CPU leg; runs on every platform.
#[test]
fn subject_and_people_masks_are_the_same_files_through_the_app_path_and_the_binary() {
    graph_leg_full(
        "subject and people",
        node_2_doc(vision_rows()),
        2,
        false,
        Some(Vision::Fake),
        true,
        |_| {},
    );
}

/// The device leg of the Subject and People masks.
#[test]
fn subject_and_people_masks_are_the_same_files_through_the_app_path_and_the_binary_on_the_device() {
    graph_leg_full(
        "subject and people",
        node_2_doc(vision_rows()),
        2,
        true,
        Some(Vision::Fake),
        true,
        |_| {},
    );
}

/// This Mac's own Vision framework on both sides: the app and `ym` ask Vision for the same picture and write the same files. Where
/// Vision does not run the requests (a virtual Mac, a computer that is not a Mac) the test says so plainly and passes; set
/// `YM_REQUIRE_VISION=1` to make that a failure. Vision may find nothing in the synthetic gradient, which is the same file as no row
/// (the legs print which); `ym-apple`'s own tests check what Vision makes.
#[test]
fn subject_and_people_masks_are_the_same_files_through_the_app_path_and_the_binary_with_this_computers_vision()
 {
    let support = ym_apple::vision::probe();
    println!("[vision] this computer: {support:?}");
    if !(support.subject.is_supported() && support.people.is_supported()) {
        assert!(
            std::env::var_os("YM_REQUIRE_VISION").is_none(),
            "YM_REQUIRE_VISION is set and this computer does not support Subject and People: {support:?}"
        );
        println!(
            "\n* VISION LEG SKIPPED in parity: this computer does not support the Vision requests ({support:?})\n"
        );
        return;
    }
    graph_leg_full(
        "subject and people, this computer's Vision",
        node_2_doc(vision_rows()),
        2,
        false,
        Some(Vision::System),
        false,
        |_| {},
    );
}

#[test]
fn masked_nodes_are_the_same_files_through_the_app_path_and_the_binary_on_the_device() {
    graph_leg("masked nodes", masked_nodes(), 6, true);
}

/// The device leg: both sides have a device and both render the nodes on it, so the files are the same.
#[test]
fn three_correctors_are_the_same_files_through_the_app_path_and_the_binary_on_the_device() {
    three_correctors_leg(true);
}

#[test]
fn a_colour_warp_is_the_same_picture_through_the_app_and_the_binary() {
    let sb = Sandbox::new();
    let arw = sb.arw("DSC00001.ARW");
    let app = App::open(&[&arw]);
    // A white balance far from the light gives the photo colours for the pins to move: a moving
    // pin on the neutral axis with the largest range (it reaches every colour of the photo), a
    // second one with tonal weights, a pivot, an exposure and a saturation, a third that only
    // changes saturation, and a lock. Pins written short, as a settings file may.
    let doc = common::field_doc(serde_json::json!({
        "white_balance": { "kind": "manual", "kelvin": 2500, "tint": 30 },
        "color_warp": { "pins": [
            { "source": [0.0, 0.0], "target": [0.06, 0.04], "range": 0.25 },
            { "source": [-0.1, -0.1], "target": [-0.05, -0.12], "range": 0.1, "shadows": 0.3,
              "highlights": 0.9, "pivot_ev": -1.0, "exposure_ev": 0.4, "saturation": -0.6 },
            { "source": [0.05, 0.1], "range": 0.2, "saturation": 0.5 },
            { "source": [0.15, -0.05], "locked": true }
        ] }
    }));
    let (settings, clamped) =
        DevelopSettings::from_value(doc.clone(), &app.core.looks.effective_default())
            .expect("settings");
    assert!(clamped.is_empty(), "{clamped:?}");
    assert_eq!(settings.main_grade().color_warp.pins.len(), 4);
    app.core
        .session
        .replace_settings(app.photo, settings)
        .expect("replace");
    // The JPEG (the finished picture) and the DNG (the warp travels in the LogC4 grade into it).
    let file = sb.root().join("warp.json");
    std::fs::write(&file, doc.to_string()).expect("write");
    for (name, format) in [("jpeg", FormatKind::Jpeg), ("dng", FormatKind::Dng)] {
        let options = ExportOptions {
            format,
            ..ExportOptions::default()
        };
        let from_app = app.export(options, &sb.root().join(format!("app-{name}")));
        let from_cli = cli_export(
            &sb,
            &[&arw],
            &[
                "--format",
                name,
                "--settings",
                file.to_str().expect("utf-8"),
            ],
            &sb.root().join(format!("cli-{name}")),
        );
        assert_same_files(&from_app, &from_cli, &format!("colour warp, {name}"));
        // The pins, saturation included, are in the file's `ym:Settings`, as every setting is.
        if name == "jpeg" {
            let xmp = String::from_utf8_lossy(&bytes(&from_app[0])).replace("&quot;", "\"");
            for saturation in ["\"saturation\":-0.6", "\"saturation\":0.5"] {
                assert!(xmp.contains(saturation), "ym:Settings holds {saturation}");
            }
        }
    }
}

#[test]
fn a_lens_is_the_same_file_through_the_app_the_options_and_a_settings_file() {
    // `docs/21` section 14: `ym render --lens vintage --flares 60 --anamorphic 50` equals the same
    // settings through `--settings` and the app's export, byte for byte. The settings travel in the
    // file's `ym:Settings`, so the three agree on every part of the lens; the pixels follow when
    // the stages arrive (the lens is deterministic, so the bytes will keep agreeing).
    let sb = Sandbox::new();
    let arw = sb.arw("DSC00001.ARW");
    let data = sb.data_with_lenses(true);
    let library = ym_lens::LensLibrary::from_dir(&data.join("lenses")).expect("lenses");
    let start_edges = library
        .lenses()
        .iter()
        .find(|l| l.kind == ym_core::settings::LensKind::Vintage)
        .expect("vintage")
        .start_edges;
    let app = App::open_in(&[&arw], &data);
    let patch = SettingsPatch::default().and(
        NodeId::Input,
        NodePatch {
            lens: Some(ym_core::settings::LensPatch {
                kind: Some(ym_core::settings::LensKind::Vintage),
                flares: Some(60.0),
                edges: Some(start_edges),
                anamorphic: Some(50.0),
            }),
            ..NodePatch::default()
        },
    );
    app.core
        .session
        .update_settings(app.photo, &patch)
        .expect("patch");
    let data_env = data.to_str().expect("utf-8");
    let from_app = app.export(ExportOptions::default(), &sb.root().join("app"));
    let from_options = cli_export_env(
        &sb,
        &[&arw],
        &["--lens", "vintage", "--flares", "60", "--anamorphic", "50"],
        &sb.root().join("options"),
        &[("YELLOWMOON_DATA", data_env)],
    );
    assert_same_files(&from_app, &from_options, "lens as options");
    let file = sb.root().join("lens.json");
    std::fs::write(
        &file,
        common::field_doc(serde_json::json!({ "lens": { "kind": "vintage", "flares": 60, "edges": start_edges, "anamorphic": 50 } }))
            .to_string(),
    )
    .expect("write");
    let from_file = cli_export_env(
        &sb,
        &[&arw],
        &["--settings", file.to_str().expect("utf-8")],
        &sb.root().join("file"),
        &[("YELLOWMOON_DATA", data_env)],
    );
    assert_same_files(&from_app, &from_file, "lens as a settings file");
    // And the file says which lens it was made with.
    let xmp = String::from_utf8_lossy(&bytes(&from_app[0])).into_owned();
    assert!(xmp.contains("vintage"), "ym:Settings holds the lens");
}

#[test]
fn a_lens_is_the_same_picture_through_the_app_and_the_binary() {
    // `docs/21` sections 9 and 14: the lens is deterministic, so the app's export and `ym render
    // --lens vintage --flares 60 --edges 50 --anamorphic 50` write the same bytes, with the shipped
    // lens data. A 640 x 480 photo is a frame whose long edge is under 1024 px (its lens plane is the
    // frame) and whose tiles cross the 512 px line. The JPEG is the finished picture, the DNG carries
    // the lens in its raw pixels and in its preview; both must also differ from the same export
    // without a lens, or the test would pass for a lens that does nothing.
    let sb = Sandbox::new();
    let arw = sb.arw_sized("DSC00001.ARW", 640, 480, 11);
    let app = App::open(&[&arw]);
    let vintage = ym_core::settings::LensPatch {
        kind: Some(ym_core::settings::LensKind::Vintage),
        flares: Some(60.0),
        edges: Some(50.0),
        anamorphic: Some(50.0),
    };
    let off: Vec<(&str, Vec<u8>)> = ["jpeg", "dng"]
        .into_iter()
        .map(|name| {
            let format = if name == "dng" {
                FormatKind::Dng
            } else {
                FormatKind::Jpeg
            };
            let options = ExportOptions {
                format,
                ..ExportOptions::default()
            };
            let files = app.export(options, &sb.root().join(format!("off-{name}")));
            (name, bytes(&files[0]))
        })
        .collect();
    app.core
        .session
        .update_settings(
            app.photo,
            &SettingsPatch::default().and(
                NodeId::Input,
                NodePatch {
                    lens: Some(vintage),
                    ..NodePatch::default()
                },
            ),
        )
        .expect("patch");
    for (name, plain) in off {
        let format = if name == "dng" {
            FormatKind::Dng
        } else {
            FormatKind::Jpeg
        };
        let options = ExportOptions {
            format,
            ..ExportOptions::default()
        };
        let from_app = app.export(options, &sb.root().join(format!("app-{name}")));
        let from_cli = cli_export(
            &sb,
            &[&arw],
            &[
                "--format",
                name,
                "--lens",
                "vintage",
                "--flares",
                "60",
                "--edges",
                "50",
                "--anamorphic",
                "50",
            ],
            &sb.root().join(format!("cli-{name}")),
        );
        assert_same_files(&from_app, &from_cli, &format!("lens, {name}"));
        assert!(
            bytes(&from_app[0]) != plain,
            "{name}: the lens changed nothing in the file"
        );
    }
    // The file says which lens data made it (`ym:Models`, `docs/09`).
    let jpeg = app.export(ExportOptions::default(), &sb.root().join("app-models"));
    let text = String::from_utf8_lossy(&bytes(&jpeg[0])).into_owned();
    assert!(
        text.contains("lens vintage ") && text.contains(", anamorphic "),
        "ym:Models names the lens data"
    );
}

#[test]
fn a_full_anamorphic_bend_is_the_same_picture_through_the_app_and_the_binary() {
    // `docs/21` sections 4 and 14 (L4): the bend, the lens's distortion and the zoom are geometry,
    // applied while the export samples its tiles. At Anamorphic 100 (the strongest bend and zoom), with
    // a quarter turn and a straighten angle (the bend acts on the turned canvas and the straightening
    // acts on the bent one), the app's export and `ym` write the same bytes: the JPEG (the baked
    // picture) and the DNG (the bend travels in `Geometry::upright`, in the buffer's own orientation,
    // for the raw pixels and the preview). A 640 x 480 photo is a frame whose tiles cross the 512 px
    // line. Both must differ from the same export without the bend, or the test would pass for a bend
    // that does nothing.
    let sb = Sandbox::new();
    let arw = sb.arw_sized("DSC00001.ARW", 640, 480, 11);
    let app = App::open(&[&arw]);
    let doc = |anamorphic: f64, edges: f64| {
        common::field_doc(serde_json::json!({
            "lens": { "kind": "vintage", "flares": 0, "edges": edges, "anamorphic": anamorphic },
            "crop": { "rect": { "x": 0.04, "y": 0.06, "width": 0.84, "height": 0.86 }, "angle_deg": 3.0, "aspect": "free" },
            "orientation": 90
        }))
    };
    let apply = |value: serde_json::Value| {
        let (settings, clamped) =
            DevelopSettings::from_value(value, &app.core.looks.effective_default())
                .expect("settings");
        assert!(clamped.is_empty(), "{clamped:?}");
        app.core
            .session
            .replace_settings(app.photo, settings)
            .expect("replace");
    };
    let file_of = |name: &str, value: serde_json::Value| {
        let file = sb.root().join(name);
        std::fs::write(&file, value.to_string()).expect("write");
        file
    };
    let straight = file_of("straight.json", doc(0.0, 0.0));
    let bent = file_of("bent.json", doc(100.0, 50.0));
    for (name, format) in [("jpeg", FormatKind::Jpeg), ("dng", FormatKind::Dng)] {
        let options = || ExportOptions {
            format,
            ..ExportOptions::default()
        };
        apply(doc(0.0, 0.0));
        let without = cli_export(
            &sb,
            &[&arw],
            &[
                "--format",
                name,
                "--settings",
                straight.to_str().expect("utf-8"),
            ],
            &sb.root().join(format!("without-{name}")),
        );
        apply(doc(100.0, 50.0));
        let from_app = app.export(options(), &sb.root().join(format!("app-{name}")));
        let from_cli = cli_export(
            &sb,
            &[&arw],
            &[
                "--format",
                name,
                "--settings",
                bent.to_str().expect("utf-8"),
            ],
            &sb.root().join(format!("cli-{name}")),
        );
        assert_same_files(&from_app, &from_cli, &format!("anamorphic bend, {name}"));
        assert!(
            bytes(&from_app[0]) != bytes(&without[0]),
            "{name}: the bend changed nothing in the file"
        );
        // And with the options instead of the file, for the JPEG: the same picture.
        if name == "jpeg" {
            apply(common::field_doc(serde_json::json!({
                "lens": { "kind": "vintage", "flares": 0, "edges": 50, "anamorphic": 100 },
            })));
            let plain_app = app.export(options(), &sb.root().join("app-options"));
            let plain_cli = cli_export(
                &sb,
                &[&arw],
                &[
                    "--format",
                    "jpeg",
                    "--lens",
                    "vintage",
                    "--flares",
                    "0",
                    "--edges",
                    "50",
                    "--anamorphic",
                    "100",
                ],
                &sb.root().join("cli-options"),
            );
            assert_same_files(&plain_app, &plain_cli, "anamorphic bend as options");
        }
    }
}

#[test]
fn the_settings_an_export_records_reproduce_the_file() {
    let sb = Sandbox::new();
    let arw = sb.arw("DSC00001.ARW");
    let out = sb
        .ym()
        .args([
            "render",
            "--json",
            "--auto-provider",
            "yellowmoon",
            "--format",
            "png",
            "--bits",
            "8",
            "--contrast",
            "25",
            "--out",
        ])
        .arg(sb.root().join("second"))
        .arg(&arw)
        .assert()
        .success();
    let reported = common::json_lines(&out.get_output().stdout).remove(0);
    let settings_file = sb.root().join("again.json");
    std::fs::write(&settings_file, reported["settings"].to_string()).expect("write");
    let again = sb
        .ym()
        .args([
            "render",
            "--json",
            "--auto-provider",
            "yellowmoon",
            "--format",
            "png",
            "--bits",
            "8",
            "--out",
        ])
        .arg(sb.root().join("third"))
        .arg("--settings")
        .arg(&settings_file)
        .arg(&arw)
        .assert()
        .success();
    let again = common::json_lines(&again.get_output().stdout).remove(0);
    // The pinned settings (exposure as a number, white balance still Auto) render the same bytes.
    assert_eq!(again["settings"], reported["settings"]);
    assert_eq!(
        bytes(Path::new(again["output"].as_str().expect("output"))),
        bytes(Path::new(reported["output"].as_str().expect("output")))
    );
    // The auto exposure is pinned as the number it resolved to.
    assert!(matches!(
        serde_json::from_value::<DevelopSettings>(reported["settings"].clone())
            .expect("settings")
            .input()
            .exposure,
        AutoOr::Manual(_)
    ));
}

// ---------------------------------------------------------------------------------------------
// The R6 method forced for the whole render (the default is the adaptive ratio demosaic)
// ---------------------------------------------------------------------------------------------

/// The test-only switch: the `ym` binary of this test run reads it (the `force-demosaic` feature of
/// `ym-core`, enabled by this crate's dev-dependency), and the app's side sets the same method on
/// `RenderOptions::demosaic`. The default method is the adaptive ratio demosaic, which every other
/// test of this file runs; these tests keep the methods that are no longer the default honest.
const FORCE_MENON: [(&str, &str); 1] = [("YM_FORCE_DEMOSAIC", "menon")];
/// The default method forced by name.
const FORCE_ADAPTIVE_RATIO: [(&str, &str); 1] = [("YM_FORCE_DEMOSAIC", "adaptive_ratio")];

#[test]
fn every_format_is_byte_identical_with_menon_forced() {
    assert_eq!(
        DEFAULT_METHOD,
        Method::AdaptiveRatio,
        "Menon is not the default"
    );
    let sb = Sandbox::new();
    let arw = sb.arw("DSC00001.ARW");
    let app = App::open_with(&[&arw], Method::Menon);
    for case in cases() {
        let app_dir = sb.root().join("app").join(case.name);
        let cli_dir = sb.root().join("cli").join(case.name);
        let from_app = app.export(case.options.clone(), &app_dir);
        let from_cli = cli_export_env(&sb, &[&arw], case.flags, &cli_dir, &FORCE_MENON);
        assert_same_files(&from_app, &from_cli, case.name);
        assert_eq!(names_in(&app_dir).len(), 1, "{}", case.name);
        assert_eq!(names_in(&cli_dir).len(), 1, "{}", case.name);
    }
}

/// The default of R6 is the adaptive ratio demosaic: the binary without the switch gives the bytes
/// of `YM_FORCE_DEMOSAIC=adaptive_ratio`, and Menon, forced, gives other bytes for the same photo
/// (so the switch does something).
#[test]
fn the_default_export_is_the_adaptive_ratio_demosaic_and_menon_is_another_picture() {
    let sb = Sandbox::new();
    let arw = sb.arw("DSC00001.ARW");
    let png = ["--format", "png", "--bits", "16"];
    let default = cli_export(&sb, &[&arw], &png, &sb.root().join("default"));
    let forced_ar = cli_export_env(
        &sb,
        &[&arw],
        &png,
        &sb.root().join("adaptive_ratio"),
        &FORCE_ADAPTIVE_RATIO,
    );
    let forced_menon = cli_export_env(&sb, &[&arw], &png, &sb.root().join("menon"), &FORCE_MENON);
    assert_eq!(
        bytes(&forced_ar[0]),
        bytes(&default[0]),
        "the adaptive ratio demosaic is the default"
    );
    assert_ne!(
        bytes(&forced_menon[0]),
        bytes(&default[0]),
        "forced Menon = default?"
    );
}

/// The tile seams and the thread layout are part of the contract for every method: with Menon
/// forced a photo of 3 x 2 export tiles is the same bytes through the app and through the binary
/// at `--jobs 1` and `--jobs 3`. (`a_photo_of_several_export_tiles_is_identical_whatever_the_thread_layout`
/// above is the same photo with the default method, the adaptive ratio demosaic with its halo of 13.)
#[test]
fn a_photo_of_several_export_tiles_is_identical_with_menon_forced() {
    let sb = Sandbox::new();
    let arw = sb.arw_sized("big.ARW", 2100, 1100, 9);
    let app = App::open_with(&[&arw], Method::Menon);
    let png = ExportOptions {
        format: FormatKind::Png,
        bit_depth: BitDepth::Sixteen,
        ..ExportOptions::default()
    };
    let from_app = app.export(png, &sb.root().join("app"));
    let flags = |jobs: &'static str| ["--format", "png", "--bits", "16", "--jobs", jobs];
    let one = cli_export_env(
        &sb,
        &[&arw],
        &flags("1"),
        &sb.root().join("one"),
        &FORCE_MENON,
    );
    assert_same_files(&from_app, &one, "several tiles, Menon");
    let three = cli_export_env(
        &sb,
        &[&arw],
        &flags("3"),
        &sb.root().join("three"),
        &FORCE_MENON,
    );
    assert_same_files(&one, &three, "jobs 1 and 3, Menon");
}

/// A bad value in the switch is an error, not a silent default (the switch exists only in test
/// builds, where a typo must show).
#[test]
fn the_test_switch_refuses_a_name_that_is_not_a_method() {
    let sb = Sandbox::new();
    let arw = sb.arw("DSC00001.ARW");
    sb.ym()
        .env("YM_FORCE_DEMOSAIC", "adaptive-ratio")
        .arg("render")
        .arg(&arw)
        .arg("--out")
        .arg(sb.root().join("out"))
        .assert()
        .failure();
}
