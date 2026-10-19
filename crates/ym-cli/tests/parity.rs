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
//! Apple Auto (`docs/19` section 7). The app's side here has no advisor (Apple Auto off) and the
//! binary is given `--apple-auto off`, so a Mac with Apple Intelligence compares today's bytes too.
//! The Apple Auto leg gives both sides the test advisor (`FakeAdvisor` in the core, the binary
//! `YELLOWMOON_TEST_APPLE_AUTO=fake` and no option, so the app's setting decides as in the app): the
//! same answers, the same files.

mod common;

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use common::{Sandbox, data_dir, names_in};
use ym_core::app::gpu::{GpuPolicy, GpuSlot, open_gpu};
use ym_core::app::{Core, CoreOptions};
use ym_core::apple_auto::FakeAdvisor;
use ym_core::export_queue::ExportJob;
use ym_core::scheduler::MemoryBudget;
use ym_core::session::{ApplyScope, OpenOptions, PhotoId};
use ym_core::settings::{AutoOr, DevelopSettings, SettingsPatch};
use ym_export::{BitDepth, ColorSpace, ExportOptions, FormatKind, Size, TiffCompression};
use ym_gpu::cpu::demosaic::{DEFAULT_METHOD, Method};
use ym_gpu::device::gpu_required_by_env;

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
        Self::open_full(files, DEFAULT_METHOD, None, true)
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
        Self::open_full(files, method, slot, false)
    }

    fn open_full(files: &[&Path], method: Method, slot: Option<GpuSlot>, fake: bool) -> Self {
        let home = tempfile::tempdir().expect("tempdir");
        let mut options = CoreOptions::new(data_dir(), home.path().to_path_buf());
        // What the CLI does: no thumbnail renders competing with the export.
        options.thumbnails = false;
        options.render.demosaic = method;
        if fake {
            options.advisor = Some(Arc::new(FakeAdvisor));
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
    // Apple Auto off, as on the app's side, unless the test gives both sides the test advisor.
    let apple_auto: &[&str] = if env.iter().any(|(k, _)| *k == common::FAKE_APPLE_AUTO.0) {
        &[]
    } else {
        &["--apple-auto", "off"]
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
    assert!(
        app.core.apple_auto_status().active,
        "the app's Apple Auto acts"
    );
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
                text.contains("<ym:Auto>apple</ym:Auto>") && text.contains("<ym:AppleAuto>"),
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
            "--apple-auto",
            "off",
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
    let patch: SettingsPatch = serde_json::from_value(serde_json::json!({
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
    }))
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
                &SettingsPatch {
                    output_sharpening: Some(amount),
                    ..SettingsPatch::default()
                },
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
    let doc = serde_json::json!({
        "exposure": { "kind": "manual", "ev": -0.25 },
        "color_balance": { "shadows": [0.1, -0.1], "midtones": [0.0, 0.0], "highlights": [-0.2, 0.15] },
        "curve": { "luma": [[0.0, 0.0], [0.25, 0.2], [0.75, 0.8], [1.0, 1.0]] },
        "crop": { "rect": { "x": 0.1, "y": 0.1, "width": 0.6, "height": 0.7 }, "angle_deg": 2.0, "aspect": "free" },
        "orientation": 90
    });
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

#[test]
fn a_colour_warp_is_the_same_picture_through_the_app_and_the_binary() {
    let sb = Sandbox::new();
    let arw = sb.arw("DSC00001.ARW");
    let app = App::open(&[&arw]);
    // A white balance far from the light gives the photo colours for the pins to move: a moving
    // pin on the neutral axis with the largest range (it reaches every colour of the photo), a
    // second one with tonal weights, a pivot and an exposure, and a lock. Pins written short, as a
    // settings file may.
    let doc = serde_json::json!({
        "white_balance": { "kind": "manual", "kelvin": 2500, "tint": 30 },
        "color_warp": { "pins": [
            { "source": [0.0, 0.0], "target": [0.06, 0.04], "range": 0.25 },
            { "source": [-0.1, -0.1], "target": [-0.05, -0.12], "range": 0.1, "shadows": 0.3,
              "highlights": 0.9, "pivot_ev": -1.0, "exposure_ev": 0.4 },
            { "source": [0.15, -0.05], "locked": true }
        ] }
    });
    let (settings, clamped) =
        DevelopSettings::from_value(doc.clone(), &app.core.looks.effective_default())
            .expect("settings");
    assert!(clamped.is_empty(), "{clamped:?}");
    assert_eq!(settings.color_warp.pins.len(), 3);
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
            "--apple-auto",
            "off",
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
            "--apple-auto",
            "off",
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
