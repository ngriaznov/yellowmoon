//! App and CLI parity (`docs/09`, `docs/13`, M3-14): the same job exported through the app's path
//! (`ym_core::app::Core::export_start`, what the Tauri commands call) and through the `ym` binary
//! gives byte-identical files, for every format and option.
//!
//! The app side runs in this process; the CLI side is a separate process. Each side has its own
//! app folder, its own renderer and its own thread layout, so the test also stands for "different
//! machines of the same build".

mod common;

use std::path::{Path, PathBuf};
use std::time::Duration;

use common::{Sandbox, data_dir, names_in};
use ym_core::app::{Core, CoreOptions};
use ym_core::export_queue::ExportJob;
use ym_core::session::{ApplyScope, OpenOptions, PhotoId};
use ym_core::settings::{AutoOr, DevelopSettings, SettingsPatch};
use ym_export::{BitDepth, ColorSpace, ExportOptions, FormatKind, Size, TiffCompression};

/// The app's side: a core over its own app folder with one photo open.
struct App {
    core: Core,
    photo: PhotoId,
    _home: tempfile::TempDir,
}

impl App {
    fn open(files: &[&Path]) -> Self {
        let home = tempfile::tempdir().expect("tempdir");
        let mut options = CoreOptions::new(data_dir(), home.path().to_path_buf());
        // What the CLI does: no thumbnail renders competing with the export.
        options.thumbnails = false;
        let core = Core::open(options).expect("core");
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
    let out = sb
        .ym()
        .arg("render")
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
fn the_settings_an_export_records_reproduce_the_file() {
    let sb = Sandbox::new();
    let arw = sb.arw("DSC00001.ARW");
    let out = sb
        .ym()
        .args([
            "render",
            "--json",
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
            "render", "--json", "--format", "png", "--bits", "8", "--out",
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
