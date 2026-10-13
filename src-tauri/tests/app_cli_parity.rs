//! App and CLI parity across separate builds (`docs/13` "App/CLI parity", `AGENTS.md` rule 6).
//!
//! The same synthetic ARW is exported in every format through the app's export path and through a `ym` binary, and the
//! files must be byte for byte the same.
//!
//! * The app side is `ym_core::app::Core::export_start`, what the Tauri commands call, compiled here in the dependency
//!   graph of `yellowmoon-app` (cargo unifies the features of a shared crate within one build, so this graph can differ from
//!   the one `ym` is built in: that is what this test exists to catch).
//! * The CLI side is the binary named by `YM_BIN`, which the CI job `parity` builds by itself with `cargo build -p ym-cli`
//!   (the way `tools/dev/prepare-sidecar.sh` builds the app's sidecar), in a cargo invocation that shares nothing with this
//!   one. `crates/ym-cli/tests/parity.rs` is the same comparison in one invocation, where both sides get the same features.
//!
//! Headless: nothing here opens a window (no webview, no display; the Tauri runtime is not started). Without `YM_BIN` the
//! test says so and passes, unless `YM_REQUIRE_PARITY_BIN` is set (the CI job sets it).
//!
//! `YM_BIN=$PWD/target/debug/ym cargo test -p yellowmoon-app --test app_cli_parity`

use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Duration;

use ym_core::app::{Core, CoreOptions};
use ym_core::export_queue::ExportJob;
use ym_core::session::{ApplyScope, OpenOptions};
use ym_export::{
    BitDepth, ColorSpace, ExportOptions, FormatKind, JpegQuality, Size, TiffCompression,
};
use ym_raw::testgen::{ArwBuilder, gradient_mosaic};

/// The `ym` binary under test, or `None` (with a message) when `YM_BIN` is not set.
fn command_line_tool() -> Option<PathBuf> {
    match std::env::var_os("YM_BIN") {
        Some(path) => {
            let path = PathBuf::from(path);
            assert!(
                path.is_file(),
                "YM_BIN={} is not a file (build it with `cargo build -p ym-cli`)",
                path.display()
            );
            Some(path)
        }
        None => {
            assert!(
                std::env::var_os("YM_REQUIRE_PARITY_BIN").is_none(),
                "YM_REQUIRE_PARITY_BIN is set but YM_BIN is not"
            );
            eprintln!(
                "skipped: YM_BIN is not set (cargo build -p ym-cli, then YM_BIN=target/debug/ym)"
            );
            None
        }
    }
}

fn data_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../data")
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
                jpeg_quality: JpegQuality::new(70).expect("quality"),
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

/// Exports `arw` once with the app's path and once with `ym render`, and returns the two lists of files.
fn export_both(
    core: &Core,
    ym: &Path,
    arw: &Path,
    root: &Path,
    case: &Case,
) -> (Vec<PathBuf>, Vec<PathBuf>) {
    let app_dir = root.join("app").join(case.name);
    let job = ExportJob {
        scope: ApplyScope::All,
        options: ExportOptions {
            destination: Some(app_dir),
            ..case.options.clone()
        },
    };
    let id = core.export_start(&job).expect("the export starts");
    assert!(
        core.exports.wait_idle(Duration::from_secs(300)),
        "{}: the export finishes",
        case.name
    );
    let summary = core.export_summary(id).expect("summary");
    assert_eq!(summary.failed, 0, "{}: {:?}", case.name, summary.errors);

    let cli_dir = root.join("cli").join(case.name);
    let output = Command::new(ym)
        .arg("render")
        .arg(arw)
        .args(case.flags)
        .arg("--out")
        .arg(&cli_dir)
        .env("YELLOWMOON_HOME", root.join("cli-home"))
        .env("YELLOWMOON_DATA", data_dir())
        .env("HOME", root)
        .env_remove("RUST_LOG")
        .env_remove("XDG_DATA_HOME")
        .current_dir(root)
        .output()
        .expect("run ym");
    assert!(
        output.status.success(),
        "{}: ym render failed: {}",
        case.name,
        String::from_utf8_lossy(&output.stderr)
    );
    let from_cli = String::from_utf8_lossy(&output.stdout)
        .lines()
        .map(PathBuf::from)
        .collect();
    (summary.files, from_cli)
}

#[test]
fn every_format_is_byte_identical_between_the_app_graph_and_a_separately_built_ym() {
    let Some(ym) = command_line_tool() else {
        return;
    };
    let dir = tempfile::tempdir().expect("tempdir");
    let root = dir.path();
    std::fs::create_dir_all(root.join("photos")).expect("photos");
    std::fs::create_dir_all(root.join("cli-home")).expect("cli home");
    let arw = root.join("photos").join("DSC00001.ARW");
    ArwBuilder::new("ILCE-7RM4", 160, 120)
        .write_to(&arw, &gradient_mosaic(160, 120, 512, 16_000, 11))
        .expect("write a synthetic ARW");

    // The app's side: a core over its own app folder, with one photo open and developed. No thumbnails, as in the CLI.
    let app_home = tempfile::tempdir().expect("app home");
    let mut options = CoreOptions::new(data_dir(), app_home.path().to_path_buf());
    options.thumbnails = false;
    let core = Core::open(options).expect("core");
    let outcome = core
        .session
        .open(std::slice::from_ref(&arw), OpenOptions::default());
    assert_eq!(outcome.added.len(), 1, "{outcome:?}");
    assert!(
        core.session.wait_idle(Duration::from_secs(120)),
        "the photo develops"
    );

    let cases = cases();
    assert_eq!(
        cases.len(),
        11,
        "JPEG (2), TIFF (5), PNG (3) and DNG, as in crates/ym-cli/tests/parity.rs"
    );
    for case in &cases {
        let (from_app, from_cli) = export_both(&core, &ym, &arw, root, case);
        assert_eq!(
            from_app.len(),
            1,
            "{}: one file from the app: {from_app:?}",
            case.name
        );
        assert_eq!(
            from_cli.len(),
            1,
            "{}: one file from ym: {from_cli:?}",
            case.name
        );
        assert_eq!(
            from_app[0].file_name(),
            from_cli[0].file_name(),
            "{}: the same file name",
            case.name
        );
        let (a, c) = (
            std::fs::read(&from_app[0]).expect("read the app's file"),
            std::fs::read(&from_cli[0]).expect("read ym's file"),
        );
        assert!(!a.is_empty(), "{}", case.name);
        assert!(
            a == c,
            "{}: {} ({} bytes) and {} ({} bytes) differ",
            case.name,
            from_app[0].display(),
            a.len(),
            from_cli[0].display(),
            c.len()
        );
    }
}
