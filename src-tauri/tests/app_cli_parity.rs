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
//! Apple Auto (`docs/19`) is off on both sides: the app's core has no advisor and its mode is off, and `ym` runs with
//! `--apple-auto off`, so the bytes are today's on every machine, a Mac with Apple Intelligence included.
//!
//! Headless: nothing here opens a window (no webview, no display; the Tauri runtime is not started). Without `YM_BIN` the
//! test says so and passes, unless `YM_REQUIRE_PARITY_BIN` is set (the CI job sets it).
//!
//! `YM_BIN=$PWD/target/debug/ym cargo test -p yellowmoon-app --test app_cli_parity`
//!
//! The R6 demosaic method forced (`docs/algorithms/01-adaptive-ratio-demosaic.md` section 10). The default is the
//! adaptive ratio demosaic, which the first test runs on both sides. The second test keeps the method that is no longer the
//! default honest: it gives the app's renderer `RenderOptions::demosaic = menon` and starts `ym` with
//! `YM_FORCE_DEMOSAIC=menon`, and it checks that forcing the adaptive ratio demosaic by name gives the default's bytes while
//! forcing Menon gives other bytes. That variable is read only by a `ym` built with the test-only `force-demosaic` feature, so
//! the binary for this test must be built `cargo build -p ym-cli --features force-demosaic` (a plain build ignores the
//! variable: the test notices that, says so and passes, unless `YM_REQUIRE_FORCE_DEMOSAIC` is set, which the CI job sets once
//! it builds the binary that way).
//!
//! Two legs for the graphics policy (`docs/17-gpu.md` section 12). The CPU leg starts `ym` with `YELLOWMOON_GPU=off`
//! (the app's core here has no device): without it `ym` would run `auto`, which on a Mac with Metal puts every stage on the
//! device, and the comparison would be the device against the CPU. The GPU leg opens the device the way the application
//! does (`open_gpu(On)`, attached to the core) and starts `ym` with `YELLOWMOON_GPU=on`: the same eleven cases, byte for
//! byte, with the two sides built apart. Without an adapter it skips with a notice, unless `YM_REQUIRE_GPU=1`.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Duration;

use ym_core::app::gpu::{GpuPolicy, GpuSlot, open_gpu};
use ym_core::app::{Core, CoreOptions};
use ym_core::apple_auto::AppleAutoMode;
use ym_core::export_queue::ExportJob;
use ym_core::scheduler::MemoryBudget;
use ym_core::session::{ApplyScope, OpenOptions};
use ym_export::{
    BitDepth, ColorSpace, ExportOptions, FormatKind, JpegQuality, Size, TiffCompression,
};
use ym_gpu::cpu::demosaic::{DEFAULT_METHOD, Method};
use ym_gpu::device::gpu_required_by_env;
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

/// Exports `arw` once with the app's path and once with `ym render` (started with the extra
/// environment `env`), and returns the two lists of files.
fn export_both(
    core: &Core,
    ym: &Path,
    arw: &Path,
    root: &Path,
    case: &Case,
    env: &[(&str, &str)],
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
        // Apple Auto off on both sides (`docs/19` section 7): today's bytes on every machine, a Mac
        // with Apple Intelligence included.
        .args(["--apple-auto", "off"])
        // The CPU reference unless the leg says otherwise (`env` comes after and wins).
        .env("YELLOWMOON_GPU", "off")
        .envs(env.iter().copied())
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

/// The sandbox of one test: a folder with one synthetic photo and the CLI's home folder.
fn sandbox() -> (tempfile::TempDir, PathBuf) {
    let dir = tempfile::tempdir().expect("tempdir");
    let root = dir.path();
    std::fs::create_dir_all(root.join("photos")).expect("photos");
    std::fs::create_dir_all(root.join("cli-home")).expect("cli home");
    let arw = root.join("photos").join("DSC00001.ARW");
    ArwBuilder::new("ILCE-7RM4", 160, 120)
        .write_to(&arw, &gradient_mosaic(160, 120, 512, 16_000, 11))
        .expect("write a synthetic ARW");
    (dir, arw)
}

/// The app's side: a core over its own app folder, with one photo open and developed, whose
/// full-resolution renders use the R6 demosaic `method`. No thumbnails, as in the CLI.
fn open_core(arw: &Path, method: Method) -> (Core, tempfile::TempDir) {
    open_core_on(arw, method, None)
}

/// [`open_core`] with the device of `slot` attached, the way the application attaches it.
fn open_core_on(arw: &Path, method: Method, slot: Option<&GpuSlot>) -> (Core, tempfile::TempDir) {
    let app_home = tempfile::tempdir().expect("app home");
    let mut options = CoreOptions::new(data_dir(), app_home.path().to_path_buf());
    options.thumbnails = false;
    options.render.demosaic = method;
    // Apple Auto off, as `ym --apple-auto off` (`docs/19` section 7): no advisor, and the mode off
    // should one ever be given here.
    options.advisor = None;
    options.apple_auto = AppleAutoMode::Off;
    let core = Core::open(options).expect("core");
    if let Some(slot) = slot {
        core.attach_gpu(slot);
    }
    let outcome = core.session.open(
        std::slice::from_ref(&arw.to_path_buf()),
        OpenOptions::default(),
    );
    assert_eq!(outcome.added.len(), 1, "{outcome:?}");
    assert!(
        core.session.wait_idle(Duration::from_secs(120)),
        "the photo develops"
    );
    (core, app_home)
}

/// Every case of [`cases`] through the app and through `ym` (with `env` added to its environment),
/// byte for byte.
fn assert_every_format_is_identical(
    core: &Core,
    ym: &Path,
    arw: &Path,
    root: &Path,
    env: &[(&str, &str)],
) {
    let cases = cases();
    assert_eq!(
        cases.len(),
        11,
        "JPEG (2), TIFF (5), PNG (3) and DNG, as in crates/ym-cli/tests/parity.rs"
    );
    for case in &cases {
        let (from_app, from_cli) = export_both(core, ym, arw, root, case, env);
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

#[test]
fn every_format_is_byte_identical_between_the_app_graph_and_a_separately_built_ym() {
    let Some(ym) = command_line_tool() else {
        return;
    };
    let (dir, arw) = sandbox();
    let (core, _app_home) = open_core(&arw, DEFAULT_METHOD);
    assert_every_format_is_identical(&core, &ym, &arw, dir.path(), &[]);
}

/// The GPU leg: the device attached to the app's core and `YELLOWMOON_GPU=on` for the separately built `ym`.
#[test]
fn every_format_is_byte_identical_between_the_app_graph_and_a_separately_built_ym_on_the_device() {
    let Some(ym) = command_line_tool() else {
        return;
    };
    let slot = open_gpu(GpuPolicy::On, MemoryBudget::detect());
    if slot.gpu().is_none() {
        assert!(
            !gpu_required_by_env(),
            "YM_REQUIRE_GPU is set but no device opened: {:?}",
            slot.missing_device_reason()
        );
        eprintln!("skipped: no graphics adapter for the GPU leg");
        return;
    }
    let (dir, arw) = sandbox();
    let (core, _app_home) = open_core_on(&arw, DEFAULT_METHOD, Some(&slot));
    assert_every_format_is_identical(&core, &ym, &arw, dir.path(), &[("YELLOWMOON_GPU", "on")]);
}

/// The same with Menon, the previous default, forced on both sides; and the switch itself: forcing the
/// adaptive ratio demosaic by name is the default, forcing Menon is another picture.
#[test]
fn every_format_is_byte_identical_with_menon_forced_and_the_default_is_the_adaptive_ratio_demosaic()
{
    let Some(ym) = command_line_tool() else {
        return;
    };
    assert_eq!(DEFAULT_METHOD, Method::AdaptiveRatio);
    let (dir, arw) = sandbox();
    let force_menon = [("YM_FORCE_DEMOSAIC", "menon")];
    let force_adaptive_ratio = [("YM_FORCE_DEMOSAIC", "adaptive_ratio")];

    // Does this `ym` read the switch? A binary without the `force-demosaic` feature ignores the
    // variable and gives the default method's bytes.
    let probe = |name: &str, env: &[(&str, &str)]| {
        let out = dir.path().join("probe").join(name);
        let output = Command::new(&ym)
            .args(["render"])
            .arg(&arw)
            .args(["--apple-auto", "off"])
            .args(["--format", "png", "--bits", "16", "--out"])
            .arg(&out)
            .env("YELLOWMOON_GPU", "off")
            .envs(env.iter().copied())
            .env("YELLOWMOON_HOME", dir.path().join("cli-home"))
            .env("YELLOWMOON_DATA", data_dir())
            .env("HOME", dir.path())
            .env_remove("RUST_LOG")
            .env_remove("XDG_DATA_HOME")
            .current_dir(dir.path())
            .output()
            .expect("run ym");
        assert!(
            output.status.success(),
            "probe {name}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        let file = String::from_utf8_lossy(&output.stdout)
            .lines()
            .next()
            .map(PathBuf::from)
            .expect("one file");
        std::fs::read(file).expect("read the probe's file")
    };
    let plain = probe("plain", &[]);
    let forced_menon = probe("menon", &force_menon);
    if plain == forced_menon {
        assert!(
            std::env::var_os("YM_REQUIRE_FORCE_DEMOSAIC").is_none(),
            "YM_REQUIRE_FORCE_DEMOSAIC is set but this ym ignores YM_FORCE_DEMOSAIC \
             (build it with `cargo build -p ym-cli --features force-demosaic`)"
        );
        eprintln!(
            "skipped: this ym was built without the test-only force-demosaic feature \
             (cargo build -p ym-cli --features force-demosaic)"
        );
        return;
    }
    // The switch works: Menon is another picture than the default, and the default is what
    // forcing the adaptive ratio demosaic by name gives.
    assert!(
        plain == probe("adaptive_ratio", &force_adaptive_ratio),
        "forcing the adaptive ratio demosaic gives the default's bytes"
    );

    let (core, _app_home) = open_core(&arw, Method::Menon);
    assert_every_format_is_identical(&core, &ym, &arw, dir.path(), &force_menon);
}
