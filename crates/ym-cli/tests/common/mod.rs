//! Helpers shared by the integration tests of the `ym` binary: an isolated app folder, synthetic
//! ARW files and a way to run the binary against both.
//!
//! Every test runs in its own temporary folder. `YELLOWMOON_HOME` points the preferences and the
//! look library into it (so a test never touches the real app folder) and `YELLOWMOON_DATA` at the
//! repository's `data/`.
//!
//! Apple Auto (`docs/19`): [`Sandbox::ym`] sets `YELLOWMOON_TEST_APPLE_AUTO=none`, so Apple Auto is
//! not available, for the reason a build without the bridge gives on this platform (`not_a_mac` on
//! Linux), even on a Mac with Apple Intelligence: every test gets the rule unless it asks for the
//! deterministic test advisor ([`Sandbox::ym_fake`]). A test that compares bytes or values with the
//! core's (parity, the snapshots) also passes `--apple-auto off`.

#![allow(dead_code, reason = "each test binary uses some of the helpers")]

use std::path::{Path, PathBuf};

use assert_cmd::Command;
use assert_cmd::cargo::cargo_bin_cmd;
use serde_json::Value;
use ym_raw::testgen::{ArwBuilder, gradient_mosaic};

/// The repository's shipped `data/` folder.
pub fn data_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data")
}

/// A temporary world: an app folder, a folder of photos and the binary set up to use both.
pub struct Sandbox {
    dir: tempfile::TempDir,
}

impl Sandbox {
    /// A fresh sandbox with an empty `photos/` folder.
    pub fn new() -> Self {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::create_dir_all(dir.path().join("photos")).expect("photos folder");
        std::fs::create_dir_all(dir.path().join("home")).expect("home folder");
        Self { dir }
    }

    /// The sandbox folder.
    pub fn root(&self) -> &Path {
        self.dir.path()
    }

    /// The app folder (`YELLOWMOON_HOME`): preferences and imported looks.
    pub fn home(&self) -> PathBuf {
        self.dir.path().join("home")
    }

    /// The folder the photos are in.
    pub fn photos(&self) -> PathBuf {
        self.dir.path().join("photos")
    }

    /// The folder `ym render` writes to by default for photos in `photos()`.
    pub fn default_export(&self) -> PathBuf {
        self.dir.path().join("photos export")
    }

    /// Writes a synthetic ARW (an A7R IV, 14-bit uncompressed, `width` x `height`) into `photos/`.
    pub fn arw_sized(&self, name: &str, width: u32, height: u32, seed: u64) -> PathBuf {
        let path = self.photos().join(name);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).expect("create folder");
        }
        ArwBuilder::new("ILCE-7RM4", width, height)
            .write_to(&path, &gradient_mosaic(width, height, 512, 16_000, seed))
            .expect("write synthetic ARW");
        path
    }

    /// Like [`Sandbox::arw_sized`] with a chosen white level of the mosaic: a lower one is a darker
    /// photo, which the automatic exposure lifts by a different amount. It is taken a day later
    /// than the other synthetic photos: photos shot together share their median exposure (a
    /// burst, `docs/02`), and this one is not part of that burst.
    pub fn arw_with_white(&self, name: &str, white: u16) -> PathBuf {
        let path = self.photos().join(name);
        ArwBuilder::new("ILCE-7RM4", 160, 120)
            .capture_time("2026:10:05 18:30:00", "000", Some("+02:00"))
            .write_to(&path, &gradient_mosaic(160, 120, 512, white, 21))
            .expect("write synthetic ARW");
        path
    }

    /// A synthetic ARW that is clipped everywhere (every photosite at the white level).
    pub fn arw_clipped(&self, name: &str) -> PathBuf {
        let path = self.photos().join(name);
        ArwBuilder::new("ILCE-7RM4", 160, 120)
            .write_to(&path, &vec![16_300u16; 160 * 120])
            .expect("write synthetic ARW");
        path
    }

    /// Writes the small synthetic ARW most tests use (160 x 120).
    pub fn arw(&self, name: &str) -> PathBuf {
        self.arw_sized(name, 160, 120, 11)
    }

    /// The `ym` binary with this sandbox's folders and a quiet environment.
    pub fn ym(&self) -> Command {
        let mut cmd = cargo_bin_cmd!("ym");
        cmd.env("YELLOWMOON_HOME", self.home())
            .env("YELLOWMOON_DATA", data_dir())
            .env("HOME", self.root())
            .env_remove("RUST_LOG")
            .env_remove("XDG_DATA_HOME")
            .env(NO_APPLE_AUTO.0, NO_APPLE_AUTO.1)
            .current_dir(self.root());
        cmd
    }

    /// [`Sandbox::ym`] with the test advisor of Apple Auto (`YELLOWMOON_TEST_APPLE_AUTO=fake`):
    /// available on every machine, model `Test advisor`, steps from the picture's statistics.
    pub fn ym_fake(&self) -> Command {
        let mut cmd = self.ym();
        cmd.env(FAKE_APPLE_AUTO.0, FAKE_APPLE_AUTO.1);
        cmd
    }

    /// Writes the app's `preferences.json` with the `apple_auto` preference (`docs/19` section 2).
    pub fn set_apple_auto_preference(&self, on: bool) {
        std::fs::write(
            self.home().join("preferences.json"),
            serde_json::json!({ "apple_auto": on }).to_string(),
        )
        .expect("write preferences");
    }
}

/// The environment variable and value that give `ym` the test advisor of Apple Auto.
pub const FAKE_APPLE_AUTO: (&str, &str) = ("YELLOWMOON_TEST_APPLE_AUTO", "fake");

/// The environment variable and value that make Apple Auto unavailable to `ym`, as on a computer
/// without it, whatever this one can do.
pub const NO_APPLE_AUTO: (&str, &str) = ("YELLOWMOON_TEST_APPLE_AUTO", "none");

/// The lines of a captured stdout.
pub fn lines(bytes: &[u8]) -> Vec<String> {
    String::from_utf8_lossy(bytes)
        .lines()
        .map(str::to_owned)
        .collect()
}

/// The JSON objects of a captured stdout, one per line.
pub fn json_lines(bytes: &[u8]) -> Vec<Value> {
    lines(bytes)
        .iter()
        .map(|l| serde_json::from_str(l).unwrap_or_else(|e| panic!("not JSON ({e}): {l}")))
        .collect()
}

/// The text of a captured stderr.
pub fn text(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes).into_owned()
}

/// The names of the files in a folder, sorted.
pub fn names_in(dir: &Path) -> Vec<String> {
    let mut names: Vec<String> = std::fs::read_dir(dir)
        .map(|rd| {
            rd.flatten()
                .map(|e| e.file_name().to_string_lossy().into_owned())
                .collect()
        })
        .unwrap_or_default();
    names.sort();
    names
}

/// The first bytes of a file.
pub fn head(path: &Path, n: usize) -> Vec<u8> {
    let mut bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    bytes.truncate(n);
    bytes
}

/// A JSON document reduced to its shape: every leaf is the name of its type, an array is the shape
/// of its first element (or nothing). What a snapshot of a report that depends on the machine can
/// pin without pinning the machine.
pub fn shape(v: &Value) -> Value {
    match v {
        Value::Null => Value::from("null"),
        Value::Bool(_) => Value::from("bool"),
        Value::Number(_) => Value::from("number"),
        Value::String(_) => Value::from("string"),
        Value::Array(items) => Value::Array(items.first().map(shape).into_iter().collect()),
        Value::Object(map) => {
            Value::Object(map.iter().map(|(k, v)| (k.clone(), shape(v))).collect())
        }
    }
}
