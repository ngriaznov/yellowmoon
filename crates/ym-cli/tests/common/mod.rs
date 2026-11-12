//! Helpers shared by the integration tests of the `ym` binary: an isolated app folder, synthetic
//! ARW files and a way to run the binary against both.
//!
//! Every test runs in its own temporary folder. `YELLOWMOON_HOME` points the preferences and the
//! look library into it (so a test never touches the real app folder) and `YELLOWMOON_DATA` at the
//! repository's `data/`.
//!
//! Subject and People (`docs/23` section 4.5): [`Sandbox::ym`] also sets `YELLOWMOON_TEST_VISION=none`, so a computer
//! that supports the Vision requests answers as one that does not, and a test that expects the sentences of section
//! 3.10 gives them everywhere; [`Sandbox::ym_fake_vision`] gives the deterministic fake provider and
//! [`Sandbox::ym_system_vision`] this computer's own Vision.
//!
//! Auto providers (`docs/19`, `docs/22`): [`Sandbox::ym`] sets `YELLOWMOON_TEST_APPLE_AUTO=none`, so Apple
//! Auto is not available, for the reason a build without the bridge gives on this platform
//! (`not_a_mac` on Linux), even on a Mac with Apple Intelligence, and `YELLOWMOON_TEST_AUTO_PROVIDERS=none`,
//! so Claude Code and Antigravity are not installed, even on a computer that has them: every test gets
//! the rule unless it asks for the deterministic test advisor ([`Sandbox::ym_fake`]) or a provider
//! ([`Sandbox::ym_provider`]). A test that compares bytes or values with the core's (parity, the
//! snapshots) also passes `--auto-provider yellowmoon`.

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

    /// Writes a synthetic ARW of defocused lights (`docs/21` section 4, L1): a dark ground with 4 x 6 flat discs of
    /// 1.5 % of the width, the blur the bokeh detector finds, so Anamorphic reshapes the photo.
    pub fn arw_bokeh(&self, name: &str, width: u32, height: u32, seed: u64) -> PathBuf {
        let path = self.photos().join(name);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).expect("create folder");
        }
        let mut mosaic = gradient_mosaic(width, height, 512, 700, seed);
        let radius = 0.015 * f64::from(width);
        for fy in [0.18, 0.4, 0.62, 0.84] {
            for fx in [0.1, 0.26, 0.42, 0.58, 0.74, 0.9] {
                let (cx, cy) = (fx * f64::from(width), fy * f64::from(height));
                for y in 0..height {
                    for x in 0..width {
                        let d = (f64::from(x) + 0.5 - cx).hypot(f64::from(y) + 0.5 - cy);
                        if d <= radius {
                            mosaic[(y * width + x) as usize] = 7_000;
                        }
                    }
                }
            }
        }
        ArwBuilder::new("ILCE-7RM4", width, height)
            .write_to(&path, &mosaic)
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
            .env(NO_AUTO_PROVIDERS.0, NO_AUTO_PROVIDERS.1)
            .env(NO_VISION.0, NO_VISION.1)
            .current_dir(self.root());
        cmd
    }

    /// [`Sandbox::ym`] with the fake Vision (`YELLOWMOON_TEST_VISION=fake`): a computer that supports Subject and People
    /// and makes up a soft mask from the picture, on every platform.
    pub fn ym_fake_vision(&self) -> Command {
        let mut cmd = self.ym();
        cmd.env(FAKE_VISION.0, FAKE_VISION.1);
        cmd
    }

    /// [`Sandbox::ym`] with this computer's own Vision framework (`YELLOWMOON_TEST_VISION=system`, which is anything but
    /// `fake` and `none`): what a Mac answers; on any other computer, that it is not a Mac.
    pub fn ym_system_vision(&self) -> Command {
        let mut cmd = self.ym();
        cmd.env(SYSTEM_VISION.0, SYSTEM_VISION.1);
        cmd
    }

    /// [`Sandbox::ym`] with the test advisor of Apple Auto (`YELLOWMOON_TEST_APPLE_AUTO=fake`):
    /// available on every machine, model `Test advisor`, steps from the picture's statistics.
    pub fn ym_fake(&self) -> Command {
        let mut cmd = self.ym();
        cmd.env(FAKE_APPLE_AUTO.0, FAKE_APPLE_AUTO.1);
        cmd
    }

    /// [`Sandbox::ym`] with the programs as the test hook says (`docs/22` section 15.3): for example
    /// `claude=available,agy=signed_out`, or `claude=exec:/path/to/ym-fake-claude`.
    pub fn ym_provider(&self, hook: &str) -> Command {
        let mut cmd = self.ym();
        cmd.env(AUTO_PROVIDERS_ENV, hook);
        cmd
    }

    /// Writes the app's `preferences.json` with the `apple_auto` preference of before the providers
    /// (`docs/19` section 2), which reads as `auto_provider` (`docs/22` section 10.1).
    pub fn set_apple_auto_preference(&self, on: bool) {
        self.write_preferences(&serde_json::json!({ "apple_auto": on }));
    }

    /// Writes the app's `preferences.json` with the saved choice and, for a program, its consent
    /// (`docs/22` section 10).
    pub fn set_auto_provider(&self, provider: &str, consent: &[&str]) {
        self.write_preferences(&serde_json::json!({
            "auto_provider": provider,
            "auto_consent": { "claude": consent.contains(&"claude"), "agy": consent.contains(&"agy") },
        }));
    }

    /// Writes the app's `preferences.json` as given.
    pub fn write_preferences(&self, preferences: &Value) {
        std::fs::write(
            self.home().join("preferences.json"),
            preferences.to_string(),
        )
        .expect("write preferences");
    }
}

/// Copies a folder and everything in it.
fn copy_tree(from: &Path, to: &Path) {
    std::fs::create_dir_all(to).expect("create folder");
    for entry in std::fs::read_dir(from).expect("read folder") {
        let entry = entry.expect("entry");
        let target = to.join(entry.file_name());
        if entry.file_type().expect("type").is_dir() {
            copy_tree(&entry.path(), &target);
        } else {
            std::fs::copy(entry.path(), target).expect("copy file");
        }
    }
}

impl Sandbox {
    /// A copy of the shipped data folder in `<root>/data` whose `lenses` folder is a synthetic one
    /// (`ym_lens::test_support`, seven lenses that say they are not lens data), or no `lenses` folder
    /// at all, whatever the shipped one is. For the tests of `--lens` and `ym lenses`.
    pub fn data_with_lenses(&self, synthetic: bool) -> PathBuf {
        let data = self.root().join("data");
        for entry in std::fs::read_dir(data_dir()).expect("the data folder") {
            let entry = entry.expect("entry");
            if entry.file_name() == "lenses" {
                continue;
            }
            let target = data.join(entry.file_name());
            if entry.file_type().expect("type").is_dir() {
                copy_tree(&entry.path(), &target);
            } else {
                std::fs::create_dir_all(&data).expect("data folder");
                std::fs::copy(entry.path(), target).expect("copy file");
            }
        }
        if synthetic {
            ym_lens::test_support::write_synthetic_lenses(&data.join("lenses"))
                .expect("write the synthetic lenses");
        }
        data
    }

    /// The shipped data folder in `<root>/data-classical` without the learned bokeh map (`models/bokeh`), so the
    /// classical detector decides which photos have bokeh whatever the weights say: the same files, linked and not
    /// copied. For the tests that need a photo with bokeh to be reshaped (`docs/21` section 7).
    #[cfg(unix)]
    pub fn data_without_the_bokeh_model(&self) -> PathBuf {
        use std::os::unix::fs::symlink;
        let data = self.root().join("data-classical");
        std::fs::create_dir_all(data.join("models")).expect("data folder");
        for entry in std::fs::read_dir(data_dir()).expect("the data folder") {
            let entry = entry.expect("entry");
            if entry.file_name() != "models" {
                symlink(entry.path(), data.join(entry.file_name())).expect("link");
            }
        }
        for entry in std::fs::read_dir(data_dir().join("models")).expect("the models") {
            let entry = entry.expect("entry");
            if entry.file_name() != "bokeh" {
                symlink(entry.path(), data.join("models").join(entry.file_name())).expect("link");
            }
        }
        data
    }

    /// The shipped data folder in `<root>/data-no-sam` without SAM's weights (`models/sam2`): the same files, linked and not copied. For the
    /// tests of what a copy of Yellowmoon with no Object model says and does (`docs/10`, `docs/23` section 9).
    #[cfg(unix)]
    pub fn data_without_the_sam_weights(&self) -> PathBuf {
        use std::os::unix::fs::symlink;
        let data = self.root().join("data-no-sam");
        std::fs::create_dir_all(data.join("models")).expect("data folder");
        for entry in std::fs::read_dir(data_dir()).expect("the data folder") {
            let entry = entry.expect("entry");
            if entry.file_name() != "models" {
                symlink(entry.path(), data.join(entry.file_name())).expect("link");
            }
        }
        for entry in std::fs::read_dir(data_dir().join("models")).expect("the models") {
            let entry = entry.expect("entry");
            if entry.file_name() != "sam2" {
                symlink(entry.path(), data.join("models").join(entry.file_name())).expect("link");
            }
        }
        data
    }

    /// The shipped data folder in `<root>/data-no-depth` without the depth weights (`models/depth`): the same files, linked and
    /// not copied. For the tests of what a copy of Yellowmoon with no Depth model says and does (`docs/10`, `docs/23`
    /// section 3.10a).
    #[cfg(unix)]
    pub fn data_without_the_depth_weights(&self) -> PathBuf {
        use std::os::unix::fs::symlink;
        let data = self.root().join("data-no-depth");
        std::fs::create_dir_all(data.join("models")).expect("data folder");
        for entry in std::fs::read_dir(data_dir()).expect("the data folder") {
            let entry = entry.expect("entry");
            if entry.file_name() != "models" {
                symlink(entry.path(), data.join(entry.file_name())).expect("link");
            }
        }
        for entry in std::fs::read_dir(data_dir().join("models")).expect("the models") {
            let entry = entry.expect("entry");
            if entry.file_name() != "depth" {
                symlink(entry.path(), data.join("models").join(entry.file_name())).expect("link");
            }
        }
        data
    }

    /// The shipped data folder in `<root>/data-no-faces` without the face weights (`models/mediapipe`): the same files, linked
    /// and not copied. For the tests of what a copy of Yellowmoon with no face models says and does (`docs/27`, S-11b).
    #[cfg(unix)]
    pub fn data_without_the_face_weights(&self) -> PathBuf {
        use std::os::unix::fs::symlink;
        let data = self.root().join("data-no-faces");
        std::fs::create_dir_all(data.join("models")).expect("data folder");
        for entry in std::fs::read_dir(data_dir()).expect("the data folder") {
            let entry = entry.expect("entry");
            if entry.file_name() != "models" {
                symlink(entry.path(), data.join(entry.file_name())).expect("link");
            }
        }
        for entry in std::fs::read_dir(data_dir().join("models")).expect("the models") {
            let entry = entry.expect("entry");
            if entry.file_name() != "mediapipe" {
                symlink(entry.path(), data.join("models").join(entry.file_name())).expect("link");
            }
        }
        data
    }

    /// [`Sandbox::ym`] with the fake face finder (`YELLOWMOON_TEST_FACES=fake`): a computer that has the Face Refinement node's
    /// models, with a finder that gives two faces at fixed places, without the weights, on every platform.
    pub fn ym_fake_faces(&self) -> Command {
        let mut cmd = self.ym();
        cmd.env(FAKE_FACES.0, FAKE_FACES.1);
        cmd
    }

    /// [`Sandbox::ym`] with the fake depth model (`YELLOWMOON_TEST_DEPTH=fake`): a computer that has the Depth mask and makes
    /// its map up from the picture, without the weights, on every platform.
    pub fn ym_fake_depth(&self) -> Command {
        let mut cmd = self.ym();
        cmd.env(FAKE_DEPTH.0, FAKE_DEPTH.1);
        cmd
    }

    /// [`Sandbox::ym`] reading the data folder `data` ([`Sandbox::data_with_lenses`]).
    pub fn ym_data(&self, data: &Path) -> Command {
        let mut cmd = self.ym();
        cmd.env("YELLOWMOON_DATA", data);
        cmd
    }
}

/// The environment variable and value that make Subject and People not supported for `ym`, as on a computer that
/// cannot make them, whatever this one can do (`ym_apple::vision::TEST_VISION_NONE`).
pub const NO_VISION: (&str, &str) = ("YELLOWMOON_TEST_VISION", "none");

/// The environment variable and value that give `ym` the fake depth model (`ym_core::pipeline::DepthModel::TEST_ENV`).
pub const FAKE_DEPTH: (&str, &str) = ("YELLOWMOON_TEST_DEPTH", "fake");

/// The environment variable and value that give `ym` the fake face finder (`ym_core::pipeline::FaceModel::TEST_ENV`).
pub const FAKE_FACES: (&str, &str) = ("YELLOWMOON_TEST_FACES", "fake");

/// The environment variable and value that give `ym` the fake Vision (`ym_apple::vision::TEST_VISION_FAKE`).
pub const FAKE_VISION: (&str, &str) = ("YELLOWMOON_TEST_VISION", "fake");

/// The environment variable and value that give `ym` this computer's own Vision: any value but `fake` and `none`.
pub const SYSTEM_VISION: (&str, &str) = ("YELLOWMOON_TEST_VISION", "system");

/// The environment variable and value that give `ym` the test advisor of Apple Auto.
pub const FAKE_APPLE_AUTO: (&str, &str) = ("YELLOWMOON_TEST_APPLE_AUTO", "fake");

/// The environment variable and value that make Apple Auto unavailable to `ym`, as on a computer
/// without it, whatever this one can do.
pub const NO_APPLE_AUTO: (&str, &str) = ("YELLOWMOON_TEST_APPLE_AUTO", "none");

/// The environment variable and value that make Claude Code and Antigravity not installed for `ym`
/// (`docs/22` section 13.5), whatever this computer has: a developer's own programs are never started by a
/// test. Every test that starts the binary sets it with [`NO_APPLE_AUTO`].
pub const NO_AUTO_PROVIDERS: (&str, &str) = ("YELLOWMOON_TEST_AUTO_PROVIDERS", "none");

/// The environment variable of the test hook for the two programs (`docs/22` section 15.3).
pub const AUTO_PROVIDERS_ENV: &str = "YELLOWMOON_TEST_AUTO_PROVIDERS";

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

/// [`json_lines`] without the `low_space` event that a run to a small, read-only or vanished destination
/// writes first (`docs/10`, H-07): the photos' records and the `stopped` event.
pub fn json_records(bytes: &[u8]) -> Vec<Value> {
    json_lines(bytes)
        .into_iter()
        .filter(|r| r["event"] != "low_space")
        .collect()
}

/// The node `id` (`"input"`, `"output"` or a number) among the items of the `settings` of a
/// `--json` record; `null` when the record has no such node.
pub fn node(record: &Value, id: impl Into<Value>) -> &Value {
    static NULL: Value = Value::Null;
    let id = id.into();
    record["settings"]["nodes"]["items"]
        .as_array()
        .and_then(|items| items.iter().find(|i| i["id"] == id))
        .unwrap_or(&NULL)
}

/// A node-first settings document that holds the given fields (a JSON object of field names and
/// values), each in the node it belongs to: the Input's, the Output's or Main's grade.
pub fn field_doc(fields: Value) -> Value {
    let (mut input, mut output, mut grade) = (
        serde_json::json!({ "id": "input" }),
        serde_json::json!({ "id": "output" }),
        serde_json::json!({}),
    );
    for (k, v) in fields.as_object().expect("a JSON object of fields") {
        match k.as_str() {
            "look" | "look_intensity" | "output_sharpening" => output[k] = v.clone(),
            "contrast" | "brightness" | "highlights" | "shadows" | "white" | "black"
            | "color_balance" | "saturation" | "color_warp" | "curve" => grade[k] = v.clone(),
            _ => input[k] = v.clone(),
        }
    }
    serde_json::json!({ "nodes": { "items": [input, { "id": 1, "grade": grade }, output] } })
}

/// A node patch that holds the given fields (a JSON object of field names and values), each on the
/// node it belongs to.
pub fn field_patch(fields: Value) -> Value {
    let doc = field_doc(fields);
    let items = doc["nodes"]["items"].as_array().expect("items");
    let mut nodes = serde_json::Map::new();
    for (key, item) in [("input", &items[0]), ("output", &items[2])] {
        let mut node = item.as_object().expect("a node").clone();
        node.remove("id");
        if !node.is_empty() {
            nodes.insert(key.to_owned(), Value::Object(node));
        }
    }
    let grade = items[1]["grade"].clone();
    if grade.as_object().is_some_and(|g| !g.is_empty()) {
        nodes.insert("1".to_owned(), serde_json::json!({ "grade": grade }));
    }
    serde_json::json!({ "nodes": nodes })
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
