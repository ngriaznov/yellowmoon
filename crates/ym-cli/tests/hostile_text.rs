//! `ym info` and `ym render --json` on a file whose text tags carry terminal control bytes.
//!
//! The model, firmware and lens strings of a downloaded file reach a terminal through these two
//! commands. The decoder turns every control character of a text tag into U+FFFD
//! (`ym_raw::tiff::TiffReader::ascii`, tested in `ym-raw`); this test pins the other end: whatever
//! the commands print, text or JSON, holds no byte below 0x20 but the newline that ends a line (the
//! formats use no tab), and no DEL or C1 control character (U+0080 to U+009F, which a terminal
//! treats as escape introducers when it reads bytes as Latin-1, and which JSON does not escape).
//! A newline hidden in a tag must not add a line either: a forged line could pose as another row.
//!
//! The names of the files are text from outside too (a folder unpacked from a downloaded archive).
//! The tests at the end pin what every command does with a hostile path: lines for a person show
//! the controls as escapes (`\x1b`, `\u{9b}`, `crate::text`), JSON keeps the exact name and escapes
//! DEL and the C1 characters as `\u` sequences, and the lines of data on stdout (the files that were
//! written) are the exact path when stdout is a pipe.

mod common;

use std::path::{Path, PathBuf};

use common::{Sandbox, json_lines, lines, text as text_of};
use ym_raw::testgen::{ArwBuilder, gradient_mosaic};

/// Every C0 control, DEL and the C1 range as characters (the UTF-8 form of a C1 control is two
/// bytes, 0xC2 and 0x80 to 0x9F), with a newline, a carriage return and a tab among them.
fn hostile_chars() -> String {
    let mut text = String::from("A\u{1b}[31mB\u{1b}]0;PWNED\u{7}\u{1b}[2J\n\r\tC");
    text.extend((0x01u32..0x20).filter_map(char::from_u32));
    text.push('\u{7f}');
    text.extend((0x80u32..0xA0).filter_map(char::from_u32));
    text.push('Z');
    text
}

/// Control and C1 bytes that are not UTF-8 at all (a stray 0x9B is no character), as a byte string.
fn hostile_bytes() -> Vec<u8> {
    let mut bytes = b"A\x1b[2J\x07\x7f\x0a\x0d\x09".to_vec();
    bytes.extend(0x80u8..=0x9F);
    bytes.extend(b"Z");
    bytes
}

/// Writes a synthetic A7R IV file whose model, firmware and lens strings are `hostile_chars()`
/// (so that they sit in the file as UTF-8).
fn write_hostile_utf8(dir: &Path) -> std::path::PathBuf {
    let path = dir.join("DSC00001.ARW");
    let text = hostile_chars();
    ArwBuilder::new(&format!("ILCE-7RM4{text}"), 128, 96)
        .model_id(375)
        .software(&format!("ILCE-7RM4 v1.00 {text}"))
        .lens_model(Some(&format!("FE {text}")))
        .write_to(&path, &gradient_mosaic(128, 96, 512, 16_000, 11))
        .expect("write hostile ARW");
    // The file really holds them (a writer that cleaned the strings would turn every check below
    // into one that cannot fail): the whole text, controls and C1 characters as UTF-8, in the
    // model, the firmware and the lens.
    let file = std::fs::read(&path).expect("read back");
    let found = file
        .windows(text.len())
        .filter(|w| *w == text.as_bytes())
        .count();
    assert!(found >= 3, "the hostile text is in the file {found} times");
    path
}

/// Writes a normal file, then overwrites the start of the firmware and lens strings in place with
/// `hostile_bytes()`: single control bytes and C1 bytes as a damaged or crafted file holds them.
fn write_hostile_raw_bytes(dir: &Path) -> std::path::PathBuf {
    let path = dir.join("DSC00002.ARW");
    let firmware = format!("v1.00 {}", "y".repeat(80));
    let lens = format!("FE {}", "x".repeat(80));
    ArwBuilder::new("ILCE-7RM4", 128, 96)
        .software(&firmware)
        .lens_model(Some(&lens))
        .write_to(&path, &gradient_mosaic(128, 96, 512, 16_000, 11))
        .expect("write ARW");
    let mut file = std::fs::read(&path).expect("read back");
    // The tags' counts are fixed and the text ends at the first NUL: change the bytes, not the length.
    let hostile = hostile_bytes();
    for text in [&firmware, &lens] {
        let at = file
            .windows(text.len())
            .position(|w| w == text.as_bytes())
            .expect("the string is in the file once");
        let from = at + text.find(' ').expect("a space") + 1;
        assert!(from + hostile.len() <= at + text.len());
        file[from..from + hostile.len()].copy_from_slice(&hostile);
    }
    std::fs::write(&path, &file).expect("write hostile ARW");
    // Twice (firmware and lens), and not as UTF-8: the lone C1 bytes are what make the file invalid text.
    assert_eq!(
        file.windows(hostile.len())
            .filter(|w| *w == hostile.as_slice())
            .count(),
        2
    );
    path
}

/// A bad byte of a terminal-bound stream: below 0x20 but a newline, or DEL, or a C1 character (two
/// bytes in UTF-8, 0xC2 0x80..=0x9F) or a lone 0x80..=0x9F byte.
fn bad_bytes(stream: &[u8]) -> Vec<(usize, u8)> {
    let mut found = Vec::new();
    for (i, &b) in stream.iter().enumerate() {
        let c1_utf8 = b == 0xC2 && stream.get(i + 1).is_some_and(|n| (0x80..=0x9F).contains(n));
        let c1_lone = (0x80..=0x9F).contains(&b)
            && !(i > 0 && stream[i - 1] == 0xC2)
            && !is_continuation_of_longer_sequence(stream, i);
        if (b < 0x20 && b != b'\n') || b == 0x7F || c1_utf8 || c1_lone {
            found.push((i, b));
        }
    }
    found
}

/// True when byte `i` (0x80..=0x9F) is a continuation byte of a valid multi-byte character, which
/// is no C1 control: for example the 0x80 of U+2014 (E2 80 94).
fn is_continuation_of_longer_sequence(stream: &[u8], i: usize) -> bool {
    // Look back up to three bytes for a lead byte whose sequence reaches i.
    (1..=3.min(i)).any(|back| {
        let lead = stream[i - back];
        let needed = match lead {
            0xC0..=0xDF => 1,
            0xE0..=0xEF => 2,
            0xF0..=0xF7 => 3,
            _ => return false,
        };
        needed >= back
            && stream[i - back + 1..=i]
                .iter()
                .all(|b| (0x80..=0xBF).contains(b))
    })
}

fn assert_clean(what: &str, stream: &[u8]) {
    let bad = bad_bytes(stream);
    assert!(
        bad.is_empty(),
        "{what} holds control bytes {bad:02x?} in {:?}",
        String::from_utf8_lossy(stream)
    );
}

#[test]
fn the_helper_finds_what_it_is_looking_for() {
    assert_clean("plain text", "Lens  FE 24-70mm — 35 mm\n".as_bytes());
    for dirty in [
        &b"a\x1b[2Jb"[..],
        b"a\x07b",
        b"a\x7fb",
        b"a\rb",
        b"a\tb",
        "a\u{85}b".as_bytes(),
        "a\u{9b}b".as_bytes(),
        b"a\x9bb",
    ] {
        assert!(!bad_bytes(dirty).is_empty(), "{dirty:02x?} went unnoticed");
    }
}

#[test]
fn info_prints_no_control_bytes_for_a_hostile_file() {
    let sandbox = Sandbox::new();
    let clean = sandbox.arw("DSC00009.ARW");
    let baseline = sandbox
        .ym()
        .arg("info")
        .arg(&clean)
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    for (name, path) in [
        ("utf-8", write_hostile_utf8(&sandbox.photos())),
        ("raw bytes", write_hostile_raw_bytes(&sandbox.photos())),
    ] {
        let run = sandbox.ym().arg("info").arg(&path).assert().success();
        let out = &run.get_output().stdout;
        assert_clean(&format!("`ym info` ({name}), stdout"), out);
        assert_clean(
            &format!("`ym info` ({name}), stderr"),
            &run.get_output().stderr,
        );
        // Text that came from the file stays on its row: as many lines as for a file with clean tags.
        assert_eq!(
            lines(out).len(),
            lines(&baseline).len(),
            "`ym info` ({name}) was given more lines by a tag:\n{}",
            String::from_utf8_lossy(out)
        );
        // The row of the lens still shows what could be read.
        let text = String::from_utf8_lossy(out);
        assert!(text.contains("Lens"), "{text}");
        assert!(
            text.contains('\u{FFFD}'),
            "the controls should show as U+FFFD:\n{text}"
        );
    }
}

#[test]
fn info_json_prints_no_control_bytes_for_a_hostile_file() {
    let sandbox = Sandbox::new();
    for (name, path) in [
        ("utf-8", write_hostile_utf8(&sandbox.photos())),
        ("raw bytes", write_hostile_raw_bytes(&sandbox.photos())),
    ] {
        let run = sandbox
            .ym()
            .args(["info", "--json"])
            .arg(&path)
            .assert()
            .success();
        assert_clean(
            &format!("`ym info --json` ({name}), stdout"),
            &run.get_output().stdout,
        );
        assert_clean(
            &format!("`ym info --json` ({name}), stderr"),
            &run.get_output().stderr,
        );
        // One object per file, on one line.
        let objects = json_lines(&run.get_output().stdout);
        assert_eq!(objects.len(), 1, "{name}");
        assert_eq!(objects[0]["status"], "ok", "{name}");
        // The values themselves (after JSON decoding) hold none either.
        let model = objects[0]["camera"]["model"].as_str().expect("model");
        assert!(
            model.starts_with("ILCE-7RM4") || name == "raw bytes",
            "{model:?}"
        );
        assert!(
            !objects[0]
                .to_string()
                .chars()
                .any(|c| c.is_control() && c != '\n'),
            "{name}: a control character in the decoded JSON: {}",
            objects[0]
        );
    }
}

#[test]
fn render_json_prints_no_control_bytes_for_a_hostile_file() {
    let sandbox = Sandbox::new();
    for (name, path) in [
        ("utf-8", write_hostile_utf8(&sandbox.photos())),
        ("raw bytes", write_hostile_raw_bytes(&sandbox.photos())),
    ] {
        let out_dir = sandbox
            .root()
            .join(format!("out-{}", name.replace(' ', "-")));
        let run = sandbox
            .ym()
            .args(["render", "--json", "--format", "jpeg", "--out"])
            .arg(&out_dir)
            .arg(&path)
            .assert()
            .success();
        assert_clean(
            &format!("`ym render --json` ({name}), stdout"),
            &run.get_output().stdout,
        );
        assert_clean(
            &format!("`ym render --json` ({name}), stderr"),
            &run.get_output().stderr,
        );
        let objects = json_lines(&run.get_output().stdout);
        assert_eq!(objects.len(), 1, "{name}");
        assert_eq!(objects[0]["status"], "ok", "{name}: {}", objects[0]);
        assert!(
            !objects[0]
                .to_string()
                .chars()
                .any(|c| c.is_control() && c != '\n'),
            "{name}: a control character in the decoded JSON: {}",
            objects[0]
        );
    }
}

/// A file name with an escape sequence, a newline (a forged line), DEL and two C1 characters in it.
const HOSTILE_NAME: &str = "DSC\u{1b}[31mRED\u{1b}[0m\nError: forged.\u{7f} \u{85}\u{9b}.ARW";

/// The way the name shows on a line for a person.
const HOSTILE_NAME_SHOWN: &str =
    "DSC\\x1b[31mRED\\x1b[0m\\x0aError: forged.\\x7f \\u{85}\\u{9b}.ARW";

/// A folder name with the same kinds of characters, for the folders the tool prints.
const HOSTILE_DIR: &str = "out\u{1b}]0;PWNED\u{7}\u{9b}2J";

/// The name of [`HOSTILE_DIR`] as a line for a person shows it.
const HOSTILE_DIR_SHOWN: &str = "out\\x1b]0;PWNED\\x07\\u{9b}2J";

/// A hostile-named file that is not an ARW file at all (the tool says so, naming the file).
fn hostile_text_file(sandbox: &Sandbox) -> PathBuf {
    let path = sandbox.photos().join("not\u{1b}[2J\u{9b}raw.ARW");
    std::fs::write(&path, b"this is not a raw file").expect("write text file");
    path
}

/// Asserts that neither stream of a run holds a control byte, a newline in stdout excepted.
fn assert_run_clean(what: &str, run: &assert_cmd::assert::Assert) {
    let output = run.get_output();
    assert_clean(&format!("{what}, stdout"), &output.stdout);
    assert_clean(&format!("{what}, stderr"), &output.stderr);
}

/// The text of `ym info` for a file name that holds controls shows them as escapes, on one row, and
/// the JSON says the exact name.
#[test]
fn info_prints_no_control_bytes_for_a_hostile_file_name() {
    let sandbox = Sandbox::new();
    let clean = sandbox.arw("DSC00009.ARW");
    let baseline = sandbox
        .ym()
        .arg("info")
        .arg(&clean)
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let path = sandbox.arw(HOSTILE_NAME);
    assert!(path.is_file());

    let run = sandbox.ym().arg("info").arg(&path).assert().success();
    assert_run_clean("`ym info` (text) of a hostile file name", &run);
    let out = &run.get_output().stdout;
    let text = String::from_utf8(out.clone()).expect("utf-8");
    // The name is on the first line, as escapes, and the newline in it added no line.
    let first = text.lines().next().expect("a first line");
    assert!(
        first.ends_with(HOSTILE_NAME_SHOWN),
        "the first line is {first:?}"
    );
    assert_eq!(lines(out).len(), lines(&baseline).len(), "{text}");

    let run = sandbox
        .ym()
        .args(["info", "--json"])
        .arg(&path)
        .assert()
        .success();
    assert_run_clean("`ym info --json` of a hostile file name", &run);
    let stdout = text_of(&run.get_output().stdout);
    // DEL and the C1 characters are escaped, like the C0 controls.
    for escape in [r"\u001b", r"\u007f", r"\u0085", r"\u009b", r"\n"] {
        assert!(stdout.contains(escape), "{escape} is not in {stdout}");
    }
    // The value is the exact name.
    let objects = json_lines(&run.get_output().stdout);
    assert_eq!(objects.len(), 1);
    assert_eq!(objects[0]["status"], "ok");
    let input = objects[0]["input"].as_str().expect("input");
    assert!(input.ends_with(HOSTILE_NAME), "{input:?}");
}

/// `ym render` names the input in its progress and error lines, in the verbose lines and in JSON.
#[test]
fn render_prints_no_control_bytes_for_a_hostile_file_name() {
    let sandbox = Sandbox::new();
    let path = sandbox.arw(HOSTILE_NAME);
    for (what, extra) in [
        ("text", vec!["--verbose"]),
        ("JSON", vec!["--json"]),
        ("JSON and verbose", vec!["--json", "--verbose"]),
    ] {
        let out_dir = sandbox
            .root()
            .join(format!("out-{}", what.replace(' ', "-")));
        let run = sandbox
            .ym()
            .args(["render", "--format", "jpeg", "--out"])
            .arg(&out_dir)
            .args(&extra)
            .arg(&path)
            .assert()
            .success();
        assert_run_clean(
            &format!("`ym render` ({what}) of a hostile file name"),
            &run,
        );
        let stderr = text_of(&run.get_output().stderr);
        if what == "text" {
            // The detail line names the input, with the escapes.
            assert!(stderr.contains(HOSTILE_NAME_SHOWN), "{stderr}");
            // Data on stdout: the file that was written, a normal name (the stem is made safe).
            let written = lines(&run.get_output().stdout);
            assert_eq!(written.len(), 1, "{written:?}");
            assert!(Path::new(&written[0]).is_file(), "{written:?}");
        } else {
            let objects = json_lines(&run.get_output().stdout);
            assert_eq!(objects.len(), 1);
            assert_eq!(objects[0]["status"], "ok", "{}", objects[0]);
            let input = objects[0]["input"].as_str().expect("input");
            assert!(input.ends_with(HOSTILE_NAME), "{input:?}");
            let stdout = text_of(&run.get_output().stdout);
            for escape in [r"\u001b", r"\u007f", r"\u0085", r"\u009b"] {
                assert!(stdout.contains(escape), "{escape} is not in {stdout}");
            }
        }
    }
}

/// A file that cannot be developed, and one that is not there, are named in an error line.
#[test]
fn errors_name_a_hostile_path_without_control_bytes() {
    let sandbox = Sandbox::new();
    let not_raw = hostile_text_file(&sandbox);
    let missing = sandbox.photos().join(HOSTILE_NAME);
    assert!(!missing.exists());
    for (command, args) in [
        ("render", vec!["render"]),
        ("render --json", vec!["render", "--json"]),
        ("info", vec!["info"]),
        ("info --json", vec!["info", "--json"]),
    ] {
        for path in [&not_raw, &missing] {
            let run = sandbox.ym().args(&args).arg(path).assert().failure();
            assert_run_clean(&format!("`ym {command}` of {path:?}"), &run);
            if !command.ends_with("--json") {
                // The error line says which file, as escapes.
                let shown = text_of(&run.get_output().stderr);
                assert!(
                    shown.contains("\\x1b") || shown.contains("\\u{9b}"),
                    "{shown}"
                );
            }
        }
    }
}

/// The folder an export went to is named in the summary on stderr; the line of data on stdout is
/// the exact path of the file when stdout is a pipe.
#[test]
fn a_hostile_output_folder_is_escaped_for_a_person_and_exact_for_a_pipe() {
    let sandbox = Sandbox::new();
    let path = sandbox.arw("DSC00001.ARW");
    let out_dir = sandbox.root().join(HOSTILE_DIR);
    let run = sandbox
        .ym()
        .args(["render", "--format", "jpeg", "--out"])
        .arg(&out_dir)
        .arg(&path)
        .assert()
        .success();
    let output = run.get_output();
    assert_clean("`ym render` into a hostile folder, stderr", &output.stderr);
    let stderr = text_of(&output.stderr);
    assert!(
        stderr.contains(&format!(
            "to {}/{HOSTILE_DIR_SHOWN}",
            sandbox.root().display()
        )),
        "{stderr}"
    );
    // The pipe gets the file name as it is: the next program has to find the file.
    let stdout = text_of(&output.stdout);
    let written = stdout.trim_end_matches('\n');
    assert!(written.contains(HOSTILE_DIR), "{stdout:?}");
    assert!(Path::new(written).is_file(), "{stdout:?}");
}

/// `ym doctor` prints the folders it uses, and the folder of the app is chosen by the environment.
#[test]
fn doctor_prints_no_control_bytes_for_a_hostile_folder() {
    let sandbox = Sandbox::new();
    let home = sandbox.root().join(HOSTILE_DIR);
    std::fs::create_dir_all(&home).expect("home folder");
    let run = sandbox
        .ym()
        .env("YELLOWMOON_HOME", &home)
        .arg("doctor")
        .assert()
        .success();
    assert_run_clean("`ym doctor` with a hostile app folder", &run);
    let text = text_of(&run.get_output().stdout);
    assert!(text.contains(HOSTILE_DIR_SHOWN), "{text}");

    let run = sandbox
        .ym()
        .env("YELLOWMOON_HOME", &home)
        .args(["doctor", "--json"])
        .assert()
        .success();
    assert_run_clean("`ym doctor --json` with a hostile app folder", &run);
    let objects = json_lines(&run.get_output().stdout);
    assert_eq!(objects.len(), 1);
    let folder = objects[0]["looks_folder"].as_str().expect("looks_folder");
    assert!(folder.contains(HOSTILE_DIR), "{folder:?}");
}

/// `ym looks import` names the path it could not read.
#[test]
fn looks_import_names_a_hostile_path_without_control_bytes() {
    let sandbox = Sandbox::new();
    let missing = sandbox.root().join(HOSTILE_DIR).join("look.cube");
    let run = sandbox
        .ym()
        .args(["looks", "import"])
        .arg(&missing)
        .assert()
        .failure();
    assert_run_clean("`ym looks import` of a hostile path", &run);
    let stderr = text_of(&run.get_output().stderr);
    assert!(stderr.contains(HOSTILE_DIR_SHOWN), "{stderr}");
}

/// The name of an imported `.cube` is its file name, so a name with controls in it reaches the
/// table of `ym looks list`: escaped for a person, exact (and `\u`-escaped) in JSON.
#[test]
fn looks_list_names_from_a_hostile_file_name_are_escaped() {
    let sandbox = Sandbox::new();
    let dir = sandbox.root().join("cubes");
    std::fs::create_dir_all(&dir).expect("folder");
    let path = dir.join("ev\u{1b}[2Jil\u{9b}.cube");
    let mut cube = String::from("TITLE \"Plain\"\nLUT_3D_SIZE 2\n");
    for b in 0..2 {
        for g in 0..2 {
            for r in 0..2 {
                cube.push_str(&format!("{r} {g} {b}\n"));
            }
        }
    }
    std::fs::write(&path, cube).expect("write cube");
    let run = sandbox
        .ym()
        .args(["looks", "import", "--input", "logc4"])
        .arg(&path)
        .assert()
        .success();
    assert_run_clean("`ym looks import` of a hostile file name", &run);

    let run = sandbox.ym().args(["looks", "list"]).assert().success();
    assert_run_clean("`ym looks list` with a hostile look name", &run);
    let shown = text_of(&run.get_output().stdout);
    assert!(shown.contains("ev\\x1b[2Jil\\u{9b}"), "{shown}");
    // The escape added no row: the header, the built-in look and the imported one.
    assert_eq!(lines(&run.get_output().stdout).len(), 3, "{shown}");

    let run = sandbox
        .ym()
        .args(["looks", "list", "--json"])
        .assert()
        .success();
    assert_run_clean("`ym looks list --json` with a hostile look name", &run);
    let stdout = text_of(&run.get_output().stdout);
    assert!(
        stdout.contains(r"\u001b") && stdout.contains(r"\u009b"),
        "{stdout}"
    );
    let objects = json_lines(&run.get_output().stdout);
    let name = objects[0]["looks"][1]["name"].as_str().expect("name");
    assert_eq!(name, "ev\u{1b}[2Jil\u{9b}");
}
