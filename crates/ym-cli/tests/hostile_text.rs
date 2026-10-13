//! `ym info` and `ym render --json` on a file whose text tags carry terminal control bytes.
//!
//! The model, firmware and lens strings of a downloaded file reach a terminal through these two
//! commands. The decoder turns every control character of a text tag into U+FFFD
//! (`ym_raw::tiff::TiffReader::ascii`, tested in `ym-raw`); this test pins the other end: whatever
//! the commands print, text or JSON, holds no byte below 0x20 but the newline that ends a line (the
//! formats use no tab), and no DEL or C1 control character (U+0080 to U+009F, which a terminal
//! treats as escape introducers when it reads bytes as Latin-1, and which JSON does not escape).
//! A newline hidden in a tag must not add a line either: a forged line could pose as another row.
//! The names of the files are text from outside too and are not covered by the tests that run: see the ignored
//! test at the end, which pins a known gap.

mod common;

use std::path::Path;

use common::{Sandbox, json_lines, lines};
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

/// A file name is text from outside as well (a folder unpacked from a downloaded zip), and the
/// commands print it: `ym info` as the first line of its text, both commands in their JSON, and
/// every command in its error lines. Text mode prints it as it is, so an escape sequence in a name
/// reaches the terminal (`ym render` is safe for the files it writes: the stem is made safe, see
/// docs/09). JSON escapes the C0 controls but prints DEL and the C1 characters as they are.
#[test]
#[ignore = "known gap, found in the review of the control-byte test: the commands print an input path as it is; un-ignore when crates/ym-cli/src prints paths through one function that replaces control characters"]
fn info_prints_no_control_bytes_for_a_hostile_file_name() {
    let sandbox = Sandbox::new();
    let path = sandbox.arw("DSC\u{1b}[31mRED\u{1b}[0m \u{85}\u{9b}.ARW");
    for (what, args) in [("text", vec!["info"]), ("JSON", vec!["info", "--json"])] {
        let run = sandbox.ym().args(args).arg(&path).assert().success();
        assert_clean(
            &format!("`ym info` ({what}) of a hostile file name, stdout"),
            &run.get_output().stdout,
        );
        assert_clean(
            &format!("`ym info` ({what}) of a hostile file name, stderr"),
            &run.get_output().stderr,
        );
    }
}
