//! Text from outside, made safe to print.
//!
//! A file name is text somebody else chose (a folder unpacked from a downloaded archive, a card
//! from another camera). A terminal obeys the control characters in it: ESC starts a sequence that
//! can move the cursor, rewrite the title or clear the screen, and the C1 characters (U+0080 to
//! U+009F, of which U+009B is a one-character ESC `[`) do the same on terminals that read them.
//! Every line the tool prints for a person goes through the functions here, so such a name shows
//! up as `\x1b` or `\u{9b}` and does nothing.
//!
//! The escapes are fixed so that the output can be tested and read back:
//!
//! - a C0 control or DEL (U+0000 to U+001F, U+007F) is `\x` and two lower-case hex digits
//!   (`\x1b`, `\x0a`, `\x7f`);
//! - a C1 control (U+0080 to U+009F) is `\u{` and its hex digits `}` (`\u{9b}`);
//! - a byte that is not UTF-8 (a path on macOS or Linux may hold any bytes but NUL and `/`) is
//!   `\x` and two lower-case hex digits too, which cannot be mistaken for a C1 character because
//!   that one has the `\u{…}` form;
//! - everything else, a backslash included, is left as it is, so ordinary paths read and copy as
//!   they always did. A name that holds the four characters `\x1b` looks like one that holds ESC;
//!   the line is for reading, the JSON output is the exact one.
//!
//! JSON is a different case ([`json_line`]): the values keep their exact characters and are
//! escaped the way JSON does it. `serde_json` escapes the C0 controls but prints DEL and the C1
//! characters as they are, and a terminal showing a line of JSON would still obey them.

use std::fmt::Write as _;
use std::path::Path;

use serde_json::Value;

/// Appends `c` to `out`, escaped when it is a control character.
fn push_char(out: &mut String, c: char) {
    match c {
        '\0'..='\x1f' | '\x7f' => {
            // Writing to a `String` cannot fail.
            let _ = write!(out, "\\x{:02x}", u32::from(c));
        }
        '\u{80}'..='\u{9f}' => {
            let _ = write!(out, "\\u{{{:x}}}", u32::from(c));
        }
        _ => out.push(c),
    }
}

/// `text` with every control character (C0, DEL, C1) replaced by its visible escape.
///
/// Meant for a line of text that may hold something from a file: it never contains a control
/// character afterwards, a newline included, so it cannot add a line either. Text that holds none
/// comes back unchanged.
#[must_use]
pub fn line_text(text: &str) -> String {
    if !text.chars().any(is_control) {
        return text.to_owned();
    }
    let mut out = String::with_capacity(text.len() + 8);
    for c in text.chars() {
        push_char(&mut out, c);
    }
    out
}

/// A control character in the sense of this module: C0, DEL or C1.
fn is_control(c: char) -> bool {
    matches!(c, '\0'..='\x1f' | '\x7f'..='\u{9f}')
}

/// `bytes` as text: valid UTF-8 as [`line_text`] does it, every byte that is not part of a UTF-8
/// character as `\x` and two hex digits.
#[must_use]
pub fn bytes_text(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() + 8);
    for chunk in bytes.utf8_chunks() {
        for c in chunk.valid().chars() {
            push_char(&mut out, c);
        }
        for byte in chunk.invalid() {
            let _ = write!(out, "\\x{byte:02x}");
        }
    }
    out
}

/// A path as one printable line: the form every human-readable line of the tool uses for a path
/// (progress, `ym info`, errors, `ym doctor`).
///
/// Controls and bytes that are not UTF-8 are shown as escapes (see the module documentation); a
/// normal path comes back as it is. Not for JSON (use [`lossy`] there) and not for a path handed
/// to another program (see [`data_path`]).
#[must_use]
pub fn path_text(path: &Path) -> String {
    #[cfg(unix)]
    {
        use std::os::unix::ffi::OsStrExt;
        bytes_text(path.as_os_str().as_bytes())
    }
    #[cfg(not(unix))]
    {
        // Platforms where a path is not bytes: what `Display` makes of it, then the same escapes.
        line_text(&path.display().to_string())
    }
}

/// A path as the value of a JSON string: exact characters, bytes that are not UTF-8 as U+FFFD.
/// [`json_line`] escapes the controls when the object is written.
#[must_use]
pub fn lossy(path: &Path) -> String {
    path.to_string_lossy().into_owned()
}

/// A path on a line of data on stdout (the files a command wrote, for `xargs` and pipes).
///
/// A pipe or a file gets the path as it is: a program that reads the line needs the exact name.
/// A terminal gets [`path_text`], because a person reads it there and the terminal obeys what it
/// is given. `terminal` is whether stdout is one.
#[must_use]
pub fn data_path(path: &Path, terminal: bool) -> String {
    if terminal {
        path_text(path)
    } else {
        lossy(path)
    }
}

/// One JSON value as a line of text, with DEL and the C1 characters escaped as `\u007f`,
/// `\u0080` … `\u009f`.
///
/// The result parses to the same value. Only the form of the text changes, so that the line is safe
/// to show on a terminal; `serde_json` leaves those characters as they are.
#[must_use]
pub fn json_line(value: &Value) -> String {
    escape_json_controls(&value.to_string())
}

/// Escapes DEL and the C1 characters in text that is already JSON.
///
/// A character outside ASCII or DEL can only be inside a string of valid JSON (every other
/// character of the syntax is ASCII, and the C0 controls are escaped by the serialiser), so
/// replacing it by the `\u` escape of the same character leaves the document valid and equal.
#[must_use]
pub fn escape_json_controls(json: &str) -> String {
    if !json.chars().any(|c| matches!(c, '\x7f'..='\u{9f}')) {
        return json.to_owned();
    }
    let mut out = String::with_capacity(json.len() + 16);
    for c in json.chars() {
        if matches!(c, '\x7f'..='\u{9f}') {
            let _ = write!(out, "\\u{:04x}", u32::from(c));
        } else {
            out.push(c);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ordinary_text_and_paths_are_left_alone() {
        for text in [
            "/Users/anna/Photos/2026-10-04/DSC00001.ARW",
            "Fotos/Größe — 35 mm.ARW",
            "写真/DSC00001.ARW",
            r"C:\Photos\a.ARW",
            "",
        ] {
            assert_eq!(line_text(text), text);
            assert_eq!(path_text(Path::new(text)), text);
            assert_eq!(bytes_text(text.as_bytes()), text);
        }
    }

    #[test]
    fn c0_controls_and_del_show_as_hex_escapes() {
        assert_eq!(line_text("a\u{1b}[31mb"), "a\\x1b[31mb");
        assert_eq!(line_text("\u{0}\u{7}\u{7f}"), "\\x00\\x07\\x7f");
        // A newline or a tab cannot add a line or a column.
        assert_eq!(line_text("a\nb\rc\td"), "a\\x0ab\\x0dc\\x09d");
    }

    #[test]
    fn c1_controls_show_as_unicode_escapes() {
        assert_eq!(line_text("a\u{85}b\u{9b}c"), "a\\u{85}b\\u{9b}c");
        // The neighbours of the range are ordinary characters.
        assert_eq!(line_text("\u{a0}\u{7e}"), "\u{a0}\u{7e}");
        let all: String = ('\u{80}'..='\u{9f}').collect();
        let shown = line_text(&all);
        assert!(shown.is_ascii(), "{shown}");
        assert_eq!(shown.matches("\\u{").count(), 32);
    }

    #[test]
    fn nothing_that_a_terminal_obeys_survives() {
        let mut every = String::new();
        every.extend((0u32..0xa0).filter_map(char::from_u32));
        let shown = line_text(&every);
        assert!(
            !shown.chars().any(|c| is_control(c)),
            "a control character is left in {shown:?}"
        );
        // Printable ASCII keeps its place.
        assert!(shown.contains(" !\"#$%&'()*+,-./0123456789:;<=>?@ABC"));
    }

    #[test]
    fn bytes_that_are_not_utf8_show_as_byte_escapes() {
        assert_eq!(bytes_text(b"a\x9bb"), "a\\x9bb");
        assert_eq!(bytes_text(b"a\xffb\xc0\xafc"), "a\\xffb\\xc0\\xafc");
        // A truncated character at the end.
        assert_eq!(bytes_text(b"ab\xe2\x80"), "ab\\xe2\\x80");
        // The C1 character U+009B as UTF-8 (C2 9B) is a character, not a stray byte.
        assert_eq!(bytes_text("a\u{9b}b".as_bytes()), "a\\u{9b}b");
        // A character above the range is kept.
        assert_eq!(bytes_text("a\u{2014}b".as_bytes()), "a\u{2014}b");
    }

    #[cfg(unix)]
    #[test]
    fn a_path_with_bytes_that_are_not_utf8_is_shown_without_loss() {
        use std::ffi::OsStr;
        use std::os::unix::ffi::OsStrExt;
        let path = Path::new(OsStr::from_bytes(b"/x/\x1b[2J\x9b\xff.ARW"));
        assert_eq!(path_text(path), "/x/\\x1b[2J\\x9b\\xff.ARW");
        // JSON gets the replacement character for the byte that is not text.
        assert_eq!(lossy(path), "/x/\u{1b}[2J\u{fffd}\u{fffd}.ARW");
    }

    #[test]
    fn data_lines_are_exact_for_a_pipe_and_escaped_for_a_terminal() {
        let path = Path::new("/x/a\u{1b}[2Jb\u{9b}.jpg");
        assert_eq!(data_path(path, false), "/x/a\u{1b}[2Jb\u{9b}.jpg");
        assert_eq!(data_path(path, true), "/x/a\\x1b[2Jb\\u{9b}.jpg");
    }

    #[test]
    fn json_keeps_the_value_and_escapes_del_and_c1() {
        let hostile = "a\u{1b}b\u{7f}c\u{85}d\u{9b}e\n\u{2014}\u{fffd}";
        let value = serde_json::json!({ "input": hostile, hostile: [hostile] });
        let line = json_line(&value);
        // Nothing on the line is a control character, and it is one line.
        assert!(
            !line.chars().any(is_control),
            "a control character in {line:?}"
        );
        // The escapes are the JSON ones, and other characters stay as they are.
        assert!(
            line.contains("a\\u001bb\\u007fc\\u0085d\\u009be\\n\u{2014}\u{fffd}"),
            "{line}"
        );
        // It is still valid JSON and says the same.
        let back: Value = serde_json::from_str(&line).expect("valid JSON");
        assert_eq!(back, value);
    }

    #[test]
    fn json_without_such_characters_is_unchanged() {
        let value = serde_json::json!({ "a": "b\u{1b}", "n": 1.5, "t": true, "z": null });
        assert_eq!(json_line(&value), value.to_string());
        assert_eq!(escape_json_controls("{\"a\":1}"), "{\"a\":1}");
    }
}
