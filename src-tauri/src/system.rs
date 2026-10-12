//! What the webview asks the shell to do on the desktop: open a page in the browser, show a file in
//! Finder, move a file to the Trash, copy text, run `ym doctor`.
//!
//! Nothing here touches the network: a page opens in the system browser because the person
//! clicked it (`AGENTS.md` rule 7), and only a short list of addresses is opened at all.

use std::io::Write;
use std::path::Path;
use std::process::{Command, Stdio};

use crate::wire::{AppError, ErrorScope};

/// The project's home, opened by the Help menu and by About.
pub const GITHUB_URL: &str = "https://github.com/ngriaznov/yellowmoon";

/// The bug report form (`.github/ISSUE_TEMPLATE/bug_report.yml`).
pub const ISSUE_URL: &str =
    "https://github.com/ngriaznov/yellowmoon/issues/new?template=bug_report.yml";

/// The hosts `open_url` opens: ARRI's look page (`docs/08`) and GitHub's pages and help
/// (`AGENTS.md` rule 7). Anything else is refused.
const ALLOWED_HOSTS: [&str; 4] = ["www.arri.com", "arri.com", "github.com", "docs.github.com"];

/// Checks that `url` is an `https` address on one of the allowed hosts.
///
/// # Errors
///
/// [`AppError`] with a plain sentence when it is not.
pub fn check_url(url: &str) -> Result<(), AppError> {
    let refuse = || {
        AppError::new(
            ErrorScope::App,
            "Yellowmoon only opens pages on arri.com and github.com.",
        )
    };
    let rest = url.strip_prefix("https://").ok_or_else(refuse)?;
    if rest.chars().any(|c| c.is_control() || c.is_whitespace()) {
        return Err(refuse());
    }
    let authority = rest.split(['/', '?', '#']).next().unwrap_or("");
    // No user name or password, and the standard port only.
    if authority.contains('@') || authority.contains(':') {
        return Err(refuse());
    }
    if ALLOWED_HOSTS
        .iter()
        .any(|h| h.eq_ignore_ascii_case(authority))
    {
        Ok(())
    } else {
        Err(refuse())
    }
}

fn failed(what: &str, e: impl std::fmt::Display) -> AppError {
    tracing::warn!("{what}: {e}");
    AppError::new(
        ErrorScope::App,
        format!("{what}. Try again, or do it by hand."),
    )
}

/// The command that opens a thing with the system's default application.
fn opener() -> &'static str {
    if cfg!(target_os = "macos") {
        "open"
    } else {
        "xdg-open"
    }
}

/// Opens an address in the system browser.
///
/// # Errors
///
/// [`AppError`] when the address is not allowed or the browser cannot be started.
pub fn open_url(url: &str) -> Result<(), AppError> {
    check_url(url)?;
    Command::new(opener())
        .arg(url)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .map(|_child| ())
        .map_err(|e| failed("The page could not be opened in your browser", e))
}

/// Shows a file in Finder (selected), or opens a folder.
///
/// # Errors
///
/// [`AppError`] when the path is not there or Finder cannot be started.
pub fn reveal_in_finder(path: &Path) -> Result<(), AppError> {
    let meta = std::fs::metadata(path)
        .map_err(|_| AppError::new(ErrorScope::App, "That file or folder isn't there any more."))?;
    let mut command = Command::new(opener());
    if cfg!(target_os = "macos") {
        if meta.is_dir() {
            command.arg(path);
        } else {
            command.arg("-R").arg(path);
        }
    } else if meta.is_dir() {
        command.arg(path);
    } else if let Some(parent) = path.parent() {
        command.arg(parent);
    }
    command
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .map(|_child| ())
        .map_err(|e| failed("Finder could not be opened", e))
}

/// Moves a downloaded zip to the Trash (recoverable). Only `.zip` files: that is all the import
/// wizard offers to trash (`docs/11`).
///
/// # Errors
///
/// [`AppError`] when the file is not a zip, is not there, or the system refuses.
pub fn trash_path(path: &Path) -> Result<(), AppError> {
    let is_zip = path
        .extension()
        .is_some_and(|e| e.eq_ignore_ascii_case("zip"));
    if !is_zip {
        return Err(AppError::new(
            ErrorScope::App,
            "Only the zip you downloaded can be moved to the Trash from here.",
        ));
    }
    ym_looks::import::trash::move_to_trash(path)
        .map_err(|e| AppError::new(ErrorScope::Looks, e.to_string()))
}

/// Copies text to the system clipboard.
///
/// # Errors
///
/// [`AppError`] when no clipboard tool is available.
pub fn copy_to_clipboard(text: &str) -> Result<(), AppError> {
    let candidates: &[(&str, &[&str])] = if cfg!(target_os = "macos") {
        &[("pbcopy", &[])]
    } else {
        &[
            ("wl-copy", &[]),
            ("xclip", &["-selection", "clipboard"]),
            ("xsel", &["--clipboard", "--input"]),
        ]
    };
    for (program, args) in candidates {
        let Ok(mut child) = Command::new(program)
            .args(*args)
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
        else {
            continue;
        };
        if let Some(mut stdin) = child.stdin.take()
            && stdin.write_all(text.as_bytes()).is_err()
        {
            let _ = child.kill();
            continue;
        }
        if child.wait().is_ok_and(|s| s.success()) {
            return Ok(());
        }
    }
    Err(AppError::new(
        ErrorScope::App,
        "The text could not be copied to the clipboard.",
    ))
}

/// Runs `ym doctor` (the CLI that ships in the app) and returns what it prints.
///
/// # Errors
///
/// [`AppError`] when the command cannot be run or fails.
pub fn doctor_output(ym: &Path) -> Result<String, AppError> {
    let output = Command::new(ym)
        .arg("doctor")
        .stdin(Stdio::null())
        .output()
        .map_err(|e| failed("The `ym doctor` report could not be made", e))?;
    if !output.status.success() {
        return Err(failed(
            "The `ym doctor` report could not be made",
            String::from_utf8_lossy(&output.stderr),
        ));
    }
    Ok(String::from_utf8_lossy(&output.stdout).into_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_https_pages_on_the_allowed_hosts_open() {
        for ok in [
            "https://www.arri.com/en/learn/camera-systems/image-science/look-files",
            "https://github.com/ngriaznov/yellowmoon",
            ISSUE_URL,
            "https://docs.github.com/en/issues",
            "https://GitHub.com/ngriaznov/yellowmoon/issues",
        ] {
            assert!(check_url(ok).is_ok(), "{ok}");
        }
        for bad in [
            "http://github.com/ngriaznov/yellowmoon",
            "file:///etc/passwd",
            "javascript:alert(1)",
            "https://evil.example/github.com",
            "https://github.com.evil.example/",
            "https://user@github.com/",
            "https://github.com:8443/",
            "https://github.com@evil.example/",
            "https://github.com/a b",
            "https://",
            "",
            "-a Calculator",
            "https://github.com/\nopen",
        ] {
            let err = check_url(bad).expect_err(bad);
            assert_eq!(err.scope, ErrorScope::App, "{bad}");
        }
    }

    #[test]
    fn only_zips_go_to_the_trash() {
        let dir = tempfile::tempdir().expect("tempdir");
        let doc = dir.path().join("notes.txt");
        std::fs::write(&doc, b"keep me").expect("write");
        let err = trash_path(&doc).expect_err("not a zip");
        assert!(err.message.contains("zip"), "{err}");
        assert!(doc.exists());
        let missing = trash_path(&dir.path().join("gone.zip")).expect_err("missing");
        assert_eq!(missing.scope, ErrorScope::Looks);
    }

    #[test]
    fn revealing_something_that_is_gone_says_so() {
        let dir = tempfile::tempdir().expect("tempdir");
        let err = reveal_in_finder(&dir.path().join("nope")).expect_err("missing");
        assert!(err.message.contains("isn't there"), "{err}");
    }

    #[cfg(unix)]
    #[test]
    fn doctor_output_is_what_the_command_prints() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir().expect("tempdir");
        let fake = dir.path().join("ym");
        std::fs::write(&fake, "#!/bin/sh\necho \"doctor says $1\"\n").expect("write");
        std::fs::set_permissions(&fake, std::fs::Permissions::from_mode(0o755)).expect("chmod");
        assert_eq!(doctor_output(&fake).expect("runs"), "doctor says doctor\n");
        let failing = dir.path().join("ym-fails");
        std::fs::write(&failing, "#!/bin/sh\nexit 3\n").expect("write");
        std::fs::set_permissions(&failing, std::fs::Permissions::from_mode(0o755)).expect("chmod");
        assert!(doctor_output(&failing).is_err());
        assert!(doctor_output(&dir.path().join("missing")).is_err());
    }
}
