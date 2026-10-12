//! "Install Command-Line Tool…" (`docs/11` §Menus, `docs/12` §Install paths): a link called `ym` to
//! the command-line tool that ships inside the app (`Yellowmoon.app/Contents/MacOS/ym`).
//!
//! The link goes to `/usr/local/bin/ym` when that works. A folder only an administrator can
//! write to is handled on macOS with the system's password prompt (`osascript … with
//! administrator privileges`); if the person cancels the prompt, or the prompt fails, the link
//! goes to `~/.local/bin/ym` instead, which needs no rights, and the answer says where it is.
//! Elsewhere (Linux development builds) only the second place is used.
//!
//! An existing `ym` that is not a link to a working file (a different program the person put
//! there) is never replaced.

use std::path::{Path, PathBuf};
#[cfg(target_os = "macos")]
use std::process::{Command, Stdio};

use crate::wire::{AppError, CliLinkStatus, ErrorScope};

/// Why the system's password prompt did not give us the rights.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ElevateError {
    /// The person cancelled the prompt.
    #[cfg_attr(not(target_os = "macos"), allow(dead_code))]
    Cancelled,
    /// The prompt, or the command behind it, failed.
    Failed(String),
}

/// Creates a link in a folder that needs more rights than the app has.
pub trait Elevate {
    /// Makes `target` a link to `source`, creating its folder, with the rights of an administrator.
    ///
    /// # Errors
    ///
    /// [`ElevateError::Cancelled`] when the person declined, [`ElevateError::Failed`] otherwise.
    fn link(&self, source: &Path, target: &Path) -> Result<(), ElevateError>;
}

/// Never gets more rights (Linux, tests).
#[derive(Debug, Clone, Copy, Default)]
#[cfg_attr(target_os = "macos", allow(dead_code))]
pub struct NoElevation;

impl Elevate for NoElevation {
    fn link(&self, _source: &Path, _target: &Path) -> Result<(), ElevateError> {
        Err(ElevateError::Failed(
            "There is no way to ask for an administrator's rights here.".to_owned(),
        ))
    }
}

/// `'…'` for a POSIX shell: single quotes, with `'` written as `'\''`.
#[cfg(any(target_os = "macos", test))]
#[must_use]
pub fn shell_quote(s: &str) -> String {
    format!("'{}'", s.replace('\'', r"'\''"))
}

/// `"…"` for AppleScript: backslash and double quote escaped.
#[cfg(any(target_os = "macos", test))]
#[must_use]
pub fn applescript_quote(s: &str) -> String {
    format!("\"{}\"", s.replace('\\', r"\\").replace('"', r#"\""#))
}

/// The AppleScript that makes the link with an administrator's rights.
#[cfg(any(target_os = "macos", test))]
#[must_use]
pub fn admin_script(source: &Path, target: &Path) -> String {
    let parent = target.parent().unwrap_or_else(|| Path::new("/"));
    let shell = format!(
        "mkdir -p {} && ln -sf {} {}",
        shell_quote(&parent.to_string_lossy()),
        shell_quote(&source.to_string_lossy()),
        shell_quote(&target.to_string_lossy()),
    );
    format!(
        "do shell script {} with administrator privileges",
        applescript_quote(&shell)
    )
}

/// The system's password prompt, through `osascript` (macOS).
#[cfg(target_os = "macos")]
#[derive(Debug, Clone, Copy, Default)]
pub struct Osascript;

#[cfg(target_os = "macos")]
impl Elevate for Osascript {
    fn link(&self, source: &Path, target: &Path) -> Result<(), ElevateError> {
        let output = Command::new("osascript")
            .arg("-e")
            .arg(admin_script(source, target))
            .stdin(Stdio::null())
            .output()
            .map_err(|e| ElevateError::Failed(e.to_string()))?;
        if output.status.success() {
            return Ok(());
        }
        let stderr = String::from_utf8_lossy(&output.stderr);
        // "execution error: User canceled. (-128)"
        if stderr.contains("-128") {
            Err(ElevateError::Cancelled)
        } else {
            Err(ElevateError::Failed(stderr.trim().to_owned()))
        }
    }
}

/// Where the tool is and where its link goes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CliLink {
    /// The `ym` that ships with the app.
    pub source: PathBuf,
    /// `/usr/local/bin/ym` (macOS only).
    pub system_link: Option<PathBuf>,
    /// `~/.local/bin/ym`.
    pub user_link: PathBuf,
    /// Other places that count as installed: the Homebrew cask's link.
    pub other_links: Vec<PathBuf>,
}

impl CliLink {
    /// The places of this machine: the tool next to the running executable (resolved through links,
    /// because the app may itself be started through one) and the home folder.
    ///
    /// # Errors
    ///
    /// [`AppError`] when the executable's place or the home folder cannot be found.
    pub fn locate() -> Result<Self, AppError> {
        let exe = std::env::current_exe()
            .and_then(std::fs::canonicalize)
            .map_err(|e| {
                AppError::new(
                    ErrorScope::App,
                    format!("Yellowmoon could not find itself on disk ({e}). Reinstall the app."),
                )
            })?;
        let dir = exe.parent().unwrap_or_else(|| Path::new("/"));
        let home = ym_export::naming::home_dir().ok_or_else(|| {
            AppError::new(
                ErrorScope::App,
                "Yellowmoon could not find your home folder.",
            )
        })?;
        Ok(Self::new(dir.join("ym"), &home))
    }

    /// The places for a tool at `source` and a home folder.
    #[must_use]
    pub fn new(source: PathBuf, home: &Path) -> Self {
        let macos = cfg!(target_os = "macos");
        Self {
            source,
            system_link: macos.then(|| PathBuf::from("/usr/local/bin/ym")),
            user_link: home.join(".local").join("bin").join("ym"),
            other_links: if macos {
                vec![PathBuf::from("/opt/homebrew/bin/ym")]
            } else {
                Vec::new()
            },
        }
    }

    /// The links that count, in the order they are looked at.
    fn candidates(&self) -> impl Iterator<Item = &Path> {
        self.system_link
            .iter()
            .chain(&self.other_links)
            .chain(std::iter::once(&self.user_link))
            .map(PathBuf::as_path)
    }

    /// Whether a working `ym` link exists, and where.
    #[must_use]
    pub fn status(&self) -> CliLinkStatus {
        let found = self.candidates().find(|p| works(p));
        CliLinkStatus {
            installed: found.is_some(),
            path: found.map(|p| p.to_string_lossy().into_owned()),
        }
    }

    /// Makes the link (see the module documentation) and returns where it is.
    ///
    /// # Errors
    ///
    /// [`AppError`] when the tool is not in the app, a different file is in the way, or no place
    /// can be written.
    pub fn install(&self, elevate: &dyn Elevate) -> Result<PathBuf, AppError> {
        // An app opened straight from its download or from the disk image runs from a temporary,
        // read-only copy that macOS removes when the app quits: a link to it would be dead the
        // next time.
        if self
            .source
            .components()
            .any(|c| c.as_os_str() == "AppTranslocation")
        {
            return Err(AppError::new(
                ErrorScope::App,
                "Yellowmoon is running from a temporary place. Move it to your Applications folder, open it from there, then install the command-line tool.",
            ));
        }
        if !self.source.is_file() {
            return Err(AppError::new(
                ErrorScope::App,
                "The command-line tool isn't in this copy of Yellowmoon. Reinstall the app.",
            ));
        }
        if let Some(system) = &self.system_link {
            match self.link_system(system, elevate) {
                Ok(()) => return Ok(system.clone()),
                // A different program is there: say so instead of hiding it with another link.
                Err(SystemFailure::InTheWay) => return Err(in_the_way(system)),
                Err(SystemFailure::Declined) => {
                    tracing::info!(
                        "the administrator prompt was declined; linking in ~/.local/bin"
                    );
                }
                Err(SystemFailure::Failed(why)) => {
                    tracing::warn!("could not link in /usr/local/bin ({why}); using ~/.local/bin");
                }
            }
        }
        self.link_user()?;
        Ok(self.user_link.clone())
    }

    fn link_system(&self, target: &Path, elevate: &dyn Elevate) -> Result<(), SystemFailure> {
        match claim(&self.source, target) {
            Claim::Done => return Ok(()),
            Claim::InTheWay => return Err(SystemFailure::InTheWay),
            Claim::Denied => {}
        }
        match elevate.link(&self.source, target) {
            Ok(()) if works(target) => Ok(()),
            Ok(()) => Err(SystemFailure::Failed("the link was not created".to_owned())),
            Err(ElevateError::Cancelled) => Err(SystemFailure::Declined),
            Err(ElevateError::Failed(why)) => Err(SystemFailure::Failed(why)),
        }
    }

    fn link_user(&self) -> Result<(), AppError> {
        match claim(&self.source, &self.user_link) {
            Claim::Done => Ok(()),
            Claim::InTheWay => Err(in_the_way(&self.user_link)),
            Claim::Denied => Err(AppError::new(
                ErrorScope::App,
                format!(
                    "The link could not be made in {}. Check that you can write to it, then try again.",
                    self.user_link
                        .parent()
                        .unwrap_or_else(|| Path::new("."))
                        .display()
                ),
            )),
        }
    }
}

enum SystemFailure {
    Declined,
    InTheWay,
    Failed(String),
}

fn in_the_way(target: &Path) -> AppError {
    AppError::new(
        ErrorScope::App,
        format!(
            "A different file named ym is already in {}. Move it away, then install the tool again.",
            target.parent().unwrap_or_else(|| Path::new(".")).display()
        ),
    )
}

/// A link that leads to a file.
fn works(link: &Path) -> bool {
    std::fs::symlink_metadata(link).is_ok() && std::fs::metadata(link).is_ok_and(|m| m.is_file())
}

enum Claim {
    /// The link is there and leads to the tool.
    Done,
    /// A file that is not a link to a working file is in the way.
    InTheWay,
    /// The folder cannot be written without more rights.
    Denied,
}

/// Makes `target` a link to `source`, replacing an old link (a stale one from a moved app too).
fn claim(source: &Path, target: &Path) -> Claim {
    match std::fs::symlink_metadata(target) {
        Ok(meta) if meta.file_type().is_symlink() => {
            // Ours, or another copy's, or dangling: a link is ours to move.
            if std::fs::remove_file(target).is_err() {
                return Claim::Denied;
            }
        }
        Ok(_) => return Claim::InTheWay,
        Err(_) => {}
    }
    if let Some(parent) = target.parent()
        && std::fs::create_dir_all(parent).is_err()
    {
        return Claim::Denied;
    }
    #[cfg(unix)]
    {
        match std::os::unix::fs::symlink(source, target) {
            Ok(()) => Claim::Done,
            Err(_) => Claim::Denied,
        }
    }
    #[cfg(not(unix))]
    {
        let _ = source;
        Claim::Denied
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> (tempfile::TempDir, CliLink) {
        let dir = tempfile::tempdir().expect("tempdir");
        let source = dir.path().join("app").join("ym");
        std::fs::create_dir_all(source.parent().expect("parent")).expect("mkdir");
        std::fs::write(&source, b"#!/bin/sh\n").expect("write");
        let home = dir.path().join("home");
        let mut link = CliLink::new(source, &home);
        // Stand in for /usr/local/bin with a folder of the test's own.
        link.system_link = Some(dir.path().join("usr-local-bin").join("ym"));
        link.other_links.clear();
        (dir, link)
    }

    /// An elevator that can make the system link, declines, or fails.
    struct Fake(Result<(), ElevateError>, std::cell::Cell<u32>);

    impl Elevate for Fake {
        fn link(&self, source: &Path, target: &Path) -> Result<(), ElevateError> {
            self.1.set(self.1.get() + 1);
            if self.0.is_ok() {
                // An administrator can write where the app cannot.
                let parent = target.parent().expect("parent");
                std::fs::create_dir_all(parent).expect("mkdir");
                #[cfg(unix)]
                {
                    use std::os::unix::fs::PermissionsExt;
                    std::fs::set_permissions(parent, std::fs::Permissions::from_mode(0o755))
                        .expect("chmod");
                    std::os::unix::fs::symlink(source, target).expect("link");
                }
            }
            self.0.clone()
        }
    }

    #[cfg(unix)]
    #[test]
    fn the_link_goes_to_the_system_folder_when_it_can_be_written() {
        let (_dir, link) = fixture();
        assert_eq!(
            link.status(),
            CliLinkStatus {
                installed: false,
                path: None
            }
        );
        let elevate = Fake(Ok(()), std::cell::Cell::new(0));
        let made = link.install(&elevate).expect("installs");
        assert_eq!(Some(&made), link.system_link.as_ref());
        // The folder was ours to write: no prompt.
        assert_eq!(elevate.1.get(), 0);
        let status = link.status();
        assert!(status.installed);
        assert_eq!(status.path.as_deref(), made.to_str());
        assert_eq!(
            std::fs::canonicalize(&made).expect("leads"),
            std::fs::canonicalize(&link.source).expect("source")
        );
        // Installing again replaces the link without complaint.
        assert_eq!(link.install(&elevate).expect("again"), made);
    }

    #[cfg(unix)]
    #[test]
    fn a_folder_that_needs_rights_asks_for_them() {
        use std::os::unix::fs::PermissionsExt;
        let (dir, link) = fixture();
        let system = link.system_link.clone().expect("system");
        let folder = system.parent().expect("folder");
        std::fs::create_dir_all(folder).expect("mkdir");
        std::fs::set_permissions(folder, std::fs::Permissions::from_mode(0o555)).expect("chmod");
        // The tests run as root in some environments, where a read-only folder is still writable.
        let writable = std::fs::write(folder.join("probe"), b"").is_ok();
        let elevate = Fake(Ok(()), std::cell::Cell::new(0));
        let made = link.install(&elevate).expect("installs");
        std::fs::set_permissions(folder, std::fs::Permissions::from_mode(0o755))
            .expect("chmod back");
        assert_eq!(made, system);
        assert_eq!(elevate.1.get(), u32::from(!writable));
        drop(dir);
    }

    #[cfg(unix)]
    #[test]
    fn declining_the_prompt_links_in_the_home_folder() {
        use std::os::unix::fs::PermissionsExt;
        let (dir, link) = fixture();
        let system = link.system_link.clone().expect("system");
        let folder = system.parent().expect("folder");
        std::fs::create_dir_all(folder).expect("mkdir");
        std::fs::set_permissions(folder, std::fs::Permissions::from_mode(0o555)).expect("chmod");
        let writable = std::fs::write(folder.join("probe"), b"").is_ok();
        let elevate = Fake(Err(ElevateError::Cancelled), std::cell::Cell::new(0));
        let made = link.install(&elevate).expect("falls back");
        std::fs::set_permissions(folder, std::fs::Permissions::from_mode(0o755))
            .expect("chmod back");
        if writable {
            assert_eq!(made, system);
        } else {
            assert_eq!(made, link.user_link);
            assert!(works(&link.user_link));
            assert_eq!(link.status().path.as_deref(), made.to_str());
        }
        drop(dir);
    }

    #[cfg(unix)]
    #[test]
    fn without_a_system_place_the_home_folder_is_used() {
        let (_dir, mut link) = fixture();
        link.system_link = None;
        let made = link.install(&NoElevation).expect("installs");
        assert_eq!(made, link.user_link);
        assert!(link.status().installed);
        assert!(made.ends_with(".local/bin/ym"));
    }

    #[cfg(unix)]
    #[test]
    fn a_different_program_named_ym_is_never_replaced() {
        let (_dir, mut link) = fixture();
        link.system_link = None;
        let user = link.user_link.clone();
        std::fs::create_dir_all(user.parent().expect("parent")).expect("mkdir");
        std::fs::write(&user, b"someone else's ym").expect("write");
        let err = link.install(&NoElevation).expect_err("refuses");
        assert!(err.message.contains("already in"), "{err}");
        assert_eq!(std::fs::read(&user).expect("kept"), b"someone else's ym");
        // A plain file is not an installed link either.
        assert!(
            link.status().installed,
            "a file that works counts as a tool on the path"
        );
    }

    #[test]
    fn an_app_that_runs_from_a_temporary_copy_is_not_linked() {
        let dir = tempfile::tempdir().expect("tempdir");
        let temporary = dir
            .path()
            .join("AppTranslocation")
            .join("0F3A")
            .join("d")
            .join("Yellowmoon.app")
            .join("Contents")
            .join("MacOS");
        std::fs::create_dir_all(&temporary).expect("mkdir");
        let source = temporary.join("ym");
        std::fs::write(&source, b"#!/bin/sh\n").expect("write");
        let home = dir.path().join("home");
        let link = CliLink::new(source, &home);
        let err = link.install(&NoElevation).expect_err("refuses");
        assert!(err.message.contains("Applications folder"), "{err}");
        assert!(!home.exists(), "nothing was made");
    }

    #[test]
    fn a_missing_tool_says_to_reinstall() {
        let dir = tempfile::tempdir().expect("tempdir");
        let link = CliLink::new(dir.path().join("nope"), dir.path());
        let err = link.install(&NoElevation).expect_err("missing");
        assert!(err.message.contains("Reinstall"), "{err}");
    }

    #[cfg(unix)]
    #[test]
    fn a_dangling_link_is_not_installed_and_is_replaced() {
        let (_dir, mut link) = fixture();
        link.system_link = None;
        let user = link.user_link.clone();
        std::fs::create_dir_all(user.parent().expect("parent")).expect("mkdir");
        std::os::unix::fs::symlink("/nonexistent/Yellowmoon.app/Contents/MacOS/ym", &user)
            .expect("link");
        assert!(!link.status().installed);
        assert_eq!(link.install(&NoElevation).expect("replaces"), user);
        assert!(link.status().installed);
    }

    #[test]
    fn the_administrator_script_quotes_awkward_paths() {
        let script = admin_script(
            Path::new("/Applications/My \"Apps\"/Yellowmoon.app/Contents/MacOS/ym"),
            Path::new("/usr/local/bin/ym"),
        );
        assert!(script.starts_with("do shell script \""));
        assert!(script.ends_with("\" with administrator privileges"));
        // The shell sees one quoted word; the double quotes of the path survive AppleScript.
        assert!(
            script.contains(r#"'/Applications/My \"Apps\"/Yellowmoon.app/Contents/MacOS/ym'"#),
            "{script}"
        );
        assert!(script.contains("mkdir -p '/usr/local/bin' && ln -sf"));
        assert_eq!(shell_quote("it's"), r"'it'\''s'");
        assert_eq!(applescript_quote(r#"a\b"c"#), r#""a\\b\"c""#);
    }
}
