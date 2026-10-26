//! Finding a program (`docs/22` section 4): the known install folders, then `PATH`, then the login shell
//! (macOS and Linux). The result is an absolute path to a real executable, never a script, whose
//! `--version` is at least the minimum.

use std::ffi::OsString;
use std::path::{Path, PathBuf};

use semver::Version;

use crate::process::{EnvProfile, Outcome, RunSpec, Workdir, run};
use crate::tool::{Tool, Tuning};

/// Whose rules apply. The tests run Windows' rules on any system; production uses the system's own.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Os {
    /// macOS and Linux.
    Unix,
    /// Windows.
    Windows,
}

impl Os {
    /// The rules of the system this was built for.
    #[must_use]
    pub const fn current() -> Self {
        if cfg!(windows) { Os::Windows } else { Os::Unix }
    }
}

/// Where to look, read from the environment once (`docs/22` section 4). A field is `None` when its
/// variable is not set; the tests give their own.
#[derive(Debug, Clone)]
pub struct Locations {
    /// Which rules apply.
    pub os: Os,
    /// The home folder: `HOME` (macOS, Linux) or `USERPROFILE` (Windows).
    pub home: Option<PathBuf>,
    /// `LOCALAPPDATA` (Windows).
    pub local_app_data: Option<PathBuf>,
    /// `APPDATA` (Windows).
    pub app_data: Option<PathBuf>,
    /// `PATH`.
    pub path: Option<OsString>,
    /// `SHELL` (macOS, Linux).
    pub shell: Option<OsString>,
}

impl Locations {
    /// The environment of this process.
    #[must_use]
    pub fn from_env() -> Self {
        let os = Os::current();
        let var = |name: &str| std::env::var_os(name).filter(|v| !v.is_empty());
        Self {
            os,
            home: var(if os == Os::Windows { "USERPROFILE" } else { "HOME" }).map(PathBuf::from),
            local_app_data: var("LOCALAPPDATA").map(PathBuf::from),
            app_data: var("APPDATA").map(PathBuf::from),
            path: var("PATH"),
            shell: var("SHELL"),
        }
    }
}

/// What the search found.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Discovery {
    /// A real executable whose version is high enough.
    Found {
        /// Its absolute path.
        path: PathBuf,
        /// Its version.
        version: Version,
    },
    /// Nothing was found.
    NotInstalled,
    /// Windows: only a `.cmd`, `.bat` or `.ps1` shim was found and no real program next to it.
    UnsafeLauncher,
    /// A program was found but its version is below the minimum; the highest one is reported.
    TooOld {
        /// Its path.
        path: PathBuf,
        /// Its version.
        version: Version,
    },
    /// Programs were found but none printed a version in time.
    ProbeFailed {
        /// The first of them.
        path: PathBuf,
    },
}

/// The first `\d+\.\d+\.\d+` in `text`, as a version.
#[must_use]
pub fn parse_version(text: &str) -> Option<Version> {
    let bytes = text.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i].is_ascii_digit() && (i == 0 || !bytes[i - 1].is_ascii_digit()) {
            if let Some(version) = version_at(text, i) {
                return Some(version);
            }
            // Skip this run of digits.
            while i < bytes.len() && bytes[i].is_ascii_digit() {
                i += 1;
            }
        } else {
            i += 1;
        }
    }
    None
}

/// The `a.b.c` that starts at byte `start`, when there is one.
fn version_at(text: &str, start: usize) -> Option<Version> {
    let bytes = text.as_bytes();
    let mut parts = [0_u64; 3];
    let mut at = start;
    for (n, part) in parts.iter_mut().enumerate() {
        let from = at;
        while at < bytes.len() && bytes[at].is_ascii_digit() {
            at += 1;
        }
        if at == from {
            return None;
        }
        *part = text[from..at].parse().ok()?;
        if n < 2 {
            if bytes.get(at) != Some(&b'.') {
                return None;
            }
            at += 1;
        }
    }
    Some(Version::new(parts[0], parts[1], parts[2]))
}

/// The folders where `tool`'s installers put it (`docs/22` section 4, step 1), in the order they are
/// tried. Folders that do not exist are included; the caller looks.
#[must_use]
pub fn known_candidates(tool: Tool, loc: &Locations) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut push = |base: &Option<PathBuf>, parts: &[&str]| {
        if let Some(base) = base {
            let mut path = base.clone();
            path.extend(parts);
            out.push(path);
        }
    };
    match (loc.os, tool) {
        (Os::Unix, Tool::Claude) => {
            push(&loc.home, &[".local", "bin", "claude"]);
            push(&loc.home, &[".claude", "local", "claude"]);
            let root = Some(PathBuf::from("/"));
            push(&root, &["opt", "homebrew", "bin", "claude"]);
            push(&root, &["usr", "local", "bin", "claude"]);
            push(&root, &["usr", "bin", "claude"]);
            push(&loc.home, &[".npm-global", "bin", "claude"]);
            push(&loc.home, &[".bun", "bin", "claude"]);
        }
        (Os::Unix, Tool::Agy) => push(&loc.home, &[".local", "bin", "agy"]),
        (Os::Windows, Tool::Claude) => {
            push(&loc.home, &[".local", "bin", "claude.exe"]);
            push(&loc.local_app_data, &["Microsoft", "WinGet", "Links", "claude.exe"]);
            push(&loc.app_data, &["npm", "claude.cmd"]);
        }
        (Os::Windows, Tool::Agy) => push(&loc.local_app_data, &["agy", "bin", "agy.exe"]),
    }
    out
}

/// What a Windows candidate is.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Launcher {
    /// A real program that can be started.
    Program(PathBuf),
    /// A script shim with no real program next to it.
    Script,
}

/// Windows: a candidate that is not an `.exe` is a script and is never started. The npm shim
/// `claude.cmd` is resolved to the program next to it, `node_modules\@anthropic-ai\claude-code\bin\
/// claude.exe`, when that exists.
#[must_use]
pub fn resolve_launcher(tool: Tool, candidate: &Path) -> Launcher {
    let is_exe = candidate
        .extension()
        .is_some_and(|e| e.eq_ignore_ascii_case("exe"));
    if is_exe {
        return Launcher::Program(candidate.to_path_buf());
    }
    if tool == Tool::Claude {
        if let Some(dir) = candidate.parent() {
            let real = dir
                .join("node_modules")
                .join("@anthropic-ai")
                .join("claude-code")
                .join("bin")
                .join("claude.exe");
            if real.is_file() {
                return Launcher::Program(real);
            }
        }
    }
    Launcher::Script
}

/// What was learned about the candidates so far.
#[derive(Default)]
struct Tally {
    seen: Vec<PathBuf>,
    highest: Option<(Version, PathBuf)>,
    first_silent: Option<PathBuf>,
    script_seen: bool,
}

impl Tally {
    /// Remembers `path` and says whether it is new (compared by canonical path).
    fn fresh(&mut self, path: &Path) -> bool {
        let canonical = std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
        if self.seen.contains(&canonical) {
            return false;
        }
        self.seen.push(canonical);
        true
    }
}

/// Looks for `tool` (`docs/22` section 4): each candidate in order runs `--version`, and the first
/// whose version is at least [`Tool::min_version`] is used. The login shell is asked only when the
/// known folders and `PATH` gave no usable program.
#[must_use]
pub fn locate(tool: Tool, loc: &Locations, tuning: &Tuning) -> Discovery {
    let Ok(workdir) = Workdir::new() else {
        return Discovery::ProbeFailed {
            path: PathBuf::new(),
        };
    };
    let mut tally = Tally::default();

    let mut candidates = known_candidates(tool, loc)
        .into_iter()
        .filter(|p| p.is_file())
        .collect::<Vec<_>>();
    if let Some(found) = loc.path.as_ref().and_then(|path| {
        which::which_in_global(tool.program(), Some(path))
            .ok()
            .and_then(|mut matches| matches.next())
    }) {
        candidates.push(found);
    }
    for candidate in candidates {
        if let Some(found) = try_candidate(tool, loc.os, tuning, &workdir, &mut tally, candidate) {
            return found;
        }
    }

    if loc.os == Os::Unix {
        for candidate in shell_candidates(tool, loc, tuning, &workdir) {
            if let Some(found) = try_candidate(tool, loc.os, tuning, &workdir, &mut tally, candidate) {
                return found;
            }
        }
    }

    finish(tally)
}

/// The result for one program that is already known (`ClaudeAdvisor::with_program`, `exec:`): its
/// `--version` is run and judged as in [`locate`], with no search.
#[must_use]
pub fn check_program(tool: Tool, program: &Path, tuning: &Tuning) -> Discovery {
    if !program.is_file() {
        return Discovery::NotInstalled;
    }
    let Ok(workdir) = Workdir::new() else {
        return Discovery::ProbeFailed {
            path: program.to_path_buf(),
        };
    };
    let mut tally = Tally::default();
    if let Some(found) = try_candidate(
        tool,
        Os::Unix,
        tuning,
        &workdir,
        &mut tally,
        program.to_path_buf(),
    ) {
        return found;
    }
    finish(tally)
}

/// What a search that found nothing usable says.
fn finish(tally: Tally) -> Discovery {
    if let Some((version, path)) = tally.highest {
        Discovery::TooOld { path, version }
    } else if let Some(path) = tally.first_silent {
        Discovery::ProbeFailed { path }
    } else if tally.script_seen {
        Discovery::UnsafeLauncher
    } else {
        Discovery::NotInstalled
    }
}

/// Runs one candidate's `--version`; `Some` when it is usable.
fn try_candidate(
    tool: Tool,
    os: Os,
    tuning: &Tuning,
    workdir: &Workdir,
    tally: &mut Tally,
    candidate: PathBuf,
) -> Option<Discovery> {
    let program = if os == Os::Windows {
        match resolve_launcher(tool, &candidate) {
            Launcher::Program(path) => path,
            Launcher::Script => {
                tally.script_seen = true;
                return None;
            }
        }
    } else {
        candidate
    };
    if !tally.fresh(&program) {
        return None;
    }
    let output = run(
        RunSpec::new(&program, workdir.cwd(), profile_for(tool))
            .args(["--version"])
            .timeout(tuning.probe.version)
            .extra_env(tuning.extra_env.clone()),
    );
    let version = (output.outcome == Outcome::Exited)
        .then(|| {
            parse_version(&output.stdout_text())
                .or_else(|| parse_version(&String::from_utf8_lossy(&output.stderr)))
        })
        .flatten();
    let Some(version) = version else {
        tally.first_silent.get_or_insert(program);
        return None;
    };
    if version >= tool.min_version() {
        return Some(Discovery::Found {
            path: program,
            version,
        });
    }
    if tally.highest.as_ref().is_none_or(|(best, _)| version > *best) {
        tally.highest = Some((version, program));
    }
    None
}

/// The environment profile of `tool`'s own runs.
#[must_use]
pub fn profile_for(tool: Tool) -> EnvProfile {
    match tool {
        Tool::Claude => EnvProfile::Claude,
        Tool::Agy => EnvProfile::Agy,
    }
}

/// The shells whose `command -v` is trusted, by file name.
const KNOWN_SHELLS: [&str; 6] = ["sh", "bash", "zsh", "dash", "ksh", "fish"];

/// Asks the person's login shell where `tool` is (`docs/22` section 4, step 3): `"$SHELL" -l -c 'command -v
/// <tool>'`, then, for zsh and bash, `-i -l -c`, each limited to the shell limit (3 s) with the whole
/// process group killed on expiry. The answer is the last output line that is an absolute path to an
/// existing executable file; nothing is evaluated.
fn shell_candidates(tool: Tool, loc: &Locations, tuning: &Tuning, workdir: &Workdir) -> Vec<PathBuf> {
    let Some(shell) = loc.shell.as_ref().map(PathBuf::from) else {
        return Vec::new();
    };
    let name = shell
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    if !KNOWN_SHELLS.contains(&name.as_str()) || !shell.is_absolute() {
        return Vec::new();
    }
    let command = format!("command -v {}", tool.program());
    let interactive_too = matches!(name.as_str(), "zsh" | "bash");
    let attempts: &[&[&str]] = if interactive_too {
        &[&["-l", "-c"], &["-i", "-l", "-c"]]
    } else {
        &[&["-l", "-c"]]
    };
    for flags in attempts {
        let mut spec = RunSpec::new(&shell, workdir.cwd(), EnvProfile::Inherit)
            .args(flags.iter().copied().chain([command.as_str()]))
            .timeout(tuning.probe.shell);
        spec.stdout_cap = 64 * 1024;
        let output = run(spec);
        if output.outcome != Outcome::Exited {
            continue;
        }
        if let Some(path) = last_executable_path(&output.stdout_text()) {
            return vec![path];
        }
    }
    Vec::new()
}

/// The last line of `text` that is an absolute path to an existing file (banners and warnings come
/// first). On Unix the file must be executable.
fn last_executable_path(text: &str) -> Option<PathBuf> {
    text.lines().rev().find_map(|line| {
        let path = PathBuf::from(line.trim());
        (path.is_absolute() && is_executable_file(&path)).then_some(path)
    })
}

/// Whether `path` is a file that can be run.
fn is_executable_file(path: &Path) -> bool {
    let Ok(meta) = std::fs::metadata(path) else {
        return false;
    };
    if !meta.is_file() {
        return false;
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        meta.permissions().mode() & 0o111 != 0
    }
    #[cfg(not(unix))]
    {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn versions_are_read_from_the_first_triple() {
        assert_eq!(
            parse_version("2.1.310 (Claude Code)"),
            Some(Version::new(2, 1, 310))
        );
        assert_eq!(parse_version("1.4.7\n"), Some(Version::new(1, 4, 7)));
        assert_eq!(
            parse_version("agy version v1.2.14-beta (2026-10-26)"),
            Some(Version::new(1, 2, 14))
        );
        assert_eq!(parse_version("2026-10-26 and 3.4"), None);
        assert_eq!(parse_version("no digits"), None);
        assert_eq!(parse_version("1.2"), None);
        assert_eq!(parse_version("build 12 then 7.8.9.10"), Some(Version::new(7, 8, 9)));
    }

    fn locations(os: Os) -> Locations {
        Locations {
            os,
            home: Some(PathBuf::from("/h")),
            local_app_data: Some(PathBuf::from("/la")),
            app_data: Some(PathBuf::from("/a")),
            path: None,
            shell: None,
        }
    }

    #[test]
    fn the_known_folders_follow_the_spec() {
        let unix = known_candidates(Tool::Claude, &locations(Os::Unix));
        assert_eq!(unix[0], PathBuf::from("/h/.local/bin/claude"));
        assert_eq!(unix.len(), 7);
        assert!(unix.contains(&PathBuf::from("/opt/homebrew/bin/claude")));
        assert_eq!(
            known_candidates(Tool::Agy, &locations(Os::Unix)),
            vec![PathBuf::from("/h/.local/bin/agy")]
        );
        let win = known_candidates(Tool::Claude, &locations(Os::Windows));
        assert_eq!(win.len(), 3);
        assert!(win[2].ends_with("claude.cmd"));
        assert_eq!(
            known_candidates(Tool::Agy, &locations(Os::Windows)),
            vec![PathBuf::from("/la").join("agy").join("bin").join("agy.exe")]
        );
    }

    #[test]
    fn only_an_exe_or_the_npm_layout_is_a_launcher() {
        let dir = tempfile::tempdir().unwrap();
        let exe = dir.path().join("claude.exe");
        assert_eq!(
            resolve_launcher(Tool::Claude, &exe),
            Launcher::Program(exe.clone())
        );
        let shim = dir.path().join("claude.cmd");
        assert_eq!(resolve_launcher(Tool::Claude, &shim), Launcher::Script);
        assert_eq!(
            resolve_launcher(Tool::Agy, &dir.path().join("agy.ps1")),
            Launcher::Script
        );
        let real = dir
            .path()
            .join("node_modules")
            .join("@anthropic-ai")
            .join("claude-code")
            .join("bin");
        std::fs::create_dir_all(&real).unwrap();
        std::fs::write(real.join("claude.exe"), b"x").unwrap();
        assert_eq!(
            resolve_launcher(Tool::Claude, &shim),
            Launcher::Program(real.join("claude.exe"))
        );
    }

    #[test]
    fn the_last_absolute_executable_line_wins() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("claude");
        std::fs::write(&file, b"#!/bin/sh\n").unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&file, std::fs::Permissions::from_mode(0o755)).unwrap();
        }
        let text = format!("Welcome to zsh\nalias claude='x'\n{}\n", file.display());
        assert_eq!(last_executable_path(&text), Some(file));
        assert_eq!(last_executable_path("claude\n/nonexistent/claude\n"), None);
    }
}
