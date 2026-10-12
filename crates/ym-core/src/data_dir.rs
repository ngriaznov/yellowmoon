//! Finding the shipped `data/` folder (`docs/01-architecture.md` section Finding `data/`).
//!
//! The app and the CLI look for the folder in this order and use the first candidate that exists
//! as a directory:
//!
//! 1. the `YELLOWMOON_DATA` environment variable (tests, development);
//! 2. next to the canonicalized executable inside an app bundle:
//!    `Contents/MacOS/<exe>` becomes `Contents/Resources/data/`. The executable is canonicalized
//!    first because the CLI on `PATH` is a symlink into the bundle;
//! 3. debug builds only: `<workspace root>/data/`, derived from `CARGO_MANIFEST_DIR` at compile
//!    time.
//!
//! If none exists, [`find_data_dir`] fails with [`DataDirError::Missing`].

use std::ffi::OsStr;
use std::path::{Path, PathBuf};

/// Name of the environment variable that overrides the data folder location.
pub const DATA_DIR_ENV: &str = "YELLOWMOON_DATA";

/// Why the data folder could not be found.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum DataDirError {
    /// None of the lookup locations holds a `data/` folder.
    #[error("Yellowmoon's data folder is missing. Reinstall the app.")]
    Missing,
}

/// Locates the shipped `data/` folder for the running process.
///
/// Reads `YELLOWMOON_DATA` and the current executable path, and in debug builds falls back to the
/// workspace `data/` folder. See the module documentation for the order.
///
/// # Errors
///
/// Returns [`DataDirError::Missing`] when no candidate exists.
pub fn find_data_dir() -> Result<PathBuf, DataDirError> {
    let env_value = std::env::var_os(DATA_DIR_ENV);
    let current_exe = std::env::current_exe().ok();
    let workspace_root = debug_workspace_root();
    resolve_data_dir(
        env_value.as_deref(),
        current_exe.as_deref(),
        workspace_root.as_deref(),
    )
    .ok_or(DataDirError::Missing)
}

/// The workspace root as known at compile time, in debug builds only.
///
/// `ym-core` lives at `<workspace root>/crates/ym-core`, so the root is two levels above the
/// manifest directory. Release builds return `None`, so a shipped binary never reaches into the
/// build machine's source tree.
fn debug_workspace_root() -> Option<PathBuf> {
    if !cfg!(debug_assertions) {
        return None;
    }
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .map(Path::to_path_buf)
}

/// Pure lookup logic behind [`find_data_dir`]: every input is passed in, nothing is read from the
/// process environment.
///
/// Candidates are tried in order and the first one that is an existing directory wins:
///
/// 1. `env_value`, the value of `YELLOWMOON_DATA` (`None` or empty counts as unset);
/// 2. `current_exe`, canonicalized (symlinks resolved); when it has the shape
///    `<bundle>/Contents/MacOS/<exe>`, the candidate is `<bundle>/Contents/Resources/data`. A path
///    that cannot be canonicalized is used as given;
/// 3. `debug_workspace_root`, which callers pass only in debug builds: `<root>/data`.
///
/// An explicit but non-existent `YELLOWMOON_DATA` falls through to the next candidate instead of
/// failing, matching the documented order.
pub fn resolve_data_dir(
    env_value: Option<&OsStr>,
    current_exe: Option<&Path>,
    debug_workspace_root: Option<&Path>,
) -> Option<PathBuf> {
    let from_env = env_value
        .filter(|value| !value.is_empty())
        .map(PathBuf::from);
    let from_bundle = current_exe.and_then(bundle_data_dir);
    let from_workspace = debug_workspace_root.map(|root| root.join("data"));

    [from_env, from_bundle, from_workspace]
        .into_iter()
        .flatten()
        .find(|candidate| candidate.is_dir())
}

/// Maps an executable path inside an app bundle to the bundle's `Resources/data` folder.
///
/// Returns `None` when the (canonicalized) path is not `<bundle>/Contents/MacOS/<exe>`.
fn bundle_data_dir(exe: &Path) -> Option<PathBuf> {
    let exe = exe.canonicalize().unwrap_or_else(|_| exe.to_path_buf());
    let macos_dir = exe.parent()?;
    if macos_dir.file_name()? != "MacOS" {
        return None;
    }
    let contents_dir = macos_dir.parent()?;
    if contents_dir.file_name()? != "Contents" {
        return None;
    }
    Some(contents_dir.join("Resources").join("data"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::os::unix::ffi::OsStrExt;

    /// Creates `<root>/<relative>` as a directory and returns its path.
    fn make_dir(root: &Path, relative: &str) -> PathBuf {
        let path = root.join(relative);
        fs::create_dir_all(&path).expect("create test directory");
        path
    }

    /// Creates a fake bundle with an executable and a `Resources/data` folder.
    /// Returns `(executable, data folder)`.
    fn make_bundle(root: &Path, name: &str) -> (PathBuf, PathBuf) {
        let macos = make_dir(root, &format!("{name}.app/Contents/MacOS"));
        let data = make_dir(root, &format!("{name}.app/Contents/Resources/data"));
        let exe = macos.join("Yellowmoon");
        fs::write(&exe, b"#!/bin/sh\n").expect("create test executable");
        (exe, data)
    }

    #[test]
    fn env_variable_wins_over_everything() {
        let dir = tempfile::tempdir().expect("tempdir");
        let from_env = make_dir(dir.path(), "env-data");
        let (exe, _bundle_data) = make_bundle(dir.path(), "Yellowmoon");
        let workspace = dir.path().join("workspace");
        make_dir(&workspace, "data");

        let found = resolve_data_dir(Some(from_env.as_os_str()), Some(&exe), Some(&workspace));
        assert_eq!(found, Some(from_env));
    }

    #[test]
    fn empty_env_variable_counts_as_unset() {
        let dir = tempfile::tempdir().expect("tempdir");
        let (exe, bundle_data) = make_bundle(dir.path(), "Yellowmoon");

        let found = resolve_data_dir(Some(OsStr::new("")), Some(&exe), None);
        assert_eq!(
            found,
            Some(bundle_data.canonicalize().expect("canonicalize"))
        );
    }

    #[test]
    fn missing_env_directory_falls_through_to_the_bundle() {
        let dir = tempfile::tempdir().expect("tempdir");
        let (exe, bundle_data) = make_bundle(dir.path(), "Yellowmoon");
        let missing = dir.path().join("nope");

        let found = resolve_data_dir(Some(missing.as_os_str()), Some(&exe), None);
        assert_eq!(
            found,
            Some(bundle_data.canonicalize().expect("canonicalize"))
        );
    }

    #[test]
    fn env_pointing_at_a_file_is_not_a_data_dir() {
        let dir = tempfile::tempdir().expect("tempdir");
        let file = dir.path().join("data");
        fs::write(&file, b"not a directory").expect("write file");

        assert_eq!(resolve_data_dir(Some(file.as_os_str()), None, None), None);
    }

    #[test]
    fn bundle_executable_resolves_to_resources_data() {
        let dir = tempfile::tempdir().expect("tempdir");
        let (exe, bundle_data) = make_bundle(dir.path(), "Yellowmoon");

        let found = resolve_data_dir(None, Some(&exe), None);
        assert_eq!(
            found,
            Some(bundle_data.canonicalize().expect("canonicalize"))
        );
    }

    #[test]
    fn symlink_to_the_bundled_executable_is_canonicalized() {
        // The CLI on PATH is a symlink into the bundle.
        let dir = tempfile::tempdir().expect("tempdir");
        let (exe, bundle_data) = make_bundle(dir.path(), "Yellowmoon");
        let bin_dir = make_dir(dir.path(), "usr/local/bin");
        let link = bin_dir.join("yellowmoon");
        std::os::unix::fs::symlink(&exe, &link).expect("create symlink");

        let found = resolve_data_dir(None, Some(&link), None);
        assert_eq!(
            found,
            Some(bundle_data.canonicalize().expect("canonicalize"))
        );
    }

    #[test]
    fn executable_outside_a_bundle_finds_nothing() {
        let dir = tempfile::tempdir().expect("tempdir");
        let bin = make_dir(dir.path(), "bin");
        let exe = bin.join("yellowmoon");
        fs::write(&exe, b"").expect("create test executable");
        // Even a sibling `Resources/data` must not be picked up outside Contents/MacOS.
        make_dir(dir.path(), "Resources/data");

        assert_eq!(resolve_data_dir(None, Some(&exe), None), None);
    }

    #[test]
    fn bundle_without_a_data_folder_falls_through() {
        let dir = tempfile::tempdir().expect("tempdir");
        let macos = make_dir(dir.path(), "Yellowmoon.app/Contents/MacOS");
        let exe = macos.join("Yellowmoon");
        fs::write(&exe, b"").expect("create test executable");
        let workspace = dir.path().join("workspace");
        let workspace_data = make_dir(&workspace, "data");

        let found = resolve_data_dir(None, Some(&exe), Some(&workspace));
        assert_eq!(found, Some(workspace_data));
    }

    #[test]
    fn unresolvable_executable_path_is_used_as_given() {
        // A path that does not exist cannot be canonicalized; it is not an error.
        let dir = tempfile::tempdir().expect("tempdir");
        let data = make_dir(dir.path(), "Yellowmoon.app/Contents/Resources/data");
        let exe = dir.path().join("Yellowmoon.app/Contents/MacOS/Yellowmoon");

        let found = resolve_data_dir(None, Some(&exe), None);
        assert_eq!(found, Some(data));
    }

    #[test]
    fn debug_workspace_root_is_the_last_resort() {
        let dir = tempfile::tempdir().expect("tempdir");
        let workspace_data = make_dir(dir.path(), "data");

        let found = resolve_data_dir(None, None, Some(dir.path()));
        assert_eq!(found, Some(workspace_data));
    }

    #[test]
    fn nothing_found_returns_none() {
        let dir = tempfile::tempdir().expect("tempdir");
        assert_eq!(resolve_data_dir(None, None, Some(dir.path())), None);
        assert_eq!(resolve_data_dir(None, None, None), None);
    }

    #[test]
    fn non_utf8_env_value_is_accepted() {
        let dir = tempfile::tempdir().expect("tempdir");
        let name = OsStr::from_bytes(b"data-\xff");
        let data = dir.path().join(name);
        fs::create_dir_all(&data).expect("create directory");

        let found = resolve_data_dir(Some(data.as_os_str()), None, None);
        assert_eq!(found, Some(data));
    }

    #[test]
    fn error_message_is_the_documented_text() {
        assert_eq!(
            DataDirError::Missing.to_string(),
            "Yellowmoon's data folder is missing. Reinstall the app."
        );
    }

    #[test]
    fn workspace_root_matches_the_repository_in_debug_builds() {
        let root = debug_workspace_root();
        if cfg!(debug_assertions) {
            let root = root.expect("debug builds know the workspace root");
            assert!(root.join("Cargo.toml").is_file(), "{}", root.display());
            assert!(root.join("data").is_dir(), "{}", root.display());
        } else {
            assert_eq!(root, None);
        }
    }

    #[test]
    fn find_data_dir_succeeds_from_the_source_tree() {
        // Under `cargo test` the debug fallback (or YELLOWMOON_DATA) finds the repository's data/.
        if cfg!(debug_assertions) {
            let found = find_data_dir().expect("data folder");
            assert!(found.is_dir(), "{}", found.display());
        }
    }
}
