//! Finding the shipped `data/` folder, the way the other crates do (`docs/01-architecture.md`, *Finding
//! `data/`*), without depending on `ym-core`.
//!
//! The order is: the `YELLOWMOON_DATA` environment variable; next to the executable inside an app bundle
//! (`Contents/MacOS/<exe>` becomes `Contents/Resources/data`); in debug builds the workspace `data/` folder.

use std::path::{Path, PathBuf};

/// The environment variable that overrides the data folder.
pub const DATA_DIR_ENV: &str = "YELLOWMOON_DATA";

/// The data folder could not be found.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum DataDirError {
    /// None of the places holds a `data/` folder.
    #[error("Yellowmoon's data folder is missing. Reinstall the app.")]
    Missing,
}

/// Locates the `data/` folder.
///
/// # Errors
///
/// [`DataDirError::Missing`] when no candidate is a directory.
pub fn find_data_dir() -> Result<PathBuf, DataDirError> {
    if let Some(v) = std::env::var_os(DATA_DIR_ENV) {
        let p = PathBuf::from(v);
        if !p.as_os_str().is_empty() && p.is_dir() {
            return Ok(p);
        }
    }
    if let Ok(exe) = std::env::current_exe() {
        let exe = exe.canonicalize().unwrap_or(exe);
        if let Some(macos) = exe.parent()
            && macos.file_name().is_some_and(|n| n == "MacOS")
            && let Some(contents) = macos.parent()
        {
            let p = contents.join("Resources").join("data");
            if p.is_dir() {
                return Ok(p);
            }
        }
    }
    if cfg!(debug_assertions)
        && let Some(root) = Path::new(env!("CARGO_MANIFEST_DIR")).ancestors().nth(2)
    {
        let p = root.join("data");
        if p.is_dir() {
            return Ok(p);
        }
    }
    Err(DataDirError::Missing)
}
