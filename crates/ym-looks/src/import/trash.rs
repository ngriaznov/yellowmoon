//! Move-to-Trash for the downloaded zip, after a successful import (`docs/08-looks.md` wizard step 5).
//!
//! The wizard offers to move the zip to the Trash or keep it. Moving to the Trash is recoverable
//! (the `trash` crate calls the system's Trash, so "Put Back" works in Finder), unlike deleting.
//! The app calls [`move_to_trash`]; nothing in the importer does it by itself.

use std::path::{Path, PathBuf};

/// Why a file could not be moved to the Trash.
#[derive(Debug, thiserror::Error)]
pub enum TrashError {
    /// The file is not there (already moved or deleted).
    #[error("{} is not there any more.", .0.display())]
    Missing(PathBuf),
    /// The system could not move it.
    #[error("Could not move {} to the Trash: {reason}", .path.display())]
    Failed {
        /// The file.
        path: PathBuf,
        /// What the system said.
        reason: String,
    },
}

/// Moves a file or folder to the system Trash. The item stays recoverable.
///
/// # Errors
///
/// [`TrashError::Missing`] when `path` does not exist, [`TrashError::Failed`] when the system
/// refuses (for example a read-only volume without a Trash).
pub fn move_to_trash(path: &Path) -> Result<(), TrashError> {
    if std::fs::symlink_metadata(path).is_err() {
        return Err(TrashError::Missing(path.to_path_buf()));
    }
    ::trash::delete(path).map_err(|e| TrashError::Failed {
        path: path.to_path_buf(),
        reason: e.to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_missing_file_is_reported_as_such() {
        let dir = tempfile::tempdir().unwrap();
        let err = move_to_trash(&dir.path().join("nope.zip")).unwrap_err();
        assert!(matches!(err, TrashError::Missing(_)), "{err}");
    }
}
