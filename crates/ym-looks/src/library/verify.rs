//! `looks verify`: re-hash every imported file and report what is missing or changed (`docs/08-looks.md` section Missing looks).

use super::ids::LookId;

/// What `verify` found for one file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VerifyOutcome {
    /// The file is there and has the imported checksum.
    Ok,
    /// The file is gone.
    Missing,
    /// The file is there but its checksum differs from the one recorded at import.
    Changed {
        /// The checksum recorded in the manifest.
        expected: String,
        /// The checksum of the file now.
        actual: String,
    },
    /// The file exists but could not be read.
    Unreadable {
        /// What the system said.
        reason: String,
    },
}

/// The result for one imported look.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifyEntry {
    /// Its id.
    pub id: LookId,
    /// Its file, relative to the `looks` folder.
    pub file: String,
    /// What was found.
    pub outcome: VerifyOutcome,
}

/// The result of re-hashing the whole library.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct VerifyReport {
    /// One entry per imported look, in manifest order.
    pub entries: Vec<VerifyEntry>,
}

impl VerifyReport {
    /// Files checked.
    #[must_use]
    pub fn checked(&self) -> usize {
        self.entries.len()
    }

    /// Files that are fine.
    #[must_use]
    pub fn ok_count(&self) -> usize {
        self.entries
            .iter()
            .filter(|e| e.outcome == VerifyOutcome::Ok)
            .count()
    }

    /// Entries that are not fine.
    pub fn problems(&self) -> impl Iterator<Item = &VerifyEntry> {
        self.entries
            .iter()
            .filter(|e| e.outcome != VerifyOutcome::Ok)
    }

    /// Ids of files that are gone.
    #[must_use]
    pub fn missing(&self) -> Vec<&LookId> {
        self.entries
            .iter()
            .filter(|e| e.outcome == VerifyOutcome::Missing)
            .map(|e| &e.id)
            .collect()
    }

    /// Ids of files whose checksum changed.
    #[must_use]
    pub fn changed(&self) -> Vec<&LookId> {
        self.entries
            .iter()
            .filter(|e| matches!(e.outcome, VerifyOutcome::Changed { .. }))
            .map(|e| &e.id)
            .collect()
    }

    /// Whether every file is fine.
    #[must_use]
    pub fn is_clean(&self) -> bool {
        self.problems().next().is_none()
    }
}
