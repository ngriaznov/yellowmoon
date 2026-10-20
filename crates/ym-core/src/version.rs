//! Version constants written into outputs and shown by `ym --version`
//! (`docs/01-architecture.md` section Versions written into outputs).

/// Semver of the processing chain.
///
/// Bump it whenever the output pixels change for the same input and settings. It is written into
/// every export's XMP (`ym:PipelineVersion`) and printed by `ym --version`.
pub const PIPELINE_VERSION: &str = "0.4.4";

/// Version of the application and the command-line tool: the workspace package version.
pub fn app_version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// True for `major.minor.patch` with numeric parts only.
    fn is_plain_semver(s: &str) -> bool {
        let parts: Vec<&str> = s.split('.').collect();
        parts.len() == 3
            && parts
                .iter()
                .all(|p| !p.is_empty() && p.chars().all(|c| c.is_ascii_digit()))
    }

    #[test]
    fn pipeline_version_is_semver() {
        assert!(is_plain_semver(PIPELINE_VERSION), "{PIPELINE_VERSION}");
    }

    #[test]
    fn app_version_is_the_workspace_version() {
        assert!(is_plain_semver(app_version()), "{}", app_version());
        assert_eq!(app_version(), env!("CARGO_PKG_VERSION"));
    }
}
