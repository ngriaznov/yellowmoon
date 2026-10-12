//! Tauri commands exposed to the webview. Each one is a thin wrapper over `ym-core`
//! (`docs/01-architecture.md` section IPC between webview and core).

use tauri_specta::{Builder, collect_commands};

/// Returns the application version (the workspace package version).
#[tauri::command]
#[specta::specta]
pub fn app_version() -> String {
    ym_core::version::app_version().to_owned()
}

/// Builds the typed command registry shared by the app and the TypeScript bindings exporter.
///
/// Add every new command to [`collect_commands!`] so its TypeScript signature is generated.
pub fn builder() -> Builder<tauri::Wry> {
    Builder::<tauri::Wry>::new().commands(collect_commands![app_version])
}

#[cfg(test)]
mod tests {
    use super::*;
    use specta_typescript::Typescript;

    #[test]
    fn app_version_matches_the_core_version() {
        assert_eq!(app_version(), ym_core::version::app_version());
    }

    #[test]
    fn typescript_bindings_cover_the_registered_commands() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("bindings.ts");
        builder()
            .export(Typescript::default(), &path)
            .expect("export TypeScript bindings");

        let bindings = std::fs::read_to_string(&path).expect("read bindings");
        assert!(bindings.contains("appVersion"), "{bindings}");
    }
}
