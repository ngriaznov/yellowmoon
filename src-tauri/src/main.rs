//! Yellowmoon desktop app: the Tauri 2 shell around `ym-core`.
//!
//! Hides the extra console window on Windows release builds (unused on macOS, kept for
//! completeness) and otherwise only wires plugins and commands together. All image processing
//! lives in the `ym-*` crates.

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod commands;

fn main() {
    let builder = commands::builder();

    tauri::Builder::default()
        .plugin(tauri_plugin_updater::Builder::new().build())
        .plugin(tauri_plugin_process::init())
        .plugin(tauri_plugin_dialog::init())
        .invoke_handler(builder.invoke_handler())
        .setup(move |app| {
            builder.mount_events(app);
            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("error while running Yellowmoon");
}
