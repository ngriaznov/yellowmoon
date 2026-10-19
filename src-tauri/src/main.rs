//! Yellowmoon desktop app: the Tauri 2 shell around `ym-core`.
//!
//! Hides the extra console window on Windows release builds (unused on macOS, kept for
//! completeness) and otherwise only wires the window, the plugins, the menu and the commands
//! together (`app`). All image processing lives in the `ym-*` crates; `api` is the one place the
//! commands meet the core, `viewport` draws the photo and `wire` is the shape of everything that
//! crosses the bridge.

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod about;
mod api;
mod app;
mod bindings;
mod cli_link;
mod commands;
mod events;
mod logging;
mod menu;
#[cfg(target_os = "macos")]
mod native;
mod perf;
#[cfg(target_os = "macos")]
mod present;
mod system;
#[cfg(test)]
mod testing;
mod thumbs;
mod viewport;
mod wire;
mod wizard;

fn main() {
    app::run();
}
