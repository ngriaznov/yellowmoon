//! Runs the Tauri build step: validates `tauri.conf.json` (including that the sidecar and the
//! bundled resources exist, see `tools/dev/prepare-sidecar.sh`) and generates the capability
//! schemas.

fn main() {
    tauri_build::build();
}
