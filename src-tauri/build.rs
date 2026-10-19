//! Runs the Tauri build step: validates `tauri.conf.json` (including that the sidecar and the
//! bundled resources exist, see `tools/dev/prepare-sidecar.sh`) and generates the capability
//! schemas. Also links FoundationModels weakly when Apple Auto's Swift bridge is built.

fn main() {
    tauri_build::build();
    weak_link_foundation_models();
}

/// Apple Auto (`docs/19-apple-auto.md` section 11): when `ym-apple` compiled its Swift bridge, which
/// it says through its `links` metadata (`DEP_YM_APPLE_AUTO_BRIDGE=1`, only on Apple silicon with
/// the macOS 27 SDK), link FoundationModels weakly, so that the app still starts on macOS 14 to 26
/// where the framework does not exist. The bridge's Swift object carries no link to the framework,
/// and a library's build script cannot pass linker arguments to the binaries that use it, so each
/// binary does it in its own build script (`crates/ym-cli/build.rs` too; `crates/ym-apple/build.rs`
/// explains more). Nothing happens while the app does not depend on `ym-apple`, on the x86_64 half
/// of the universal build, or on any other target.
fn weak_link_foundation_models() {
    if std::env::var("DEP_YM_APPLE_AUTO_BRIDGE").is_ok_and(|bridge| bridge == "1") {
        println!("cargo:rustc-link-arg=-Wl,-weak_framework,FoundationModels");
    }
}
