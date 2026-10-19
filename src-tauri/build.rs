//! Runs the Tauri build step: validates `tauri.conf.json` (including that the sidecar and the
//! bundled resources exist, see `tools/dev/prepare-sidecar.sh`) and generates the capability
//! schemas. Also links FoundationModels weakly and finds the Swift runtime in macOS when Apple
//! Auto's Swift bridge is built.

fn main() {
    tauri_build::build();
    link_apple_auto_bridge();
}

/// Apple Auto (`docs/19-apple-auto.md` section 11): when `ym-apple` compiled its Swift bridge, which
/// it says through its `links` metadata (`DEP_YM_APPLE_AUTO_BRIDGE=1`, only on Apple silicon with
/// the macOS 27 SDK), link FoundationModels weakly, so that the app still starts on macOS 14 to 26
/// where the framework does not exist, and add `/usr/lib/swift` to the run path, where macOS keeps
/// the Swift runtime (a binary linked for a macOS older than 12 loads the concurrency runtime through
/// `@rpath`; `crates/ym-cli/build.rs` says more). The bridge's Swift object carries no link to the
/// framework, and a library's build script cannot pass linker arguments to the binaries that use it,
/// so each binary does it in its own build script (`crates/ym-cli/build.rs` too;
/// `crates/ym-apple/build.rs` explains more). Nothing happens while the app does not depend on
/// `ym-apple`, on the x86_64 half of the universal build, or on any other target.
fn link_apple_auto_bridge() {
    if std::env::var("DEP_YM_APPLE_AUTO_BRIDGE").is_ok_and(|bridge| bridge == "1") {
        println!("cargo:rustc-link-arg=-Wl,-weak_framework,FoundationModels");
        println!("cargo:rustc-link-arg=-Wl,-rpath,/usr/lib/swift");
    }
}
