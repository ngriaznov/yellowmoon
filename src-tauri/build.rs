//! Runs the Tauri build step: validates `tauri.conf.json` (including that the sidecar and the
//! bundled resources exist, see `tools/dev/prepare-sidecar.sh`) and generates the capability
//! schemas. Also links FoundationModels weakly and finds the Swift runtime in macOS when Apple
//! Auto's Swift bridge or the Vision bridge (the Subject and People masks) is built.

fn main() {
    tauri_build::build();
    link_apple_bridges();
}

/// The Swift bridges of `ym-apple`, which say whether they were compiled through its `links` metadata.
///
/// * Apple Auto (`docs/19-apple-auto.md` section 11; `DEP_YM_APPLE_AUTO_BRIDGE=1`, only on Apple silicon with the
///   macOS 27 SDK): link FoundationModels weakly, so that the app still starts on macOS 14 to 26 where the framework
///   does not exist, and add `/usr/lib/swift` to the run path.
/// * Vision (`docs/23-nodes.md` section 4.5; `DEP_YM_APPLE_AUTO_VISION=1`, on either half of the universal build with
///   the macOS 14 SDK or later): add `/usr/lib/swift` to the run path. Vision itself is a normal link, made by
///   `ym-apple`'s own build script, because the framework is on every macOS the app runs on.
///
/// `/usr/lib/swift` is where macOS keeps the Swift runtime (a binary linked for a macOS older than 12 loads the
/// concurrency runtime through `@rpath`; `crates/ym-cli/build.rs` says more). A bridge's Swift object carries no link to
/// FoundationModels, and a library's build script cannot pass linker arguments to the binaries that use it, so each
/// binary does it in its own build script (`crates/ym-cli/build.rs` too; `crates/ym-apple/build.rs` explains more).
/// Nothing happens while the app does not depend on `ym-apple`, or on any other target.
fn link_apple_bridges() {
    let built = |name: &str| std::env::var(name).is_ok_and(|v| v == "1");
    let auto = built("DEP_YM_APPLE_AUTO_BRIDGE");
    let vision = built("DEP_YM_APPLE_AUTO_VISION");
    if auto {
        println!("cargo:rustc-link-arg=-Wl,-weak_framework,FoundationModels");
    }
    if auto || vision {
        println!("cargo:rustc-link-arg=-Wl,-rpath,/usr/lib/swift");
    }
}
