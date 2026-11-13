//! Builds the two Swift bridges of this crate. They are separate: each has its own gate, its own static
//! library and its own `cfg`, so one can be built where the other cannot.
//!
//! **Apple Auto's bridge**, `swift/AppleAuto.swift` (`docs/19-apple-auto.md` section 11), is compiled
//! only for Apple silicon Macs (`aarch64-apple-darwin`) with the macOS 27 SDK or later (Xcode 27), and
//! not when `YM_APPLE_BRIDGE=off`. Then this script
//!
//! * compiles the Swift file into a static library in `OUT_DIR` (deployment target macOS 14, or the
//!   `MACOSX_DEPLOYMENT_TARGET` when that is later), without an autolink entry for FoundationModels;
//! * links that library, the Swift runtime (the paths `swiftc -print-target-info` reports), the
//!   compiler runtime (`clang_rt.osx`, which holds `__isPlatformVersionAtLeast`, the function behind
//!   Swift's `#available`), Foundation and CoreGraphics;
//! * links FoundationModels **weakly** (`-weak_framework`) and adds `/usr/lib/swift` to the run path
//!   for this crate's own tests and examples, sets `cfg(ym_apple_bridge)` and tells the binaries
//!   that depend on this crate, through the `links` metadata `DEP_YM_APPLE_AUTO_BRIDGE=1`, that they
//!   must do both too.
//!
//! **The Vision bridge**, `swift/VisionMasks.swift` (`docs/23-nodes.md` sections 3.10 and 4.5: the Subject
//! and People masks), is compiled for **both** architectures of the universal binary (`aarch64` and
//! `x86_64`; cargo builds each separately) wherever the SDK has the macOS 14 Vision requests (the macOS 14
//! SDK, Xcode 15, or later: every image `macos-latest` has, and the macOS 27 SDK Apple Auto uses), and not
//! when `YM_APPLE_BRIDGE=off` or `YM_APPLE_VISION=off`. It is a normal link: Vision exists on every macOS
//! the app supports, so there is no weak link and nothing like FoundationModels' macOS 27 gate; the Swift
//! guards its calls with `#available(macOS 14.0, *)`. The script compiles it into `libym_apple_vision.a`
//! (deployment target macOS 14, or the later `MACOSX_DEPLOYMENT_TARGET`), links it with the same Swift
//! runtime and compiler runtime as above, plus Vision, CoreGraphics, CoreVideo and Foundation, sets
//! `cfg(ym_apple_vision)` and tells the binaries `DEP_YM_APPLE_AUTO_VISION=1`, so that they add
//! `/usr/lib/swift` to their run path (the reason below). A SDK too old for the requests builds the stub,
//! with a warning on a Mac; the core then reports Subject and People as not supported.
//!
//! Why the run path: macOS keeps the Swift runtime in `/usr/lib/swift`. A binary linked for a macOS
//! older than 12 (rustc's default deployment target is 11.0 when `MACOSX_DEPLOYMENT_TARGET` is not
//! set, as in a plain `cargo build`) gets the concurrency runtime from the SDK's back-deployment
//! name, `@rpath/libswift_Concurrency.dylib`, and stops at launch ("no LC_RPATH's found") unless a
//! run path leads to it. Linked for macOS 12 or later it loads `/usr/lib/swift` directly, and the
//! run path is unused.
//!
//! Why the binaries do the weak link themselves: FoundationModels exists only from macOS 26, and the
//! app and `ym` start on macOS 14. A weak link (`LC_LOAD_WEAK_DYLIB`) lets them start where the
//! framework is missing; the Swift code reaches it only behind `#available(macOS 27.0, *)`. There is
//! no weak kind of `rustc-link-lib`, and `rustc-link-arg` from a library's build script reaches only
//! that library's own tests and examples, never the binaries of the crates that depend on it. So the
//! Swift object carries no autolink entry for the framework (`-disable-autolink-framework`, which
//! would ask the linker for a strong link), and each final binary passes
//! `-Wl,-weak_framework,FoundationModels` from its own build script when `DEP_YM_APPLE_AUTO_BRIDGE`
//! is `1` (`crates/ym-cli/build.rs`, `src-tauri/build.rs`). A binary that forgets fails to link
//! (undefined FoundationModels symbols) instead of silently linking the framework strongly.
//!
//! **The Metal bridge**, `swift/MetalDevices.swift` (`docs/26-hardening.md` H-17: the GPU budget from Metal's
//! working-set limit), is built exactly like the Vision one: both architectures, a normal link (Metal is on every Mac the
//! app supports; the Swift needs no `#available`), the macOS 14 SDK or later (every Metal property it reads is older),
//! not when `YM_APPLE_BRIDGE=off` or `YM_APPLE_METAL=off`. It compiles into `libym_apple_metal.a`, links Metal and
//! Foundation, sets `cfg(ym_apple_metal)` and tells the binaries `DEP_YM_APPLE_AUTO_METAL=1` (the run path). Without it
//! the crate reports no Metal devices and the budget rule stands.
//!
//! Everywhere else (Linux, Intel Macs for Apple Auto, an older SDK, `YM_APPLE_BRIDGE=off`) nothing is
//! compiled and the crate reports `not_a_mac`, `intel_mac` or `built_without_support` for Apple Auto and
//! `not_a_mac` or `not_supported` for Subject and People. A Mac build without a bridge prints a
//! `cargo:warning`; other platforms build silently.

use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicBool, Ordering};

/// The oldest macOS the Swift code is compiled for: the app's `minimumSystemVersion`.
const MIN_MACOS: (u32, u32) = (14, 0);

/// The first SDK with every Foundation Models API the bridge uses (`Attachment`, `capabilities`,
/// `variant`, `LanguageModelError`).
const MIN_SDK_MAJOR: u32 = 27;

/// The first SDK with the Vision requests the Subject and People masks use
/// (`VNGenerateForegroundInstanceMaskRequest`, `VNGeneratePersonInstanceMaskRequest`, `VNInstanceMaskObservation`:
/// macOS 14, Xcode 15).
const VISION_MIN_SDK_MAJOR: u32 = 14;

/// The SDK the Metal bridge is built with: the same floor as the Vision bridge, so that one image builds both.
const METAL_MIN_SDK_MAJOR: u32 = 14;

fn main() {
    println!("cargo::rustc-check-cfg=cfg(ym_apple_bridge)");
    println!("cargo::rustc-check-cfg=cfg(ym_apple_vision)");
    println!("cargo::rustc-check-cfg=cfg(ym_apple_metal)");
    println!("cargo::rerun-if-changed=build.rs");
    println!("cargo::rerun-if-changed=swift");
    for variable in [
        "YM_APPLE_BRIDGE",
        "YM_APPLE_VISION",
        "YM_APPLE_METAL",
        "DEVELOPER_DIR",
        "SDKROOT",
        "MACOSX_DEPLOYMENT_TARGET",
    ] {
        println!("cargo::rerun-if-env-changed={variable}");
    }

    let os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    let mac = os == "macos";
    let auto = mac && arch == "aarch64" && build_auto_if_possible();
    let vision = mac && swift_arch(&arch).is_some() && build_vision_if_possible(&arch);
    let metal = mac && swift_arch(&arch).is_some() && build_metal_if_possible(&arch);
    // Read by the build scripts of the binaries as DEP_YM_APPLE_AUTO_BRIDGE, _VISION and _METAL.
    println!("cargo::metadata=bridge={}", u8::from(auto));
    println!("cargo::metadata=vision={}", u8::from(vision));
    println!("cargo::metadata=metal={}", u8::from(metal));
}

/// The architecture name Swift's `-target` uses for a Rust one, for the two Mac architectures.
fn swift_arch(arch: &str) -> Option<&'static str> {
    match arch {
        "aarch64" => Some("arm64"),
        "x86_64" => Some("x86_64"),
        _ => None,
    }
}

/// Builds Apple Auto's bridge when this Mac can, and says why not when it cannot. A failure to compile
/// with a recent enough SDK is an error, not a warning: it is a bug, and a silent stub would hide it.
fn build_auto_if_possible() -> bool {
    if bridge_off() {
        warn_without_auto("YM_APPLE_BRIDGE=off");
        return false;
    }
    let sdk_version = match xcrun(&["--sdk", "macosx", "--show-sdk-version"]) {
        Ok(version) => version,
        Err(why) => {
            warn_without_auto(&why);
            return false;
        }
    };
    if !sdk_major(&sdk_version).is_some_and(|major| major >= MIN_SDK_MAJOR) {
        warn_without_auto(&format!(
            "the macOS SDK is {sdk_version}; Apple Auto needs the macOS {MIN_SDK_MAJOR} SDK (Xcode {MIN_SDK_MAJOR}) or later"
        ));
        return false;
    }
    if let Err(why) = build_auto_bridge() {
        panic!(
            "Apple Auto's Swift bridge did not build with the macOS {sdk_version} SDK.\n{why}\n\
             Set YM_APPLE_BRIDGE=off to build Yellowmoon without Apple Auto."
        );
    }
    true
}

/// Builds the Vision bridge for `arch` when this Mac's SDK has the requests, and says why not when it
/// does not. As for Apple Auto, a failure to compile with a recent enough SDK is an error.
fn build_vision_if_possible(arch: &str) -> bool {
    if bridge_off() {
        warn_without_vision("YM_APPLE_BRIDGE=off");
        return false;
    }
    if env::var("YM_APPLE_VISION").is_ok_and(|value| value == "off") {
        warn_without_vision("YM_APPLE_VISION=off");
        return false;
    }
    let sdk_version = match xcrun(&["--sdk", "macosx", "--show-sdk-version"]) {
        Ok(version) => version,
        Err(why) => {
            warn_without_vision(&why);
            return false;
        }
    };
    if !sdk_major(&sdk_version).is_some_and(|major| major >= VISION_MIN_SDK_MAJOR) {
        warn_without_vision(&format!(
            "the macOS SDK is {sdk_version}; Subject and People need the macOS {VISION_MIN_SDK_MAJOR} SDK (Xcode 15) or later"
        ));
        return false;
    }
    if let Err(why) = build_vision_bridge(arch) {
        panic!(
            "The Vision bridge (Subject and People masks) did not build with the macOS {sdk_version} SDK.\n{why}\n\
             Set YM_APPLE_VISION=off to build Yellowmoon without Subject and People."
        );
    }
    true
}

/// Builds the Metal bridge for `arch` when this Mac's SDK is the macOS 14 one or later, and says why not when it is
/// not. As for the others, a failure to compile with a recent enough SDK is an error.
fn build_metal_if_possible(arch: &str) -> bool {
    if bridge_off() {
        warn_without_metal("YM_APPLE_BRIDGE=off");
        return false;
    }
    if env::var("YM_APPLE_METAL").is_ok_and(|value| value == "off") {
        warn_without_metal("YM_APPLE_METAL=off");
        return false;
    }
    let sdk_version = match xcrun(&["--sdk", "macosx", "--show-sdk-version"]) {
        Ok(version) => version,
        Err(why) => {
            warn_without_metal(&why);
            return false;
        }
    };
    if !sdk_major(&sdk_version).is_some_and(|major| major >= METAL_MIN_SDK_MAJOR) {
        warn_without_metal(&format!(
            "the macOS SDK is {sdk_version}; the Metal limit needs the macOS {METAL_MIN_SDK_MAJOR} SDK (Xcode 15) or later"
        ));
        return false;
    }
    if let Err(why) = build_metal_bridge(arch) {
        panic!(
            "The Metal bridge (the GPU memory limit) did not build with the macOS {sdk_version} SDK.\n{why}\n\
             Set YM_APPLE_METAL=off to build Yellowmoon without it."
        );
    }
    true
}

/// Whether `YM_APPLE_BRIDGE=off` switches every Swift bridge off.
fn bridge_off() -> bool {
    env::var("YM_APPLE_BRIDGE").is_ok_and(|value| value == "off")
}

/// The major version of an SDK version string ("15.4" is 15).
fn sdk_major(version: &str) -> Option<u32> {
    version
        .split('.')
        .next()
        .and_then(|major| major.parse().ok())
}

/// What `swiftc` compiles: one Swift file into a static library of one module.
struct SwiftLibrary<'a> {
    /// The file under `swift/`.
    file: &'a str,
    /// The Swift module name.
    module: &'a str,
    /// The library's name: `lib<name>.a` in `OUT_DIR`, linked as `name`.
    name: &'a str,
    /// The Swift architecture of the target (`arm64`, `x86_64`).
    arch: &'a str,
    /// Further arguments for `swiftc`.
    extra: &'a [&'a str],
}

/// Compiles a Swift file into a static library in `OUT_DIR` for macOS 14 (or later) and tells cargo to link it. The
/// Swift runtime and the compiler runtime are [`link_swift_runtime`]'s.
fn compile_swift(library: &SwiftLibrary<'_>) -> Result<(), String> {
    let out_dir = PathBuf::from(env::var_os("OUT_DIR").ok_or("cargo did not set OUT_DIR")?);
    let manifest_dir = PathBuf::from(
        env::var_os("CARGO_MANIFEST_DIR").ok_or("cargo did not set CARGO_MANIFEST_DIR")?,
    );
    let source = manifest_dir.join("swift").join(library.file);
    let sdk = xcrun(&["--sdk", "macosx", "--show-sdk-path"])?;
    let target = swift_target(library.arch);
    let archive = out_dir.join(format!("lib{}.a", library.name));
    let module_cache = out_dir.join(format!("swift-module-cache-{}", library.name));

    let output = Command::new("xcrun")
        .args(["--sdk", "macosx", "swiftc"])
        .args(["-parse-as-library", "-emit-library", "-static"])
        .args(["-module-name", library.module, "-swift-version", "5", "-O"])
        .args(["-target", &target, "-sdk", &sdk])
        .arg("-module-cache-path")
        .arg(&module_cache)
        .args(library.extra)
        .arg(&source)
        .arg("-o")
        .arg(&archive)
        .output()
        .map_err(|e| format!("swiftc could not run: {e}"))?;
    // swiftc's warnings go to this script's stderr (shown by `cargo build -vv` and on failure);
    // a line on stdout could be read as a cargo instruction.
    eprintln!("{}", String::from_utf8_lossy(&output.stdout));
    eprintln!("{}", String::from_utf8_lossy(&output.stderr));
    if !output.status.success() {
        return Err(format!(
            "swiftc failed ({}):\n{}{}",
            output.status,
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    println!("cargo::rustc-link-search=native={}", out_dir.display());
    println!("cargo::rustc-link-lib=static={}", library.name);
    link_swift_runtime(library.arch)
}

/// The `-target` of a Swift object: the architecture and the macOS it is compiled for.
fn swift_target(arch: &str) -> String {
    format!("{arch}-apple-macos{}", deployment_target())
}

/// Builds Apple Auto's bridge (`swift/AppleAuto.swift`) and links what it needs.
fn build_auto_bridge() -> Result<(), String> {
    compile_swift(&SwiftLibrary {
        file: "AppleAuto.swift",
        module: "YmAppleAuto",
        name: "ym_apple_auto",
        arch: "arm64",
        // No autolink entry for FoundationModels: the final binaries link it weakly (see the top).
        extra: &[
            "-Xfrontend",
            "-disable-autolink-framework",
            "-Xfrontend",
            "FoundationModels",
        ],
    })?;
    println!("cargo::rustc-link-lib=framework=Foundation");
    println!("cargo::rustc-link-lib=framework=CoreGraphics");
    // This crate's own tests and examples; the binaries of other crates add both themselves.
    println!("cargo::rustc-link-arg=-Wl,-weak_framework,FoundationModels");
    add_swift_run_path();
    println!("cargo::rustc-cfg=ym_apple_bridge");
    Ok(())
}

/// Adds `/usr/lib/swift` to the run path of this crate's own tests and examples, once whichever bridges are built (a
/// duplicate run path only makes the linker warn).
fn add_swift_run_path() {
    static ADDED: AtomicBool = AtomicBool::new(false);
    if !ADDED.swap(true, Ordering::SeqCst) {
        println!("cargo::rustc-link-arg=-Wl,-rpath,/usr/lib/swift");
    }
}

/// Builds the Vision bridge (`swift/VisionMasks.swift`) for `arch` and links what it needs. A normal link: Vision is
/// on every macOS the app runs on.
fn build_vision_bridge(arch: &str) -> Result<(), String> {
    let swift = swift_arch(arch).ok_or_else(|| format!("no Swift architecture for {arch}"))?;
    compile_swift(&SwiftLibrary {
        file: "VisionMasks.swift",
        module: "YmAppleVision",
        name: "ym_apple_vision",
        arch: swift,
        extra: &[],
    })?;
    for framework in ["Foundation", "CoreGraphics", "CoreVideo", "Vision"] {
        println!("cargo::rustc-link-lib=framework={framework}");
    }
    // This crate's own tests and examples; the binaries of other crates add it themselves.
    add_swift_run_path();
    println!("cargo::rustc-cfg=ym_apple_vision");
    Ok(())
}

/// Builds the Metal bridge (`swift/MetalDevices.swift`) for `arch` and links what it needs. A normal link.
fn build_metal_bridge(arch: &str) -> Result<(), String> {
    let swift = swift_arch(arch).ok_or_else(|| format!("no Swift architecture for {arch}"))?;
    compile_swift(&SwiftLibrary {
        file: "MetalDevices.swift",
        module: "YmAppleMetal",
        name: "ym_apple_metal",
        arch: swift,
        extra: &[],
    })?;
    for framework in ["Foundation", "Metal"] {
        println!("cargo::rustc-link-lib=framework={framework}");
    }
    // This crate's own tests and examples; the binaries of other crates add it themselves.
    add_swift_run_path();
    println!("cargo::rustc-cfg=ym_apple_metal");
    Ok(())
}

/// Tells cargo where the Swift runtime and the compiler runtime are, once for every Swift library of this build (both
/// bridges are for the same architecture whenever both are built).
fn link_swift_runtime(arch: &str) -> Result<(), String> {
    static LINKED: AtomicBool = AtomicBool::new(false);
    if LINKED.swap(true, Ordering::SeqCst) {
        return Ok(());
    }
    let target = swift_target(arch);
    // The Swift runtime: the libraries live in macOS (/usr/lib/swift, found through the SDK's .tbd
    // stubs) and the toolchain (the back-deployment archives).
    let info = xcrun(&[
        "--sdk",
        "macosx",
        "swiftc",
        "-target",
        &target,
        "-print-target-info",
    ])?;
    let mut paths = json_string_array(&info, "runtimeLibraryPaths");
    paths.extend(json_string_array(&info, "runtimeLibraryImportPaths"));
    if paths.is_empty() {
        return Err(format!(
            "swiftc -print-target-info named no runtime library paths:\n{info}"
        ));
    }
    for path in paths {
        if Path::new(&path).exists() {
            println!("cargo::rustc-link-search=native={path}");
        }
    }

    // The compiler runtime, for `#available` (rustc links with -nodefaultlibs, so clang does not
    // add it). Not bundled into the rlib: the final link finds it on the search path.
    let resource_dir = xcrun(&["--sdk", "macosx", "clang", "-print-resource-dir"])?;
    let darwin = Path::new(&resource_dir).join("lib").join("darwin");
    if !darwin.join("libclang_rt.osx.a").is_file() {
        return Err(format!("no libclang_rt.osx.a in {}", darwin.display()));
    }
    println!("cargo::rustc-link-search=native={}", darwin.display());
    println!("cargo::rustc-link-lib=static:-bundle=clang_rt.osx");
    Ok(())
}

/// The macOS version the Swift code is compiled for: [`MIN_MACOS`], or `MACOSX_DEPLOYMENT_TARGET`
/// when that is later, so the Swift object never targets an older macOS than the binary.
fn deployment_target() -> String {
    let requested = env::var("MACOSX_DEPLOYMENT_TARGET").ok().and_then(|value| {
        let mut parts = value.trim().split('.');
        let major = parts.next()?.parse::<u32>().ok()?;
        let minor = parts.next().map_or(Some(0), |m| m.parse::<u32>().ok())?;
        Some((major, minor))
    });
    let (major, minor) = requested.filter(|&v| v > MIN_MACOS).unwrap_or(MIN_MACOS);
    format!("{major}.{minor}")
}

/// Says that this Mac build has no Apple Auto bridge, and why.
fn warn_without_auto(why: &str) {
    println!(
        "cargo::warning=Apple Auto is built without its bridge to Apple's model ({why}); \
         the app and ym will report it as not available."
    );
}

/// Says that this Mac build has no Vision bridge, and why.
fn warn_without_vision(why: &str) {
    println!(
        "cargo::warning=Subject and People masks are built without their bridge to Apple's Vision framework ({why}); \
         the app and ym will report them as not supported."
    );
}

/// Says that this Mac build has no Metal bridge, and why.
fn warn_without_metal(why: &str) {
    println!(
        "cargo::warning=The GPU budget is built without its bridge to Metal's memory limit ({why}); \
         the app and ym will use the budget rule alone."
    );
}

/// Runs `xcrun` and returns its trimmed standard output.
fn xcrun(args: &[&str]) -> Result<String, String> {
    let output = Command::new("xcrun")
        .args(args)
        .output()
        .map_err(|e| format!("xcrun could not run: {e}"))?;
    if !output.status.success() {
        return Err(format!(
            "`xcrun {}` failed: {}",
            args.join(" "),
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    Ok(String::from_utf8_lossy(&output.stdout).trim().to_owned())
}

/// The strings of the JSON array under `key` (the first occurrence), e.g. `"key": ["a", "b"]`.
/// Enough for `swiftc -print-target-info`, whose arrays hold plain path strings.
fn json_string_array(json: &str, key: &str) -> Vec<String> {
    let needle = format!("\"{key}\"");
    let Some(start) = json.find(&needle) else {
        return Vec::new();
    };
    let rest = json[start + needle.len()..].trim_start();
    let Some(rest) = rest.strip_prefix(':') else {
        return Vec::new();
    };
    let Some(body) = rest.trim_start().strip_prefix('[') else {
        return Vec::new();
    };
    let mut strings = Vec::new();
    let mut chars = body.chars();
    while let Some(c) = chars.next() {
        match c {
            ']' => break,
            '"' => {
                let mut text = String::new();
                loop {
                    match chars.next() {
                        None => return strings,
                        Some('"') => break,
                        Some('\\') => match chars.next() {
                            None => return strings,
                            Some('n') => text.push('\n'),
                            Some('t') => text.push('\t'),
                            Some(other) => text.push(other),
                        },
                        Some(other) => text.push(other),
                    }
                }
                strings.push(text);
            }
            _ => {}
        }
    }
    strings
}
