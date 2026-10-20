//! Builds Apple Auto's Swift bridge, `swift/AppleAuto.swift` (`docs/19-apple-auto.md` section 11).
//!
//! The bridge is compiled only for Apple silicon Macs (`aarch64-apple-darwin`) with the macOS 27 SDK
//! or later (Xcode 27), and not when `YM_APPLE_BRIDGE=off`. Then this script
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
//! Everywhere else (Linux, Intel Macs, an older SDK, `YM_APPLE_BRIDGE=off`) nothing is compiled and
//! the crate reports `not_a_mac`, `intel_mac` or `built_without_support`. A Mac build without the
//! bridge prints a `cargo:warning`; other platforms build silently.

use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

/// The oldest macOS the Swift code is compiled for: the app's `minimumSystemVersion`.
const MIN_MACOS: (u32, u32) = (14, 0);

/// The first SDK with every Foundation Models API the bridge uses (`Attachment`, `capabilities`,
/// `variant`, `LanguageModelError`).
const MIN_SDK_MAJOR: u32 = 27;

fn main() {
    println!("cargo::rustc-check-cfg=cfg(ym_apple_bridge)");
    println!("cargo::rerun-if-changed=build.rs");
    println!("cargo::rerun-if-changed=swift");
    for variable in [
        "YM_APPLE_BRIDGE",
        "DEVELOPER_DIR",
        "SDKROOT",
        "MACOSX_DEPLOYMENT_TARGET",
    ] {
        println!("cargo::rerun-if-env-changed={variable}");
    }

    let os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    let built = os == "macos" && arch == "aarch64" && build_if_possible();
    // Read by the build scripts of the binaries as DEP_YM_APPLE_AUTO_BRIDGE.
    println!("cargo::metadata=bridge={}", u8::from(built));
}

/// Builds the bridge when this Mac can, and says why not when it cannot. A failure to compile with a
/// recent enough SDK is an error, not a warning: it is a bug, and a silent stub would hide it.
fn build_if_possible() -> bool {
    if env::var("YM_APPLE_BRIDGE").is_ok_and(|value| value == "off") {
        warn_without("YM_APPLE_BRIDGE=off");
        return false;
    }
    let sdk_version = match xcrun(&["--sdk", "macosx", "--show-sdk-version"]) {
        Ok(version) => version,
        Err(why) => {
            warn_without(&why);
            return false;
        }
    };
    let major = sdk_version
        .split('.')
        .next()
        .and_then(|major| major.parse::<u32>().ok());
    if !major.is_some_and(|major| major >= MIN_SDK_MAJOR) {
        warn_without(&format!(
            "the macOS SDK is {sdk_version}; Apple Auto needs the macOS {MIN_SDK_MAJOR} SDK (Xcode {MIN_SDK_MAJOR}) or later"
        ));
        return false;
    }
    if let Err(why) = build_bridge() {
        panic!(
            "Apple Auto's Swift bridge did not build with the macOS {sdk_version} SDK.\n{why}\n\
             Set YM_APPLE_BRIDGE=off to build Yellowmoon without Apple Auto."
        );
    }
    true
}

/// Compiles `swift/AppleAuto.swift` and tells cargo how to link it.
fn build_bridge() -> Result<(), String> {
    let out_dir = PathBuf::from(env::var_os("OUT_DIR").ok_or("cargo did not set OUT_DIR")?);
    let manifest_dir = PathBuf::from(
        env::var_os("CARGO_MANIFEST_DIR").ok_or("cargo did not set CARGO_MANIFEST_DIR")?,
    );
    let source = manifest_dir.join("swift").join("AppleAuto.swift");
    let sdk = xcrun(&["--sdk", "macosx", "--show-sdk-path"])?;
    let target = format!("arm64-apple-macos{}", deployment_target());
    let library = out_dir.join("libym_apple_auto.a");
    let module_cache = out_dir.join("swift-module-cache");

    let output = Command::new("xcrun")
        .args(["--sdk", "macosx", "swiftc"])
        .args(["-parse-as-library", "-emit-library", "-static"])
        .args(["-module-name", "YmAppleAuto", "-swift-version", "5", "-O"])
        .args(["-target", &target, "-sdk", &sdk])
        .arg("-module-cache-path")
        .arg(&module_cache)
        // No autolink entry for FoundationModels: the final binaries link it weakly (see the top).
        .args(["-Xfrontend", "-disable-autolink-framework"])
        .args(["-Xfrontend", "FoundationModels"])
        .arg(&source)
        .arg("-o")
        .arg(&library)
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
    println!("cargo::rustc-link-lib=static=ym_apple_auto");

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

    println!("cargo::rustc-link-lib=framework=Foundation");
    println!("cargo::rustc-link-lib=framework=CoreGraphics");
    // This crate's own tests and examples; the binaries of other crates add both themselves.
    println!("cargo::rustc-link-arg=-Wl,-weak_framework,FoundationModels");
    println!("cargo::rustc-link-arg=-Wl,-rpath,/usr/lib/swift");
    println!("cargo::rustc-cfg=ym_apple_bridge");
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

/// Says that this Mac build has no bridge, and why.
fn warn_without(why: &str) {
    println!(
        "cargo::warning=Apple Auto is built without its bridge to Apple's model ({why}); \
         the app and ym will report it as not available."
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
