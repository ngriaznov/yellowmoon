# ym-apple

Apple Auto's bridge to Apple Intelligence's on-device model (`docs/19-apple-auto.md`).
`ym_apple::system_advisor()` gives the app and `ym` the `AutoAdvisor` of this computer.

- `swift/AppleAuto.swift`: the Foundation Models code (the answer's schema, the instructions,
  availability, the request), exposed as four C functions.
- `swift/ym_apple.h`: the C declarations of those functions and of `YmAppleAnswer`.
- `src/bridge.rs`: the Rust side of the C interface; `src/abi.rs`: the codes and conversions.
- `build.rs`: compiles the Swift on Apple silicon with the macOS 27 SDK (Xcode 27) or later and
  explains the weak link to FoundationModels. Everywhere else the crate is a stub that reports why
  Apple Auto is not available. `YM_APPLE_BRIDGE=off` builds the stub on any Mac.

## Trying it on a Mac

```sh
cargo run -p ym-apple --release --example probe
```

The probe prints whether Apple Auto is available (and why not), the model and the macOS build,
then asks twice about a synthetic 768 × 512 picture with the default instructions and prints the
steps (a step the model left empty shows as not set), the scene, the plan, the time per request and
whether the second answer repeated the first. `YELLOWMOON_TEST_APPLE_AUTO=fake` runs it against the
test advisor.

A binary that depends on this crate must link FoundationModels weakly and find the Swift runtime in
macOS when the bridge is built: its build script passes `-Wl,-weak_framework,FoundationModels` and
`-Wl,-rpath,/usr/lib/swift` when `DEP_YM_APPLE_AUTO_BRIDGE` is `1` (see `crates/ym-cli/build.rs`).
Without the first the link fails; without the second a binary linked for a macOS older than 12
(rustc's default when `MACOSX_DEPLOYMENT_TARGET` is not set) stops at launch, as it loads the
concurrency runtime through `@rpath`.
