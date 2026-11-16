# ym-apple

Three bridges from Swift to Apple's frameworks, each with its own gate in `build.rs`:

- **Apple Auto's** bridge to Apple Intelligence's on-device model (`docs/19-apple-auto.md`).
  `ym_apple::system_advisor()` gives the app and `ym` the `AutoAdvisor` of this computer.
- **The Subject and People masks'** bridge to Apple's Vision framework (`docs/23-nodes.md` sections 3.10 and 4.5).
  `ym_apple::system_vision()` gives them the `VisionProvider` of this computer (`ym_core::vision`): its probe runs
  each request on a 64 × 64 picture, and a request makes the union of the instances Vision finds in a picture.
  `ym_apple::vision::probe()` and `ym_apple::vision::mask(kind, &picture)` are the safe API behind it.

- `swift/VisionMasks.swift`: the Vision code (`VNGenerateForegroundInstanceMaskRequest` and
  `VNGeneratePersonInstanceMaskRequest`, `#available(macOS 14.0, *)`), exposed as two C functions,
  `ym_apple_mask_probe` and `ym_apple_mask_make`. No Swift error crosses the interface.
- `src/vision.rs`, `src/vision_abi.rs`, `src/vision_bridge.rs`: the safe API and the provider, the codes and
  conversions (built and tested on every platform), and the unsafe calls (the only `unsafe` of the Vision side).
- `build.rs` compiles it for macOS 14 and later for **both** architectures of the universal binary (cargo builds
  each separately) wherever the SDK has the macOS 14 Vision requests, with a normal link (Vision exists on every
  macOS the app supports): `cfg(ym_apple_vision)`, `DEP_YM_APPLE_AUTO_VISION=1` for the binaries.
  `YM_APPLE_VISION=off` (or `YM_APPLE_BRIDGE=off`) builds the stub: nothing is supported, for the reason of the
  platform (`not_a_mac`, or `not_supported` on a Mac).
- `YELLOWMOON_TEST_VISION=fake` gives any platform a computer that supports both requests and makes up a soft
  mask from the picture; `none` a computer that supports nothing.

- **The Metal bridge** lists the Metal devices with `recommendedMaxWorkingSetSize`, which wgpu does not expose
  (`docs/26-hardening.md` H-17, finding P5-2). `ym_apple::system_metal_devices()` gives the core the
  `MetalDevices` of this computer (`ym_core::app::metal`), and `ym_apple::gpu_opener(policy, budget)` is the one
  place the app and `ym` build their `GpuOpener`, with that list: the core matches the device wgpu opened to a Metal
  device by name and caps the GPU budget at 0.75 of its limit (`ym_core::app::metal::gpu_budget_bytes`). The list is
  empty off macOS and when Metal does not answer; the budget rule then stands.

  - `swift/MetalDevices.swift`: `MTLCopyAllDevices()`, exposed as one C function, `ym_apple_metal_devices`, which
    fills a caller-owned array of `YmAppleMetalDevice` (name, working-set limit, `hasUnifiedMemory`, `isLowPower`,
    `isRemovable`; 152 bytes, the layout asserted in the header and in `src/metal_abi.rs`). Nothing is allocated for
    the caller and no pointer is kept.
  - `src/metal.rs`, `src/metal_abi.rs`, `src/metal_bridge.rs`: the safe API, the struct and conversion (built and
    tested on every platform), and the one unsafe call.
  - `build.rs` compiles it like the Vision bridge: both architectures (an Intel Mac has Metal, and its AMD GPU is not
    the default device), macOS 14 SDK or later, normal link against Metal: `cfg(ym_apple_metal)`,
    `DEP_YM_APPLE_AUTO_METAL=1` for the binaries. `YM_APPLE_METAL=off` (or `YM_APPLE_BRIDGE=off`) builds the stub.
  - The Swift cannot be compiled off a Mac; the first compile is on the Mac runner (`mac-gpu.yml` runs
    `ym doctor --json` and asserts the limit). `cargo run -p ym-apple --release --example probe` prints the list.

Apple Auto's bridge:

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

The probe first prints what Vision says for Subject and People (whether the bridge is built, each request's probe answer and, for each that runs, the mask it makes of the synthetic picture, twice), then whether Apple Auto is available (and why not), the model and the macOS build,
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
