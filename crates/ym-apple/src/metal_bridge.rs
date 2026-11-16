//! The call into the Metal bridge (`swift/MetalDevices.swift`, declared in `swift/ym_apple.h`). Compiled only when
//! build.rs compiled the Swift (`cfg(ym_apple_metal)`: a Mac whose SDK is the macOS 14 one or later).

use ym_core::app::metal::MetalDevice;

use crate::metal_abi::{self, MAX_DEVICES, YmAppleMetalDevice};

// The function of ym_apple.h. A test (src/source_tests.rs) checks that the header and the Swift file name it.
unsafe extern "C" {
    /// Writes the Metal devices into `out` (`capacity` entries) and the number of entries written into `*count`.
    /// 0 on success (also with no device), non-zero when an argument is missing.
    fn ym_apple_metal_devices(
        out: *mut YmAppleMetalDevice,
        capacity: usize,
        count: *mut usize,
    ) -> i32;
}

/// Every Metal device of this Mac; empty when Metal does not answer.
pub(crate) fn devices() -> Vec<MetalDevice> {
    let mut entries = [YmAppleMetalDevice::empty(); MAX_DEVICES];
    let mut count = 0usize;
    // SAFETY: `entries` is writable for `MAX_DEVICES` entries of the layout of `YmAppleMetalDevice` (asserted equal to
    // the header's), which is the `capacity` passed, and `count` is a valid `usize`. Both live until the call returns;
    // the Swift side writes them only on this thread and only before it returns, writes at most `capacity` entries,
    // and keeps no pointer to either.
    let code = unsafe { ym_apple_metal_devices(entries.as_mut_ptr(), entries.len(), &mut count) };
    if code != 0 {
        return Vec::new();
    }
    metal_abi::devices(&entries, count)
}
