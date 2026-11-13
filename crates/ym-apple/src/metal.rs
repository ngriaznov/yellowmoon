//! The Metal devices of this Mac, for the GPU budget (`docs/26-hardening.md` H-17, finding P5-2; `docs/17-gpu.md`
//! section 8).
//!
//! [`devices`] lists them with the most memory Metal recommends each may use (`recommendedMaxWorkingSetSize`, which wgpu
//! does not expose). [`system_metal_devices`] gives the app and `ym` the [`MetalDevices`] of this computer, and
//! [`gpu_opener`] is the one place that hands it to the core's [`GpuOpener`], so that the app and `ym` have the same
//! budget (hard rule 6). The core matches the device wgpu opened to one of these by name and caps the budget
//! (`ym_core::app::metal`).
//!
//! The Swift (`swift/MetalDevices.swift`) is compiled for every Mac whose SDK is the macOS 14 one or later, both
//! architectures (`build.rs`); elsewhere this module lists nothing. Metal runs on this Mac and nothing is sent anywhere
//! (hard rule 7).

use std::sync::Arc;

use ym_core::app::gpu::{GpuOpener, GpuPolicy};
use ym_core::app::metal::{MetalDevice, MetalDevices};
use ym_core::scheduler::MemoryBudget;

/// Whether this build carries the Metal bridge (a Mac whose SDK is recent enough). Without it [`AppleMetal`] lists no
/// device.
pub const METAL_BUILT: bool = cfg!(ym_apple_metal);

/// Every Metal device of this computer, in the order `MTLCopyAllDevices` gives them. Empty on a platform without Metal
/// and when Metal does not answer; never an error.
#[must_use]
pub fn devices() -> Vec<MetalDevice> {
    platform::devices()
}

/// Metal, on this Mac.
#[derive(Debug, Clone, Copy, Default)]
pub struct AppleMetal;

impl MetalDevices for AppleMetal {
    fn devices(&self) -> Vec<MetalDevice> {
        devices()
    }
}

/// The Metal devices the app and `ym` hand to the core: [`AppleMetal`].
#[must_use]
pub fn system_metal_devices() -> Arc<dyn MetalDevices> {
    Arc::new(AppleMetal)
}

/// The opener the app and `ym` use for the graphics device: [`GpuOpener::new`] with
/// [`system_metal_devices`]. Every place that opens the device calls this (a test reads their sources), so that
/// both front ends cap the GPU budget the same way.
#[must_use]
pub fn gpu_opener(policy: GpuPolicy, budget: MemoryBudget) -> GpuOpener {
    GpuOpener::new(policy, budget).with_metal_devices(system_metal_devices())
}

/// With the Swift bridge: Metal answers.
#[cfg(ym_apple_metal)]
mod platform {
    pub(crate) use crate::metal_bridge::devices;
}

/// Without the Swift bridge: no devices.
#[cfg(not(ym_apple_metal))]
mod platform {
    use ym_core::app::metal::MetalDevice;

    pub(crate) fn devices() -> Vec<MetalDevice> {
        Vec::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn this_computer_lists_its_devices_or_none() {
        let list = devices();
        println!("Metal devices: {list:#?}");
        if cfg!(ym_apple_metal) {
            // A Mac, a CI runner and a VM all have at least one Metal device (a VM: "Apple Paravirtual device").
            assert!(!list.is_empty(), "a Mac lists a Metal device");
            for d in &list {
                assert!(!d.name.is_empty());
            }
            assert!(
                list.iter().any(|d| d.recommended_max_working_set_bytes > 0),
                "one device has a limit: {list:?}"
            );
        } else {
            assert!(list.is_empty());
        }
        assert_eq!(METAL_BUILT, cfg!(ym_apple_metal));
        assert_eq!(AppleMetal.devices(), list);
        assert_eq!(system_metal_devices().devices(), list);
    }
}
