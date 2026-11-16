//! The C interface of the Metal bridge (`swift/ym_apple.h`, `ym_apple_metal_devices`; `docs/26-hardening.md` H-17)
//! without the call: the struct's layout and the conversion. Kept apart from `metal_bridge.rs` so that it builds and is
//! tested on every platform.

use ym_core::app::metal::MetalDevice;

use crate::vision_abi::c_string;

/// The bytes of [`YmAppleMetalDevice::name`], the NUL included (`YM_APPLE_METAL_NAME_BYTES`).
pub(crate) const NAME_BYTES: usize = 128;

/// The most devices asked for in one call. A Mac has a handful (an iGPU, a dGPU, external GPUs).
pub(crate) const MAX_DEVICES: usize = 16;

/// `YmAppleMetalDevice` of `ym_apple.h`.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct YmAppleMetalDevice {
    /// `recommendedMaxWorkingSetSize`; 0 when Metal gave none.
    pub(crate) recommended_max_working_set_bytes: u64,
    /// `hasUnifiedMemory`, 0 or 1.
    pub(crate) has_unified_memory: u32,
    /// `isLowPower`, 0 or 1.
    pub(crate) is_low_power: u32,
    /// `isRemovable`, 0 or 1.
    pub(crate) is_removable: u32,
    /// Always 0.
    pub(crate) reserved: u32,
    /// `MTLDevice.name`, NUL-terminated (`char name[128]`).
    pub(crate) name: [u8; NAME_BYTES],
}

// The layout the Swift side writes to (ym_apple.h asserts the same).
const _: () = assert!(size_of::<YmAppleMetalDevice>() == 152);
const _: () = assert!(align_of::<YmAppleMetalDevice>() == 8);
const _: () = assert!(std::mem::offset_of!(YmAppleMetalDevice, has_unified_memory) == 8);
const _: () = assert!(std::mem::offset_of!(YmAppleMetalDevice, is_low_power) == 12);
const _: () = assert!(std::mem::offset_of!(YmAppleMetalDevice, is_removable) == 16);
const _: () = assert!(std::mem::offset_of!(YmAppleMetalDevice, reserved) == 20);
const _: () = assert!(std::mem::offset_of!(YmAppleMetalDevice, name) == 24);

impl YmAppleMetalDevice {
    /// An empty entry: what the buffer holds before the Swift side writes.
    pub(crate) const fn empty() -> Self {
        Self {
            recommended_max_working_set_bytes: 0,
            has_unified_memory: 0,
            is_low_power: 0,
            is_removable: 0,
            reserved: 0,
            name: [0; NAME_BYTES],
        }
    }

    /// The device as the core knows it.
    pub(crate) fn device(&self) -> MetalDevice {
        MetalDevice {
            name: c_string(&self.name),
            recommended_max_working_set_bytes: self.recommended_max_working_set_bytes,
            has_unified_memory: self.has_unified_memory != 0,
            is_low_power: self.is_low_power != 0,
            is_removable: self.is_removable != 0,
        }
    }
}

/// The devices of the first `count` entries (`ym_apple_metal_devices` wrote them; a `count` past the buffer is cut to
/// it).
pub(crate) fn devices(entries: &[YmAppleMetalDevice], count: usize) -> Vec<MetalDevice> {
    entries
        .iter()
        .take(count)
        .map(YmAppleMetalDevice::device)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(name: &[u8], limit: u64, flags: [u32; 3]) -> YmAppleMetalDevice {
        let mut e = YmAppleMetalDevice::empty();
        e.name[..name.len()].copy_from_slice(name);
        e.recommended_max_working_set_bytes = limit;
        [e.has_unified_memory, e.is_low_power, e.is_removable] = flags;
        e
    }

    #[test]
    fn an_entry_is_a_metal_device() {
        let e = entry(b"AMD Radeon Pro 5500M", 4 << 30, [0, 0, 7]);
        assert_eq!(
            e.device(),
            MetalDevice {
                name: "AMD Radeon Pro 5500M".to_owned(),
                recommended_max_working_set_bytes: 4 << 30,
                has_unified_memory: false,
                is_low_power: false,
                // Any non-zero value is true.
                is_removable: true,
            }
        );
        let unified = entry(b"Apple M2 Pro", 10 << 30, [1, 1, 0]).device();
        assert!(unified.has_unified_memory && unified.is_low_power && !unified.is_removable);
    }

    #[test]
    fn only_the_entries_written_are_read() {
        let list = [
            entry(b"A", 1, [0; 3]),
            entry(b"B", 2, [0; 3]),
            entry(b"C", 3, [0; 3]),
        ];
        let names =
            |count| -> Vec<String> { devices(&list, count).into_iter().map(|d| d.name).collect() };
        assert_eq!(names(0), Vec::<String>::new());
        assert_eq!(names(2), ["A", "B"]);
        // A count past the buffer reads what there is.
        assert_eq!(names(99), ["A", "B", "C"]);
    }

    #[test]
    fn a_name_without_a_nul_is_read_whole() {
        let mut e = YmAppleMetalDevice::empty();
        e.name = [b'x'; NAME_BYTES];
        assert_eq!(e.device().name.len(), NAME_BYTES);
    }
}
