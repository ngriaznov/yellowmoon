//! The GPU budget from Metal (`docs/26-hardening.md` H-17, finding P5-2; `docs/17-gpu.md` section 8).
//!
//! wgpu does not expose Metal's `recommendedMaxWorkingSetSize`, the most memory the system
//! recommends one device to use. `ym-apple` reads it through Swift and hands the list of Metal
//! devices to [`GpuOpener`](super::gpu::GpuOpener) behind [`MetalDevices`], the way the Vision
//! framework and Apple's model are handed over; `ym-core` does not depend on `ym-apple`. This module
//! is the part that is the same on every platform and is tested everywhere:
//!
//! * [`metal_limit`] says which Metal device is the one wgpu opened (by name) and what its limit is,
//!   or why there is none ([`MetalLimit`]);
//! * [`gpu_budget_bytes`] is the cap: `min(rule, factor × limit)`.
//!
//! The cap changes only where work runs (tile sizes, the size of the buffer pool), never a pixel.

use std::fmt;

use ym_gpu::device::GpuInfo;

/// The share of Metal's working-set limit the GPU budget may use.
///
/// **A starting value, not measured.** It leaves a quarter of what Metal recommends for the system's
/// own use and for the buffers the pool does not count (textures, the swap chain). A real Intel Mac
/// with an iGPU and one with an AMD GPU set it (`docs/26` MC-6, the owner's); until then it is a
/// guess, and the rule of `MemoryBudget` still applies as the upper bound.
pub const METAL_WORKING_SET_FACTOR: f64 = 0.75;

/// What `MTLDevice` says about one Metal device.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MetalDevice {
    /// `MTLDevice.name`. wgpu's Metal backend reports the same string as the adapter's name.
    pub name: String,
    /// `recommendedMaxWorkingSetSize`, in bytes: 0 when Metal gave none.
    pub recommended_max_working_set_bytes: u64,
    /// `hasUnifiedMemory`.
    pub has_unified_memory: bool,
    /// `isLowPower`.
    pub is_low_power: bool,
    /// `isRemovable`.
    pub is_removable: bool,
}

/// The Metal devices of this computer. `ym-apple` implements it with Swift (`MTLCopyAllDevices`).
pub trait MetalDevices: Send + Sync {
    /// Every Metal device. Empty where there is no Metal (not a Mac) and where Metal did not answer.
    fn devices(&self) -> Vec<MetalDevice>;
}

/// A computer with no Metal devices: the default of the opener, and what a build for another
/// platform has.
#[derive(Clone, Copy, Debug, Default)]
pub struct NoMetalDevices;

impl MetalDevices for NoMetalDevices {
    fn devices(&self) -> Vec<MetalDevice> {
        Vec::new()
    }
}

/// A fixed list, for tests and for tools that already have the answer.
#[derive(Clone, Debug, Default)]
pub struct FixedMetalDevices(pub Vec<MetalDevice>);

impl MetalDevices for FixedMetalDevices {
    fn devices(&self) -> Vec<MetalDevice> {
        self.0.clone()
    }
}

/// The Metal working-set limit of the adapter wgpu opened, or why there is none.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MetalLimit {
    /// Exactly one Metal device has the adapter's name and gave a limit above 0.
    Known {
        /// The device's name.
        name: String,
        /// Its `recommendedMaxWorkingSetSize`, in bytes.
        bytes: u64,
    },
    /// The adapter is not a Metal one (no adapter, a Vulkan device, Mesa lavapipe): not a Mac.
    NotMetal,
    /// The adapter is a Metal one and Metal listed no devices: it did not answer.
    NoAnswer,
    /// No Metal device has the adapter's name.
    NoMatch,
    /// More than one Metal device has the adapter's name, so the limit of ours is not known.
    Ambiguous(usize),
    /// The one device with the adapter's name gave no limit (0).
    Zero,
}

impl MetalLimit {
    /// The limit in bytes, when there is one.
    #[must_use]
    pub fn bytes(&self) -> Option<u64> {
        match self {
            Self::Known { bytes, .. } => Some(*bytes),
            _ => None,
        }
    }

    /// Why there is no limit, in a short sentence (`ym doctor`); `None` when there is one.
    #[must_use]
    pub fn why_none(&self) -> Option<String> {
        match self {
            Self::Known { .. } => None,
            Self::NotMetal => Some("not a Mac, so no Metal limit".to_owned()),
            Self::NoAnswer => Some("Metal did not answer".to_owned()),
            Self::NoMatch => Some("no Metal device has the adapter's name".to_owned()),
            Self::Ambiguous(n) => Some(format!(
                "{n} Metal devices have the adapter's name, so the right one is not known"
            )),
            Self::Zero => Some("Metal gave no limit for the device".to_owned()),
        }
    }
}

impl fmt::Display for MetalLimit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Known { bytes, .. } => write!(f, "{bytes} bytes"),
            other => f.write_str(&other.why_none().unwrap_or_default()),
        }
    }
}

/// Which Metal device is the adapter wgpu opened, by name, and what its limit is.
///
/// `adapter_is_metal` is whether wgpu's backend for the adapter is Metal. The name has to match
/// exactly one device (compared as given, after trimming spaces): on an Intel Mac with an AMD GPU the
/// device wgpu opened is not the default one, and two devices with one name (two identical cards)
/// cannot be told apart from here. With no match, several matches or a zero limit there is no limit
/// and the rule stays.
#[must_use]
pub fn metal_limit(
    adapter_name: &str,
    adapter_is_metal: bool,
    devices: &[MetalDevice],
) -> MetalLimit {
    if !adapter_is_metal {
        return MetalLimit::NotMetal;
    }
    if devices.is_empty() {
        return MetalLimit::NoAnswer;
    }
    let wanted = adapter_name.trim();
    let mut matches = devices.iter().filter(|d| d.name.trim() == wanted);
    match (matches.next(), matches.next()) {
        (None, _) => MetalLimit::NoMatch,
        (Some(device), None) => match device.recommended_max_working_set_bytes {
            0 => MetalLimit::Zero,
            bytes => MetalLimit::Known {
                name: device.name.clone(),
                bytes,
            },
        },
        (Some(_), Some(_)) => {
            MetalLimit::Ambiguous(devices.iter().filter(|d| d.name.trim() == wanted).count())
        }
    }
}

/// [`metal_limit`] for an adapter as wgpu described it.
#[must_use]
pub fn metal_limit_of(adapter: &GpuInfo, devices: &[MetalDevice]) -> MetalLimit {
    metal_limit(&adapter.name, adapter.backend == "Metal", devices)
}

/// The GPU budget: `min(rule_bytes, factor × limit)`. Without a limit (`None`, or 0 which Metal gives
/// where it has no answer) it is `rule_bytes`. `rule_bytes` is what the budget rule gave: 4 GiB, or
/// `RAM / 4` between 512 MiB and 4 GiB on a unified-memory machine.
#[must_use]
pub fn gpu_budget_bytes(rule_bytes: u64, metal_limit: Option<u64>, factor: f64) -> u64 {
    match metal_limit {
        Some(limit) if limit > 0 => {
            // Rounded down; an f64 holds every limit a GPU has to the byte or within a few.
            let capped = (limit as f64 * factor).max(0.0) as u64;
            rule_bytes.min(capped)
        }
        _ => rule_bytes,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const GIB: u64 = 1 << 30;

    fn device(name: &str, limit: u64) -> MetalDevice {
        MetalDevice {
            name: name.to_owned(),
            recommended_max_working_set_bytes: limit,
            has_unified_memory: false,
            is_low_power: false,
            is_removable: false,
        }
    }

    #[test]
    fn the_budget_is_the_smaller_of_the_rule_and_the_share_of_the_limit() {
        // (rule, limit, expected, what it shows)
        let cases: [(u64, Option<u64>, u64, &str); 9] = [
            (
                4 * GIB,
                Some(2 * GIB),
                3 * GIB / 2,
                "a limit below the rule: 0.75 of it",
            ),
            (
                4 * GIB,
                Some(16 * GIB),
                4 * GIB,
                "a limit above the rule: the rule",
            ),
            (
                3 * GIB,
                Some(4 * GIB),
                3 * GIB,
                "0.75 of the limit is exactly the rule",
            ),
            (4 * GIB, None, 4 * GIB, "no limit: the rule"),
            (4 * GIB, Some(0), 4 * GIB, "a zero limit is none"),
            (
                2 * GIB,
                Some(1536 << 20),
                1152 << 20,
                "a unified rule above 0.75 of the limit",
            ),
            (2 * GIB, Some(12 * GIB), 2 * GIB, "a unified rule below it"),
            (0, Some(GIB), 0, "a rule of 0 stays 0"),
            (
                4 * GIB,
                Some(1),
                0,
                "a limit of one byte caps to 0 (the pool has its own floor)",
            ),
        ];
        for (rule, limit, want, what) in cases {
            assert_eq!(
                gpu_budget_bytes(rule, limit, METAL_WORKING_SET_FACTOR),
                want,
                "{what}"
            );
        }
        // The factor is the argument, not hidden.
        assert_eq!(gpu_budget_bytes(4 * GIB, Some(2 * GIB), 0.5), GIB);
        assert_eq!(METAL_WORKING_SET_FACTOR, 0.75);
    }

    #[test]
    fn the_device_wgpu_opened_is_found_by_name() {
        let amd = "AMD Radeon Pro 5500M";
        let intel = "Intel(R) UHD Graphics 630";
        let para = "Apple Paravirtual device";
        let both = [device(intel, 1536 << 20), device(amd, 4 * GIB)];
        let twins = [device(amd, 4 * GIB), device(amd, 8 * GIB)];

        // One match: the one wgpu opened, not the default (first) device.
        assert_eq!(
            metal_limit(amd, true, &both),
            MetalLimit::Known {
                name: amd.to_owned(),
                bytes: 4 * GIB
            }
        );
        assert_eq!(metal_limit(amd, true, &both).bytes(), Some(4 * GIB));
        assert_eq!(metal_limit(intel, true, &both).bytes(), Some(1536 << 20));
        // Spaces around a name do not matter.
        assert_eq!(
            metal_limit(" AMD Radeon Pro 5500M ", true, &both).bytes(),
            Some(4 * GIB)
        );
        // None: a name Metal does not list.
        assert_eq!(metal_limit("Apple M2", true, &both), MetalLimit::NoMatch);
        // Two of the same name: no limit, and the count is said.
        assert_eq!(metal_limit(amd, true, &twins), MetalLimit::Ambiguous(2));
        assert_eq!(metal_limit(amd, true, &twins).bytes(), None);
        // The Paravirtual device of a VM matches itself like any other, and a limit of 0 is none.
        assert_eq!(
            metal_limit(para, true, &[device(para, 3 * GIB)]).bytes(),
            Some(3 * GIB)
        );
        assert_eq!(
            metal_limit(para, true, &[device(para, 0)]),
            MetalLimit::Zero
        );
        // A VM whose wgpu adapter is the Paravirtual device does not borrow another device's limit.
        assert_eq!(metal_limit(para, true, &both), MetalLimit::NoMatch);
        // Not Metal at all, and Metal listing nothing.
        assert_eq!(
            metal_limit("llvmpipe (LLVM 17.0.6, 256 bits)", false, &both),
            MetalLimit::NotMetal
        );
        assert_eq!(metal_limit(amd, false, &[]), MetalLimit::NotMetal);
        assert_eq!(metal_limit(amd, true, &[]), MetalLimit::NoAnswer);
    }

    #[test]
    fn every_reason_has_a_sentence_and_only_a_limit_has_none() {
        let known = MetalLimit::Known {
            name: "x".to_owned(),
            bytes: GIB,
        };
        assert_eq!(known.why_none(), None);
        for none in [
            MetalLimit::NotMetal,
            MetalLimit::NoAnswer,
            MetalLimit::NoMatch,
            MetalLimit::Ambiguous(2),
            MetalLimit::Zero,
        ] {
            let text = none.why_none().expect("a reason");
            assert!(!text.is_empty() && !text.ends_with('.'), "{none:?}: {text}");
            assert_eq!(none.bytes(), None);
        }
    }
}
