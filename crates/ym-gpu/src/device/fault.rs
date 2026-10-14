//! The fault surface of the device: what the tests can break on purpose, to prove that a GPU error
//! ends in the CPU fallback and never in a wrong picture (`docs/gpu-plan.md` sections 4.9 and 7).
//!
//! Everything that can fail is a *hook* in the code that runs in a normal build too: an allocation
//! ([`FaultState::on_allocation`]), a pipeline compile ([`FaultState::on_pipeline`]), the start of a
//! job ([`FaultState::on_job`]) and a tile of a chain ([`FaultState::on_tile`]). In a normal build
//! the hooks are empty functions the optimiser removes. With the cargo feature `fault-injection` a
//! test arms one fault ([`GpuFault`], [`crate::device::Gpu::set_fault`], or the environment
//! variable `YM_GPU_FAULT` read when the device opens) and the hooks start failing as described on
//! the variants. No feature, no fault: a release build cannot be made to fail this way.

use crate::device::{DeviceHealth, GpuError};
use crate::queue::CancelToken;

#[cfg(feature = "fault-injection")]
use std::sync::Mutex;

/// A fault a test can arm (feature `fault-injection`). `OutOfBudget`, `Shader` and `Lost` stay
/// armed until [`crate::device::Gpu::set_fault`] clears them, so a "retry once, then the CPU" rule
/// sees the failure twice; `CancelAtJob` and `FailTile` fire once, on the hit they name.
#[cfg(feature = "fault-injection")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GpuFault {
    /// Every allocation fails with [`GpuError::OutOfBudget`] (`oom`).
    OutOfBudget,
    /// Every pipeline compile fails with [`GpuError::Shader`] (`shader`).
    Shader,
    /// The device is lost at the start of the next job: it is marked lost and destroyed, and every
    /// later job reports [`GpuError::DeviceLost`] (`lost`).
    Lost,
    /// The `n`th job after arming (counting from 1) finds its cancel token set (`cancel@n`).
    CancelAtJob(u64),
    /// The `n`th tile after arming (counting from 1, as counted by
    /// [`crate::device::GpuContext::inject_tile_fault`]) fails with [`GpuError::Readback`]
    /// (`tile@n`).
    FailTile(u64),
}

#[cfg(feature = "fault-injection")]
impl std::str::FromStr for GpuFault {
    type Err = String;

    /// Parses `oom`, `shader`, `lost`, `cancel@N` or `tile@N` (N from 1).
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let s = s.trim();
        let at = |rest: &str| {
            rest.parse::<u64>()
                .ok()
                .filter(|&n| n >= 1)
                .ok_or_else(|| format!("`{s}`: the count after `@` is a whole number from 1"))
        };
        match s {
            "oom" => Ok(GpuFault::OutOfBudget),
            "shader" => Ok(GpuFault::Shader),
            "lost" => Ok(GpuFault::Lost),
            _ => {
                if let Some(n) = s.strip_prefix("cancel@") {
                    Ok(GpuFault::CancelAtJob(at(n)?))
                } else if let Some(n) = s.strip_prefix("tile@") {
                    Ok(GpuFault::FailTile(at(n)?))
                } else {
                    Err(format!(
                        "`{s}` is not a fault: use oom, shader, lost, cancel@N or tile@N"
                    ))
                }
            }
        }
    }
}

/// The armed fault and its hit counter, shared by the pool, the kernels and the job wrapper.
#[derive(Debug, Default)]
pub(crate) struct FaultState {
    #[cfg(feature = "fault-injection")]
    armed: Mutex<Option<(GpuFault, u64)>>,
}

impl FaultState {
    /// Arms `fault` (or clears the armed one) and restarts the hit counter.
    #[cfg(feature = "fault-injection")]
    pub(crate) fn arm(&self, fault: Option<GpuFault>) {
        *self
            .armed
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = fault.map(|f| (f, 0));
    }

    /// The fault armed from `YM_GPU_FAULT`, when the variable is set and valid.
    #[cfg(feature = "fault-injection")]
    pub(crate) fn arm_from_env(&self) {
        if let Ok(v) = std::env::var("YM_GPU_FAULT") {
            match v.parse::<GpuFault>() {
                Ok(f) => self.arm(Some(f)),
                Err(e) => tracing::warn!("YM_GPU_FAULT ignored: {e}"),
            }
        }
    }

    /// When the armed fault is of the kind `matches` selects, counts the hit and runs `f` on the
    /// fault and its hit number (from 1); other kinds are not counted.
    #[cfg(feature = "fault-injection")]
    fn hit<R>(
        &self,
        matches: fn(GpuFault) -> bool,
        f: impl FnOnce(GpuFault, u64) -> R,
    ) -> Option<R> {
        let mut guard = self
            .armed
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let (fault, hits) = guard.as_mut()?;
        if !matches(*fault) {
            return None;
        }
        *hits += 1;
        Some(f(*fault, *hits))
    }

    /// Called before every new allocation; `OutOfBudget` makes it fail.
    #[inline]
    #[allow(unused_variables)]
    pub(crate) fn on_allocation(
        &self,
        requested: u64,
        allocated: u64,
        budget: u64,
    ) -> Result<(), GpuError> {
        #[cfg(feature = "fault-injection")]
        if self
            .hit(|f| f == GpuFault::OutOfBudget, |_, _| ())
            .is_some()
        {
            return Err(GpuError::OutOfBudget {
                requested,
                allocated,
                budget,
            });
        }
        Ok(())
    }

    /// Called before every pipeline compile; `Shader` makes it fail.
    #[inline]
    #[allow(unused_variables)]
    pub(crate) fn on_pipeline(&self, kernel: &str) -> Result<(), GpuError> {
        #[cfg(feature = "fault-injection")]
        if self.hit(|f| f == GpuFault::Shader, |_, _| ()).is_some() {
            return Err(GpuError::Shader(format!("{kernel}: injected fault")));
        }
        Ok(())
    }

    /// Called at the start of every job on the device thread. `Lost` loses the device,
    /// `CancelAtJob(n)` sets the job's token on its `n`th hit.
    #[inline]
    #[allow(unused_variables)]
    pub(crate) fn on_job(&self, health: &DeviceHealth, device: &wgpu::Device, token: &CancelToken) {
        #[cfg(feature = "fault-injection")]
        {
            let lose = self.hit(
                |f| matches!(f, GpuFault::Lost | GpuFault::CancelAtJob(_)),
                |f, n| match f {
                    GpuFault::CancelAtJob(at) if at == n => {
                        token.cancel();
                        false
                    }
                    GpuFault::Lost => true,
                    _ => false,
                },
            );
            if lose == Some(true) {
                health.mark_lost("injected fault: the device was lost".to_string());
                device.destroy();
            }
        }
    }

    /// Called once per tile by the chains; `FailTile(n)` fails the `n`th.
    #[inline]
    pub(crate) fn on_tile(&self) -> Result<(), GpuError> {
        #[cfg(feature = "fault-injection")]
        if let Some(Some(n)) = self.hit(
            |f| matches!(f, GpuFault::FailTile(_)),
            |f, n| match f {
                GpuFault::FailTile(at) if at == n => Some(n),
                _ => None,
            },
        ) {
            return Err(GpuError::Readback(format!("injected fault at tile {n}")));
        }
        Ok(())
    }
}

#[cfg(all(test, feature = "fault-injection"))]
mod tests {
    use super::*;

    #[test]
    fn faults_parse_from_the_environment_spelling() {
        assert_eq!("oom".parse::<GpuFault>(), Ok(GpuFault::OutOfBudget));
        assert_eq!("shader".parse::<GpuFault>(), Ok(GpuFault::Shader));
        assert_eq!(" lost ".parse::<GpuFault>(), Ok(GpuFault::Lost));
        assert_eq!("cancel@3".parse::<GpuFault>(), Ok(GpuFault::CancelAtJob(3)));
        assert_eq!("tile@12".parse::<GpuFault>(), Ok(GpuFault::FailTile(12)));
        for bad in [
            "", "oom@1", "cancel@0", "cancel@x", "tile@", "tile@-1", "boom",
        ] {
            assert!(bad.parse::<GpuFault>().is_err(), "{bad:?}");
        }
    }

    #[test]
    fn sticky_faults_keep_failing_and_counted_ones_fire_once() {
        let f = FaultState::default();
        assert!(f.on_allocation(1, 0, 10).is_ok(), "nothing armed");
        f.arm(Some(GpuFault::OutOfBudget));
        assert!(f.on_allocation(1, 0, 10).is_err());
        assert!(f.on_allocation(1, 0, 10).is_err(), "sticky");
        assert!(f.on_pipeline("k").is_ok(), "only the armed kind fails");
        f.arm(Some(GpuFault::FailTile(2)));
        assert!(f.on_tile().is_ok());
        assert!(f.on_tile().is_err(), "the second tile");
        assert!(f.on_tile().is_ok(), "once");
        f.arm(None);
        assert!(f.on_allocation(1, 0, 10).is_ok());
    }
}
