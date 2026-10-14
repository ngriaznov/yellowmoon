//! Reserved for G-07: R1 normalise, R3 vignetting, R4 gains, the quantisation step and the clip mask in one pass.
//!
//! Registered by the device foundation (G-02) so that the module, the `KernelId` and its warm-up set
//! exist before the kernel does; the kernel's WGSL is still a placeholder stub (see
//! [`KernelId::is_placeholder`](super::KernelId::is_placeholder) and `docs/gpu-plan.md` section 11).
