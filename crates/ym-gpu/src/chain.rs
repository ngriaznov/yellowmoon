//! Reserved for G-03, G-06 and G-07: the device chain (`GpuChain`, `StageRange`, `GpuReduced`;
//! `docs/gpu-plan.md` sections 4.3 and 11).
//!
//! The foundation (G-02) provides what the chain is built from: [`GpuImage`](crate::image::GpuImage)
//! and [`GpuRaw`](crate::image::GpuRaw), [`PassList`](crate::stream::PassList) and
//! [`Pending`](crate::stream::Pending), the per-tile fault hook
//! ([`GpuContext::inject_tile_fault`](crate::device::GpuContext::inject_tile_fault)), the shared cancel
//! token ([`Gpu::run_with_token`](crate::device::Gpu::run_with_token)) and the tile planner
//! ([`crate::tiles`]). The chain itself starts as this empty module so that later tasks add their
//! types without touching `lib.rs`.
