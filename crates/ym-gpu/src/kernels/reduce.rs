//! The fixed-order reduction of the R5 chroma offsets for the reduced path (G-07, decision D2 of
//! `docs/gpu-plan.md`): the kernels [`KernelId::Reduce`](super::KernelId::Reduce) (one partial per
//! workgroup: a binary tree of compensated sums in thread-index order) and
//! [`KernelId::ReduceFold`](super::KernelId::ReduceFold) (the partials folded in index order, the
//! offsets and their counts written) share `highlights.wgsl` with the R5 apply pass, because the
//! terms they sum are the opposed estimates that pass computes. The host side is in
//! [`highlights`](super::highlights): [`OffsetReducer`] records the slabs of a frame and the fold,
//! [`reduce_offsets`] is the host wrapper the parity tests use. Re-exported here so that the family
//! has the name the plan gives it.

pub use super::highlights::{OffsetReducer, reduce_offsets};
