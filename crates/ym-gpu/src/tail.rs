//! Reserved for G-08: the export tail on the device (slabs, strip sharpening, resize, C13 bands;
//! `docs/gpu-plan.md` section 4.10).
//!
//! Registered by the device foundation (G-02) as an empty module so that G-08 adds its types without
//! touching `lib.rs`. The pieces it builds on are the chunked readback
//! ([`read_bytes`](crate::stream::read_bytes)), the pass lists ([`crate::stream`]) and the design
//! binding size ([`DESIGN_BINDING_BYTES`](crate::tiles::DESIGN_BINDING_BYTES)) that fixes the slab size.
