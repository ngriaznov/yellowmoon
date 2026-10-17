//! The binding size the kernels are designed for (`docs/17-gpu.md` K4 and section 4).
//!
//! Lavapipe, the CI adapter, reports a 128 MiB storage binding although the device asks for more.
//! A kernel module exports `max_binding_bytes(rect)`, the size of its largest binding for a compute
//! rectangle; these helpers check it against the **constant** [`DESIGN_BINDING_BYTES`], not against
//! the device in hand, so the check does not depend on the machine it runs on.

#![allow(dead_code)]

use ym_gpu::cpu::tiling::Rect;
pub use ym_gpu::tiles::DESIGN_BINDING_BYTES;

/// Panics unless `bytes` fits one binding of the design size.
pub fn assert_fits_design_binding(what: &str, bytes: u64) {
    assert!(
        ym_gpu::tiles::fits_design_binding(bytes),
        "{what}: a {bytes} byte binding ({:.1} MiB) is above the {} MiB design size",
        bytes as f64 / f64::from(1u32 << 20),
        DESIGN_BINDING_BYTES >> 20
    );
}

/// K4: checks a kernel module's `max_binding_bytes(rect)` at `rect` (the largest compute rectangle
/// of `docs/17-gpu.md` section 4) against the design binding, and that it grows with the rectangle
/// (a function that ignores its argument would pass the first check for the wrong reason).
pub fn assert_max_binding_bytes(what: &str, rect: Rect, max_binding_bytes: impl Fn(Rect) -> u64) {
    let at_max = max_binding_bytes(rect);
    assert_fits_design_binding(&format!("{what} at {}x{}", rect.w, rect.h), at_max);
    let smaller = Rect::new(rect.x, rect.y, rect.w / 2, rect.h / 2);
    assert!(
        max_binding_bytes(smaller) < at_max,
        "{what}: max_binding_bytes does not depend on the rectangle"
    );
}
