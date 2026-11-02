//! The calls into the Vision bridge (`swift/VisionMasks.swift`, declared in `swift/ym_apple.h`). Compiled only when
//! build.rs compiled the Swift (`cfg(ym_apple_vision)`: a Mac whose SDK has the macOS 14 Vision requests).

use std::ffi::c_char;

use ym_core::vision::{
    VisionError, VisionKind, VisionMask, VisionPicture, VisionStatus, VisionSupport,
};

use crate::vision_abi::{self, MESSAGE_BYTES};

// The two functions of ym_apple.h. A test (src/source_tests.rs) checks that the header and the Swift file name the
// same ones.
unsafe extern "C" {
    /// 0 when the request runs on a 64 × 64 picture, else the reason code (`vision_abi::status_of_probe`). It takes
    /// no pointers and has no preconditions, so it is safe to call.
    safe fn ym_apple_mask_probe(kind: i32) -> i32;

    /// Runs the request of `kind` on `width × height` RGB pixels and writes the union of its instances into `out`
    /// (`out_len` floats, at least `width × height`). 0 with the mask in `out`, 1 for nothing found, else an error
    /// code (`vision_abi::MAKE_*`) with Apple's message in `message` (`message_len` bytes, NUL-terminated).
    fn ym_apple_mask_make(
        kind: i32,
        rgb: *const u8,
        width: u32,
        height: u32,
        out: *mut f32,
        out_len: usize,
        message: *mut c_char,
        message_len: usize,
    ) -> i32;
}

/// Whether this Mac runs each request: the Swift runs it on a 64 × 64 picture.
pub(crate) fn probe() -> VisionSupport {
    let status = |kind: VisionKind| -> VisionStatus {
        vision_abi::status_of_probe(ym_apple_mask_probe(vision_abi::kind_code(kind)))
    };
    VisionSupport {
        subject: status(VisionKind::Subject),
        people: status(VisionKind::People),
    }
}

/// The union of the instances `kind` finds in `picture`, at the picture's size.
pub(crate) fn mask(kind: VisionKind, picture: &VisionPicture) -> Result<VisionMask, VisionError> {
    let (width, height) = vision_abi::picture_size(kind, picture)?;
    let mut plane = vec![0.0f32; picture.width * picture.height];
    let mut message = [0u8; MESSAGE_BYTES];
    // SAFETY: `picture.rgb` holds `width × height × 3` bytes (checked by `picture_size`), `plane` is writable for
    // `plane.len()` floats, which is what `out_len` says and at least `width × height`, and `message` is writable for
    // its full size. All three live until the call returns; the Swift side reads `rgb` (it copies it first) and writes
    // `out` and `message` only on this thread and only before it returns, and keeps no pointer to any of them.
    let code = unsafe {
        ym_apple_mask_make(
            vision_abi::kind_code(kind),
            picture.rgb.as_ptr(),
            width,
            height,
            plane.as_mut_ptr(),
            plane.len(),
            message.as_mut_ptr().cast::<c_char>(),
            message.len(),
        )
    };
    vision_abi::outcome(kind, code, &message, plane, picture.width, picture.height)
}
