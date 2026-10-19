//! The calls into the Swift bridge (`swift/AppleAuto.swift`, declared in `swift/ym_apple.h`).
//! Compiled only when build.rs compiled the Swift (`cfg(ym_apple_bridge)`: Apple silicon, the
//! macOS 27 SDK or later).

use std::ffi::c_char;

use ym_core::apple_auto::{
    AdvisorAnswer, AdvisorError, AdvisorInfo, AdvisorPicture, Availability, REQUEST_TIMEOUT_MS,
};

use crate::abi::{self, INFO_BYTES, MESSAGE_BYTES, YmAppleAnswer};

// The four functions of ym_apple.h. A test (src/source_tests.rs) checks that the header and the
// Swift file name the same ones.
unsafe extern "C" {
    /// 0 when Apple Auto can run, else the reason code (`abi::UNAVAILABLE_CODES`). It takes no
    /// pointers and has no preconditions, so it is safe to call.
    safe fn ym_apple_auto_availability() -> i32;

    /// The model variant's display name into `buf` (`len` bytes, NUL-terminated); 0 on success.
    fn ym_apple_auto_model(buf: *mut c_char, len: usize) -> i32;

    /// The macOS version and build into `buf` (`len` bytes, NUL-terminated); 0 on success.
    fn ym_apple_auto_os(buf: *mut c_char, len: usize) -> i32;

    /// Asks about `width × height` RGB pixels with `instructions` (NUL-terminated UTF-8); blocks
    /// until the answer, an error or `timeout_ms`. 0 with the answer in `*out`, else an error code
    /// (`abi::FAILURE_CODES`) with Apple's message.
    fn ym_apple_auto_ask(
        rgb: *const u8,
        width: u32,
        height: u32,
        instructions: *const c_char,
        timeout_ms: u32,
        out: *mut YmAppleAnswer,
        message: *mut c_char,
        message_len: usize,
    ) -> i32;
}

/// The availability Apple's model reports (conditions 3 to 5 of `docs/19` section 1).
pub(crate) fn availability() -> Availability {
    abi::availability_from_code(ym_apple_auto_availability(), info)
}

/// Which model answers and on which macOS.
fn info() -> AdvisorInfo {
    AdvisorInfo::new(
        read_text(ym_apple_auto_model).unwrap_or_else(|| "unknown".to_owned()),
        read_text(ym_apple_auto_os).unwrap_or_else(|| "unknown".to_owned()),
    )
}

/// Calls one of the string functions with a buffer of [`INFO_BYTES`].
fn read_text(read: unsafe extern "C" fn(*mut c_char, usize) -> i32) -> Option<String> {
    let mut buf = [0u8; INFO_BYTES];
    // SAFETY: `buf` is writable for `buf.len()` bytes and lives until the call returns; the Swift
    // side writes at most that many bytes (the NUL included) and keeps no pointer to it.
    let code = unsafe { read(buf.as_mut_ptr().cast::<c_char>(), buf.len()) };
    (code == 0).then(|| abi::c_string(&buf))
}

/// Asks Apple's model about one picture with `instructions` (no retries: the core retries).
pub(crate) fn ask(
    picture: &AdvisorPicture,
    instructions: &str,
) -> Result<AdvisorAnswer, AdvisorError> {
    let (width, height) = abi::picture_size(picture)?;
    let instructions = abi::instructions_c_string(instructions);
    let mut answer = YmAppleAnswer::unset();
    let mut message = [0u8; MESSAGE_BYTES];
    // SAFETY: `picture.rgb` holds `width × height × 3` bytes (checked by `picture_size`) and
    // `instructions` is a NUL-terminated string; the Swift side reads both only before it returns
    // (it copies them before the request starts).
    // `answer` and `message` are writable for their full sizes and live until the call returns;
    // the Swift side writes them only on this thread and only before it returns, even after a
    // timeout (the request's task writes to a box of its own, never to these buffers).
    let code = unsafe {
        ym_apple_auto_ask(
            picture.rgb.as_ptr(),
            width,
            height,
            instructions.as_ptr(),
            REQUEST_TIMEOUT_MS,
            &raw mut answer,
            message.as_mut_ptr().cast::<c_char>(),
            message.len(),
        )
    };
    abi::outcome(code, &answer, &message)
}
