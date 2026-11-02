//! The C interface of the Vision bridge (`swift/ym_apple.h`, `ym_apple_mask_*`; `docs/23-nodes.md` sections 3.10
//! and 4.5) without the calls: the codes and the conversions. Kept apart from `vision_bridge.rs` so that it builds
//! and is tested on every platform.

use ym_core::settings::MaskUnavailable;
use ym_core::vision::{
    VisionError, VisionKind, VisionMask, VisionPicture, VisionPlane, VisionStatus,
};

/// The side of the picture the probe runs on, in pixels (`YM_APPLE_MASK_PROBE_PX`).
pub(crate) const PROBE_PX: usize = 64;

/// The bytes of the buffer for Apple's message.
pub(crate) const MESSAGE_BYTES: usize = 1024;

/// The kind argument of `ym_apple_mask_probe` and `ym_apple_mask_make` (`YM_APPLE_MASK_SUBJECT`,
/// `YM_APPLE_MASK_PEOPLE`).
pub(crate) const fn kind_code(kind: VisionKind) -> i32 {
    match kind {
        VisionKind::Subject => 0,
        VisionKind::People => 1,
    }
}

/// `ym_apple_mask_probe`: the request ran on the probe picture.
pub(crate) const PROBE_SUPPORTED: i32 = 0;
/// `ym_apple_mask_probe`: macOS is older than 14.
pub(crate) const PROBE_MACOS_TOO_OLD: i32 = 1;
/// `ym_apple_mask_probe`: the request failed on the probe picture, or the kind is unknown.
pub(crate) const PROBE_NOT_SUPPORTED: i32 = 2;

/// `ym_apple_mask_make`: the union mask is in the buffer.
pub(crate) const MAKE_MADE: i32 = 0;
/// `ym_apple_mask_make`: Vision found no instance.
pub(crate) const MAKE_NOTHING_FOUND: i32 = 1;
/// `ym_apple_mask_make`: macOS is older than 14.
pub(crate) const MAKE_MACOS_TOO_OLD: i32 = 2;
/// `ym_apple_mask_make`: the request could not run.
pub(crate) const MAKE_NOT_SUPPORTED: i32 = 3;
/// `ym_apple_mask_make`: the arguments are wrong.
pub(crate) const MAKE_BAD_INPUT: i32 = 4;
/// `ym_apple_mask_make`: the request ran and its mask could not be read.
pub(crate) const MAKE_FAILED: i32 = 5;

/// What a code of `ym_apple_mask_probe` says. A code the bridge should never give reads as not supported.
pub(crate) fn status_of_probe(code: i32) -> VisionStatus {
    match code {
        PROBE_SUPPORTED => VisionStatus::Supported,
        PROBE_MACOS_TOO_OLD => VisionStatus::Unsupported(MaskUnavailable::MacosTooOld),
        _ => VisionStatus::Unsupported(MaskUnavailable::NotSupported),
    }
}

/// The width and height to hand to the bridge, or an error when the picture is not `width × height × 3` bytes, is
/// empty, or does not fit the C types.
pub(crate) fn picture_size(
    kind: VisionKind,
    picture: &VisionPicture,
) -> Result<(u32, u32), VisionError> {
    let width = u32::try_from(picture.width).map_err(|_| VisionError::BadPicture(kind))?;
    let height = u32::try_from(picture.height).map_err(|_| VisionError::BadPicture(kind))?;
    let expected = picture
        .width
        .checked_mul(picture.height)
        .and_then(|pixels| pixels.checked_mul(3));
    if width == 0 || height == 0 || expected != Some(picture.rgb.len()) {
        return Err(VisionError::BadPicture(kind));
    }
    Ok((width, height))
}

/// The text before the first NUL (all of it without one), as UTF-8 with invalid bytes replaced.
pub(crate) fn c_string(bytes: &[u8]) -> String {
    let end = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
    String::from_utf8_lossy(&bytes[..end]).into_owned()
}

/// What `ym_apple_mask_make` returned: the plane for code 0, [`VisionMask::NothingFound`] for code 1, else the error
/// with Apple's message. `plane` is the buffer the bridge wrote (`width * height` floats).
pub(crate) fn outcome(
    kind: VisionKind,
    code: i32,
    message: &[u8],
    plane: Vec<f32>,
    width: usize,
    height: usize,
) -> Result<VisionMask, VisionError> {
    let text = c_string(message);
    let failed = |default: &str| {
        VisionError::Failed(if text.is_empty() {
            default.to_owned()
        } else {
            text.clone()
        })
    };
    match code {
        MAKE_MADE => Ok(VisionMask::Plane(VisionPlane {
            width,
            height,
            data: plane,
        })),
        MAKE_NOTHING_FOUND => Ok(VisionMask::NothingFound),
        MAKE_MACOS_TOO_OLD => Err(VisionError::Unavailable(kind, MaskUnavailable::MacosTooOld)),
        MAKE_NOT_SUPPORTED => Err(failed("Vision could not run the request.")),
        MAKE_BAD_INPUT => Err(failed("Vision was given something it cannot read.")),
        MAKE_FAILED => Err(failed("Vision's mask could not be read.")),
        other => Err(failed(&format!("Vision gave the unknown code {other}."))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn picture(width: usize, height: usize, bytes: usize) -> VisionPicture {
        VisionPicture {
            width,
            height,
            rgb: vec![0; bytes],
        }
    }

    #[test]
    fn the_kinds_have_the_codes_of_the_header() {
        assert_eq!(kind_code(VisionKind::Subject), 0);
        assert_eq!(kind_code(VisionKind::People), 1);
    }

    #[test]
    fn a_probe_code_is_a_status() {
        assert_eq!(status_of_probe(0), VisionStatus::Supported);
        assert_eq!(
            status_of_probe(1),
            VisionStatus::Unsupported(MaskUnavailable::MacosTooOld)
        );
        for code in [2, 3, -1, 99] {
            assert_eq!(
                status_of_probe(code),
                VisionStatus::Unsupported(MaskUnavailable::NotSupported),
                "{code}"
            );
        }
    }

    #[test]
    fn a_picture_that_does_not_fit_its_size_is_refused_before_the_bridge() {
        let kind = VisionKind::Subject;
        assert_eq!(picture_size(kind, &picture(4, 3, 36)), Ok((4, 3)));
        for bad in [
            picture(4, 3, 35),
            picture(4, 3, 37),
            picture(0, 3, 0),
            picture(4, 0, 0),
            picture(usize::MAX, 2, 6),
        ] {
            assert_eq!(
                picture_size(kind, &bad),
                Err(VisionError::BadPicture(kind)),
                "{}x{} with {} bytes",
                bad.width,
                bad.height,
                bad.rgb.len()
            );
        }
    }

    #[test]
    fn text_stops_at_the_first_nul() {
        assert_eq!(c_string(b"abc\0def"), "abc");
        assert_eq!(c_string(b"abc"), "abc");
        assert_eq!(c_string(b"\0"), "");
        assert_eq!(c_string(&[0x61, 0xff, 0x62, 0]), "a\u{fffd}b");
    }

    #[test]
    fn a_made_mask_nothing_found_and_every_failure_are_told_apart() {
        let kind = VisionKind::People;
        let plane = vec![0.0, 0.5, 1.0, 0.25];
        assert_eq!(
            outcome(kind, MAKE_MADE, b"", plane.clone(), 2, 2),
            Ok(VisionMask::Plane(VisionPlane {
                width: 2,
                height: 2,
                data: plane.clone()
            }))
        );
        assert_eq!(
            outcome(kind, MAKE_NOTHING_FOUND, b"", plane.clone(), 2, 2),
            Ok(VisionMask::NothingFound)
        );
        assert_eq!(
            outcome(kind, MAKE_MACOS_TOO_OLD, b"", plane.clone(), 2, 2),
            Err(VisionError::Unavailable(kind, MaskUnavailable::MacosTooOld))
        );
        // Apple's text is kept; without it a plain default says what happened.
        assert_eq!(
            outcome(
                kind,
                MAKE_NOT_SUPPORTED,
                b"No inference context.\0",
                plane.clone(),
                2,
                2
            ),
            Err(VisionError::Failed("No inference context.".to_owned()))
        );
        for code in [MAKE_NOT_SUPPORTED, MAKE_BAD_INPUT, MAKE_FAILED, 77] {
            let Err(VisionError::Failed(text)) = outcome(kind, code, b"\0", plane.clone(), 2, 2)
            else {
                panic!("code {code} is a failure");
            };
            assert!(!text.is_empty() && text.ends_with('.'), "{code}: {text}");
        }
    }
}
