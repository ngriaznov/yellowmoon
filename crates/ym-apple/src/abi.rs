//! The C interface of the Swift bridge (`swift/ym_apple.h`, `docs/19-apple-auto.md` section 11)
//! without the calls: the answer's layout, the meaning of the codes and the conversions. Kept apart
//! from `bridge.rs` so that it builds and is tested on every platform.

use ym_core::apple_auto::{
    AdvisorAnswer, AdvisorError, AdvisorInfo, AdvisorPicture, AutoSteps, Availability,
    EXPOSURE_STEPS, FailureKind, OTHER_STEPS, UnavailableReason,
};

/// The bytes of [`YmAppleAnswer::scene`], the NUL included (`YM_APPLE_SCENE_BYTES`).
pub(crate) const SCENE_BYTES: usize = 128;

/// The buffer for the model's name and the macOS version.
pub(crate) const INFO_BYTES: usize = 256;

/// The buffer for Apple's error message.
pub(crate) const MESSAGE_BYTES: usize = 1024;

/// `YmAppleAnswer` of `ym_apple.h`: the four steps and the scene, NUL-terminated UTF-8.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct YmAppleAnswer {
    /// `muchDarker` −3 … `muchBrighter` +3.
    pub(crate) exposure: i32,
    /// `muchCooler` −2 … `muchWarmer` +2.
    pub(crate) white: i32,
    /// `muchLess` −2 … `muchMore` +2.
    pub(crate) contrast: i32,
    /// `muchLess` −2 … `muchMore` +2.
    pub(crate) saturation: i32,
    /// What the photo shows, NUL-terminated (`char scene[128]`).
    pub(crate) scene: [u8; SCENE_BYTES],
}

// The layout the Swift side writes to (ym_apple.h asserts the same).
const _: () = assert!(size_of::<YmAppleAnswer>() == 144);
const _: () = assert!(align_of::<YmAppleAnswer>() == 4);
const _: () = assert!(std::mem::offset_of!(YmAppleAnswer, white) == 4);
const _: () = assert!(std::mem::offset_of!(YmAppleAnswer, contrast) == 8);
const _: () = assert!(std::mem::offset_of!(YmAppleAnswer, saturation) == 12);
const _: () = assert!(std::mem::offset_of!(YmAppleAnswer, scene) == 16);

impl YmAppleAnswer {
    /// Every step 0, the scene empty.
    pub(crate) const fn zeroed() -> Self {
        Self {
            exposure: 0,
            white: 0,
            contrast: 0,
            saturation: 0,
            scene: [0; SCENE_BYTES],
        }
    }
}

/// The reason codes of `ym_apple_auto_availability` (0 is available).
pub(crate) const UNAVAILABLE_CODES: [(i32, UnavailableReason); 6] = [
    (3, UnavailableReason::MacosTooOld),
    (4, UnavailableReason::DeviceNotEligible),
    (5, UnavailableReason::AppleIntelligenceOff),
    (6, UnavailableReason::ModelNotReady),
    (7, UnavailableReason::Unavailable),
    (8, UnavailableReason::NoImageInput),
];

/// The error codes of `ym_apple_auto_ask` (0 is an answer).
pub(crate) const FAILURE_CODES: [(i32, FailureKind); 11] = [
    (1, FailureKind::Unavailable),
    (2, FailureKind::RateLimited),
    (3, FailureKind::Timeout),
    (4, FailureKind::Refused),
    (5, FailureKind::Guardrail),
    (6, FailureKind::UnsupportedLanguage),
    (7, FailureKind::ContextTooLarge),
    (8, FailureKind::AssetsUnavailable),
    (9, FailureKind::ConcurrentRequests),
    (10, FailureKind::BadAnswer),
    (11, FailureKind::Other),
];

/// The availability a code of `ym_apple_auto_availability` stands for; `info` names the model when
/// it is available. A code the bridge should never give reads as [`UnavailableReason::Unavailable`].
pub(crate) fn availability_from_code(
    code: i32,
    info: impl FnOnce() -> AdvisorInfo,
) -> Availability {
    if code == 0 {
        return Availability::Available(info());
    }
    let reason = UNAVAILABLE_CODES
        .iter()
        .find(|(c, _)| *c == code)
        .map_or(UnavailableReason::Unavailable, |&(_, reason)| reason);
    Availability::Unavailable(reason)
}

/// The failure an error code of `ym_apple_auto_ask` stands for, `None` for a code the bridge should
/// never give.
pub(crate) fn failure_kind(code: i32) -> Option<FailureKind> {
    FAILURE_CODES
        .iter()
        .find(|(c, _)| *c == code)
        .map(|&(_, kind)| kind)
}

/// The width and height to hand to the bridge, or an error when the picture is not
/// `width × height × 3` bytes, is empty, or does not fit the C types.
pub(crate) fn picture_size(picture: &AdvisorPicture) -> Result<(u32, u32), AdvisorError> {
    let bad = || {
        AdvisorError::new(
            FailureKind::Other,
            format!(
                "a {}×{} picture with {} bytes",
                picture.width,
                picture.height,
                picture.rgb.len()
            ),
        )
    };
    let width = u32::try_from(picture.width).map_err(|_| bad())?;
    let height = u32::try_from(picture.height).map_err(|_| bad())?;
    let expected = picture
        .width
        .checked_mul(picture.height)
        .and_then(|pixels| pixels.checked_mul(3));
    if width == 0 || height == 0 || expected != Some(picture.rgb.len()) {
        return Err(bad());
    }
    Ok((width, height))
}

/// The text before the first NUL (all of it without one), as UTF-8 with invalid bytes replaced.
pub(crate) fn c_string(bytes: &[u8]) -> String {
    let end = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
    String::from_utf8_lossy(&bytes[..end]).into_owned()
}

/// One step of an answer as `i8`, or [`FailureKind::BadAnswer`] when it is outside its range.
fn step(field: &str, value: i32, (lo, hi): (i8, i8)) -> Result<i8, AdvisorError> {
    i8::try_from(value)
        .ok()
        .filter(|v| (lo..=hi).contains(v))
        .ok_or_else(|| {
            AdvisorError::new(
                FailureKind::BadAnswer,
                format!("the {field} step {value} is outside {lo} … {hi}"),
            )
        })
}

/// The answer the bridge wrote, checked ([`AdvisorAnswer::checked`]: the scene cut and trimmed).
pub(crate) fn answer_from(raw: &YmAppleAnswer) -> Result<AdvisorAnswer, AdvisorError> {
    let steps = AutoSteps {
        exposure: step("exposure", raw.exposure, EXPOSURE_STEPS)?,
        white: step("white", raw.white, OTHER_STEPS)?,
        contrast: step("contrast", raw.contrast, OTHER_STEPS)?,
        saturation: step("saturation", raw.saturation, OTHER_STEPS)?,
    };
    AdvisorAnswer {
        steps,
        scene: c_string(&raw.scene),
    }
    .checked()
}

/// What `ym_apple_auto_ask` returned: the answer for code 0, else the failure with Apple's message.
pub(crate) fn outcome(
    code: i32,
    raw: &YmAppleAnswer,
    message: &[u8],
) -> Result<AdvisorAnswer, AdvisorError> {
    if code == 0 {
        return answer_from(raw);
    }
    let text = c_string(message);
    Err(match failure_kind(code) {
        Some(kind) => AdvisorError::new(kind, text),
        None if text.is_empty() => {
            AdvisorError::new(FailureKind::Other, format!("unknown error code {code}"))
        }
        None => AdvisorError::new(
            FailureKind::Other,
            format!("unknown error code {code}: {text}"),
        ),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn raw(
        exposure: i32,
        white: i32,
        contrast: i32,
        saturation: i32,
        scene: &[u8],
    ) -> YmAppleAnswer {
        let mut answer = YmAppleAnswer::zeroed();
        answer.exposure = exposure;
        answer.white = white;
        answer.contrast = contrast;
        answer.saturation = saturation;
        answer.scene[..scene.len()].copy_from_slice(scene);
        answer
    }

    fn info() -> AdvisorInfo {
        AdvisorInfo {
            model: "AFM 3 Core".to_owned(),
            os: "27.0.1 (27A404)".to_owned(),
        }
    }

    #[test]
    fn availability_codes_map_to_the_reasons_of_section_1() {
        assert_eq!(
            availability_from_code(0, info),
            Availability::Available(info())
        );
        let expected = [
            (3, UnavailableReason::MacosTooOld),
            (4, UnavailableReason::DeviceNotEligible),
            (5, UnavailableReason::AppleIntelligenceOff),
            (6, UnavailableReason::ModelNotReady),
            (7, UnavailableReason::Unavailable),
            (8, UnavailableReason::NoImageInput),
        ];
        for (code, reason) in expected {
            assert_eq!(
                availability_from_code(code, || unreachable!("not available")),
                Availability::Unavailable(reason),
                "code {code}"
            );
        }
        for code in [-1, 1, 2, 9, 100, i32::MIN, i32::MAX] {
            assert_eq!(
                availability_from_code(code, || unreachable!("not available")),
                Availability::Unavailable(UnavailableReason::Unavailable),
                "code {code}"
            );
        }
    }

    #[test]
    fn error_codes_map_to_failure_kinds() {
        let expected = [
            (1, FailureKind::Unavailable),
            (2, FailureKind::RateLimited),
            (3, FailureKind::Timeout),
            (4, FailureKind::Refused),
            (5, FailureKind::Guardrail),
            (6, FailureKind::UnsupportedLanguage),
            (7, FailureKind::ContextTooLarge),
            (8, FailureKind::AssetsUnavailable),
            (9, FailureKind::ConcurrentRequests),
            (10, FailureKind::BadAnswer),
            (11, FailureKind::Other),
        ];
        for (code, kind) in expected {
            assert_eq!(failure_kind(code), Some(kind), "code {code}");
            let error =
                outcome(code, &YmAppleAnswer::zeroed(), b"Apple says no\0junk").unwrap_err();
            assert_eq!(error, AdvisorError::new(kind, "Apple says no"));
        }
        for code in [-1, 12, 99] {
            assert_eq!(failure_kind(code), None);
            let error = outcome(code, &YmAppleAnswer::zeroed(), b"\0").unwrap_err();
            assert_eq!(error.kind, FailureKind::Other);
            assert_eq!(error.message, format!("unknown error code {code}"));
        }
        let error = outcome(42, &YmAppleAnswer::zeroed(), b"odd\0").unwrap_err();
        assert_eq!(error.message, "unknown error code 42: odd");
    }

    #[test]
    fn an_answer_converts_its_steps_and_scene() {
        let answer = outcome(0, &raw(-3, 2, -1, 1, b"  Backlit portrait at dusk \0"), &[]).unwrap();
        assert_eq!(
            answer.steps,
            AutoSteps {
                exposure: -3,
                white: 2,
                contrast: -1,
                saturation: 1
            }
        );
        assert_eq!(answer.scene, "Backlit portrait at dusk");
        let keep = outcome(0, &raw(0, 0, 0, 0, b""), &[]).unwrap();
        assert_eq!(keep.steps, AutoSteps::default());
        assert_eq!(keep.scene, "");
        let top = outcome(0, &raw(3, -2, 2, -2, b"x"), &[]).unwrap();
        assert_eq!(top.steps.exposure, 3);
    }

    #[test]
    fn steps_outside_their_range_are_a_bad_answer() {
        for bad in [
            raw(4, 0, 0, 0, b""),
            raw(-4, 0, 0, 0, b""),
            raw(0, 3, 0, 0, b""),
            raw(0, 0, -3, 0, b""),
            raw(0, 0, 0, 3, b""),
            raw(300, 0, 0, 0, b""),
            raw(0, i32::MIN, 0, 0, b""),
            raw(0, 0, 0, i32::MAX, b""),
        ] {
            let error = outcome(0, &bad, &[]).unwrap_err();
            assert_eq!(error.kind, FailureKind::BadAnswer, "{bad:?}");
        }
    }

    #[test]
    fn the_scene_is_lossy_utf8_and_cut_to_80_characters() {
        // Invalid UTF-8 becomes U+FFFD; a scene without a NUL is read to the end of the buffer.
        let answer = outcome(0, &raw(0, 0, 0, 0, b"caf\xc3\xa9 \xff\0"), &[]).unwrap();
        assert_eq!(answer.scene, "café \u{fffd}");
        let full = raw(0, 0, 0, 0, &[b'a'; SCENE_BYTES]);
        assert_eq!(
            outcome(0, &full, &[]).unwrap().scene.chars().count(),
            ym_core::apple_auto::SCENE_MAX_CHARS
        );
        assert_eq!(c_string(b"no nul"), "no nul");
        assert_eq!(c_string(b""), "");
    }

    #[test]
    fn pictures_are_checked_before_they_reach_the_bridge() {
        let picture = |width: usize, height: usize, len: usize| AdvisorPicture {
            width,
            height,
            rgb: vec![0; len],
        };
        assert_eq!(
            picture_size(&picture(768, 512, 768 * 512 * 3)),
            Ok((768, 512))
        );
        assert_eq!(picture_size(&picture(1, 1, 3)), Ok((1, 1)));
        for bad in [
            picture(0, 512, 0),
            picture(768, 0, 0),
            picture(4, 3, 35),
            picture(4, 3, 37),
            picture(usize::MAX, 2, 6),
            picture(1 << 33, 1, 0),
        ] {
            let error = picture_size(&bad).unwrap_err();
            assert_eq!(error.kind, FailureKind::Other);
        }
    }
}
