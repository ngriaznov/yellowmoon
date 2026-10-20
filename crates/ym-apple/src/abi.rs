//! The C interface of the Swift bridge (`swift/ym_apple.h`, `docs/19-apple-auto.md` section 11)
//! without the calls: the answer's layout, the meaning of the codes and the conversions. Kept apart
//! from `bridge.rs` so that it builds and is tested on every platform.

use std::ffi::CString;

use ym_core::apple_auto::{
    AdvisorAnswer, AdvisorError, AdvisorInfo, AdvisorPicture, AutoSteps, Availability,
    EXPOSURE_STEPS, FailureKind, HighlightColor, OTHER_STEPS, STRENGTHS, ShadowColor,
    UnavailableReason,
};

/// The bytes of [`YmAppleAnswer::scene`], the NUL included (`YM_APPLE_SCENE_BYTES`).
pub(crate) const SCENE_BYTES: usize = 128;

/// The bytes of [`YmAppleAnswer::plan`], the NUL included (`YM_APPLE_PLAN_BYTES`).
pub(crate) const PLAN_BYTES: usize = 256;

/// The code of a field the model left empty (`YM_APPLE_UNSET`): nothing is set for that control.
pub(crate) const UNSET: i32 = i32::MIN;

/// The buffer for the model's name and the macOS version.
pub(crate) const INFO_BYTES: usize = 256;

/// The buffer for Apple's error message.
pub(crate) const MESSAGE_BYTES: usize = 1024;

/// `YmAppleAnswer` of `ym_apple.h`: the codes of the twelve values ([`UNSET`] for a field the model
/// left empty), and the scene and the plan, NUL-terminated UTF-8.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct YmAppleAnswer {
    /// `muchDarker` −3 … `muchBrighter` +3.
    pub(crate) exposure: i32,
    /// `muchDarker` −2 … `muchBrighter` +2.
    pub(crate) highlights: i32,
    /// `muchDarker` −2 … `muchBrighter` +2.
    pub(crate) shadows: i32,
    /// `muchDeeper` −2 … `muchLifted` +2.
    pub(crate) blacks: i32,
    /// `muchLess` −2 … `muchMore` +2.
    pub(crate) contrast: i32,
    /// The model's warmth: `muchCooler` −2 … `muchWarmer` +2.
    pub(crate) white: i32,
    /// `muchGreener` −2 … `muchMoreMagenta` +2.
    pub(crate) tint: i32,
    /// `muchLess` −2 … `muchMore` +2.
    pub(crate) saturation: i32,
    /// none 0, teal 1, blue 2, green 3, amber 4, magenta 5.
    pub(crate) shadow_color: i32,
    /// subtle 1, medium 2, strong 3.
    pub(crate) shadow_color_strength: i32,
    /// none 0, warm 1, gold 2, cool 3, pink 4.
    pub(crate) highlight_color: i32,
    /// subtle 1, medium 2, strong 3.
    pub(crate) highlight_color_strength: i32,
    /// What the photo shows, NUL-terminated (`char scene[128]`).
    pub(crate) scene: [u8; SCENE_BYTES],
    /// The look the model gives the photo, NUL-terminated (`char plan[256]`).
    pub(crate) plan: [u8; PLAN_BYTES],
}

// The layout the Swift side writes to (ym_apple.h asserts the same).
const _: () = assert!(size_of::<YmAppleAnswer>() == 432);
const _: () = assert!(align_of::<YmAppleAnswer>() == 4);
const _: () = assert!(std::mem::offset_of!(YmAppleAnswer, highlights) == 4);
const _: () = assert!(std::mem::offset_of!(YmAppleAnswer, white) == 20);
const _: () = assert!(std::mem::offset_of!(YmAppleAnswer, highlight_color_strength) == 44);
const _: () = assert!(std::mem::offset_of!(YmAppleAnswer, scene) == 48);
const _: () = assert!(std::mem::offset_of!(YmAppleAnswer, plan) == 176);

impl YmAppleAnswer {
    /// Every field left empty ([`UNSET`]), the scene and the plan empty: what the Swift side
    /// writes before it asks.
    pub(crate) const fn unset() -> Self {
        Self {
            exposure: UNSET,
            highlights: UNSET,
            shadows: UNSET,
            blacks: UNSET,
            contrast: UNSET,
            white: UNSET,
            tint: UNSET,
            saturation: UNSET,
            shadow_color: UNSET,
            shadow_color_strength: UNSET,
            highlight_color: UNSET,
            highlight_color_strength: UNSET,
            scene: [0; SCENE_BYTES],
            plan: [0; PLAN_BYTES],
        }
    }
}

/// The instructions as the C string the bridge passes: the text with any NUL replaced by a space
/// (a NUL would end it early).
pub(crate) fn instructions_c_string(text: &str) -> CString {
    let clean: Vec<u8> = text
        .bytes()
        .map(|b| if b == 0 { b' ' } else { b })
        .collect();
    // No NUL is left, so this cannot fail; an empty string is the fallback all the same.
    CString::new(clean).unwrap_or_default()
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

/// One numeric step of an answer: `None` for [`UNSET`], else the step as `i8`, or
/// [`FailureKind::BadAnswer`] when it is outside its range.
fn step(field: &str, value: i32, (lo, hi): (i8, i8)) -> Result<Option<i8>, AdvisorError> {
    if value == UNSET {
        return Ok(None);
    }
    i8::try_from(value)
        .ok()
        .filter(|v| (lo..=hi).contains(v))
        .map(Some)
        .ok_or_else(|| {
            AdvisorError::new(
                FailureKind::BadAnswer,
                format!("the {field} step {value} is outside {lo} … {hi}"),
            )
        })
}

/// A colour of an answer: `None` for [`UNSET`], else the colour of its code, or
/// [`FailureKind::BadAnswer`] for a code that is no colour.
fn color<C>(
    field: &str,
    value: i32,
    from_code: fn(i32) -> Option<C>,
) -> Result<Option<C>, AdvisorError> {
    if value == UNSET {
        return Ok(None);
    }
    from_code(value).map(Some).ok_or_else(|| {
        AdvisorError::new(
            FailureKind::BadAnswer,
            format!("the {field} code {value} is no colour"),
        )
    })
}

/// The answer the bridge wrote, checked ([`AdvisorAnswer::checked`]: the words cut and trimmed, a
/// strength kept only with its colour, at least one value set).
pub(crate) fn answer_from(raw: &YmAppleAnswer) -> Result<AdvisorAnswer, AdvisorError> {
    let steps = AutoSteps {
        exposure: step("exposure", raw.exposure, EXPOSURE_STEPS)?,
        highlights: step("highlights", raw.highlights, OTHER_STEPS)?,
        shadows: step("shadows", raw.shadows, OTHER_STEPS)?,
        blacks: step("blacks", raw.blacks, OTHER_STEPS)?,
        contrast: step("contrast", raw.contrast, OTHER_STEPS)?,
        white: step("warmth", raw.white, OTHER_STEPS)?,
        tint: step("tint", raw.tint, OTHER_STEPS)?,
        saturation: step("saturation", raw.saturation, OTHER_STEPS)?,
        shadow_color: color("shadow colour", raw.shadow_color, ShadowColor::from_code)?,
        shadow_color_strength: step(
            "shadow colour strength",
            raw.shadow_color_strength,
            STRENGTHS,
        )?,
        highlight_color: color(
            "highlight colour",
            raw.highlight_color,
            HighlightColor::from_code,
        )?,
        highlight_color_strength: step(
            "highlight colour strength",
            raw.highlight_color_strength,
            STRENGTHS,
        )?,
    };
    AdvisorAnswer {
        steps,
        scene: c_string(&raw.scene),
        plan: c_string(&raw.plan),
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

    /// An answer with the four steps of prompt version 1 set and the rest left empty.
    fn raw(
        exposure: i32,
        white: i32,
        contrast: i32,
        saturation: i32,
        scene: &[u8],
    ) -> YmAppleAnswer {
        let mut answer = YmAppleAnswer::unset();
        answer.exposure = exposure;
        answer.white = white;
        answer.contrast = contrast;
        answer.saturation = saturation;
        answer.scene[..scene.len()].copy_from_slice(scene);
        answer
    }

    fn info() -> AdvisorInfo {
        AdvisorInfo::new("AFM 3 Core", "27.0.1 (27A404)")
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
            let error = outcome(code, &YmAppleAnswer::unset(), b"Apple says no\0junk").unwrap_err();
            assert_eq!(error, AdvisorError::new(kind, "Apple says no"));
        }
        for code in [-1, 12, 99] {
            assert_eq!(failure_kind(code), None);
            let error = outcome(code, &YmAppleAnswer::unset(), b"\0").unwrap_err();
            assert_eq!(error.kind, FailureKind::Other);
            assert_eq!(error.message, format!("unknown error code {code}"));
        }
        let error = outcome(42, &YmAppleAnswer::unset(), b"odd\0").unwrap_err();
        assert_eq!(error.message, "unknown error code 42: odd");
    }

    #[test]
    fn an_answer_converts_its_steps_scene_and_plan() {
        let mut full = raw(-3, 2, -1, 1, b"  Backlit portrait at dusk \0");
        full.highlights = -2;
        full.shadows = 1;
        full.blacks = 2;
        full.tint = -1;
        full.shadow_color = 1;
        full.shadow_color_strength = 3;
        full.highlight_color = 2;
        full.highlight_color_strength = 1;
        full.plan[..20].copy_from_slice(b"Teal shadows, gold. ");
        let answer = outcome(0, &full, &[]).unwrap();
        assert_eq!(
            answer.steps,
            AutoSteps {
                exposure: Some(-3),
                highlights: Some(-2),
                shadows: Some(1),
                blacks: Some(2),
                contrast: Some(-1),
                white: Some(2),
                tint: Some(-1),
                saturation: Some(1),
                shadow_color: Some(ShadowColor::Teal),
                shadow_color_strength: Some(3),
                highlight_color: Some(HighlightColor::Gold),
                highlight_color_strength: Some(1),
            }
        );
        assert_eq!(answer.scene, "Backlit portrait at dusk");
        assert_eq!(answer.plan, "Teal shadows, gold.");
        let top = outcome(0, &raw(3, -2, 2, -2, b"x"), &[]).unwrap();
        assert_eq!(top.steps.exposure, Some(3));
    }

    #[test]
    fn a_field_left_empty_is_unset_and_an_answer_with_none_set_fails() {
        // The sentinel of ym_apple.h.
        assert_eq!(UNSET, i32::MIN);
        let mut only = YmAppleAnswer::unset();
        only.contrast = 1;
        let answer = outcome(0, &only, &[]).unwrap();
        assert_eq!(
            answer.steps,
            AutoSteps {
                contrast: Some(1),
                ..AutoSteps::default()
            }
        );
        // A colour without its strength is kept and sets nothing; a strength alone is dropped.
        let mut colour = YmAppleAnswer::unset();
        colour.shadow_color = 2;
        colour.highlight_color_strength = 2;
        let answer = outcome(0, &colour, &[]).unwrap();
        assert_eq!(answer.steps.shadow_color, Some(ShadowColor::Blue));
        assert_eq!(answer.steps.shadow_color_strength, None);
        assert_eq!(answer.steps.highlight_color_strength, None);
        // Nothing filled: a bad answer.
        let error = outcome(0, &YmAppleAnswer::unset(), &[]).unwrap_err();
        assert_eq!(error.kind, FailureKind::BadAnswer);
        let mut strength_only = YmAppleAnswer::unset();
        strength_only.shadow_color_strength = 1;
        let error = outcome(0, &strength_only, &[]).unwrap_err();
        assert_eq!(error.kind, FailureKind::BadAnswer);
    }

    #[test]
    fn steps_outside_their_range_are_a_bad_answer() {
        let with = |f: fn(&mut YmAppleAnswer)| {
            let mut a = raw(0, 0, 0, 0, b"");
            f(&mut a);
            a
        };
        for bad in [
            raw(4, 0, 0, 0, b""),
            raw(-4, 0, 0, 0, b""),
            raw(0, 3, 0, 0, b""),
            raw(0, 0, -3, 0, b""),
            raw(0, 0, 0, 3, b""),
            raw(300, 0, 0, 0, b""),
            raw(0, i32::MIN + 1, 0, 0, b""),
            raw(0, 0, 0, i32::MAX, b""),
            with(|a| a.highlights = 3),
            with(|a| a.blacks = -3),
            with(|a| a.tint = 7),
            with(|a| a.shadow_color = 6),
            with(|a| a.highlight_color = 5),
            with(|a| a.highlight_color = -1),
            with(|a| {
                a.shadow_color = 1;
                a.shadow_color_strength = 0;
            }),
            with(|a| {
                a.highlight_color = 1;
                a.highlight_color_strength = 4;
            }),
        ] {
            let error = outcome(0, &bad, &[]).unwrap_err();
            assert_eq!(error.kind, FailureKind::BadAnswer, "{bad:?}");
        }
    }

    #[test]
    fn the_words_are_lossy_utf8_and_cut_to_their_lengths() {
        // Invalid UTF-8 becomes U+FFFD; a scene without a NUL is read to the end of the buffer.
        let answer = outcome(0, &raw(0, 0, 0, 0, b"caf\xc3\xa9 \xff\0"), &[]).unwrap();
        assert_eq!(answer.scene, "café \u{fffd}");
        let full = raw(0, 0, 0, 0, &[b'a'; SCENE_BYTES]);
        assert_eq!(
            outcome(0, &full, &[]).unwrap().scene.chars().count(),
            ym_core::apple_auto::SCENE_MAX_CHARS
        );
        let mut long_plan = raw(0, 0, 0, 0, b"");
        long_plan.plan = [b'p'; PLAN_BYTES];
        assert_eq!(
            outcome(0, &long_plan, &[]).unwrap().plan.chars().count(),
            ym_core::apple_auto::PLAN_MAX_CHARS
        );
        assert_eq!(c_string(b"no nul"), "no nul");
        assert_eq!(c_string(b""), "");
    }

    #[test]
    fn the_instructions_reach_the_bridge_as_one_c_string() {
        assert_eq!(instructions_c_string("Grade it.").as_bytes(), b"Grade it.");
        // A NUL would end the text early: it becomes a space.
        assert_eq!(instructions_c_string("Grade\0it.").as_bytes(), b"Grade it.");
        assert_eq!(
            instructions_c_string("Gélé, très doux").to_str().unwrap(),
            "Gélé, très doux"
        );
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
