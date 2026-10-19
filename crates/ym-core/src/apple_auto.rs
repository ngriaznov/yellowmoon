//! Apple Auto: on Macs where Apple Intelligence's on-device model can look at a photo, that model
//! sets four of Auto's values (`docs/19-apple-auto.md`). This module is the platform-free half: the
//! [`AutoAdvisor`] the app and `ym` hand to the core, the answer's steps and the table that turns them
//! into values, the burst rule, the states a photo goes through and a deterministic
//! [`FakeAdvisor`] for tests. The Swift bridge that talks to Apple's model is `ym-apple`.
//!
//! Nothing here persists (hard rule 4): answers live in the session's memory.

use std::fmt;

use serde::{Deserialize, Serialize};

/// The version of the instructions and the answer's schema the bridge sends (`docs/19` section 4).
/// Part of "version" under hard rule 8: change it whenever the words or the schema change.
pub const PROMPT_VERSION: u32 = 1;

/// The version of [`TABLE`] (`docs/19` section 5). Change it whenever a value of the table changes.
pub const TABLE_VERSION: u32 = 1;

/// The long edge of the picture the model sees, in pixels (`docs/19` section 3). Smaller photos are
/// not enlarged.
pub const PICTURE_LONG_EDGE_PX: usize = 768;

/// How long one request may take before it is abandoned, in milliseconds (`docs/19` section 6).
pub const REQUEST_TIMEOUT_MS: u32 = 30_000;

/// How many times a request that macOS asked to wait (`rate_limited`, `concurrent_requests`) is tried
/// again (`docs/19` section 6).
pub const RETRIES: u32 = 2;

/// The pause before a retry, in milliseconds.
pub const RETRY_PAUSE_MS: u64 = 2_000;

/// The longest scene description kept, in characters; it is shown only in reports.
pub const SCENE_MAX_CHARS: usize = 80;

/// The range of the exposure step: `muchDarker` (−3) to `muchBrighter` (+3).
pub const EXPOSURE_STEPS: (i8, i8) = (-3, 3);

/// The range of the white, contrast and saturation steps: `muchCooler`/`muchLess` (−2) to
/// `muchWarmer`/`muchMore` (+2).
pub const OTHER_STEPS: (i8, i8) = (-2, 2);

/// The values of the steps (`docs/19` section 5, table version [`TABLE_VERSION`]). The owner tunes
/// them by eye; they are never fitted to Apple's answers.
pub struct Table {
    /// EV added to the rule's exposure, for the steps −3 … +3.
    pub exposure_ev: [f64; 7],
    /// How much warmer the picture is made, in mired, for the steps −2 … +2: the user's white moves
    /// by minus this (a warmer picture is a higher Kelvin, a lower mired).
    pub warmer_mired: [f64; 5],
    /// The Auto contrast (−100 … 100), for the steps −2 … +2.
    pub contrast: [f64; 5],
    /// The Auto saturation (−100 … 100), for the steps −2 … +2.
    pub saturation: [f64; 5],
}

/// The table in use (version [`TABLE_VERSION`]).
pub const TABLE: Table = Table {
    exposure_ev: [-1.0, -0.6, -0.3, 0.0, 0.3, 0.6, 1.0],
    warmer_mired: [-20.0, -10.0, 0.0, 10.0, 20.0],
    contrast: [-30.0, -15.0, 0.0, 15.0, 30.0],
    saturation: [-30.0, -15.0, 0.0, 15.0, 30.0],
};

/// Why Apple Auto is not available on this computer (`docs/19` section 1), in the order the
/// conditions are checked.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UnavailableReason {
    /// Not a Mac.
    NotAMac,
    /// A Mac with an Intel processor.
    IntelMac,
    /// This binary was built without the macOS 27 SDK, so it has no bridge.
    BuiltWithoutSupport,
    /// macOS older than 27.
    MacosTooOld,
    /// The Mac cannot run Apple Intelligence.
    DeviceNotEligible,
    /// Apple Intelligence is off in System Settings.
    AppleIntelligenceOff,
    /// The model is still downloading, or not ready for another system reason.
    ModelNotReady,
    /// Any other unavailability Apple reports.
    Unavailable,
    /// The model on this Mac cannot take pictures.
    NoImageInput,
}

impl UnavailableReason {
    /// The reason in words, for `ym` and the logs ("Apple Auto is not available on this computer
    /// (<words>).").
    #[must_use]
    pub fn words(self) -> &'static str {
        match self {
            Self::NotAMac => "this is not a Mac",
            Self::IntelMac => "this Mac has an Intel processor",
            Self::BuiltWithoutSupport => "this build of Yellowmoon was made without it",
            Self::MacosTooOld => "it needs macOS 27 or later",
            Self::DeviceNotEligible => "this Mac can't run Apple Intelligence",
            Self::AppleIntelligenceOff => "Apple Intelligence is off in System Settings",
            Self::ModelNotReady => "Apple Intelligence is still downloading its model",
            Self::Unavailable => "Apple Intelligence is not available",
            Self::NoImageInput => "Apple's model on this Mac can't look at pictures",
        }
    }
}

impl fmt::Display for UnavailableReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.words())
    }
}

/// Why one request failed (`docs/19` sections 6 and 8). A failed photo uses the rule.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FailureKind {
    /// Apple Auto stopped being available between the check and the request.
    Unavailable,
    /// macOS asked the app to wait (after the retries).
    RateLimited,
    /// No answer within [`REQUEST_TIMEOUT_MS`].
    Timeout,
    /// The model refused.
    Refused,
    /// Apple's safety check stopped the request.
    Guardrail,
    /// The model does not support the language or locale.
    UnsupportedLanguage,
    /// The request did not fit the model's context.
    ContextTooLarge,
    /// The model's assets were not there.
    AssetsUnavailable,
    /// Another request was running (after the retries).
    ConcurrentRequests,
    /// The answer could not be read, or a step was out of its range.
    BadAnswer,
    /// Anything else; the message says what.
    Other,
}

impl FailureKind {
    /// The failure in words, for `ym info` and the logs.
    #[must_use]
    pub fn words(self) -> &'static str {
        match self {
            Self::Unavailable => "Apple Intelligence stopped being available",
            Self::RateLimited => "macOS asked Yellowmoon to wait",
            Self::Timeout => "no answer within 30 seconds",
            Self::Refused => "the model refused",
            Self::Guardrail => "Apple's safety check stopped it",
            Self::UnsupportedLanguage => "the model does not support this language",
            Self::ContextTooLarge => "the request was too large for the model",
            Self::AssetsUnavailable => "the model's files were not available",
            Self::ConcurrentRequests => "another request was running",
            Self::BadAnswer => "the answer could not be read",
            Self::Other => "an error",
        }
    }

    /// Whether the request is tried again ([`RETRIES`] times, [`RETRY_PAUSE_MS`] apart).
    #[must_use]
    pub fn retried(self) -> bool {
        matches!(self, Self::RateLimited | Self::ConcurrentRequests)
    }
}

/// Which model answers, for reports and `ym:AppleAuto`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AdvisorInfo {
    /// The model variant's display name (`AFM 3 Core`).
    pub model: String,
    /// The macOS version and build (`27.0.1 (27A404)`).
    pub os: String,
}

impl AdvisorInfo {
    /// The version string of `docs/19` section 7:
    /// `Apple Intelligence · AFM 3 Core · macOS 27.0.1 (27A404) · prompt 1 · table 1`.
    #[must_use]
    pub fn version_string(&self) -> String {
        format!(
            "Apple Intelligence · {} · macOS {} · prompt {PROMPT_VERSION} · table {TABLE_VERSION}",
            self.model, self.os
        )
    }
}

/// Whether an advisor can answer now.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Availability {
    /// It can; which model answers.
    Available(AdvisorInfo),
    /// It cannot, and why.
    Unavailable(UnavailableReason),
}

impl Availability {
    /// Whether it can answer.
    #[must_use]
    pub fn is_available(&self) -> bool {
        matches!(self, Self::Available(_))
    }
}

/// The picture the model sees (`docs/19` section 3): 8-bit sRGB, 3 bytes per pixel, rows top to
/// bottom, no padding.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AdvisorPicture {
    /// Width in pixels.
    pub width: usize,
    /// Height in pixels.
    pub height: usize,
    /// `width * height * 3` bytes.
    pub rgb: Vec<u8>,
}

/// The four steps of an answer (`docs/19` section 4): exposure −3 … +3, the others −2 … +2, 0 for
/// keep.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct AutoSteps {
    /// `muchDarker` (−3) … `muchBrighter` (+3).
    pub exposure: i8,
    /// `muchCooler` (−2) … `muchWarmer` (+2).
    pub white: i8,
    /// `muchLess` (−2) … `muchMore` (+2).
    pub contrast: i8,
    /// `muchLess` (−2) … `muchMore` (+2).
    pub saturation: i8,
}

impl AutoSteps {
    /// Whether every step is inside its range.
    #[must_use]
    pub fn in_range(&self) -> bool {
        let inside = |v: i8, (lo, hi): (i8, i8)| (lo..=hi).contains(&v);
        inside(self.exposure, EXPOSURE_STEPS)
            && inside(self.white, OTHER_STEPS)
            && inside(self.contrast, OTHER_STEPS)
            && inside(self.saturation, OTHER_STEPS)
    }

    /// The values of the steps through [`TABLE`], or `None` when a step is out of its range.
    #[must_use]
    pub fn values(&self) -> Option<AppleValues> {
        if !self.in_range() {
            return None;
        }
        let other = |v: i8| usize::from((v - OTHER_STEPS.0).unsigned_abs());
        let exposure = usize::from((self.exposure - EXPOSURE_STEPS.0).unsigned_abs());
        Some(AppleValues {
            exposure_ev: TABLE.exposure_ev[exposure],
            white_shift_mired: -TABLE.warmer_mired[other(self.white)],
            contrast: TABLE.contrast[other(self.contrast)],
            saturation: TABLE.saturation[other(self.saturation)],
        })
    }
}

/// What an answer does to Auto (`docs/19` section 5).
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
pub struct AppleValues {
    /// EV added to the rule's exposure.
    pub exposure_ev: f64,
    /// Mired added to the user's white in Auto (negative: a higher Kelvin, a warmer picture).
    pub white_shift_mired: f64,
    /// The Auto contrast.
    pub contrast: f64,
    /// The Auto saturation.
    pub saturation: f64,
}

/// One answer.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AdvisorAnswer {
    /// The four steps.
    pub steps: AutoSteps,
    /// What the photo shows, in a few words (at most [`SCENE_MAX_CHARS`] characters).
    pub scene: String,
}

impl AdvisorAnswer {
    /// The answer with its scene cut to [`SCENE_MAX_CHARS`] characters and trimmed, or a
    /// [`FailureKind::BadAnswer`] error when a step is out of its range.
    pub fn checked(mut self) -> Result<Self, AdvisorError> {
        if !self.steps.in_range() {
            return Err(AdvisorError::new(
                FailureKind::BadAnswer,
                format!("a step is out of its range: {:?}", self.steps),
            ));
        }
        self.scene = self
            .scene
            .trim()
            .chars()
            .take(SCENE_MAX_CHARS)
            .collect::<String>()
            .trim_end()
            .to_owned();
        Ok(self)
    }
}

/// A failed request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AdvisorError {
    /// What kind of failure.
    pub kind: FailureKind,
    /// Apple's message, when there is one.
    pub message: String,
}

impl fmt::Display for AdvisorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.kind.words())?;
        if !self.message.is_empty() {
            write!(f, ": {}", self.message)?;
        }
        Ok(())
    }
}

impl std::error::Error for AdvisorError {}

impl AdvisorError {
    /// A failure with a message.
    pub fn new(kind: FailureKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
        }
    }
}

/// Something that looks at a photo and answers the four steps: Apple's on-device model through
/// `ym-apple`, or [`FakeAdvisor`] in tests. The app and `ym` hand one to the core; every other
/// caller hands none, which is the same as one that is unavailable.
pub trait AutoAdvisor: Send + Sync {
    /// Whether it can answer now. Cheap; called at launch, when Preferences opens, when a session
    /// opens and before each export.
    fn availability(&self) -> Availability;

    /// Asks about one picture. Blocks until the answer, a failure or [`REQUEST_TIMEOUT_MS`]; called
    /// from one worker thread at a time, never from the main thread. Retries are the caller's.
    fn ask(&self, picture: &AdvisorPicture) -> Result<AdvisorAnswer, AdvisorError>;
}

/// Where a photo stands with Apple Auto (`docs/19` sections 6 and 12).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum AppleAuto {
    /// The preference is off (or no advisor was given): Auto is the rule.
    Off,
    /// Apple Auto is not available on this computer: Auto is the rule.
    Unavailable {
        /// Why.
        reason: UnavailableReason,
    },
    /// Asked, or waiting for its turn: Auto is the rule until the answer.
    Pending,
    /// Answered: Auto takes the answer's values.
    Answered {
        /// The answer.
        answer: AdvisorAnswer,
        /// Which model answered.
        info: AdvisorInfo,
    },
    /// The request failed: Auto is the rule.
    Failed {
        /// Why.
        kind: FailureKind,
        /// Apple's message, when there is one.
        message: String,
        /// Which model was asked.
        info: AdvisorInfo,
    },
}

impl AppleAuto {
    /// The steps of an answer, `None` in every other state.
    #[must_use]
    pub fn steps(&self) -> Option<AutoSteps> {
        match self {
            Self::Answered { answer, .. } => Some(answer.steps),
            _ => None,
        }
    }

    /// Whether the state will not change without a new request (answered, failed, off or
    /// unavailable): what an export waits for.
    #[must_use]
    pub fn is_settled(&self) -> bool {
        !matches!(self, Self::Pending)
    }

    /// Whether Apple Auto acts on the photo: pending, answered or failed (not off or unavailable).
    /// Exactly when the Automatic card shows its Contrast and Saturation rows (`docs/11`), and when
    /// Re-run puts contrast and saturation back to Auto (`docs/19` section 5).
    #[must_use]
    pub fn acts(&self) -> bool {
        matches!(
            self,
            Self::Pending | Self::Answered { .. } | Self::Failed { .. }
        )
    }
}

impl AppleAuto {
    /// The `ym:AppleAuto` property of an export (`docs/19` section 9): compact JSON with sorted keys,
    /// written only when Apple Auto was on for the photo, so `None` for [`AppleAuto::Off`],
    /// [`AppleAuto::Unavailable`] and [`AppleAuto::Pending`] (an export waits for the answer; only a
    /// photo whose session closed under its export can still be pending).
    ///
    /// ```text
    /// {"model":"AFM 3 Core","os":"27.0.1 (27A404)","prompt":1,"result":"answered","scene":"…",
    ///  "steps":{"contrast":1,"exposure":1,"saturation":0,"white":-1},"table":1}
    /// {"model":"…","os":"…","prompt":1,"reason":"rate_limited","result":"failed","table":1}
    /// ```
    #[must_use]
    pub fn xmp_json(&self) -> Option<String> {
        use serde_json::{Value, json};
        use std::collections::BTreeMap;
        let mut map: BTreeMap<&'static str, Value> = BTreeMap::new();
        let info = match self {
            Self::Off | Self::Unavailable { .. } | Self::Pending => return None,
            Self::Answered { answer, info } => {
                map.insert("result", json!("answered"));
                map.insert("scene", json!(answer.scene));
                let steps: BTreeMap<&'static str, i8> = [
                    ("contrast", answer.steps.contrast),
                    ("exposure", answer.steps.exposure),
                    ("saturation", answer.steps.saturation),
                    ("white", answer.steps.white),
                ]
                .into_iter()
                .collect();
                map.insert("steps", json!(steps));
                info
            }
            Self::Failed { kind, info, .. } => {
                map.insert("result", json!("failed"));
                map.insert("reason", serde_json::to_value(kind).ok()?);
                info
            }
        };
        map.insert("model", json!(info.model));
        map.insert("os", json!(info.os));
        map.insert("prompt", json!(PROMPT_VERSION));
        map.insert("table", json!(TABLE_VERSION));
        serde_json::to_string(&map).ok()
    }
}

/// How a run decides whether Apple Auto is used (`docs/19` sections 2 and 8): the app follows the
/// preference; `ym --apple-auto on|off` overrides it for one run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AppleAutoMode {
    /// The `apple_auto` preference decides (the app, and `ym` without `--apple-auto`).
    #[default]
    FromPreference,
    /// On for this run, whatever the preference says (still only where it is available).
    On,
    /// Off for this run: Auto is the rule.
    Off,
}

/// Whether Apple Auto acts for a session now, and if not, why (`docs/19` sections 1 and 2).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Activity {
    /// It acts: photos are asked, answers change Auto. Which model answers.
    Active(AdvisorInfo),
    /// Off: no advisor was given, `--apple-auto off`, or the preference is off.
    Off,
    /// Not available on this computer.
    Unavailable(UnavailableReason),
}

impl Activity {
    /// Decides the activity. In this order: no advisor is off (tests, the golden tool); a run with
    /// `--apple-auto off` is off; where Apple Auto is not available it is unavailable whatever the
    /// preference says (the preference is kept and ignored there, `docs/19` section 2); the
    /// preference off is off (unless the run says `on`); else it acts. `availability` is the last
    /// one read; none read yet counts as unavailable.
    #[must_use]
    pub fn decide(
        has_advisor: bool,
        mode: AppleAutoMode,
        preference: bool,
        availability: Option<&Availability>,
    ) -> Self {
        if !has_advisor || mode == AppleAutoMode::Off {
            return Self::Off;
        }
        match availability {
            None => Self::Unavailable(UnavailableReason::Unavailable),
            Some(Availability::Unavailable(reason)) => Self::Unavailable(*reason),
            Some(Availability::Available(info)) => {
                if mode == AppleAutoMode::FromPreference && !preference {
                    Self::Off
                } else {
                    Self::Active(info.clone())
                }
            }
        }
    }

    /// Whether Apple Auto acts.
    #[must_use]
    pub fn is_active(&self) -> bool {
        matches!(self, Self::Active(_))
    }

    /// The state of a photo under this activity: its own result while Apple Auto acts (pending until
    /// there is one), else off or unavailable. `result` is [`AppleAuto::Answered`] or
    /// [`AppleAuto::Failed`] when the photo was asked; answers are kept while Apple Auto is off.
    #[must_use]
    pub fn photo_state(&self, result: Option<&AppleAuto>) -> AppleAuto {
        match self {
            Self::Off => AppleAuto::Off,
            Self::Unavailable(reason) => AppleAuto::Unavailable { reason: *reason },
            Self::Active(_) => result.cloned().unwrap_or(AppleAuto::Pending),
        }
    }
}

/// What the app's Preferences and `ym doctor` show about Apple Auto (`docs/19` sections 1, 2 and 8).
/// JSON, snake_case: `{ "available": true, "reason": null, "info": { "model": "…", "os": "…" },
/// "preference": true, "mode": "from_preference", "active": true, "prompt": 1, "table": 1 }`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AppleAutoStatus {
    /// Whether Apple Auto is available on this computer now (read again for this answer). False when
    /// the core was given no advisor.
    pub available: bool,
    /// Why it is not available; `None` when it is, and when the core was given no advisor.
    pub reason: Option<UnavailableReason>,
    /// Which model answers, when it is available.
    pub info: Option<AdvisorInfo>,
    /// The `apple_auto` preference.
    pub preference: bool,
    /// How this run decides ([`AppleAutoMode`]).
    pub mode: AppleAutoMode,
    /// Whether Apple Auto acts for the session now ([`Activity::decide`]).
    pub active: bool,
    /// [`PROMPT_VERSION`].
    pub prompt: u32,
    /// [`TABLE_VERSION`].
    pub table: u32,
}

/// The steps a burst shares (`docs/19` section 5): field by field, the median of the answered
/// frames' steps, the lower of the two middle values for an even count. Frames without an answer
/// (`None`) are left out; `None` when no frame answered.
#[must_use]
pub fn share_burst_steps(frames: &[Option<AutoSteps>]) -> Option<AutoSteps> {
    let answered: Vec<AutoSteps> = frames.iter().flatten().copied().collect();
    if answered.is_empty() {
        return None;
    }
    let median = |field: fn(&AutoSteps) -> i8| {
        let mut values: Vec<i8> = answered.iter().map(field).collect();
        values.sort_unstable();
        values[(values.len() - 1) / 2]
    };
    Some(AutoSteps {
        exposure: median(|s| s.exposure),
        white: median(|s| s.white),
        contrast: median(|s| s.contrast),
        saturation: median(|s| s.saturation),
    })
}

/// A deterministic advisor for tests (`docs/19` section 12), always available. From the picture's
/// mean sRGB luma `L` (0…1, Rec.709 weights on the 8-bit values) and its standard deviation `S`:
/// exposure +1 when `L` < 0.30, −1 when `L` > 0.65, else 0; contrast +1 when `S` < 0.15, else 0;
/// white 0; saturation 0; scene "Test picture". The app and `ym` use it when
/// `YELLOWMOON_TEST_APPLE_AUTO=fake` is set.
#[derive(Debug, Clone, Copy, Default)]
pub struct FakeAdvisor;

impl FakeAdvisor {
    /// The model name it reports.
    pub const MODEL: &'static str = "Test advisor";
    /// The operating system it reports.
    pub const OS: &'static str = "test";

    /// Its [`AdvisorInfo`].
    #[must_use]
    pub fn info() -> AdvisorInfo {
        AdvisorInfo {
            model: Self::MODEL.to_owned(),
            os: Self::OS.to_owned(),
        }
    }
}

impl AutoAdvisor for FakeAdvisor {
    fn availability(&self) -> Availability {
        Availability::Available(Self::info())
    }

    fn ask(&self, picture: &AdvisorPicture) -> Result<AdvisorAnswer, AdvisorError> {
        let pixels = picture.width * picture.height;
        if pixels == 0 || picture.rgb.len() != pixels * 3 {
            return Err(AdvisorError::new(
                FailureKind::Other,
                format!(
                    "a {}×{} picture with {} bytes",
                    picture.width,
                    picture.height,
                    picture.rgb.len()
                ),
            ));
        }
        let lumas = picture.rgb.chunks_exact(3).map(|p| {
            (0.2126 * f64::from(p[0]) + 0.7152 * f64::from(p[1]) + 0.0722 * f64::from(p[2])) / 255.0
        });
        let n = pixels as f64;
        let (sum, sum_sq) = lumas.fold((0.0, 0.0), |(s, q), l| (s + l, q + l * l));
        let mean = sum / n;
        let sd = (sum_sq / n - mean * mean).max(0.0).sqrt();
        let exposure = if mean < 0.30 {
            1
        } else if mean > 0.65 {
            -1
        } else {
            0
        };
        Ok(AdvisorAnswer {
            steps: AutoSteps {
                exposure,
                white: 0,
                contrast: i8::from(sd < 0.15),
                saturation: 0,
            },
            scene: "Test picture".to_owned(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn steps(exposure: i8, white: i8, contrast: i8, saturation: i8) -> AutoSteps {
        AutoSteps {
            exposure,
            white,
            contrast,
            saturation,
        }
    }

    #[test]
    fn keep_is_the_rule() {
        assert_eq!(AutoSteps::default().values(), Some(AppleValues::default()));
    }

    #[test]
    fn the_table_maps_every_step() {
        let v = steps(-3, -2, -2, -2).values().unwrap();
        assert_eq!(
            (v.exposure_ev, v.white_shift_mired, v.contrast, v.saturation),
            (-1.0, 20.0, -30.0, -30.0)
        );
        let v = steps(3, 2, 2, 2).values().unwrap();
        assert_eq!(
            (v.exposure_ev, v.white_shift_mired, v.contrast, v.saturation),
            (1.0, -20.0, 30.0, 30.0)
        );
        let v = steps(1, 1, -1, 1).values().unwrap();
        assert_eq!(
            (v.exposure_ev, v.white_shift_mired, v.contrast, v.saturation),
            (0.3, -10.0, -15.0, 15.0)
        );
    }

    #[test]
    fn warmer_is_a_lower_mired() {
        // A warmer picture balances the camera for a bluer light: a higher Kelvin, a lower mired.
        assert!(steps(0, 1, 0, 0).values().unwrap().white_shift_mired < 0.0);
        assert!(steps(0, -1, 0, 0).values().unwrap().white_shift_mired > 0.0);
    }

    #[test]
    fn steps_out_of_range_have_no_values() {
        for s in [
            steps(4, 0, 0, 0),
            steps(-4, 0, 0, 0),
            steps(0, 3, 0, 0),
            steps(0, 0, -3, 0),
            steps(0, 0, 0, 3),
        ] {
            assert!(!s.in_range());
            assert_eq!(s.values(), None);
        }
    }

    #[test]
    fn a_checked_answer_trims_the_scene_and_refuses_bad_steps() {
        let long = format!("  {}  ", "x".repeat(200));
        let a = AdvisorAnswer {
            steps: AutoSteps::default(),
            scene: long,
        }
        .checked()
        .unwrap();
        assert_eq!(a.scene.chars().count(), SCENE_MAX_CHARS);
        let bad = AdvisorAnswer {
            steps: steps(0, 0, 5, 0),
            scene: String::new(),
        }
        .checked()
        .unwrap_err();
        assert_eq!(bad.kind, FailureKind::BadAnswer);
    }

    #[test]
    fn burst_steps_are_field_medians_of_the_answered_frames() {
        assert_eq!(share_burst_steps(&[]), None);
        assert_eq!(share_burst_steps(&[None, None]), None);
        let a = steps(1, 0, 2, -1);
        assert_eq!(share_burst_steps(&[Some(a), None]), Some(a));
        let frames = [
            Some(steps(1, -1, 0, 0)),
            Some(steps(3, 1, 1, 0)),
            None,
            Some(steps(2, 0, -2, 1)),
        ];
        assert_eq!(share_burst_steps(&frames), Some(steps(2, 0, 0, 0)));
        // Even count: the lower of the two middle values.
        let even = [Some(steps(0, 0, 0, 0)), Some(steps(2, 2, 2, 2))];
        assert_eq!(share_burst_steps(&even), Some(steps(0, 0, 0, 0)));
    }

    fn flat(width: usize, height: usize, value: u8) -> AdvisorPicture {
        AdvisorPicture {
            width,
            height,
            rgb: vec![value; width * height * 3],
        }
    }

    #[test]
    fn the_fake_reads_brightness_and_spread() {
        let fake = FakeAdvisor;
        assert!(fake.availability().is_available());
        assert_eq!(fake.ask(&flat(4, 3, 40)).unwrap().steps, steps(1, 0, 1, 0));
        assert_eq!(
            fake.ask(&flat(4, 3, 220)).unwrap().steps,
            steps(-1, 0, 1, 0)
        );
        assert_eq!(fake.ask(&flat(4, 3, 128)).unwrap().steps, steps(0, 0, 1, 0));
        // Half black, half white: mean 0.5, standard deviation 0.5.
        let mut rgb = vec![0u8; 12];
        rgb[6..].fill(255);
        let split = AdvisorPicture {
            width: 2,
            height: 2,
            rgb,
        };
        assert_eq!(fake.ask(&split).unwrap().steps, steps(0, 0, 0, 0));
        assert_eq!(fake.ask(&flat(4, 3, 40)), fake.ask(&flat(4, 3, 40)));
    }

    #[test]
    fn the_fake_refuses_a_malformed_picture() {
        let bad = AdvisorPicture {
            width: 4,
            height: 3,
            rgb: vec![0; 5],
        };
        assert_eq!(FakeAdvisor.ask(&bad).unwrap_err().kind, FailureKind::Other);
    }

    #[test]
    fn the_version_string_names_model_os_prompt_and_table() {
        let info = AdvisorInfo {
            model: "AFM 3 Core".to_owned(),
            os: "27.0.1 (27A404)".to_owned(),
        };
        assert_eq!(
            info.version_string(),
            "Apple Intelligence · AFM 3 Core · macOS 27.0.1 (27A404) · prompt 1 · table 1"
        );
    }

    #[test]
    fn states_serialise_with_a_state_tag() {
        let v = serde_json::to_value(AppleAuto::Unavailable {
            reason: UnavailableReason::AppleIntelligenceOff,
        })
        .unwrap();
        assert_eq!(
            v,
            serde_json::json!({ "state": "unavailable", "reason": "apple_intelligence_off" })
        );
        assert!(!AppleAuto::Pending.is_settled());
        assert!(AppleAuto::Off.is_settled());
        assert_eq!(AppleAuto::Pending.steps(), None);
    }

    fn info() -> AdvisorInfo {
        AdvisorInfo {
            model: "AFM 3 Core".to_owned(),
            os: "27.0.1 (27A404)".to_owned(),
        }
    }

    #[test]
    fn the_xmp_property_is_compact_sorted_json_for_an_answer_or_a_failure_only() {
        let answered = AppleAuto::Answered {
            answer: AdvisorAnswer {
                steps: steps(1, -1, 1, 0),
                scene: "Backlit portrait \"at\" dusk".to_owned(),
            },
            info: info(),
        };
        assert_eq!(
            answered.xmp_json().unwrap(),
            r#"{"model":"AFM 3 Core","os":"27.0.1 (27A404)","prompt":1,"result":"answered","scene":"Backlit portrait \"at\" dusk","steps":{"contrast":1,"exposure":1,"saturation":0,"white":-1},"table":1}"#
        );
        let failed = AppleAuto::Failed {
            kind: FailureKind::RateLimited,
            message: "busy".to_owned(),
            info: info(),
        };
        assert_eq!(
            failed.xmp_json().unwrap(),
            r#"{"model":"AFM 3 Core","os":"27.0.1 (27A404)","prompt":1,"reason":"rate_limited","result":"failed","table":1}"#
        );
        for none in [
            AppleAuto::Off,
            AppleAuto::Pending,
            AppleAuto::Unavailable {
                reason: UnavailableReason::NotAMac,
            },
        ] {
            assert_eq!(none.xmp_json(), None);
        }
    }

    #[test]
    fn the_activity_follows_the_advisor_the_mode_the_availability_and_the_preference() {
        let available = Availability::Available(info());
        let off_here = Availability::Unavailable(UnavailableReason::AppleIntelligenceOff);
        use AppleAutoMode::{FromPreference, Off, On};
        // No advisor, or a run that says off: off.
        assert_eq!(
            Activity::decide(false, On, true, Some(&available)),
            Activity::Off
        );
        assert_eq!(
            Activity::decide(true, Off, true, Some(&available)),
            Activity::Off
        );
        // Not available: unavailable, whatever the preference.
        for (mode, pref) in [(FromPreference, true), (FromPreference, false), (On, false)] {
            assert_eq!(
                Activity::decide(true, mode, pref, Some(&off_here)),
                Activity::Unavailable(UnavailableReason::AppleIntelligenceOff)
            );
        }
        assert_eq!(
            Activity::decide(true, FromPreference, true, None),
            Activity::Unavailable(UnavailableReason::Unavailable)
        );
        // Available: the preference decides, unless the run says on.
        assert!(Activity::decide(true, FromPreference, true, Some(&available)).is_active());
        assert_eq!(
            Activity::decide(true, FromPreference, false, Some(&available)),
            Activity::Off
        );
        assert!(Activity::decide(true, On, false, Some(&available)).is_active());
        // A photo's state: its result while active, pending without one; off or unavailable else.
        let active = Activity::Active(info());
        assert_eq!(active.photo_state(None), AppleAuto::Pending);
        let failed = AppleAuto::Failed {
            kind: FailureKind::Timeout,
            message: String::new(),
            info: info(),
        };
        assert_eq!(active.photo_state(Some(&failed)), failed);
        assert_eq!(Activity::Off.photo_state(Some(&failed)), AppleAuto::Off);
        assert_eq!(
            serde_json::to_value(AppleAutoMode::FromPreference).unwrap(),
            serde_json::json!("from_preference")
        );
    }

    #[test]
    fn only_waiting_failures_are_retried() {
        assert!(FailureKind::RateLimited.retried());
        assert!(FailureKind::ConcurrentRequests.retried());
        assert!(!FailureKind::Timeout.retried());
        assert!(!FailureKind::Refused.retried());
    }
}
